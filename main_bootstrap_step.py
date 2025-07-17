# main_bootstrap_step.py
import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import timm.optim.optim_factory as optim_factory
import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler

# Import all necessary models and engines
import models_mae
import models_bmae
from engine_pretrain import train_one_epoch as train_one_epoch_mae
from engine_bmae_k_pretrain import train_one_epoch as train_one_epoch_bmae_k


def get_args_parser():
    parser = argparse.ArgumentParser('MAE Bootstrap Step Pre-training', add_help=False)
    # General arguments
    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--accum_iter', default=1, type=int)
    parser.add_argument('--model', default='deit_tiny_patch4', type=str, help='Model name prefix')
    parser.add_argument('--input_size', default=32, type=int)
    parser.add_argument('--mask_ratio', default=0.75, type=float)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--blr', type=float, default=1.5e-4)
    parser.add_argument('--min_lr', type=float, default=0.)
    parser.add_argument('--warmup_epochs', type=int, default=10)
    parser.add_argument('--data_path', default='./data', type=str)
    parser.add_argument('--output_dir', default='./output_dir_bootstrap_step', help='path where to save')
    parser.add_argument('--log_dir', default=None, help='path for tensorboard log')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true', default=True)
    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://')
    
    # --- Bootstrapping specific arguments ---
    parser.add_argument('--teacher_checkpoint_path', default='', type=str,
                        help='Path to the teacher model checkpoint. If empty, run standard MAE (pixel recon).')

    return parser


def load_teacher(teacher_path, device):
    """Loads a teacher model from a checkpoint, extracts its encoder, and sets it to eval mode."""
    teacher = models_bmae.bmae_deit_tiny_patch4() # Can be any model arch that has the encoder
    checkpoint = torch.load(teacher_path, map_location='cpu')
    
    # We only need the encoder part of the teacher
    checkpoint_model = checkpoint['model']
    # Filter out decoder keys
    encoder_keys = {k: v for k, v in checkpoint_model.items() if not k.startswith('decoder_')}
    # Adjust keys if they have 'module.' prefix from DDP
    encoder_keys = {k.replace('module.', ''): v for k, v in encoder_keys.items()}
    
    # Load the state dict into the teacher model (it will only fill encoder weights)
    teacher.load_state_dict(encoder_keys, strict=False)
    
    teacher.to(device)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False
    print(f"Teacher model loaded from {teacher_path} and set to eval mode.")
    return teacher


def main(args):
    misc.init_distributed_mode(args)

    if args.log_dir is None:
        args.log_dir = args.output_dir

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # Dataset setup
    transform_train = transforms.Compose([
            transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4913999140262604,0.4821586608886719,0.44653135538101196], std=[0.2470322698354721,0.24348516762256622,0.26158788800239563])])
    dataset_train = datasets.CIFAR10(root=args.data_path, train=True, download=True, transform=transform_train)
    
    sampler_train = torch.utils.data.DistributedSampler(dataset_train, num_replicas=misc.get_world_size(), rank=misc.get_rank(), shuffle=True)
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train, batch_size=args.batch_size,
        num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=True)
    
    log_writer = SummaryWriter(log_dir=args.log_dir) if misc.get_rank() == 0 else None

    # ---- Model and Engine setup based on bootstrap mode ----
    teacher_model = None
    if args.teacher_checkpoint_path:
        # --- Feature Reconstruction Mode (Stage > 1) ---
        print("Running in FEATURE reconstruction mode (Bootstrap Stage > 1)")
        model = models_bmae.bmae_deit_tiny_patch4()
        teacher_model = load_teacher(args.teacher_checkpoint_path, device)
        train_fn = train_one_epoch_bmae_k
    else:
        # --- Pixel Reconstruction Mode (Stage 1) ---
        print("Running in PIXEL reconstruction mode (Bootstrap Stage 1 or standard MAE)")
        model = models_mae.mae_diet_tiny_patch4()
        train_fn = train_one_epoch_mae

    model.to(device)
    model_without_ddp = model
    
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    
    # Optimizer and Scaler
    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 256
    
    print(f"Effective batch size: {eff_batch_size}, Actual LR: {args.lr:.2e}")
    param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    loss_scaler = NativeScaler()

    misc.load_model_2(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    # --- Training Loop ---
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        
        # Call the appropriate training function
        if teacher_model:
            train_stats = train_fn(
                model, teacher_model, data_loader_train,
                optimizer, device, epoch, loss_scaler,
                log_writer=log_writer, args=args
            )
        else:
             train_stats = train_fn(
                model, data_loader_train, optimizer, device, epoch, loss_scaler,
                log_writer=log_writer, args=args
            )

        # Save checkpoint periodically and at the end
        if args.output_dir and (epoch % 20 == 0 or epoch + 1 == args.epochs):
             misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch}
        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    print(f"Training time {datetime.timedelta(seconds=int(total_time))}")


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
