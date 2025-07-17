# main_bmae_pretrain.py
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

import timm
import timm.optim.optim_factory as optim_factory

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler

# Import the new BMAE model and engine
import models_bmae
from engine_bmae_pretrain import train_one_epoch


def get_args_parser():
    parser = argparse.ArgumentParser('BMAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=512, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations')

    # Model parameters
    parser.add_argument('--model', default='bmae_deit_tiny_patch4', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input_size', default=32, type=int, help='images input size')
    parser.add_argument('--mask_ratio', default=0.75, type=float, help='Masking ratio')

    # BMAE parameters
    parser.add_argument('--ema', type=float, default=0.996, help="EMA decay for teacher model")

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--lr', type=float, default=None, help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1.5e-4, help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., help='lower lr bound')
    parser.add_argument('--warmup_epochs', type=int, default=10, metavar='N', help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--data_path', default='./data', type=str, help='dataset path')
    parser.add_argument('--output_dir', default='./output_dir_bmae', help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir_bmae', help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true', help='Pin CPU memory in DataLoader')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    return parser


def main(args):
    misc.init_distributed_mode(args)
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    transform_train = transforms.Compose([
            transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4913999140262604,0.4821586608886719,0.44653135538101196], std=[0.2470322698354721,0.24348516762256622,0.26158788800239563])])
    dataset_train = datasets.CIFAR10(root=args.data_path, train=True, download=True, transform=transform_train)
    
    sampler_train = torch.utils.data.DistributedSampler(dataset_train, num_replicas=misc.get_world_size(), rank=misc.get_rank(), shuffle=True)
    
    log_writer = SummaryWriter(log_dir=args.log_dir) if misc.get_rank() == 0 else None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train, batch_size=args.batch_size,
        num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=True)
    
    # Define the online and teacher models
    model = models_bmae.__dict__[args.model]()
    teacher = models_bmae.__dict__[args.model]()
    
    # Disable gradients for the teacher model
    for param in teacher.parameters():
        param.requires_grad = False

    model.to(device)
    teacher.to(device)

    model_without_ddp = model
    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 256

    print(f"base lr: {args.blr:.2e}, actual lr: {args.lr:.2e}")
    print(f"accumulate grad iterations: {args.accum_iter}, effective batch size: {eff_batch_size}")

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    
    param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    loss_scaler = NativeScaler()

    # Load checkpoint if resuming, and sync teacher with online model
    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)
    misc.load_model(args=args, model_without_ddp=teacher, optimizer=optimizer, loss_scaler=loss_scaler)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, teacher, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.ema, log_writer=log_writer, args=args
        )
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
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
