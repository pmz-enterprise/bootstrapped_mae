# boot_mae_ema.py (EMA Version - Supports Multiple Bootstraps)

from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math # Added for EMA warmup

from models_mae import MaskedAutoencoderViT # Base MAE class

# Renamed class
class BootstrappedMaskedAutoencoderViT(MaskedAutoencoderViT):
    def __init__(self,
                patch_size=4, 
                embed_dim=192, 
                depth=12, 
                num_heads=3,
                decoder_embed_dim=192,
                decoder_depth=8, 
                decoder_num_heads=3,
                mlp_ratio=4, 
                norm_layer=partial(nn.LayerNorm, eps=1e-12),
                img_size=32,
                norm_latent_loss: bool = False,
                enable_ema: bool = True,
                ema_alpha: float = 0.999,
                ema_start_epoch: int = 10, 
                ema_decay_warmup_epochs: int = 10, 
                **kwargs):

        super().__init__(
            patch_size=patch_size, 
            embed_dim=embed_dim, 
            depth=depth, 
            num_heads=num_heads, 
            decoder_embed_dim=decoder_embed_dim,
            decoder_depth=decoder_depth,
            decoder_num_heads=decoder_num_heads,
            mlp_ratio=mlp_ratio,
            norm_layer=norm_layer,
            img_size=img_size
        )
        print(f"Initializing BootstrappedMaeEma (EMA multi-bootstrap version) "
              f"(norm_latent_loss={norm_latent_loss}, enable_ema={enable_ema}, "
              f"ema_alpha={ema_alpha}, ema_start_epoch={ema_start_epoch}, "
              f"ema_decay_warmup_epochs={ema_decay_warmup_epochs})")

        # Store args
        self.norm_latent_loss = norm_latent_loss
        self.enable_ema = enable_ema
        self.ema_alpha_base = ema_alpha # Store base alpha
        self.ema_start_epoch = ema_start_epoch
        self.ema_decay_warmup_epochs = ema_decay_warmup_epochs
        self.current_epoch = 0 # Internal epoch counter for EMA logic

        self.embed_dim = kwargs.get('embed_dim', 192)
        self.decoder_embed_dim = kwargs.get('decoder_embed_dim', 192)
        if self.embed_dim != self.decoder_embed_dim:
             raise ValueError(
                 f"Latent prediction loss requires encoder embed_dim ({self.embed_dim}) "
                 f"to match decoder decoder_embed_dim ({self.decoder_embed_dim}). "
                 f"Adjust model configuration."
             )

        # --- Shadow Model (EMA Target) ---
        # Stores the frozen EMA encoder state
        self.shadow_encoder_state = {}
        print(f"Model will perform pixel loss until epoch {self.ema_start_epoch}. "
              f"Shadow encoder will be initialized then, followed by latent loss with EMA.")

    @torch.no_grad()
    def _initialize_shadow_state(self):
        """Initialize shadow state by deep copying current encoder components.
           Should be called only once by update_shadow.
        """
        print(f"Initializing shadow encoder state via deepcopy (Epoch {self.current_epoch})...")
        current_device = next(self.parameters()).device 
        self.shadow_encoder_state['patch_embed'] = copy.deepcopy(self.patch_embed).to(current_device)
        self.shadow_encoder_state['cls_token'] = copy.deepcopy(self.cls_token).to(current_device)
        self.shadow_encoder_state['pos_embed'] = copy.deepcopy(self.pos_embed).to(current_device)
        self.shadow_encoder_state['blocks'] = copy.deepcopy(self.blocks).to(current_device)
        self.shadow_encoder_state['norm'] = copy.deepcopy(self.norm).to(current_device)
        for component in self.shadow_encoder_state.values():
            if isinstance(component, nn.Module):
                for param in component.parameters():
                    param.requires_grad = False
            elif isinstance(component, torch.Tensor):
                 component.requires_grad = False
        print("Shadow encoder state initialized and frozen.")

    @torch.no_grad()
    def update_shadow(self):
        """Update shadow state using EMA.
           Should be called by the training script once per epoch (ideally after optimizer step).
           Handles initialization at ema_start_epoch and subsequent EMA updates.
        """
        self.current_epoch += 1

        if not self.enable_ema:
            return
        
        if self.current_epoch < self.ema_start_epoch:

            return 
        elif self.current_epoch == self.ema_start_epoch:
            self._initialize_shadow_state()
            return 
        else: 

            ema_epoch = self.current_epoch - self.ema_start_epoch

            # Calculate current EMA alpha with cosine warmup based on *decay* warmup epochs
            if ema_epoch <= self.ema_decay_warmup_epochs:
                warmup_progress = (ema_epoch -1) / max(1, self.ema_decay_warmup_epochs) 
                momentum = 1.0 - self.ema_alpha_base 
                current_momentum = momentum * (1 - math.cos(math.pi * ema_epoch / self.ema_decay_warmup_epochs)) / 2
                alpha = 1.0 - current_momentum
            else:
                alpha = self.ema_alpha_base

            print(f"Updating shadow encoder state via EMA (Epoch {self.current_epoch}, EMA Epoch {ema_epoch}, Alpha {alpha:.4f})...")

            # --- EMA Update Logic ---
            shadow_components = {
                'patch_embed': self.shadow_encoder_state['patch_embed'],
                'cls_token': self.shadow_encoder_state['cls_token'],
                'pos_embed': self.shadow_encoder_state['pos_embed'],
                'blocks': self.shadow_encoder_state['blocks'],
                'norm': self.shadow_encoder_state['norm']
            }

            online_modules = {
                'patch_embed': self.patch_embed,
                'blocks': self.blocks,
                'norm': self.norm
            }
            for name in ['patch_embed', 'blocks', 'norm']:
                shadow_module = shadow_components[name]
                online_module = online_modules[name]
                for shadow_param, online_param in zip(shadow_module.parameters(), online_module.parameters()):
                    if not shadow_param.requires_grad:
                        pass

                    # Ensure matching types for calculation, if needed (e.g., mixed precision)
                    if shadow_param.dtype != online_param.dtype:
                        online_param_data = online_param.data.to(shadow_param.dtype)
                    else:
                        online_param_data = online_param.data

                    # Calculate EMA
                    new_average = alpha * shadow_param.data + (1.0 - alpha) * online_param_data
                    # Update shadow parameter in-place
                    shadow_param.data.copy_(new_average)

            # Update standalone tensors (cls_token, pos_embed)
            for name in ['cls_token', 'pos_embed']:
                shadow_tensor = shadow_components[name]
                online_tensor = getattr(self, name)
                if shadow_tensor.dtype != online_tensor.dtype:
                    online_tensor_data = online_tensor.data.to(shadow_tensor.dtype)
                else:
                    online_tensor_data = online_tensor.data

                new_average = alpha * shadow_tensor.data + (1.0 - alpha) * online_tensor_data
                shadow_tensor.data.copy_(new_average)

    # --- Forward Methods ---
    def latent_decoder_forward(self, x, ids_restore):
        """ Decoder forward pass stopping before the final pixel projection layer.
            Outputs latent representations.
        """
        enc_fea = x  # store the shadow encoder output in enc_fea for later cross-attention computation
        x = self.decoder_embed(x)
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token
        x = x + self.decoder_pos_embed
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        x = x[:, 1:, :] 
        return x

    @torch.no_grad()
    def shadow_encoder_forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Forward pass through the frozen shadow encoder state.
            Operates on the full, unmasked image.
        """
        shadow_state = self.shadow_encoder_state
        if not shadow_state:
             raise RuntimeError("Shadow encoder state is not initialized! Call update_shadow() first.")

        # Manually apply shadow encoder layers using components from shadow_encoder_state
        patch_embed = shadow_state['patch_embed']
        cls_token_base = shadow_state['cls_token']
        pos_embed = shadow_state['pos_embed']
        blocks = shadow_state['blocks']
        norm = shadow_state['norm']

        x = patch_embed(x)
        x = x + pos_embed[:, 1:, :]
        cls_token = cls_token_base + pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        for blk in blocks: # Iterate through the ModuleList stored in shadow state
            x = blk(x)
        x = norm(x)
        return x

    def latent_prediction_loss(self, target_latent, pred_latent, mask):
        """ Calculates MSE loss between predicted and target latent embeddings
            on the masked patches.
        """
        if self.norm_latent_loss:
            with torch.no_grad():
                mean = target_latent.mean(dim=-1, keepdim=True)
                var = target_latent.var(dim=-1, keepdim=True)
                target_latent = (target_latent - mean) / (var + 1.e-6)**.5

        loss = (pred_latent - target_latent) ** 2
        loss = loss.mean(dim=-1)  
        loss = (loss * mask).sum() / mask.sum().clamp(min=1.0)
        return loss

    # --- Main Forward ---

    def forward(self, imgs: torch.Tensor, mask_ratio: float = 0.75):
        """ Main forward pass. Performs pixel reconstruction if shadow state is empty,
            otherwise performs latent reconstruction against the shadow state.
        """
        in_latent_phase = bool(self.shadow_encoder_state)

        if in_latent_phase:
            # --- Latent  Reconstruction Phase ---
            latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
            pred_latent = self.latent_decoder_forward(latent, ids_restore)
            target_latent_full = self.shadow_encoder_forward(imgs)
            target_latent = target_latent_full[:, 1:, :].detach() 
            loss = self.latent_prediction_loss(target_latent, pred_latent, mask)
            return loss, pred_latent, mask 

        else:
            # --- Pixel Reconstruction Phase ---
            loss, pred_pixels, mask = super().forward(imgs, mask_ratio)
            return loss, pred_pixels, mask 



def bmae_ema_tiny_patch4(**kwargs):

    # Create the model 
    model = BootstrappedMaskedAutoencoderViT(patch_size=4, embed_dim=192, depth=12, num_heads=3,
        decoder_embed_dim=192, 
        decoder_depth=8, decoder_num_heads=3,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-12),
        img_size=32)
    return model

