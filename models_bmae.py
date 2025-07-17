# models_bmae.py
from functools import partial

import torch
import torch.nn as nn

from timm.models.vision_transformer import Block
# Inherit from the original MAE model to reuse its components
from models_mae import MaskedAutoencoderViT 

class BootstrappedMaskedAutoencoderViT(MaskedAutoencoderViT):
    """
    Bootstrapped Masked Autoencoder with VisionTransformer backbone.
    This model's decoder predicts encoder features from a target network,
    not pixels.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Redefine the decoder's prediction head to output features instead of pixels.
        # The output dimension should match the encoder's embedding dimension.
        embed_dim = self.pos_embed.shape[-1]
        decoder_embed_dim = self.decoder_pos_embed.shape[-1]
        self.decoder_pred = nn.Linear(decoder_embed_dim, embed_dim, bias=True)

        # A LayerNorm for the target features, as suggested in the problem description.
        self.target_norm = nn.LayerNorm(embed_dim)

    def forward_encoder_for_target(self, x):
        """
        Forward pass for the target encoder.
        Encodes all patches of an image without masking.
        Outputs patch features, excluding the [CLS] token.
        """
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x

    def forward_loss(self, pred, target, mask):
        """
        Calculates the loss between predicted features and target features.
        pred: [N, L, D] - Predicted features from the online decoder.
        target: [N, L, D] - Target features from the target encoder.
        mask: [N, L] - Binary mask (1 for masked patches).
        """
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], calculate mean loss per patch

        # Only consider loss on masked patches
        loss = (loss * mask).sum() / mask.sum()
        return loss

    def forward(self, imgs, target_features, mask_ratio=0.75):
        """
        The main forward pass for the BMAE model.
        """
        # Get latent representation, mask, and restore ids from the online encoder
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)

        # Get predicted features from the online decoder
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, D]

        # Normalize the target features from the teacher
        normed_target_features = self.target_norm(target_features)
        
        # Calculate the final loss
        loss = self.forward_loss(pred, normed_target_features, mask)

        return loss, pred, mask


def bmae_deit_tiny_patch4(**kwargs):
    """
    Constructor for the BMAE DeiT-Tiny model for CIFAR-10.
    """
    model = BootstrappedMaskedAutoencoderViT(
        img_size=32, patch_size=4, in_chans=3,
        embed_dim=192, depth=12, num_heads=3,
        decoder_embed_dim=96, decoder_depth=4, decoder_num_heads=3,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

