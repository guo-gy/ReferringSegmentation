"""
CRIS-Lite: Simplified lightweight CRIS for testing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.clip import build_model


class SimpleFPN(nn.Module):
    """Simple FPN with consistent dimensions"""
    def __init__(self, in_channels=[512, 1024, 1024], out_dim=256):
        super().__init__()
        self.out_dim = out_dim
        # Project all features to same dimension
        self.proj3 = nn.Sequential(
            nn.Conv2d(in_channels[0], out_dim, 1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(True)
        )
        self.proj4 = nn.Sequential(
            nn.Conv2d(in_channels[1], out_dim, 1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(True)
        )
        self.proj5 = nn.Sequential(
            nn.Conv2d(in_channels[2], out_dim, 1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(True)
        )
        self.txt_proj = nn.Linear(1024, out_dim)
        self.fusion = nn.Sequential(
            nn.Conv2d(out_dim * 2, out_dim, 3, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(True)
        )

    def forward(self, imgs, state):
        v3, v4, v5 = imgs
        # Project to same dimension
        f3 = self.proj3(v3)
        f4 = self.proj4(v4)
        f5 = self.proj5(v5)
        # Text guidance
        txt_emb = self.txt_proj(state).unsqueeze(-1).unsqueeze(-1)
        f5 = f5 * txt_emb
        # Upsample and combine
        f5_up = F.interpolate(f5, size=f4.shape[2:], mode='bilinear', align_corners=False)
        f4 = torch.cat([f4, f5_up], dim=1)
        out = self.fusion(f4)
        return out


class SimpleDecoder(nn.Module):
    """Simple decoder with cross attention"""
    def __init__(self, vis_dim=256, txt_dim=512, nhead=4):
        super().__init__()
        self.vis_dim = vis_dim
        # Project text to match visual dim
        self.txt_proj = nn.Linear(txt_dim, vis_dim)
        # Simple cross attention
        self.cross_attn = nn.MultiheadAttention(vis_dim, nhead, batch_first=True)
        self.norm = nn.LayerNorm(vis_dim)

    def forward(self, vis, txt, pad_mask):
        B, C, H, W = vis.shape
        # Reshape visual features
        vis_flat = vis.flatten(2).permute(0, 2, 1)  # B, HW, C
        # Project text
        txt_proj = self.txt_proj(txt)  # B, L, vis_dim
        # Cross attention
        attn_out, _ = self.cross_attn(vis_flat, txt_proj, txt_proj, key_padding_mask=pad_mask)
        vis_flat = self.norm(vis_flat + attn_out)
        # Reshape back
        out = vis_flat.permute(0, 2, 1).reshape(B, C, H, W)
        return out


class SimpleProjector(nn.Module):
    """Simple projector for mask prediction"""
    def __init__(self, in_dim=256, out_size=4):
        super().__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_dim, in_dim // 2, 4, 2, 1),
            nn.BatchNorm2d(in_dim // 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_dim // 2, in_dim // 4, 4, 2, 1),
            nn.BatchNorm2d(in_dim // 4),
            nn.ReLU(True),
            nn.Conv2d(in_dim // 4, 1, 1)
        )

    def forward(self, x, word):
        return self.conv(x)


class CRIS_Lite(nn.Module):
    """Lightweight CRIS for testing"""

    def __init__(self, cfg):
        super().__init__()
        # Load CLIP
        clip_model = torch.jit.load(cfg.clip_pretrain, map_location="cpu").eval()
        self.backbone = build_model(clip_model.state_dict(), cfg.word_len).float()

        # Get dimensions from config
        vis_dim = getattr(cfg, 'vis_dim', 256)

        # Simple FPN
        self.neck = SimpleFPN(cfg.fpn_in, vis_dim)

        # Simple decoder - text dim is 512 from CLIP
        self.decoder = SimpleDecoder(vis_dim, txt_dim=512, nhead=cfg.num_head)

        # Simple projector
        self.proj = SimpleProjector(vis_dim)

    def forward(self, img, word, mask=None):
        pad_mask = torch.zeros_like(word).masked_fill_(word == 0, 1).bool()

        # Encode
        vis = self.backbone.encode_image(img)
        word_enc, state = self.backbone.encode_text(word)

        # FPN
        fq = self.neck(vis, state)
        b, c, h, w = fq.size()

        # Decoder
        fq = self.decoder(fq, word_enc, pad_mask)
        fq = fq.reshape(b, c, h, w)

        # Predict
        pred = self.proj(fq, state)

        if self.training:
            if pred.shape[-2:] != mask.shape[-2:]:
                mask = F.interpolate(mask, pred.shape[-2:], mode='nearest').detach()
            loss = F.binary_cross_entropy_with_logits(pred, mask)
            return pred.detach(), mask, loss
        else:
            return pred.detach()


def build_cris_lite(cfg):
    return CRIS_Lite(cfg)
