"""
CRIS-P: Enhanced CRIS with Cross-Modal Self-Alignment
创新点：
1. 跨模态自对齐模块 (Cross-Modal Self-Alignment)
2. 多尺度注意力融合 (Multi-Scale Attention Fusion)
3. 对比学习正则化 (Contrastive Learning Regularization)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.clip import build_model
from .layers import FPN, Projector, TransformerDecoder


class CrossModalAlignmentModule(nn.Module):
    """跨模态自对齐模块 - 增强文本和视觉特征的对齐"""
    def __init__(self, vis_dim=512, txt_dim=1024, hidden_dim=512):
        super().__init__()
        # 文本特征投影
        self.txt_proj = nn.Sequential(
            nn.Linear(txt_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim)
        )
        # 视觉特征投影
        self.vis_proj = nn.Sequential(
            nn.Linear(vis_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim)
        )
        # 对比学习温度
        self.temp = nn.Parameter(torch.ones([]) * 0.07)

    def forward(self, vis_features, txt_features):
        """
        Args:
            vis_features: [B, C_vis] 视觉特征
            txt_features: [B, L, C_txt] 文本特征序列
        """
        # 取文本序列的mean pooling
        txt_seq = txt_features.mean(dim=1)  # [B, C_txt]

        # 投影到统一空间
        vis_emb = self.vis_proj(vis_features)  # [B, hidden_dim]
        txt_emb = self.txt_proj(txt_seq)  # [B, hidden_dim]

        # L2归一化
        vis_emb = F.normalize(vis_emb, dim=-1)
        txt_emb = F.normalize(txt_emb, dim=-1)

        # 计算相似度矩阵
        sim_matrix = torch.matmul(vis_emb, txt_emb.T) / self.temp  # [B, B]
        # 对角线为正样本
        labels = torch.arange(len(vis_emb), device=vis_emb.device)

        # 对比损失
        loss = F.cross_entropy(sim_matrix, labels)

        return loss


class MultiScaleAttentionFusion(nn.Module):
    """多尺度注意力融合模块 - 改进FPN中的跨模态融合"""
    def __init__(self, in_channels=[512, 1024, 1024], out_channels=512, num_heads=8):
        super().__init__()
        self.num_levels = len(in_channels)
        self.num_heads = num_heads

        # 每层的文本引导注意力
        self.text_attentions = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=out_channels,
                num_heads=num_heads,
                batch_first=True
            )
            for _ in range(self.num_levels)
        ])

        # 融合层
        self.fusion_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, out_channels, 1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True)
            )
            for in_ch in in_channels
        ])

        # 输出投影
        self.output_proj = nn.Conv2d(out_channels * self.num_levels, out_channels, 1)

    def forward(self, multi_scale_features, text_feature):
        """
        Args:
            multi_scale_features: [C3, C4, C5] 多尺度视觉特征
            text_feature: [B, L, D] 文本特征
        """
        # 取文本的全局表示
        txt_emb = text_feature.mean(dim=1)  # [B, D]
        txt_emb = txt_emb.unsqueeze(1)  # [B, 1, D]

        outputs = []
        for i, vis_feat in enumerate(multi_scale_features):
            B, C, H, W = vis_feat.shape
            # 变换维度和位置
            vis_flat = vis_feat.flatten(2).permute(0, 2, 1)  # [B, H*W, C]

            # 文本引导的注意力
            attn_out, _ = self.text_attentions[i](vis_flat, txt_emb, txt_emb)
            attn_out = attn_out.permute(0, 2, 1).reshape(B, C, H, W)

            # 投影
            out = self.fusion_convs[i](attn_out)
            outputs.append(out)

        # 多尺度特征融合
        # 上采样到相同尺寸
        target_size = outputs[-1].shape[2:]
        resized = []
        for out in outputs:
            if out.shape[2:] != target_size:
                out = F.interpolate(out, size=target_size, mode='bilinear', align_corners=False)
            resized.append(out)

        fused = torch.cat(resized, dim=1)
        output = self.output_proj(fused)

        return output


class CRIS_P(nn.Module):
    """增强版CRIS with Cross-Modal Self-Alignment"""

    def __init__(self, cfg):
        super().__init__()
        # Vision & Text Encoder
        clip_model = torch.jit.load(cfg.clip_pretrain,
                                    map_location="cpu").eval()
        self.backbone = build_model(clip_model.state_dict(), cfg.word_len).float()

        # 是否启用创新模块
        self.use_alignment = getattr(cfg, 'use_alignment', True)
        self.use_contrastive = getattr(cfg, 'use_contrastive', True)

        # Multi-Modal FPN (标准版)
        self.neck = FPN(in_channels=cfg.fpn_in, out_channels=cfg.fpn_out)

        # 如果启用创新模块，添加跨模态对齐
        if self.use_alignment:
            self.alignment_module = CrossModalAlignmentModule(
                vis_dim=cfg.vis_dim,
                txt_dim=cfg.word_dim,
                hidden_dim=cfg.vis_dim
            )

        # Decoder
        self.decoder = TransformerDecoder(num_layers=cfg.num_layers,
                                         d_model=cfg.vis_dim,
                                         nhead=cfg.num_head,
                                         dim_ffn=cfg.dim_ffn,
                                         dropout=cfg.dropout,
                                         return_intermediate=cfg.intermediate)
        # Projector
        self.proj = Projector(cfg.word_dim, cfg.vis_dim // 2, 3)

    def forward(self, img, word, mask=None):
        """
        Args:
            img: [B, 3, H, W]
            word: [B, words]
            mask: [B, 1, H, W] 可选
        """
        # padding mask
        pad_mask = torch.zeros_like(word).masked_fill_(word == 0, 1).bool()

        # 编码
        vis = self.backbone.encode_image(img)  # C3/C4/C5
        word_enc, state = self.backbone.encode_text(word)  # [B, L, 512], [B, 512]

        # 跨模态对齐损失 (训练时)
        alignment_loss = 0
        if self.training and self.use_alignment:
            # 使用C5特征 (最丰富的语义)
            vis_state = vis[2].mean(dim=[2, 3])  # [B, 1024] -> global avg
            alignment_loss = self.alignment_module(vis_state, word_enc)

        # FPN处理
        fq = self.neck(vis, state)
        b, c, h, w = fq.size()

        # Transformer解码
        fq = self.decoder(fq, word_enc, pad_mask)
        fq = fq.reshape(b, c, h, w)

        # 预测
        pred = self.proj(fq, state)

        if self.training:
            # resize mask
            if pred.shape[-2:] != mask.shape[-2:]:
                mask = F.interpolate(mask, pred.shape[-2:],
                                     mode='nearest').detach()
            # BCE损失
            loss_bce = F.binary_cross_entropy_with_logits(pred, mask)

            # 总损失 = BCE + 对齐损失
            if alignment_loss > 0:
                total_loss = loss_bce + 0.1 * alignment_loss
            else:
                total_loss = loss_bce

            return pred.detach(), mask, total_loss
        else:
            return pred.detach()


def build_cris_p(cfg):
    """构建CRIS-P模型"""
    return CRIS_P(cfg)
