"""多模態資料融合網路（文字 + 影像 + 感測器）。

修正的問題：

1. **維度對不上**：MiniLM 輸出 384 維、ResNet 輸出 2048 維，但
   ``embedding_dim`` 是 512。原本直接 ``torch.stack`` 三個不同長度的
   向量會炸掉。改成每個模態都先經過一層投影，統一到 ``embedding_dim``。
2. **注意力加權廣播錯誤**：``attended_emb`` 形狀 ``[3, B, D]``，
   ``weights.unsqueeze(-1)`` 卻是 ``[B, 3, 1]``，相乘只有在 B == 3
   時才不會報錯（而且算出來的結果是錯的）。改成 ``[3, B, 1]``。
3. ``encode_image`` 裡宣告了 ``preprocess`` 卻從來沒用到 ── 影像根本
   沒被正規化。改成真的套用正規化，且 resize/crop 移到資料載入階段
   （模型只負責 normalize，才不會把已經處理好的 batch 再切一次）。
4. ``pretrained=True`` 在新版 torchvision 已被棄用，改用 ``weights=``。
5. 文字編碼器原本永遠包在 ``torch.no_grad()`` 裡，等於無法微調；
   改成由 ``config.freeze_text_encoder`` 決定。
"""

from __future__ import annotations

import logging
from contextlib import nullcontext
from typing import List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ImageNet 的標準正規化參數
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _build_image_backbone(pretrained: bool):
    """建立 ResNet-50 骨幹，並處理新舊 torchvision API 的差異。"""
    from torchvision import models

    try:  # torchvision >= 0.13
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        return models.resnet50(weights=weights)
    except AttributeError:  # pragma: no cover - 舊版 torchvision
        return models.resnet50(pretrained=pretrained)


class MultimodalFusionNetwork(nn.Module):
    """把三種模態的資料編碼後融合成單一向量。"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding_dim = config.embedding_dim

        # ---------------- 文字編碼器 ---------------- #
        from transformers import AutoModel, AutoTokenizer

        self.text_tokenizer = AutoTokenizer.from_pretrained(config.text_model_name)
        self.text_encoder = AutoModel.from_pretrained(config.text_model_name)
        text_hidden = self.text_encoder.config.hidden_size

        self.freeze_text_encoder = config.freeze_text_encoder
        if self.freeze_text_encoder:
            for param in self.text_encoder.parameters():
                param.requires_grad = False

        # 投影層：把各模態原生的維度統一到 embedding_dim
        self.text_projection = nn.Linear(text_hidden, self.embedding_dim)

        # ---------------- 影像編碼器 ---------------- #
        backbone = _build_image_backbone(config.pretrained_image_weights)
        image_hidden = backbone.fc.in_features  # ResNet-50 是 2048
        backbone.fc = nn.Identity()  # 拿掉分類頭，只留特徵
        self.image_encoder = backbone

        if config.freeze_image_encoder:
            for param in self.image_encoder.parameters():
                param.requires_grad = False

        self.image_projection = nn.Linear(image_hidden, self.embedding_dim)

        # 正規化參數存成 buffer，才會跟著模型一起搬到 GPU
        self.register_buffer(
            "image_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "image_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False
        )

        # ---------------- 感測器編碼器 ---------------- #
        self.sensor_encoder = nn.Sequential(
            nn.Linear(config.sensor_input_dim, 256),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(256, self.embedding_dim),
        )

        # ---------------- 融合與輸出 ---------------- #
        self.attention_fusion = AttentionFusion(
            self.embedding_dim,
            num_heads=config.num_attention_heads,
            dropout=config.dropout,
        )

        self.output_layer = nn.Sequential(
            nn.Linear(self.embedding_dim, 256),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(256, config.output_dim),
        )

    # ------------------------------------------------------------------ #
    # 各模態編碼
    # ------------------------------------------------------------------ #
    def encode_text(self, text_data: Sequence[str]) -> torch.Tensor:
        """編碼文字（例如社群媒體上的路況回報）。"""
        if not text_data:
            raise ValueError("text_data 不可以是空的")

        inputs = self.text_tokenizer(
            list(text_data),
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512,
        )
        inputs = {k: v.to(self.text_projection.weight.device) for k, v in inputs.items()}

        # 凍結時才關梯度；要微調就要讓梯度流過去
        context = torch.no_grad() if self.freeze_text_encoder else nullcontext()
        with context:
            outputs = self.text_encoder(**inputs)
            # 取 [CLS] token 當作整句的表示
            pooled = outputs.last_hidden_state[:, 0, :]

        return self.text_projection(pooled)

    def normalize_image(self, image_data: torch.Tensor) -> torch.Tensor:
        """把 [0, 1] 的影像轉成 ImageNet 正規化後的數值。"""
        return (image_data - self.image_mean) / self.image_std

    def encode_image(
        self, image_data: torch.Tensor, normalize: bool = True
    ) -> torch.Tensor:
        """編碼影像（例如路口攝影機畫面）。

        預期輸入是 ``[B, 3, H, W]``、數值範圍 [0, 1]。
        單張影像 ``[3, H, W]`` 也可以，會自動補上 batch 維度。
        """
        if image_data.dim() == 3:
            image_data = image_data.unsqueeze(0)
        if image_data.dim() != 4:
            raise ValueError(
                f"image_data 必須是 [B, 3, H, W] 或 [3, H, W]，收到 {tuple(image_data.shape)}"
            )

        if normalize:
            image_data = self.normalize_image(image_data)

        features = self.image_encoder(image_data)
        return self.image_projection(features)

    def encode_sensor(self, sensor_data: torch.Tensor) -> torch.Tensor:
        """編碼 IoT 感測器讀值。"""
        if sensor_data.dim() == 1:
            sensor_data = sensor_data.unsqueeze(0)

        expected = self.config.sensor_input_dim
        if sensor_data.shape[-1] != expected:
            raise ValueError(
                f"sensor_data 最後一維必須是 {expected}，收到 {sensor_data.shape[-1]}"
            )

        return self.sensor_encoder(sensor_data)

    # ------------------------------------------------------------------ #
    # 前向傳播
    # ------------------------------------------------------------------ #
    def forward(
        self,
        text_data: Sequence[str],
        image_data: torch.Tensor,
        sensor_data: torch.Tensor,
        return_weights: bool = False,
    ):
        """三種模態一起編碼、融合、輸出。

        ``return_weights=True`` 時會多回傳一組融合權重，方便在網頁介面上
        顯示「這次決策主要參考了哪個資料源」。
        """
        text_emb = self.encode_text(text_data)
        image_emb = self.encode_image(image_data)
        sensor_emb = self.encode_sensor(sensor_data)

        batch_sizes = {text_emb.shape[0], image_emb.shape[0], sensor_emb.shape[0]}
        if len(batch_sizes) > 1:
            raise ValueError(
                f"三種模態的 batch size 必須相同，目前是 {sorted(batch_sizes)}"
            )

        fused, weights = self.attention_fusion(text_emb, image_emb, sensor_emb)
        output = self.output_layer(fused)

        if return_weights:
            return output, weights
        return output


class AttentionFusion(nn.Module):
    """以多頭注意力 + 動態權重做模態融合。"""

    MODALITIES = ("text", "image", "sensor")

    def __init__(self, embedding_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.embedding_dim = embedding_dim

        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=False,  # 輸入格式是 [序列長度, batch, 維度]
        )
        self.layer_norm = nn.LayerNorm(embedding_dim)

        # 依三個模態的內容動態算出各自的權重
        self.weight_net = nn.Sequential(
            nn.Linear(embedding_dim * 3, 256),
            nn.ReLU(),
            nn.Linear(256, len(self.MODALITIES)),
        )

    def forward(
        self,
        text_emb: torch.Tensor,
        image_emb: torch.Tensor,
        sensor_emb: torch.Tensor,
    ):
        # [3, B, D]：把三個模態當成長度 3 的序列
        stacked = torch.stack([text_emb, image_emb, sensor_emb], dim=0)

        attended, _ = self.multihead_attn(stacked, stacked, stacked)
        # 殘差連接 + LayerNorm，訓練比較穩
        attended = self.layer_norm(attended + stacked)

        # 動態權重：[B, 3]
        concat = torch.cat([text_emb, image_emb, sensor_emb], dim=-1)
        weights = F.softmax(self.weight_net(concat), dim=-1)

        # 關鍵修正：要轉成 [3, B, 1] 才能和 [3, B, D] 正確廣播
        weighted = (attended * weights.t().unsqueeze(-1)).sum(dim=0)

        return weighted, weights
