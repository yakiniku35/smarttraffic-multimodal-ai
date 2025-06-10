import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from torchvision import models, transforms
import numpy as np
from typing import Dict, List, Tuple, Optional

class MultimodalFusionNetwork(nn.Module):
    """多模態數據融合網路"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding_dim = config.embedding_dim
        
        # 文本編碼器
        self.text_tokenizer = AutoTokenizer.from_pretrained(config.text_model_name)
        self.text_encoder = AutoModel.from_pretrained(config.text_model_name)
        
        # 影像編碼器
        self.image_encoder = models.resnet50(pretrained=True)
        self.image_encoder.fc = nn.Linear(2048, self.embedding_dim)
        
        # 感測器數據編碼器
        self.sensor_encoder = nn.Sequential(
            nn.Linear(64, 256),  # 假設64個感測器特徵
            nn.ReLU(),
            nn.Linear(256, self.embedding_dim)
        )
        
        # 注意力融合機制
        self.attention_fusion = AttentionFusion(self.embedding_dim)
        
        # 輸出層
        self.output_layer = nn.Sequential(
            nn.Linear(self.embedding_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128)
        )
    
    def encode_text(self, text_data: List[str]) -> torch.Tensor:
        """編碼文本數據"""
        inputs = self.text_tokenizer(
            text_data, 
            padding=True, 
            truncation=True, 
            return_tensors="pt",
            max_length=512
        )
        
        with torch.no_grad():
            outputs = self.text_encoder(**inputs)
            # 使用CLS token的embedding
            text_embeddings = outputs.last_hidden_state[:, 0, :]
        
        return text_embeddings
    
    def encode_image(self, image_data: torch.Tensor) -> torch.Tensor:
        """編碼影像數據"""
        # 影像預處理
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        if len(image_data.shape) == 3:
            image_data = image_data.unsqueeze(0)
        
        # 通過ResNet編碼
        image_embeddings = self.image_encoder(image_data)
        return image_embeddings
    
    def encode_sensor(self, sensor_data: torch.Tensor) -> torch.Tensor:
        """編碼感測器數據"""
        return self.sensor_encoder(sensor_data)
    
    def forward(self, text_data: List[str], 
                image_data: torch.Tensor, 
                sensor_data: torch.Tensor) -> torch.Tensor:
        """前向傳播"""
        # 編碼各模態數據
        text_emb = self.encode_text(text_data)
        image_emb = self.encode_image(image_data)
        sensor_emb = self.encode_sensor(sensor_data)
        
        # 注意力融合
        fused_embedding = self.attention_fusion(text_emb, image_emb, sensor_emb)
        
        # 輸出預測
        output = self.output_layer(fused_embedding)
        return output

class AttentionFusion(nn.Module):
    """注意力融合機制"""
    
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        # 多頭注意力
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=8,
            dropout=0.1
        )
        
        # 權重計算
        self.weight_net = nn.Sequential(
            nn.Linear(embedding_dim * 3, 256),
            nn.ReLU(),
            nn.Linear(256, 3),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, text_emb: torch.Tensor, 
                image_emb: torch.Tensor, 
                sensor_emb: torch.Tensor) -> torch.Tensor:
        """多模態注意力融合"""
        
        # 堆疊所有模態
        all_embeddings = torch.stack([text_emb, image_emb, sensor_emb], dim=0)
        
        # 多頭注意力
        attended_emb, _ = self.multihead_attn(
            all_embeddings, all_embeddings, all_embeddings
        )
        
        # 計算融合權重
        concat_emb = torch.cat([text_emb, image_emb, sensor_emb], dim=-1)
        weights = self.weight_net(concat_emb)
        
        # 加權融合
        weighted_emb = (attended_emb * weights.unsqueeze(-1)).sum(dim=0)
        
        return weighted_emb
