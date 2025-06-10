import os
from dataclasses import dataclass, field  # 記得加 field
from typing import Dict, List, Tuple

@dataclass
class MultimodalConfig:
    """多模態配置"""
    text_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    image_model_name: str = "resnet50"
    fusion_method: str = "attention_based"
    embedding_dim: int = 512

@dataclass
class RLConfig:
    """強化學習配置"""
    algorithm: str = "FED_PPO"
    num_agents: int = 16
    learning_rate: float = 3e-4
    batch_size: int = 256
    n_epochs: int = 10
    clip_range: float = 0.2
    
@dataclass
class TrafficConfig:
    """交通模擬配置"""
    sumo_config_file: str = "traffic_networks/downtown.sumocfg"
    simulation_time: int = 3600  # 1小時
    time_step: int = 1
    num_intersections: int = 16
    
@dataclass
class SystemConfig:
    """系統總配置"""
    multimodal: MultimodalConfig = field(default_factory=MultimodalConfig)
    rl: RLConfig = field(default_factory=RLConfig)
    traffic: TrafficConfig = field(default_factory=TrafficConfig)
    
    # API配置
    openai_api_key: str = ""
    
    # 路徑配置
    data_dir: str = "data/"
    model_dir: str = "models/"
    log_dir: str = "logs/"
