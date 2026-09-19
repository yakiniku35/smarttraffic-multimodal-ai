"""系統配置模組。

這個檔案定義整個系統會用到的所有設定值，並提供：

* ``SystemConfig.load()``  ── 從 JSON 檔案讀取設定（找不到檔案就用預設值）
* ``SystemConfig.save()``  ── 把目前的設定寫回 JSON 檔案
* ``SystemConfig.validate()`` ── 檢查設定是否合理，避免帶著壞資料跑訓練
* ``SystemConfig.ensure_directories()`` ── 自動建立 logs/、models/ 等資料夾

所有 dataclass 的欄位都有預設值，因此 ``SystemConfig()`` 就能直接使用。
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict


# --------------------------------------------------------------------------- #
# 各子系統設定
# --------------------------------------------------------------------------- #
@dataclass
class MultimodalConfig:
    """多模態融合設定。"""

    text_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    image_model_name: str = "resnet50"
    fusion_method: str = "attention_based"
    embedding_dim: int = 512

    # 感測器輸入的特徵數量（原本寫死在模型裡，抽出來比較好調整）
    sensor_input_dim: int = 64
    # 融合後輸出的向量長度
    output_dim: int = 128
    dropout: float = 0.1
    num_attention_heads: int = 8

    # 是否凍結預訓練的文字 / 影像編碼器（只當特徵抽取器用，不參與訓練）
    freeze_text_encoder: bool = True
    freeze_image_encoder: bool = False
    # 是否下載 ImageNet 預訓練權重；離線環境請設成 False
    pretrained_image_weights: bool = True

    def validate(self) -> None:
        if self.embedding_dim <= 0:
            raise ValueError("embedding_dim 必須大於 0")
        if self.num_attention_heads <= 0:
            raise ValueError("num_attention_heads 必須大於 0")
        if self.embedding_dim % self.num_attention_heads != 0:
            raise ValueError(
                f"embedding_dim({self.embedding_dim}) 必須能被 "
                f"num_attention_heads({self.num_attention_heads}) 整除"
            )
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout 必須介於 0（含）到 1（不含）之間")


@dataclass
class RLConfig:
    """強化學習（聯邦式 PPO）設定。"""

    algorithm: str = "FED_PPO"
    num_agents: int = 16

    # 之前 state_dim / action_dim 是在 main.py 裡動態塞進來的，
    # 這樣 IDE 查不到、也容易打錯字，所以改成正式欄位。
    state_dim: int = 8
    action_dim: int = 2

    learning_rate: float = 3e-4
    batch_size: int = 256
    n_epochs: int = 10
    clip_range: float = 0.2

    # GAE 與損失權重（原本寫死在程式裡）
    gamma: float = 0.99
    gae_lambda: float = 0.95
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5

    hidden_dim: int = 128
    gnn_output_dim: int = 64
    dropout: float = 0.1

    # 聯邦式學習：每隔幾次更新做一次模型平均
    federated_sync_interval: int = 10

    def validate(self) -> None:
        if self.state_dim <= 0 or self.action_dim <= 0:
            raise ValueError("state_dim 與 action_dim 都必須大於 0")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate 必須大於 0")
        if self.batch_size <= 0:
            raise ValueError("batch_size 必須大於 0")
        if self.n_epochs <= 0:
            raise ValueError("n_epochs 必須大於 0")
        if not 0.0 < self.clip_range < 1.0:
            raise ValueError("clip_range 必須介於 0 到 1 之間")
        if not 0.0 < self.gamma <= 1.0:
            raise ValueError("gamma 必須介於 0（不含）到 1（含）之間")
        if not 0.0 <= self.gae_lambda <= 1.0:
            raise ValueError("gae_lambda 必須介於 0 到 1 之間")


@dataclass
class TrafficConfig:
    """交通模擬設定。"""

    sumo_config_file: str = "traffic_networks/downtown.sumocfg"
    simulation_time: int = 3600  # 模擬 1 小時（秒）
    time_step: int = 1
    num_intersections: int = 16

    # 原本 sumo_interface.py 讀取 config.use_gui，但這裡沒有定義，
    # 一執行就會 AttributeError，所以補上。
    use_gui: bool = False
    # 找不到 SUMO / traci 時，自動改用內建的模擬環境（方便開發與測試）
    fallback_to_mock: bool = True
    # 建圖時，兩個路口距離小於這個值（公尺）就連一條邊
    neighbor_distance_m: float = 1000.0
    # 亂數種子，None 表示不固定
    random_seed: int | None = 42

    def validate(self) -> None:
        if self.simulation_time <= 0:
            raise ValueError("simulation_time 必須大於 0")
        if self.time_step <= 0:
            raise ValueError("time_step 必須大於 0")
        if self.num_intersections <= 0:
            raise ValueError("num_intersections 必須大於 0")
        if self.neighbor_distance_m <= 0:
            raise ValueError("neighbor_distance_m 必須大於 0")


@dataclass
class UIConfig:
    """網頁介面（Streamlit）設定。"""

    theme: str = "auto"  # auto / light / dark
    language: str = "zh-TW"
    refresh_interval_s: int = 5
    auto_refresh: bool = False
    show_advanced: bool = False
    chart_height: int = 380
    decimal_places: int = 1

    def validate(self) -> None:
        if self.theme not in ("auto", "light", "dark"):
            raise ValueError("theme 只能是 auto / light / dark")
        if self.refresh_interval_s < 1:
            raise ValueError("refresh_interval_s 至少要 1 秒")
        if self.chart_height < 200:
            raise ValueError("chart_height 至少要 200 像素")


# --------------------------------------------------------------------------- #
# 總設定
# --------------------------------------------------------------------------- #
DEFAULT_CONFIG_PATH = Path("configs/system_config.json")


@dataclass
class SystemConfig:
    """系統總設定，把上面幾個子設定包在一起。"""

    multimodal: MultimodalConfig = field(default_factory=MultimodalConfig)
    rl: RLConfig = field(default_factory=RLConfig)
    traffic: TrafficConfig = field(default_factory=TrafficConfig)
    ui: UIConfig = field(default_factory=UIConfig)

    # 路徑設定
    data_dir: str = "data"
    model_dir: str = "models"
    log_dir: str = "logs"
    config_dir: str = "configs"

    log_level: str = "INFO"

    # ----------------------------------------------------------------- #
    # API 金鑰
    # ----------------------------------------------------------------- #
    # 金鑰一律從環境變數讀，不寫進 JSON，避免不小心 commit 上 GitHub。
    @property
    def openai_api_key(self) -> str:
        return os.environ.get("OPENAI_API_KEY", "")

    @property
    def weather_api_key(self) -> str:
        return os.environ.get("WEATHER_API_KEY", "")

    @property
    def maps_api_key(self) -> str:
        return os.environ.get("MAPS_API_KEY", "")

    def api_key_status(self) -> Dict[str, bool]:
        """回傳每個 API 金鑰是否已設定（只回傳有無，不回傳內容）。"""
        return {
            "OPENAI_API_KEY": bool(self.openai_api_key),
            "WEATHER_API_KEY": bool(self.weather_api_key),
            "MAPS_API_KEY": bool(self.maps_api_key),
        }

    # ----------------------------------------------------------------- #
    # 驗證與資料夾
    # ----------------------------------------------------------------- #
    def validate(self) -> None:
        """逐一驗證每個子設定，有問題就丟出 ValueError。"""
        self.multimodal.validate()
        self.rl.validate()
        self.traffic.validate()
        self.ui.validate()

        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if self.log_level.upper() not in valid_levels:
            raise ValueError(f"log_level 必須是 {sorted(valid_levels)} 其中之一")

    def ensure_directories(self) -> None:
        """建立所有需要的資料夾。

        原本 main.py 直接寫 logs/smarttraffic.log，資料夾不存在就會
        FileNotFoundError；先建立好就不會有這個問題。
        """
        for directory in (self.data_dir, self.model_dir, self.log_dir, self.config_dir):
            Path(directory).mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------------- #
    # 序列化
    # ----------------------------------------------------------------- #
    def to_dict(self) -> Dict[str, Any]:
        """轉成純 dict（可以直接丟給 json.dump）。"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SystemConfig":
        """從 dict 建立設定；不認識的鍵會被忽略，缺少的鍵用預設值。"""
        sub_configs = {
            "multimodal": MultimodalConfig,
            "rl": RLConfig,
            "traffic": TrafficConfig,
            "ui": UIConfig,
        }

        kwargs: Dict[str, Any] = {}
        for name, sub_cls in sub_configs.items():
            raw = data.get(name) or {}
            allowed = {f.name for f in fields(sub_cls)}
            kwargs[name] = sub_cls(**{k: v for k, v in raw.items() if k in allowed})

        scalar_fields = {
            f.name for f in fields(cls) if f.name not in sub_configs
        }
        for key in scalar_fields:
            if key in data:
                kwargs[key] = data[key]

        return cls(**kwargs)

    def save(self, path: str | Path = DEFAULT_CONFIG_PATH) -> Path:
        """把設定存成 JSON 檔，回傳實際寫入的路徑。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return path

    @classmethod
    def load(cls, path: str | Path | None = DEFAULT_CONFIG_PATH) -> "SystemConfig":
        """讀取 JSON 設定檔。

        找不到檔案時回傳預設設定，不會讓程式直接掛掉；
        但如果檔案存在卻是壞的 JSON，就會照實丟出錯誤。
        """
        if path is None:
            return cls()

        path = Path(path)
        if not path.exists():
            return cls()

        data = json.loads(path.read_text(encoding="utf-8"))
        return cls.from_dict(data)
