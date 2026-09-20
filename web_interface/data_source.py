"""網頁介面的資料來源。

原本的程式在每個地方都直接呼叫 ``np.random.randint(...)``，
造成一個很煩人的問題：**Streamlit 只要重新執行一次（按任何按鈕、
拉任何滑桿）整個頁面的數字就會全部亂跳**，看起來像資料在閃。

這裡的做法是：
* 所有隨機資料都由一個固定種子的產生器產生
* 只有「時間刻度」變動時才重新產生（模擬真的有新資料進來）
* 結果放在 ``st.session_state``，同一次 rerun 內保持一致
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List

import numpy as np
import pandas as pd

INTERSECTION_IDS = ["TL_001", "TL_002", "TL_003", "TL_004"]

DATA_SOURCE_LABELS = {
    "camera": "交通攝影機",
    "gps": "GPS 軌跡",
    "weather": "氣象資訊",
    "social": "社群媒體文本",
    "sensor": "IoT 感測器",
}

# 各資料源的基準特性：資料量(MB/h)、處理延遲(ms)、預設融合權重
DATA_SOURCE_PROFILE = {
    "camera": (1200, 45, 0.35),
    "gps": (800, 23, 0.25),
    "weather": (50, 12, 0.15),
    "social": (300, 67, 0.10),
    "sensor": (150, 18, 0.15),
}


@dataclass
class LiveSnapshot:
    """某個時間點的即時指標快照。"""

    timestamp: datetime
    traffic_volume: int
    average_speed: float
    waiting_time: float
    efficiency: float

    # 和上一個快照相比的變化量，用來畫 st.metric 的 delta
    traffic_delta: int = 0
    speed_delta: float = 0.0
    waiting_delta: float = 0.0
    efficiency_delta: float = 0.0


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def build_traffic_history(seed: int = 42, hours: int = 24) -> pd.DataFrame:
    """產生過去 N 小時的各路口車流量。

    日期改成以「現在」為基準往回推，原本寫死 ``2025-06-10`` 的話，
    過一陣子再打開就會看到一個奇怪的舊日期。
    """
    rng = _rng(seed)
    end = pd.Timestamp.now().floor("h")
    # pandas 2.2 起 'H' 已棄用，要用小寫 'h'
    timestamps = pd.date_range(end=end, periods=hours, freq="h")

    data: Dict[str, object] = {"timestamp": timestamps}
    # 用一條日間高、深夜低的基準曲線，比純亂數像真的車流
    hour_of_day = np.array([ts.hour for ts in timestamps])
    daily_curve = 1.0 + 0.45 * np.sin((hour_of_day - 6) / 24 * 2 * np.pi)

    for i, tl_id in enumerate(INTERSECTION_IDS):
        base = 120 + i * 25
        noise = rng.normal(0, base * 0.08, hours)
        data[tl_id] = np.clip(base * daily_curve + noise, 0, None).round(1)

    return pd.DataFrame(data)


def build_live_snapshot(
    tick: int, seed: int = 42, timestamp: datetime | None = None
) -> LiveSnapshot:
    """依照目前的時間刻度產生一組即時指標。

    同一個 ``tick`` 一定得到同一組數字，所以頁面重新執行時不會亂跳。

    ``timestamp`` 要由呼叫端傳入「這批資料實際載入的時間」。
    如果在這裡直接用 ``datetime.now()``，畫面上的「最後更新」每次
    重新執行都會變，看起來像有新資料進來，其實數字根本沒動。
    """
    rng = _rng(seed + tick)
    prev = _rng(seed + max(tick - 1, 0))

    def sample(generator: np.random.Generator):
        return (
            int(generator.integers(150, 300)),
            float(generator.uniform(25, 45)),
            float(generator.uniform(20, 60)),
            float(generator.uniform(75, 95)),
        )

    volume, speed, waiting, efficiency = sample(rng)
    p_volume, p_speed, p_waiting, p_efficiency = sample(prev)

    return LiveSnapshot(
        timestamp=timestamp if timestamp is not None else datetime.now(),
        traffic_volume=volume,
        average_speed=speed,
        waiting_time=waiting,
        efficiency=efficiency,
        traffic_delta=volume - p_volume,
        speed_delta=speed - p_speed,
        waiting_delta=waiting - p_waiting,
        efficiency_delta=efficiency - p_efficiency,
    )


def build_signal_status(tick: int, seed: int = 42) -> pd.DataFrame:
    """產生各路口的號誌即時狀態。"""
    rng = _rng(seed + tick)
    phases = ["綠燈", "黃燈", "紅燈"]
    suggestions = ["延長", "正常", "縮短", "切換"]

    rows = []
    for tl_id in INTERSECTION_IDS:
        phase = phases[int(rng.integers(0, len(phases)))]
        rows.append(
            {
                "路口ID": tl_id,
                "當前相位": phase,
                # 黃燈本來就很短，秒數要符合直覺
                "剩餘秒數": int(rng.integers(2, 6) if phase == "黃燈" else rng.integers(8, 60)),
                "等待車輛": int(rng.integers(0, 40)),
                "AI建議": suggestions[int(rng.integers(0, len(suggestions)))],
            }
        )

    return pd.DataFrame(rows)


def build_density_grid(tick: int, seed: int = 42, size: int = 10) -> np.ndarray:
    """產生交通密度熱力圖的資料。"""
    rng = _rng(seed + tick)
    grid = rng.exponential(2.0, (size, size))
    # 讓市中心（中間區塊）密度高一點，看起來比較合理
    centre = np.add.outer(*(np.exp(-0.15 * (np.arange(size) - size / 2) ** 2),) * 2)
    return grid * (1 + centre)


def build_fusion_table(enabled_sources: Dict[str, bool]) -> pd.DataFrame:
    """依照側邊欄勾選的資料源，產生多模態融合狀態表。

    關掉的資料源權重會變成 0，剩下的重新正規化 ──
    這樣勾選框才真的「有作用」，而不是只是畫面上的裝飾。
    """
    rows = []
    for key, label in DATA_SOURCE_LABELS.items():
        volume, latency, weight = DATA_SOURCE_PROFILE[key]
        active = enabled_sources.get(key, True)
        rows.append(
            {
                "資料源": label,
                "啟用": active,
                "資料量 (MB/h)": volume if active else 0,
                "處理延遲 (ms)": latency if active else 0,
                "融合權重": weight if active else 0.0,
            }
        )

    df = pd.DataFrame(rows)
    total = df["融合權重"].sum()
    if total > 0:
        df["融合權重"] = (df["融合權重"] / total).round(3)

    return df


def build_training_curves(epochs: int = 100, seed: int = 42) -> pd.DataFrame:
    """產生訓練損失曲線。"""
    rng = _rng(seed)
    x = np.arange(1, epochs + 1)
    actor = 1.0 * np.exp(-x / 20) + 0.1 + rng.normal(0, 0.03, epochs)
    critic = 0.8 * np.exp(-x / 25) + 0.08 + rng.normal(0, 0.02, epochs)

    return pd.DataFrame(
        {
            "epoch": x,
            "Actor Loss": np.clip(actor, 0, None),
            "Critic Loss": np.clip(critic, 0, None),
        }
    )


def build_efficiency_trend(days: int = 30, seed: int = 42) -> pd.DataFrame:
    """產生近 N 天的系統效率趨勢。"""
    rng = _rng(seed)
    dates = pd.date_range(end=pd.Timestamp.now().floor("D"), periods=days, freq="D")
    values = 70 + 20 * (1 - np.exp(-np.arange(days) / 10)) + rng.normal(0, 1.5, days)

    return pd.DataFrame({"日期": dates, "系統效率": np.clip(values, 0, 100).round(2)})


def build_prediction(tick: int, seed: int = 42, minutes: int = 30) -> pd.DataFrame:
    """產生「現行策略 vs AI 優化策略」的等待時間預測。"""
    rng = _rng(seed + tick)
    t = np.arange(1, minutes + 1)
    current = 100 + 10 * np.sin(t / 5) + rng.normal(0, 4, minutes)
    optimized = 80 + 8 * np.sin(t / 5) + rng.normal(0, 2.5, minutes)

    return pd.DataFrame(
        {"分鐘": t, "現行策略": current.round(1), "AI 優化策略": optimized.round(1)}
    )


def build_system_info(uptime_start: datetime, seed: int = 42) -> List[tuple]:
    """產生系統資訊。

    運行時間改成從啟動時間實際換算，原本是寫死的
    「72天 14小時 32分鐘」，不管什麼時候看都一樣。
    """
    rng = _rng(seed)
    delta = datetime.now() - uptime_start
    days = delta.days
    hours, remainder = divmod(delta.seconds, 3600)
    minutes = remainder // 60

    return [
        ("系統版本", "SmartTraffic AI v2.2.0"),
        ("部署環境", "Kubernetes Cluster"),
        ("運行時間", f"{days} 天 {hours} 小時 {minutes} 分"),
        ("CPU 使用率", f"{rng.uniform(30, 60):.1f}%"),
        ("記憶體使用", f"{rng.uniform(5, 9):.1f} GB / 16 GB"),
        ("GPU 使用率", f"{rng.uniform(60, 90):.1f}%"),
        ("網路延遲", f"{rng.uniform(1, 8):.1f} ms"),
        ("資料處理量", f"{rng.uniform(1.8, 2.8):.1f} TB/天"),
    ]
