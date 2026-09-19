"""網頁介面資料來源的單元測試。

這一組測試不需要 PyTorch，也不需要真的啟動 Streamlit。
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from web_interface import data_source as ds


def test_traffic_history_shape_and_columns():
    df = ds.build_traffic_history(seed=1, hours=24)

    assert len(df) == 24
    assert "timestamp" in df.columns
    for tl_id in ds.INTERSECTION_IDS:
        assert tl_id in df.columns


def test_traffic_history_is_deterministic():
    """同樣的種子必須產生同樣的資料。

    舊版每次 rerun 都重新亂數，導致按任何按鈕整頁數字都在跳。
    """
    a = ds.build_traffic_history(seed=42)
    b = ds.build_traffic_history(seed=42)

    pd.testing.assert_frame_equal(a, b)


def test_traffic_history_uses_recent_timestamps():
    """日期要以「現在」為基準，不能寫死成過去的某一天。"""
    df = ds.build_traffic_history(seed=1, hours=24)
    latest = df["timestamp"].max()

    assert abs((pd.Timestamp.now() - latest).total_seconds()) < 3600 * 2


def test_traffic_volumes_are_non_negative():
    df = ds.build_traffic_history(seed=3)
    for tl_id in ds.INTERSECTION_IDS:
        assert (df[tl_id] >= 0).all()


def test_live_snapshot_is_stable_for_same_tick():
    a = ds.build_live_snapshot(tick=5, seed=42)
    b = ds.build_live_snapshot(tick=5, seed=42)

    assert a.traffic_volume == b.traffic_volume
    assert a.average_speed == pytest.approx(b.average_speed)


def test_live_snapshot_changes_between_ticks():
    a = ds.build_live_snapshot(tick=1, seed=42)
    b = ds.build_live_snapshot(tick=2, seed=42)

    assert (a.traffic_volume, a.average_speed) != (b.traffic_volume, b.average_speed)


def test_live_snapshot_values_in_range():
    for tick in range(10):
        snap = ds.build_live_snapshot(tick, seed=42)
        assert 150 <= snap.traffic_volume <= 300
        assert 25 <= snap.average_speed <= 45
        assert 0 <= snap.efficiency <= 100


def test_signal_status_columns_and_phase_values():
    df = ds.build_signal_status(tick=0, seed=42)

    assert list(df["路口ID"]) == ds.INTERSECTION_IDS
    assert set(df["當前相位"]) <= {"綠燈", "黃燈", "紅燈"}
    assert (df["等待車輛"] >= 0).all()


def test_yellow_phase_has_short_remaining_time():
    """黃燈剩 45 秒不合常理，這裡確認秒數符合直覺。"""
    for tick in range(30):
        df = ds.build_signal_status(tick, seed=42)
        yellow = df[df["當前相位"] == "黃燈"]
        assert (yellow["剩餘秒數"] <= 6).all()


def test_fusion_weights_sum_to_one():
    enabled = {key: True for key in ds.DATA_SOURCE_LABELS}
    df = ds.build_fusion_table(enabled)

    assert df["融合權重"].sum() == pytest.approx(1.0, abs=1e-2)


def test_fusion_weights_renormalise_when_sources_disabled():
    """關掉資料源之後，剩下的權重要重新分配成 1.0。"""
    enabled = {key: key in ("camera", "gps") for key in ds.DATA_SOURCE_LABELS}
    df = ds.build_fusion_table(enabled)

    active = df[df["啟用"]]
    assert len(active) == 2
    assert active["融合權重"].sum() == pytest.approx(1.0, abs=1e-2)
    assert df[~df["啟用"]]["融合權重"].sum() == 0.0


def test_fusion_table_handles_all_sources_disabled():
    """全部關掉時不可以除以零。"""
    enabled = {key: False for key in ds.DATA_SOURCE_LABELS}
    df = ds.build_fusion_table(enabled)

    assert df["融合權重"].sum() == 0.0
    assert np.isfinite(df["融合權重"]).all()


def test_density_grid_shape_and_positive():
    grid = ds.build_density_grid(tick=0, seed=42, size=10)

    assert grid.shape == (10, 10)
    assert (grid >= 0).all()
    assert np.isfinite(grid).all()


def test_training_curves_are_positive_and_decreasing():
    df = ds.build_training_curves(epochs=100, seed=42)

    assert len(df) == 100
    assert (df["Actor Loss"] >= 0).all()
    # 後段的損失應該明顯比前段低
    assert df["Actor Loss"][:10].mean() > df["Actor Loss"][-10:].mean()


def test_efficiency_trend_bounds():
    df = ds.build_efficiency_trend(days=30, seed=42)

    assert len(df) == 30
    assert df["系統效率"].between(0, 100).all()


def test_prediction_shows_improvement():
    """AI 優化策略的等待時間應該低於現行策略。"""
    df = ds.build_prediction(tick=0, seed=42, minutes=30)

    assert len(df) == 30
    assert df["AI 優化策略"].mean() < df["現行策略"].mean()


def test_system_info_uptime_is_computed():
    """運行時間要真的從啟動時間算出來，不是寫死的字串。"""
    started = datetime.now() - timedelta(days=2, hours=3, minutes=15)
    info = dict(ds.build_system_info(started, seed=42))

    assert info["運行時間"].startswith("2 天 3 小時")


def test_system_info_has_expected_keys():
    info = dict(ds.build_system_info(datetime.now(), seed=42))

    for key in ("系統版本", "CPU 使用率", "記憶體使用", "網路延遲"):
        assert key in info


def test_snapshot_timestamp_is_caller_supplied():
    """同一批資料重複取用時，「最後更新」時間不可以自己往前跑。

    之前 build_live_snapshot 內部直接呼叫 datetime.now()，
    畫面每重新執行一次就顯示一個新時間，看起來像有新資料進來。
    """
    fixed = datetime(2026, 1, 1, 12, 30, 0)

    a = ds.build_live_snapshot(tick=3, seed=42, timestamp=fixed)
    b = ds.build_live_snapshot(tick=3, seed=42, timestamp=fixed)

    assert a == b  # 整個 dataclass 相等，包含時間戳
    assert a.timestamp == fixed


def test_snapshot_timestamp_defaults_to_now():
    """沒有傳時間戳時仍然可用（維持向後相容）。"""
    before = datetime.now()
    snap = ds.build_live_snapshot(tick=0, seed=42)
    assert before <= snap.timestamp <= datetime.now()
