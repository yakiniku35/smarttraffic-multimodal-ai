"""交通模擬環境的單元測試。

這些測試只用 MockTrafficEnvironment，所以不需要安裝 SUMO。
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="需要安裝 PyTorch")

from config import TrafficConfig  # noqa: E402
from traffic_simulation.sumo_interface import (  # noqa: E402
    NODE_FEATURE_DIM,
    BaseTrafficEnvironment,
    MockTrafficEnvironment,
    _safe_mean,
    _safe_std,
    create_environment,
)


@pytest.fixture
def config():
    return TrafficConfig(num_intersections=6, simulation_time=20, random_seed=7)


def test_module_imports_without_sumo():
    """以前檔案開頭有 `import sumo`，沒裝就整包 import 失敗。"""
    import traffic_simulation.sumo_interface as module

    assert not hasattr(module, "sumo")


def test_safe_mean_and_std_on_empty_input():
    """空 list 要回傳 0.0，不是 nan。"""
    assert _safe_mean([]) == 0.0
    assert _safe_std([]) == 0.0
    assert _safe_mean([1.0, 3.0]) == 2.0


def test_create_environment_falls_back_to_mock(config):
    env = create_environment(config, force_mock=True)
    assert isinstance(env, MockTrafficEnvironment)


def test_create_environment_raises_when_fallback_disabled(monkeypatch, config):
    import traffic_simulation.sumo_interface as module

    monkeypatch.setattr(module, "TRACI_AVAILABLE", False)
    config.fallback_to_mock = False

    with pytest.raises(RuntimeError, match="traci"):
        create_environment(config)


def test_reset_returns_correct_shapes(config):
    env = MockTrafficEnvironment(config)
    state, edge_index = env.reset()

    assert state.shape == (config.num_intersections, NODE_FEATURE_DIM)
    assert state.dtype == torch.float32
    assert edge_index.shape[0] == 2
    assert edge_index.dtype == torch.long


def test_edge_index_stays_in_range(config):
    env = MockTrafficEnvironment(config)
    _, edge_index = env.reset()

    assert int(edge_index.max()) < config.num_intersections
    assert int(edge_index.min()) >= 0


def test_edge_index_is_stable_across_steps(config):
    """路網在一個 episode 內不會變，PPOMemory 才能把它們 stack 起來。"""
    env = MockTrafficEnvironment(config)
    _, first = env.reset()

    for _ in range(3):
        _, edge_index, *_ = env.step([0] * config.num_intersections)
        assert torch.equal(first, edge_index)


def test_edge_index_falls_back_to_self_loops():
    """路口距離都超過閾值時，要退回自環而不是空的邊索引。"""
    positions = [(0.0, 0.0), (10_000.0, 0.0), (0.0, 10_000.0)]
    edge_index = BaseTrafficEnvironment._edge_index_from_positions(positions, 100.0)

    assert edge_index.shape == (2, 3)
    assert torch.equal(edge_index[0], edge_index[1])


def test_step_returns_expected_structure(config):
    env = MockTrafficEnvironment(config)
    env.reset()

    state, edge_index, rewards, dones, info = env.step([1] * config.num_intersections)

    assert state.shape == (config.num_intersections, NODE_FEATURE_DIM)
    assert len(rewards) == config.num_intersections
    assert len(dones) == config.num_intersections
    assert {"step", "total_vehicles", "average_speed"} <= set(info)
    assert all(isinstance(r, float) for r in rewards)


def test_step_rejects_wrong_action_count(config):
    """動作數量不對時要明確報錯，而不是安靜地少控制幾個路口。"""
    env = MockTrafficEnvironment(config)
    env.reset()

    with pytest.raises(ValueError, match="動作數量"):
        env.step([0, 1])


def test_done_flag_after_simulation_time(config):
    env = MockTrafficEnvironment(config)
    env.reset()

    for _ in range(config.simulation_time - 1):
        *_, dones, _ = env.step([0] * config.num_intersections)
        assert not any(dones)

    *_, dones, _ = env.step([0] * config.num_intersections)
    assert all(dones)


def test_state_has_no_nan_or_inf(config):
    env = MockTrafficEnvironment(config)
    env.reset()

    for _ in range(15):
        state, *_ = env.step([1, 0, 1, 0, 1, 0])
        assert torch.isfinite(state).all()


def test_same_seed_gives_same_trajectory(config):
    env_a = MockTrafficEnvironment(config)
    env_b = MockTrafficEnvironment(TrafficConfig(**vars(config)))

    env_a.reset()
    env_b.reset()

    actions = [1, 0, 1, 0, 1, 0]
    state_a, *_ = env_a.step(actions)
    state_b, *_ = env_b.step(actions)

    assert torch.allclose(state_a, state_b)


def test_close_is_idempotent(config):
    env = MockTrafficEnvironment(config)
    env.close()
    env.close()  # 重複呼叫不應該出錯
