"""聯邦 PPO 智能體的單元測試。

需要 torch 與 torch-geometric，沒安裝就整組跳過。
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="需要安裝 PyTorch")
pytest.importorskip("torch_geometric", reason="需要安裝 torch-geometric")

from config import RLConfig  # noqa: E402
from reinforcement_learning.fed_ppo_agent import (  # noqa: E402
    FederatedPPOAgent,
    PPOMemory,
)

NUM_NODES = 6
NUM_STEPS = 8


@pytest.fixture
def config():
    return RLConfig(
        state_dim=8,
        action_dim=2,
        batch_size=4,
        n_epochs=2,
        hidden_dim=32,
        gnn_output_dim=16,
    )


@pytest.fixture
def edge_index():
    # 一條環狀路網：0-1-2-...-5-0（雙向）
    src = list(range(NUM_NODES)) + [(i + 1) % NUM_NODES for i in range(NUM_NODES)]
    dst = [(i + 1) % NUM_NODES for i in range(NUM_NODES)] + list(range(NUM_NODES))
    return torch.tensor([src, dst], dtype=torch.long)


@pytest.fixture
def agent(config, edge_index):
    torch.manual_seed(0)
    return FederatedPPOAgent(config, edge_index)


def test_get_action_and_value_shapes(agent, edge_index, config):
    state = torch.randn(NUM_NODES, config.state_dim)
    action, logprob, value = agent.get_action_and_value(state, edge_index)

    assert action.shape == (NUM_NODES,)
    assert logprob.shape == (NUM_NODES,)
    assert value.shape == (NUM_NODES,)
    assert action.min() >= 0 and action.max() < config.action_dim


def test_deterministic_action_is_repeatable(agent, edge_index, config):
    agent.eval()
    state = torch.randn(NUM_NODES, config.state_dim)

    a1, _, _ = agent.get_action_and_value(state, edge_index, deterministic=True)
    a2, _, _ = agent.get_action_and_value(state, edge_index, deterministic=True)

    assert torch.equal(a1, a2)


def test_batch_graphs_offsets_node_indices(agent, edge_index, config):
    """T 個時間步要被攤平成一張互不相連的大圖。"""
    states = torch.randn(NUM_STEPS, NUM_NODES, config.state_dim)
    edges = edge_index.unsqueeze(0).repeat(NUM_STEPS, 1, 1)

    flat_states, flat_edges = agent.batch_graphs(states, edges)

    assert flat_states.shape == (NUM_STEPS * NUM_NODES, config.state_dim)
    assert flat_edges.shape == (2, NUM_STEPS * edge_index.shape[1])
    assert int(flat_edges.max()) < NUM_STEPS * NUM_NODES
    # 第二個時間步的節點編號要整體 +NUM_NODES
    second_block = flat_edges[:, edge_index.shape[1] : 2 * edge_index.shape[1]]
    assert torch.equal(second_block, edge_index + NUM_NODES)


def test_batch_graphs_keeps_timesteps_disconnected(agent, edge_index, config):
    """不同時間步之間不可以有邊相連，否則資訊會互相汙染。"""
    states = torch.randn(3, NUM_NODES, config.state_dim)
    edges = edge_index.unsqueeze(0).repeat(3, 1, 1)

    _, flat_edges = agent.batch_graphs(states, edges)

    src_block = flat_edges[0] // NUM_NODES
    dst_block = flat_edges[1] // NUM_NODES
    assert torch.equal(src_block, dst_block)


def test_compute_gae_returns_per_node_advantages(agent):
    """這是舊版最嚴重的 bug：torch.tensor(一串 [N] 的 tensor) 會直接爆掉。"""
    rewards = torch.randn(NUM_STEPS)
    values = torch.randn(NUM_STEPS, NUM_NODES)
    dones = torch.zeros(NUM_STEPS)

    advantages = agent.compute_gae(rewards, values, dones)

    assert advantages.shape == (NUM_STEPS, NUM_NODES)
    assert torch.isfinite(advantages).all()


def test_compute_gae_matches_manual_calculation(agent):
    """用一個小例子手算 GAE，確認公式沒寫錯。"""
    agent.config.gamma = 0.9
    agent.config.gae_lambda = 0.5

    rewards = torch.tensor([1.0, 2.0])
    values = torch.tensor([[0.5], [1.0]])
    dones = torch.tensor([0.0, 0.0])

    advantages = agent.compute_gae(rewards, values, dones)

    # t=1（最後一步，bootstrap value = 0）
    delta1 = 2.0 + 0.9 * 0.0 - 1.0
    gae1 = delta1
    # t=0
    delta0 = 1.0 + 0.9 * 1.0 - 0.5
    gae0 = delta0 + 0.9 * 0.5 * gae1

    assert advantages[0, 0].item() == pytest.approx(gae0)
    assert advantages[1, 0].item() == pytest.approx(gae1)


def test_compute_gae_respects_done_flags(agent):
    """episode 結束的那一步，不應該把下一步的價值算進來。"""
    agent.config.gamma = 0.9
    agent.config.gae_lambda = 1.0

    rewards = torch.tensor([1.0, 1.0])
    values = torch.tensor([[2.0], [2.0]])
    dones = torch.tensor([1.0, 0.0])  # 第 0 步就結束了

    advantages = agent.compute_gae(rewards, values, dones)

    # done=1 時 delta = reward - value，不加折扣後的下一個價值
    assert advantages[0, 0].item() == pytest.approx(1.0 - 2.0)


def test_update_runs_and_changes_parameters(agent, edge_index, config):
    """完整跑一次 PPO 更新，確認不會丟例外而且參數真的有變。"""
    torch.manual_seed(0)
    before = agent.actor[0].weight.detach().clone()

    for _ in range(NUM_STEPS):
        state = torch.randn(NUM_NODES, config.state_dim)
        with torch.no_grad():
            action, logprob, value = agent.get_action_and_value(state, edge_index)
        agent.memory.store(state, edge_index, action, logprob, 1.0, False, value)

    losses = agent.update()

    assert set(losses) == {"total_loss", "actor_loss", "critic_loss", "entropy_loss"}
    assert all(isinstance(v, float) for v in losses.values())
    assert not torch.equal(before, agent.actor[0].weight)
    assert len(agent.memory) == 0  # 更新後緩衝區要清空


def test_update_skips_when_not_enough_data(agent, edge_index, config):
    state = torch.randn(NUM_NODES, config.state_dim)
    with torch.no_grad():
        action, logprob, value = agent.get_action_and_value(state, edge_index)
    agent.memory.store(state, edge_index, action, logprob, 1.0, False, value)

    assert agent.update() == {}
    assert len(agent.memory) == 1  # 沒更新就不該清空


def test_memory_detaches_tensors(agent, edge_index, config):
    """緩衝區不該保留計算圖，否則記憶體會一直長大。"""
    state = torch.randn(NUM_NODES, config.state_dim)
    action, logprob, value = agent.get_action_and_value(state, edge_index)
    agent.memory.store(state, edge_index, action, logprob, 1.0, False, value)

    assert not agent.memory.logprobs[0].requires_grad
    assert not agent.memory.values[0].requires_grad


def test_memory_rejects_mismatched_edge_shapes(agent, edge_index, config):
    state = torch.randn(NUM_NODES, config.state_dim)
    with torch.no_grad():
        action, logprob, value = agent.get_action_and_value(state, edge_index)

    agent.memory.store(state, edge_index, action, logprob, 1.0, False, value)
    smaller = edge_index[:, :4]
    agent.memory.store(state, smaller, action, logprob, 1.0, False, value)

    with pytest.raises(ValueError, match="edge_index"):
        agent.memory.get_batch()


def test_empty_memory_raises():
    with pytest.raises(ValueError, match="空的"):
        PPOMemory().get_batch()


def test_federated_average_is_the_mean_of_clients(agent):
    """FedAvg：兩個 client 等權重平均，結果要落在中間。"""
    client_a = agent.clone()
    client_b = agent.clone()

    with torch.no_grad():
        for param in client_a.parameters():
            param.fill_(1.0)
        for param in client_b.parameters():
            param.fill_(3.0)

    agent.federated_average([client_a, client_b])

    for param in agent.parameters():
        assert torch.allclose(param, torch.full_like(param, 2.0))


def test_federated_average_respects_weights(agent):
    client_a = agent.clone()
    client_b = agent.clone()

    with torch.no_grad():
        for param in client_a.parameters():
            param.fill_(0.0)
        for param in client_b.parameters():
            param.fill_(4.0)

    # 權重 3:1，結果應該是 0*0.75 + 4*0.25 = 1.0
    agent.federated_average([client_a, client_b], weights=[3.0, 1.0])

    for param in agent.parameters():
        assert torch.allclose(param, torch.full_like(param, 1.0))


def test_federated_average_rejects_bad_input(agent):
    with pytest.raises(ValueError, match="空的"):
        agent.federated_average([])

    with pytest.raises(ValueError, match="長度"):
        agent.federated_average([agent.clone()], weights=[1.0, 2.0])


def test_should_sync_follows_interval(agent):
    agent.config.federated_sync_interval = 3

    agent.update_count = 0
    assert not agent.should_sync()
    agent.update_count = 3
    assert agent.should_sync()
    agent.update_count = 4
    assert not agent.should_sync()

    agent.config.federated_sync_interval = 0
    agent.update_count = 3
    assert not agent.should_sync()


def test_federated_average_rejects_negative_and_nan_weights(agent):
    """只檢查總和是不夠的。

    [-1, 2] 的總和剛好是 1，負權重會把參數外推到所有地區模型之外；
    [nan, 1] 的總和是 nan，`total <= 0` 也判斷為 False，
    結果是把 nan 寫進全域模型。
    """
    clients = [agent.clone(), agent.clone()]

    for bad in ([-1.0, 2.0], [float("nan"), 1.0], [float("inf"), 1.0]):
        with pytest.raises(ValueError, match="有限的非負數"):
            agent.federated_average(clients, weights=bad)


def test_federated_average_allows_zero_weight(agent):
    """權重 0 是合理的（某個地區這輪沒有資料），不該被擋下來。"""
    a, b = agent.clone(), agent.clone()
    with torch.no_grad():
        for p in a.parameters():
            p.fill_(5.0)
        for p in b.parameters():
            p.fill_(1.0)

    agent.federated_average([a, b], weights=[0.0, 1.0])
    for p in agent.parameters():
        assert torch.allclose(p, torch.full_like(p, 1.0))


def test_clone_keeps_source_device(agent):
    """clone() 要留在原本的裝置上。

    nn.Module 一律在 CPU 建立參數，load_state_dict 只複製數值不搬裝置，
    所以複製 GPU 模型時若少了 .to()，拿到的會是 CPU 模型。
    """
    clone = agent.clone()
    assert clone.device == agent.device

    if torch.cuda.is_available():  # pragma: no cover - CI 通常沒有 GPU
        gpu_agent = agent.to("cuda")
        assert gpu_agent.clone().device.type == "cuda"
