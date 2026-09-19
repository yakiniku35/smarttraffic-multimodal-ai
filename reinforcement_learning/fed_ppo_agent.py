"""聯邦式 PPO 智能體（搭配圖神經網路）。

修正的問題：

1. ``compute_gae`` 原本把一串 tensor 丟進 ``torch.tensor(...)``。
   因為 ``values`` 的形狀是 ``[T, N]``（T 步、N 個路口），
   每個 advantage 都是 ``[N]`` 的 tensor，這行會直接丟出
   ``ValueError: only one element tensors can be converted...``。
   改用 ``torch.stack``，並且全程保持 ``[T, N]`` 的形狀。
2. ``evaluate_actions`` 原本把 ``torch.stack`` 出來的 ``[T, 2, E]``
   邊索引直接丟給 ``GCNConv``。GCNConv 的節點特徵可以有前置的 batch 維度
   （``[T, N, F]`` 沒問題），但 ``edge_index`` 一定要是 ``[2, E]``，
   多一個維度就會 ``RuntimeError: Sizes of tensors must match...``。
   改成把 T 個時間步攤平成一張互不相連的大圖（block-diagonal），
   這是 PyTorch Geometric 標準的 batch 做法。
3. Actor 原本輸出 ``Softmax`` 機率再丟給 ``Categorical``，
   數值上不穩定。改成輸出 logits，用 ``Categorical(logits=...)``。
4. ``gamma`` / ``lambda`` / 損失權重原本寫死，改成從 config 讀。
5. 類別叫「聯邦」PPO 卻沒有任何聯邦邏輯，補上 ``federated_average``。
6. 加上 device 處理，不會 CPU / GPU tensor 混在一起。
"""

from __future__ import annotations

import copy
import logging
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import GCNConv

logger = logging.getLogger(__name__)


class FederatedPPOAgent(nn.Module):
    """以圖神經網路為骨幹的 PPO 智能體。

    每個路口是圖上的一個節點，智能體會替每個節點各輸出一個動作，
    所以這其實是「參數共享的多智能體 PPO」。
    """

    def __init__(self, config, graph_structure: Optional[torch.Tensor] = None):
        super().__init__()
        self.config = config
        # 註冊成 buffer，state_dict 存檔 / 搬到 GPU 時會一起處理
        if graph_structure is not None:
            self.register_buffer(
                "graph_structure", graph_structure.long(), persistent=False
            )
        else:
            self.graph_structure = None

        hidden = config.hidden_dim
        gnn_out = config.gnn_output_dim

        self.gnn_layers = nn.ModuleList(
            [
                GCNConv(config.state_dim, hidden),
                GCNConv(hidden, hidden),
                GCNConv(hidden, gnn_out),
            ]
        )

        # Actor：輸出 logits（不要在這裡接 Softmax）
        self.actor = nn.Sequential(
            nn.Linear(gnn_out, hidden),
            nn.ReLU(),
            nn.Linear(hidden, gnn_out),
            nn.ReLU(),
            nn.Linear(gnn_out, config.action_dim),
        )

        # Critic：輸出每個節點的狀態價值
        self.critic = nn.Sequential(
            nn.Linear(gnn_out, hidden),
            nn.ReLU(),
            nn.Linear(hidden, gnn_out),
            nn.ReLU(),
            nn.Linear(gnn_out, 1),
        )

        self.optimizer = optim.Adam(self.parameters(), lr=config.learning_rate)
        self.memory = PPOMemory()
        self.update_count = 0

    # ------------------------------------------------------------------ #
    # 前向傳播
    # ------------------------------------------------------------------ #
    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward_gnn(
        self, node_features: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        """圖神經網路前向傳播，輸入輸出都是 ``[N, F]``。"""
        x = node_features
        last = len(self.gnn_layers) - 1

        for i, layer in enumerate(self.gnn_layers):
            x = layer(x, edge_index)
            if i < last:
                x = F.relu(x)
                x = F.dropout(x, p=self.config.dropout, training=self.training)

        return x

    def forward(
        self, state: torch.Tensor, edge_index: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """回傳 (action_logits, state_values)。"""
        node_embeddings = self.forward_gnn(state, edge_index)
        return self.actor(node_embeddings), self.critic(node_embeddings).squeeze(-1)

    def get_action_and_value(
        self, state: torch.Tensor, edge_index: torch.Tensor, deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """取得動作、log 機率與狀態價值。

        ``deterministic=True`` 時直接取機率最大的動作，推論時比較穩定。
        """
        logits, state_value = self.forward(state, edge_index)
        dist = torch.distributions.Categorical(logits=logits)

        action = logits.argmax(dim=-1) if deterministic else dist.sample()
        return action, dist.log_prob(action), state_value

    def evaluate_actions(
        self, state: torch.Tensor, edge_index: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """重新評估既有動作，PPO 更新時用。"""
        logits, state_values = self.forward(state, edge_index)
        dist = torch.distributions.Categorical(logits=logits)
        return dist.log_prob(action), state_values, dist.entropy()

    # ------------------------------------------------------------------ #
    # PPO 更新
    # ------------------------------------------------------------------ #
    @staticmethod
    def batch_graphs(
        states: torch.Tensor, edge_indices: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """把 T 個時間步的小圖合併成一張大圖。

        ``states`` 形狀 ``[T, N, F]``、``edge_indices`` 形狀 ``[T, 2, E]``，
        回傳 ``[T*N, F]`` 的節點特徵，以及把第 t 步的節點編號整體
        平移 ``t * N`` 之後的邊索引，各個時間步之間彼此不連通。
        """
        num_steps, num_nodes, num_features = states.shape
        flat_states = states.reshape(num_steps * num_nodes, num_features)

        offsets = (
            torch.arange(num_steps, device=edge_indices.device).view(-1, 1, 1)
            * num_nodes
        )
        shifted = edge_indices + offsets  # [T, 2, E]
        flat_edges = shifted.permute(1, 0, 2).reshape(2, -1)

        return flat_states, flat_edges

    def compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        last_value: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """廣義優勢估計（GAE-Lambda）。

        ``rewards`` / ``dones`` 形狀 ``[T]``（整個路網共用），
        ``values`` 形狀 ``[T, N]``，回傳的 advantage 也是 ``[T, N]``。
        """
        gamma = self.config.gamma
        lam = self.config.gae_lambda

        num_steps = values.shape[0]
        advantages = torch.zeros_like(values)

        # 最後一步之後的 bootstrap value，沒提供就當作 0
        next_value = (
            torch.zeros_like(values[0]) if last_value is None else last_value
        )
        gae = torch.zeros_like(values[0])

        # rewards / dones 是 [T]，values 是 [T, N]，用 unsqueeze 讓它們廣播
        rewards = rewards.view(-1, 1)
        dones = dones.view(-1, 1)

        for t in reversed(range(num_steps)):
            non_terminal = 1.0 - dones[t]
            delta = rewards[t] + gamma * next_value * non_terminal - values[t]
            gae = delta + gamma * lam * non_terminal * gae
            advantages[t] = gae
            next_value = values[t]

        return advantages

    def update(self) -> Dict[str, float]:
        """執行一次 PPO 更新，資料不足時回傳空 dict。"""
        if len(self.memory) < self.config.batch_size:
            return {}

        states, edge_indices, actions, old_logprobs, rewards, dones, values = (
            self.memory.get_batch(self.device)
        )

        with torch.no_grad():
            advantages = self.compute_gae(rewards, values, dones)
            returns = advantages + values
            # 正規化 advantage，讓不同 batch 的梯度尺度接近
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        flat_states, flat_edges = self.batch_graphs(states, edge_indices)
        flat_actions = actions.reshape(-1)
        flat_old_logprobs = old_logprobs.reshape(-1)
        flat_advantages = advantages.reshape(-1)
        flat_returns = returns.reshape(-1)

        totals = {"total": 0.0, "actor": 0.0, "critic": 0.0, "entropy": 0.0}

        for _ in range(self.config.n_epochs):
            new_logprobs, new_values, entropy = self.evaluate_actions(
                flat_states, flat_edges, flat_actions
            )

            ratio = torch.exp(new_logprobs - flat_old_logprobs)

            surr1 = ratio * flat_advantages
            surr2 = (
                torch.clamp(
                    ratio, 1 - self.config.clip_range, 1 + self.config.clip_range
                )
                * flat_advantages
            )
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = F.mse_loss(new_values, flat_returns)
            entropy_loss = -entropy.mean()

            loss = (
                actor_loss
                + self.config.value_coef * critic_loss
                + self.config.entropy_coef * entropy_loss
            )

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), self.config.max_grad_norm)
            self.optimizer.step()

            totals["total"] += loss.item()
            totals["actor"] += actor_loss.item()
            totals["critic"] += critic_loss.item()
            totals["entropy"] += entropy_loss.item()

        self.memory.clear()
        self.update_count += 1

        n = self.config.n_epochs
        return {
            "total_loss": totals["total"] / n,
            "actor_loss": totals["actor"] / n,
            "critic_loss": totals["critic"] / n,
            "entropy_loss": totals["entropy"] / n,
        }

    # ------------------------------------------------------------------ #
    # 聯邦式學習
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def federated_average(
        self,
        client_agents: Sequence["FederatedPPOAgent"],
        weights: Optional[Sequence[float]] = None,
    ) -> None:
        """FedAvg：把多個地區模型的權重加權平均後寫回自己。

        ``weights`` 通常用各地區的資料量當權重；不給就一律等權重。
        """
        if not client_agents:
            raise ValueError("client_agents 不可以是空的")

        if weights is None:
            weights = [1.0 / len(client_agents)] * len(client_agents)
        else:
            if len(weights) != len(client_agents):
                raise ValueError("weights 長度必須和 client_agents 相同")
            total = float(sum(weights))
            if total <= 0:
                raise ValueError("weights 總和必須大於 0")
            weights = [w / total for w in weights]

        global_state = self.state_dict()
        averaged = {
            key: torch.zeros_like(value, dtype=torch.float32)
            for key, value in global_state.items()
            if value.is_floating_point()
        }

        for agent, weight in zip(client_agents, weights):
            client_state = agent.state_dict()
            for key in averaged:
                averaged[key] += client_state[key].to(averaged[key].device) * weight

        for key, value in averaged.items():
            global_state[key].copy_(value)

        logger.info("已完成 %d 個地區模型的聯邦平均", len(client_agents))

    def should_sync(self) -> bool:
        """依 ``federated_sync_interval`` 判斷這輪要不要做聯邦同步。"""
        interval = self.config.federated_sync_interval
        return interval > 0 and self.update_count > 0 and self.update_count % interval == 0

    def clone(self) -> "FederatedPPOAgent":
        """複製一份智能體，用來當作某個地區的本地模型。"""
        graph = getattr(self, "graph_structure", None)
        clone = FederatedPPOAgent(copy.deepcopy(self.config), graph)
        clone.load_state_dict(self.state_dict())
        return clone


class PPOMemory:
    """PPO 的經驗緩衝區（on-policy，每次更新後清空）。"""

    def __init__(self) -> None:
        self.states: List[torch.Tensor] = []
        self.edge_indices: List[torch.Tensor] = []
        self.actions: List[torch.Tensor] = []
        self.logprobs: List[torch.Tensor] = []
        self.rewards: List[float] = []
        self.dones: List[bool] = []
        self.values: List[torch.Tensor] = []

    def store(self, state, edge_index, action, logprob, reward, done, value) -> None:
        # detach 一下，避免把整張計算圖留在 buffer 裡造成記憶體暴增
        self.states.append(state.detach())
        self.edge_indices.append(edge_index.detach())
        self.actions.append(action.detach())
        self.logprobs.append(logprob.detach())
        self.rewards.append(float(reward))
        self.dones.append(bool(done))
        self.values.append(value.detach())

    def get_batch(self, device: Optional[torch.device] = None):
        """把緩衝區疊成 batch tensor。

        所有時間步的邊索引必須一樣長才能 stack；
        路網在一次 episode 內不會變，所以正常情況下都成立，
        不成立時給明確的錯誤訊息比 RuntimeError 好懂。
        """
        if not self.states:
            raise ValueError("緩衝區是空的，沒有資料可以取出")

        edge_sizes = {tuple(e.shape) for e in self.edge_indices}
        if len(edge_sizes) > 1:
            raise ValueError(
                f"每一步的 edge_index 形狀必須相同，目前有 {sorted(edge_sizes)}"
            )

        batch = (
            torch.stack(self.states),
            torch.stack(self.edge_indices),
            torch.stack(self.actions),
            torch.stack(self.logprobs),
            torch.tensor(self.rewards, dtype=torch.float32),
            torch.tensor(self.dones, dtype=torch.float32),
            torch.stack(self.values),
        )

        if device is not None:
            batch = tuple(t.to(device) for t in batch)

        return batch

    def clear(self) -> None:
        self.states.clear()
        self.edge_indices.clear()
        self.actions.clear()
        self.logprobs.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()

    def __len__(self) -> int:
        return len(self.states)
