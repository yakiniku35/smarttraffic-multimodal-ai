import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GraphConv
from torch_geometric.data import Data, Batch
import numpy as np
from typing import Dict, List, Tuple, Optional
import copy

class FederatedPPOAgent(nn.Module):
    """聯邦PPO智能體"""
    
    def __init__(self, config, graph_structure):
        super().__init__()
        self.config = config
        self.graph_structure = graph_structure
        
        # 圖神經網路層
        self.gnn_layers = nn.ModuleList([
            GCNConv(config.state_dim, 128),
            GCNConv(128, 128),
            GCNConv(128, 64)
        ])
        
        # Actor網路（策略網路）
        self.actor = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, config.action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Critic網路（價值網路）
        self.critic = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # 優化器
        self.optimizer = optim.Adam(self.parameters(), lr=config.learning_rate)
        
        # 經驗緩衝區
        self.memory = PPOMemory()
        
    def forward_gnn(self, node_features: torch.Tensor, 
                    edge_index: torch.Tensor) -> torch.Tensor:
        """圖神經網路前向傳播"""
        x = node_features
        
        for i, gnn_layer in enumerate(self.gnn_layers):
            x = gnn_layer(x, edge_index)
            if i < len(self.gnn_layers) - 1:
                x = F.relu(x)
                x = F.dropout(x, training=self.training)
        
        return x
    
    def get_action_and_value(self, state: torch.Tensor, 
                           edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """獲取動作和價值"""
        # GNN特徵提取
        node_embeddings = self.forward_gnn(state, edge_index)
        
        # Actor預測動作概率
        action_probs = self.actor(node_embeddings)
        action_dist = torch.distributions.Categorical(action_probs)
        
        # 採樣動作
        action = action_dist.sample()
        action_logprob = action_dist.log_prob(action)
        
        # Critic預測價值
        state_value = self.critic(node_embeddings).squeeze(-1)
        
        return action, action_logprob, state_value
    
    def evaluate_actions(self, state: torch.Tensor, 
                        edge_index: torch.Tensor, 
                        action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """評估動作"""
        node_embeddings = self.forward_gnn(state, edge_index)
        
        action_probs = self.actor(node_embeddings)
        action_dist = torch.distributions.Categorical(action_probs)
        
        action_logprobs = action_dist.log_prob(action)
        entropy = action_dist.entropy()
        state_values = self.critic(node_embeddings).squeeze(-1)
        
        return action_logprobs, state_values, entropy
    
    def update(self) -> Dict[str, float]:
        """PPO更新"""
        if len(self.memory) < self.config.batch_size:
            return {}
        
        # 獲取經驗數據
        states, edge_indices, actions, logprobs, rewards, dones, values = self.memory.get_batch()
        
        # 計算優勢函數
        advantages = self.compute_gae(rewards, values, dones)
        returns = advantages + values
        
        # 正規化優勢
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO更新
        total_loss = 0
        for epoch in range(self.config.n_epochs):
            # 重新評估動作
            new_logprobs, new_values, entropy = self.evaluate_actions(states, edge_indices, actions)
            
            # 計算比率
            ratio = torch.exp(new_logprobs - logprobs)
            
            # 計算替代損失
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.config.clip_range, 1 + self.config.clip_range) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # 價值損失
            critic_loss = F.mse_loss(new_values, returns)
            
            # 熵損失
            entropy_loss = -entropy.mean()
            
            # 總損失
            loss = actor_loss + 0.5 * critic_loss + 0.01 * entropy_loss
            
            # 反向傳播
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        # 清空記憶體
        self.memory.clear()
        
        return {
            'total_loss': total_loss / self.config.n_epochs,
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'entropy_loss': entropy_loss.item()
        }
    
    def compute_gae(self, rewards: torch.Tensor, 
                    values: torch.Tensor, 
                    dones: torch.Tensor, 
                    gamma: float = 0.99, 
                    lam: float = 0.95) -> torch.Tensor:
        """計算廣義優勢估計（GAE）"""
        advantages = []
        gae = 0
        
        for i in reversed(range(len(rewards))):
            if i == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[i + 1]
            
            delta = rewards[i] + gamma * next_value * (1 - dones[i]) - values[i]
            gae = delta + gamma * lam * (1 - dones[i]) * gae
            advantages.insert(0, gae)
        
        return torch.tensor(advantages, dtype=torch.float32)

class PPOMemory:
    """PPO經驗緩衝區"""
    
    def __init__(self):
        self.states = []
        self.edge_indices = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.dones = []
        self.values = []
    
    def store(self, state, edge_index, action, logprob, reward, done, value):
        self.states.append(state)
        self.edge_indices.append(edge_index)
        self.actions.append(action)
        self.logprobs.append(logprob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)
    
    def get_batch(self):
        return (
            torch.stack(self.states),
            torch.stack(self.edge_indices),
            torch.stack(self.actions),
            torch.stack(self.logprobs),
            torch.tensor(self.rewards, dtype=torch.float32),
            torch.tensor(self.dones, dtype=torch.float32),
            torch.stack(self.values)
        )
    
    def clear(self):
        self.states.clear()
        self.edge_indices.clear()
        self.actions.clear()
        self.logprobs.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()
    
    def __len__(self):
        return len(self.states)
