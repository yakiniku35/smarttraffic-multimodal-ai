import torch
import numpy as np
from config import SystemConfig
from multimodal.data_fusion import MultimodalFusionNetwork
from reinforcement_learning.fed_ppo_agent import FederatedPPOAgent
from traffic_simulation.sumo_interface import SUMOTrafficEnvironment
import argparse
import logging

def setup_logging():
    """設置日誌"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/smarttraffic.log'),
            logging.StreamHandler()
        ]
    )

def train_system(config: SystemConfig):
    """訓練系統"""
    logger = logging.getLogger(__name__)
    logger.info("開始訓練多模態AI交通優化系統")
    
    # 初始化環境
    env = SUMOTrafficEnvironment(config.traffic)
    
    # 初始化多模態融合網路
    multimodal_net = MultimodalFusionNetwork(config.multimodal)
    
    # 初始化聯邦PPO智能體
    graph_structure = env.create_edge_index()
    config.rl.state_dim = 8  # 特徵維度
    config.rl.action_dim = 2  # 動作維度
    agent = FederatedPPOAgent(config.rl, graph_structure)
    
    # 訓練循環
    num_episodes = 1000
    best_reward = float('-inf')
    
    for episode in range(num_episodes):
        state, edge_index = env.reset()
        episode_reward = 0
        
        for step in range(config.traffic.simulation_time):
            # 獲取動作
            with torch.no_grad():
                action, action_logprob, state_value = agent.get_action_and_value(state, edge_index)
            
            # 執行動作
            next_state, next_edge_index, rewards, dones, info = env.step(action.cpu().numpy())
            
            # 儲存經驗
            agent.memory.store(
                state, edge_index, action, action_logprob, 
                np.mean(rewards), any(dones), state_value
            )
            
            episode_reward += np.mean(rewards)
            state, edge_index = next_state, next_edge_index
            
            if any(dones):
                break
        
        # 更新智能體
        if len(agent.memory) >= config.rl.batch_size:
            loss_info = agent.update()
            
            if loss_info:
                logger.info(f"Episode {episode}: Reward={episode_reward:.2f}, Loss={loss_info['total_loss']:.4f}")
        
        # 保存最佳模型
        if episode_reward > best_reward:
            best_reward = episode_reward
            torch.save({
                'multimodal_net': multimodal_net.state_dict(),
                'agent': agent.state_dict(),
                'episode': episode,
                'reward': episode_reward
            }, 'models/best_model.pth')
            logger.info(f"新的最佳模型已保存，獎勵: {episode_reward:.2f}")
    
    env.close()
    logger.info("訓練完成")

def run_inference(config: SystemConfig):
    """運行推理"""
    logger = logging.getLogger(__name__)
    logger.info("開始運行推理模式")
    
    # 載入訓練好的模型
    checkpoint = torch.load('models/best_model.pth')
    
    # 初始化網路
    multimodal_net = MultimodalFusionNetwork(config.multimodal)
    multimodal_net.load_state_dict(checkpoint['multimodal_net'])
    
    # 初始化環境
    env = SUMOTrafficEnvironment(config.traffic)
    
    # 初始化智能體
    graph_structure = env.create_edge_index()
    config.rl.state_dim = 8
    config.rl.action_dim = 2
    agent = FederatedPPOAgent(config.rl, graph_structure)
    agent.load_state_dict(checkpoint['agent'])
    
    # 運行推理
    state, edge_index = env.reset()
    total_reward = 0
    
    for step in range(config.traffic.simulation_time):
        with torch.no_grad():
            action, _, _ = agent.get_action_and_value(state, edge_index)
        
        next_state, next_edge_index, rewards, dones, info = env.step(action.cpu().numpy())
        total_reward += np.mean(rewards)
        
        # 記錄關鍵指標
        if step % 100 == 0:
            logger.info(f"Step {step}: Average Speed={info['average_speed']:.2f}, Total Vehicles={info['total_vehicles']}")
        
        state, edge_index = next_state, next_edge_index
        
        if any(dones):
            break
    
    logger.info(f"推理完成，總獎勵: {total_reward:.2f}")
    env.close()

def main():
    parser = argparse.ArgumentParser(description='多模態AI智慧城市交通優化系統')
    parser.add_argument('--mode', choices=['train', 'inference', 'web'], default='web',
                       help='運行模式: train(訓練), inference(推理), web(網頁介面)')
    parser.add_argument('--config', type=str, help='配置文件路徑')
    
    args = parser.parse_args()
    
    # 設置日誌
    setup_logging()
    
    # 載入配置
    config = SystemConfig()
    
    if args.mode == 'train':
        train_system(config)
    elif args.mode == 'inference':
        run_inference(config)
    elif args.mode == 'web':
        # 啟動Streamlit應用
        import subprocess
        subprocess.run(['streamlit', 'run', 'web_interface/app.py'])

if __name__ == "__main__":
    main()
