import traci
import sumo
import numpy as np
from typing import Dict, List, Tuple, Optional
import xml.etree.ElementTree as ET

class SUMOTrafficEnvironment:
    """SUMO交通模擬環境"""
    
    def __init__(self, config):
        self.config = config
        self.sumo_config = config.sumo_config_file
        self.traffic_lights = []
        self.detectors = []
        self.current_step = 0
        
        # 初始化SUMO
        self.init_sumo()
        
    def init_sumo(self):
        """初始化SUMO模擬器"""
        sumo_cmd = [
            "sumo-gui" if self.config.use_gui else "sumo",
            "-c", self.sumo_config,
            "--start",
            "--quit-on-end"
        ]
        
        traci.start(sumo_cmd)
        
        # 獲取交通燈ID
        self.traffic_lights = traci.trafficlight.getIDList()
        
        # 獲取檢測器ID
        self.detectors = traci.inductionloop.getIDList()
        
        print(f"初始化完成: {len(self.traffic_lights)}個交通燈, {len(self.detectors)}個檢測器")
    
    def get_state(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """獲取當前狀態"""
        node_features = []
        
        for tl_id in self.traffic_lights:
            # 交通燈狀態
            current_phase = traci.trafficlight.getPhase(tl_id)
            time_since_last_switch = traci.trafficlight.getNextSwitch(tl_id) - traci.simulation.getTime()
            
            # 車道狀態
            controlled_lanes = traci.trafficlight.getControlledLanes(tl_id)
            
            lane_densities = []
            lane_waiting_times = []
            lane_speeds = []
            
            for lane in controlled_lanes:
                # 車道密度
                density = traci.lane.getLastStepVehicleNumber(lane) / traci.lane.getLength(lane) * 1000
                
                # 等待時間
                waiting_time = traci.lane.getWaitingTime(lane)
                
                # 平均速度
                mean_speed = traci.lane.getLastStepMeanSpeed(lane)
                
                lane_densities.append(density)
                lane_waiting_times.append(waiting_time)
                lane_speeds.append(mean_speed)
            
            # 組合特徵
            features = [
                current_phase,
                time_since_last_switch,
                np.mean(lane_densities),
                np.mean(lane_waiting_times),
                np.mean(lane_speeds),
                np.std(lane_densities),
                np.std(lane_waiting_times),
                np.std(lane_speeds)
            ]
            
            node_features.append(features)
        
        # 轉換為張量
        state = torch.tensor(node_features, dtype=torch.float32)
        
        # 創建邊索引（交通網路拓撲）
        edge_index = self.create_edge_index()
        
        return state, edge_index
    
    def create_edge_index(self) -> torch.Tensor:
        """創建圖邊索引"""
        edges = []
        
        for i, tl1 in enumerate(self.traffic_lights):
            pos1 = traci.junction.getPosition(tl1)
            
            for j, tl2 in enumerate(self.traffic_lights):
                if i != j:
                    pos2 = traci.junction.getPosition(tl2)
                    
                    # 計算距離
                    distance = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
                    
                    # 如果距離小於閾值，則連接
                    if distance < 1000:  # 1km閾值
                        edges.append([i, j])
        
        if edges:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        else:
            # 如果沒有邊，創建自環
            num_nodes = len(self.traffic_lights)
            edge_index = torch.tensor([[i for i in range(num_nodes)], 
                                     [i for i in range(num_nodes)]], dtype=torch.long)
        
        return edge_index
    
    def step(self, actions: List[int]) -> Tuple[torch.Tensor, torch.Tensor, List[float], List[bool], Dict]:
        """執行一步模擬"""
        # 執行動作
        for i, action in enumerate(actions):
            tl_id = self.traffic_lights[i]
            
            # 動作映射到交通燈相位
            if action == 1:  # 切換相位
                traci.trafficlight.setPhase(tl_id, (traci.trafficlight.getPhase(tl_id) + 1) % 4)
        
        # 前進一步
        traci.simulationStep()
        self.current_step += 1
        
        # 獲取新狀態
        next_state, edge_index = self.get_state()
        
        # 計算獎勵
        rewards = self.compute_rewards()
        
        # 檢查是否結束
        dones = [self.current_step >= self.config.simulation_time] * len(self.traffic_lights)
        
        # 額外資訊
        info = {
            'step': self.current_step,
            'total_vehicles': traci.simulation.getMinExpectedNumber(),
            'average_speed': self.get_average_speed()
        }
        
        return next_state, edge_index, rewards, dones, info
    
    def compute_rewards(self) -> List[float]:
        """計算獎勵函數"""
        rewards = []
        
        for tl_id in self.traffic_lights:
            controlled_lanes = traci.trafficlight.getControlledLanes(tl_id)
            
            # 等待時間懲罰
            total_waiting_time = sum(traci.lane.getWaitingTime(lane) for lane in controlled_lanes)
            
            # 通過量獎勵
            total_throughput = sum(traci.lane.getLastStepVehicleNumber(lane) for lane in controlled_lanes)
            
            # 綜合獎勵
            reward = total_throughput - 0.1 * total_waiting_time
            rewards.append(reward)
        
        return rewards
    
    def get_average_speed(self) -> float:
        """獲取平均速度"""
        total_speed = 0
        vehicle_count = 0
        
        for vehicle_id in traci.vehicle.getIDList():
            total_speed += traci.vehicle.getSpeed(vehicle_id)
            vehicle_count += 1
        
        return total_speed / vehicle_count if vehicle_count > 0 else 0
    
    def reset(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """重置環境"""
        traci.close()
        self.current_step = 0
        self.init_sumo()
        return self.get_state()
    
    def close(self):
        """關閉環境"""
        traci.close()
