"""SUMO 交通模擬環境介面。

重點修正：

1. 原本有 ``import sumo``，但 PyPI 上的 ``sumo`` 套件跟 SUMO 模擬器無關，
   而且整個檔案根本沒用到它 ── 只要沒裝就整包 import 失敗。已移除。
2. ``traci`` 改成「軟性匯入」：沒安裝時不會讓整個專案掛掉，
   而是自動改用內建的 :class:`MockTrafficEnvironment`（純 numpy 模擬）。
3. ``config.use_gui`` 之前沒有定義，已在 ``TrafficConfig`` 補上。
4. 相位切換原本寫死 ``% 4``，改成讀取該號誌實際的相位數量。
5. 路口沒有車道時 ``np.mean([])`` 會產生 nan，已加上保護。
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)

# traci 是 SUMO 附帶的 Python 套件，沒有安裝時不應該讓整個專案無法 import。
try:  # pragma: no cover - 取決於執行環境有沒有裝 SUMO
    import traci

    TRACI_AVAILABLE = True
except ImportError:  # pragma: no cover
    traci = None  # type: ignore[assignment]
    TRACI_AVAILABLE = False

# 每個路口會抽出來的特徵數量，要和 RLConfig.state_dim 一致
NODE_FEATURE_DIM = 8


def _safe_mean(values: Sequence[float]) -> float:
    """空陣列時回傳 0.0，避免 numpy 產生 nan 與 RuntimeWarning。"""
    return float(np.mean(values)) if len(values) else 0.0


def _safe_std(values: Sequence[float]) -> float:
    """同上，但算標準差。"""
    return float(np.std(values)) if len(values) else 0.0


class BaseTrafficEnvironment:
    """交通環境的共同介面。

    真實的 SUMO 環境與測試用的假環境都遵守同一組方法，
    這樣上層的訓練程式不用管底下到底是哪一種。
    """

    traffic_lights: List[str]

    def reset(self) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def step(
        self, actions: Sequence[int]
    ) -> Tuple[torch.Tensor, torch.Tensor, List[float], List[bool], Dict]:
        raise NotImplementedError

    def get_state(self) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def create_edge_index(self) -> torch.Tensor:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # 共用工具
    # ------------------------------------------------------------------ #
    @property
    def num_agents(self) -> int:
        return len(self.traffic_lights)

    @staticmethod
    def _edge_index_from_positions(
        positions: Sequence[Tuple[float, float]], threshold: float
    ) -> torch.Tensor:
        """依照路口座標距離建立圖的邊索引。

        距離小於 ``threshold`` 的兩個路口之間連一條邊（雙向）。
        如果完全沒有邊，就退回自環（self-loop），
        否則 GCNConv 會因為沒有鄰居而算不出東西。
        """
        edges: List[List[int]] = []
        num_nodes = len(positions)

        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                dx = positions[i][0] - positions[j][0]
                dy = positions[i][1] - positions[j][1]
                if math.hypot(dx, dy) < threshold:
                    edges.append([i, j])
                    edges.append([j, i])

        if not edges:
            idx = list(range(num_nodes))
            return torch.tensor([idx, idx], dtype=torch.long)

        return torch.tensor(edges, dtype=torch.long).t().contiguous()


class SUMOTrafficEnvironment(BaseTrafficEnvironment):
    """使用 SUMO / traci 的真實交通模擬環境。"""

    def __init__(self, config):
        if not TRACI_AVAILABLE:
            raise RuntimeError(
                "找不到 traci 套件。請先安裝 SUMO（https://eclipse.dev/sumo/），"
                "或改用 create_environment() 讓系統自動切換到模擬環境。"
            )

        self.config = config
        self.sumo_config = config.sumo_config_file
        self.traffic_lights: List[str] = []
        self.detectors: List[str] = []
        self.current_step = 0
        self._connected = False
        # 快取每個號誌的相位數量與控制車道，避免每一步都重新查詢
        self._phase_counts: Dict[str, int] = {}
        self._controlled_lanes: Dict[str, Tuple[str, ...]] = {}

        self.init_sumo()

    # ------------------------------------------------------------------ #
    # 生命週期
    # ------------------------------------------------------------------ #
    def init_sumo(self) -> None:
        """啟動 SUMO 模擬器並建立 traci 連線。"""
        binary = "sumo-gui" if getattr(self.config, "use_gui", False) else "sumo"
        sumo_cmd = [binary, "-c", self.sumo_config, "--start", "--quit-on-end"]

        seed = getattr(self.config, "random_seed", None)
        if seed is not None:
            sumo_cmd += ["--seed", str(seed)]

        traci.start(sumo_cmd)
        self._connected = True

        # 連線建立之後只要有任何一步失敗，都要把 sumo 關掉再把例外丟出去，
        # 否則呼叫端（例如 create_environment 的 fallback）會留下一個
        # 沒人管的 sumo 行程，下次 traci.start 就會連不上。
        try:
            self.traffic_lights = list(traci.trafficlight.getIDList())
            self.detectors = list(traci.inductionloop.getIDList())

            # 先把不會變動的資訊快取起來
            self._controlled_lanes.clear()
            self._phase_counts.clear()
            for tl_id in self.traffic_lights:
                self._controlled_lanes[tl_id] = tuple(
                    dict.fromkeys(traci.trafficlight.getControlledLanes(tl_id))
                )
                self._phase_counts[tl_id] = self._query_phase_count(tl_id)
        except Exception:
            self.close()
            raise

        logger.info(
            "SUMO 初始化完成：%d 個號誌、%d 個偵測器",
            len(self.traffic_lights),
            len(self.detectors),
        )

    def _query_phase_count(self, tl_id: str) -> int:
        """查出號誌實際有幾個相位（原本寫死 4 是錯的）。"""
        try:
            programs = traci.trafficlight.getAllProgramLogics(tl_id)
            if programs:
                return max(len(programs[0].phases), 1)
        except Exception:  # pragma: no cover - 不同 SUMO 版本 API 略有差異
            logger.debug("無法取得 %s 的相位數量，改用預設值 4", tl_id)
        return 4

    def close(self) -> None:
        """關閉 traci 連線；重複呼叫也不會出錯。"""
        if self._connected and TRACI_AVAILABLE:
            try:
                traci.close()
            except Exception:  # pragma: no cover
                logger.debug("關閉 traci 連線時發生例外，已忽略", exc_info=True)
            finally:
                self._connected = False

    def reset(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """重置環境。

        原本無條件呼叫 ``traci.close()``，第一次 reset 時（還沒連線）會出錯，
        改成先檢查連線狀態。
        """
        self.close()
        self.current_step = 0
        self.init_sumo()
        return self.get_state()

    # ------------------------------------------------------------------ #
    # 狀態與獎勵
    # ------------------------------------------------------------------ #
    def get_state(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """取得所有路口的節點特徵與圖結構。"""
        node_features: List[List[float]] = []
        now = traci.simulation.getTime()

        for tl_id in self.traffic_lights:
            current_phase = traci.trafficlight.getPhase(tl_id)
            time_to_switch = traci.trafficlight.getNextSwitch(tl_id) - now

            densities: List[float] = []
            waiting_times: List[float] = []
            speeds: List[float] = []

            for lane in self._controlled_lanes.get(tl_id, ()):
                length = traci.lane.getLength(lane)
                if length <= 0:
                    continue
                # 每公里車輛數
                densities.append(
                    traci.lane.getLastStepVehicleNumber(lane) / length * 1000.0
                )
                waiting_times.append(traci.lane.getWaitingTime(lane))
                speeds.append(traci.lane.getLastStepMeanSpeed(lane))

            node_features.append(
                [
                    float(current_phase),
                    float(time_to_switch),
                    _safe_mean(densities),
                    _safe_mean(waiting_times),
                    _safe_mean(speeds),
                    _safe_std(densities),
                    _safe_std(waiting_times),
                    _safe_std(speeds),
                ]
            )

        state = torch.tensor(node_features, dtype=torch.float32)
        return state, self.create_edge_index()

    def create_edge_index(self) -> torch.Tensor:
        """依照號誌座標建立路網圖。"""
        positions: List[Tuple[float, float]] = []
        for tl_id in self.traffic_lights:
            try:
                positions.append(tuple(traci.junction.getPosition(tl_id)))
            except Exception:
                # 有些路網的號誌 ID 和路口 ID 不同名，取不到座標就先放原點
                logger.debug("取不到 %s 的座標，以 (0, 0) 代替", tl_id)
                positions.append((0.0, 0.0))

        threshold = getattr(self.config, "neighbor_distance_m", 1000.0)
        return self._edge_index_from_positions(positions, threshold)

    def compute_rewards(self) -> List[float]:
        """獎勵 = 通過量 − 0.1 × 等待時間。"""
        rewards: List[float] = []
        for tl_id in self.traffic_lights:
            lanes = self._controlled_lanes.get(tl_id, ())
            waiting = sum(traci.lane.getWaitingTime(lane) for lane in lanes)
            throughput = sum(
                traci.lane.getLastStepVehicleNumber(lane) for lane in lanes
            )
            rewards.append(float(throughput - 0.1 * waiting))
        return rewards

    def get_average_speed(self) -> float:
        """全網平均車速（m/s）。"""
        vehicles = traci.vehicle.getIDList()
        if not vehicles:
            return 0.0
        return float(
            sum(traci.vehicle.getSpeed(v) for v in vehicles) / len(vehicles)
        )

    def step(
        self, actions: Sequence[int]
    ) -> Tuple[torch.Tensor, torch.Tensor, List[float], List[bool], Dict]:
        """執行一步模擬。

        ``actions`` 的長度如果和號誌數量不同，會直接報錯 ──
        原本用 ``enumerate`` 搭配索引取號誌，動作太多時會 IndexError，
        太少則是安靜地少控制幾個路口，兩種都很難debug。
        """
        if len(actions) != len(self.traffic_lights):
            raise ValueError(
                f"動作數量({len(actions)})與號誌數量({len(self.traffic_lights)})不符"
            )

        for tl_id, action in zip(self.traffic_lights, actions):
            if int(action) == 1:  # 1 = 切換到下一個相位
                num_phases = self._phase_counts.get(tl_id, 4)
                next_phase = (traci.trafficlight.getPhase(tl_id) + 1) % num_phases
                traci.trafficlight.setPhase(tl_id, next_phase)

        traci.simulationStep()
        self.current_step += 1

        next_state, edge_index = self.get_state()
        rewards = self.compute_rewards()

        done = self.current_step >= self.config.simulation_time
        dones = [done] * len(self.traffic_lights)

        info = {
            "step": self.current_step,
            "total_vehicles": traci.simulation.getMinExpectedNumber(),
            "average_speed": self.get_average_speed(),
            "mean_reward": _safe_mean(rewards),
        }

        return next_state, edge_index, rewards, dones, info


class MockTrafficEnvironment(BaseTrafficEnvironment):
    """不需要 SUMO 的替代環境。

    用簡單的排隊模型模擬車流，讓沒安裝 SUMO 的人也能跑訓練流程、
    寫單元測試、或是在網頁介面上看到會動的資料。
    """

    def __init__(self, config):
        self.config = config
        num = int(getattr(config, "num_intersections", 16))
        self.traffic_lights = [f"TL_{i + 1:03d}" for i in range(num)]
        self.current_step = 0

        seed = getattr(config, "random_seed", None)
        self._rng = np.random.default_rng(seed)

        # 把路口排成接近正方形的網格，格距 300 公尺
        side = max(1, int(math.ceil(math.sqrt(num))))
        self._positions = [
            (float((i % side) * 300), float((i // side) * 300)) for i in range(num)
        ]

        self._phase_count = 4
        self._phases = np.zeros(num, dtype=np.int64)
        self._queues = self._rng.uniform(5, 30, size=num)
        self._waiting = self._rng.uniform(5, 40, size=num)
        self._speeds = self._rng.uniform(20, 45, size=num)

    def reset(self) -> Tuple[torch.Tensor, torch.Tensor]:
        self.current_step = 0
        num = len(self.traffic_lights)
        self._phases = np.zeros(num, dtype=np.int64)
        self._queues = self._rng.uniform(5, 30, size=num)
        self._waiting = self._rng.uniform(5, 40, size=num)
        self._speeds = self._rng.uniform(20, 45, size=num)
        return self.get_state()

    def get_state(self) -> Tuple[torch.Tensor, torch.Tensor]:
        features = np.stack(
            [
                self._phases.astype(np.float32),
                np.full(len(self.traffic_lights), 10.0, dtype=np.float32),
                self._queues.astype(np.float32),
                self._waiting.astype(np.float32),
                self._speeds.astype(np.float32),
                np.full(len(self.traffic_lights), float(self._queues.std())),
                np.full(len(self.traffic_lights), float(self._waiting.std())),
                np.full(len(self.traffic_lights), float(self._speeds.std())),
            ],
            axis=1,
        )
        state = torch.tensor(features, dtype=torch.float32)
        return state, self.create_edge_index()

    def create_edge_index(self) -> torch.Tensor:
        threshold = getattr(self.config, "neighbor_distance_m", 1000.0)
        return self._edge_index_from_positions(self._positions, threshold)

    def step(
        self, actions: Sequence[int]
    ) -> Tuple[torch.Tensor, torch.Tensor, List[float], List[bool], Dict]:
        actions = np.asarray(actions, dtype=np.int64).reshape(-1)
        if actions.size != len(self.traffic_lights):
            raise ValueError(
                f"動作數量({actions.size})與號誌數量({len(self.traffic_lights)})不符"
            )

        switched = actions == 1
        self._phases = np.where(
            switched, (self._phases + 1) % self._phase_count, self._phases
        )

        arrivals = self._rng.uniform(1, 6, size=actions.size)
        # 切換相位可以放掉比較多車，但也會有一次性的損失時間
        discharge = np.where(switched, self._rng.uniform(4, 10, size=actions.size), 2.0)

        self._queues = np.clip(self._queues + arrivals - discharge, 0, 120)
        self._waiting = np.clip(
            np.where(switched, self._waiting * 0.75, self._waiting + 1.5), 0, 240
        )
        self._speeds = np.clip(60.0 - self._queues * 0.4, 5, 60)

        self.current_step += 1

        rewards = (discharge - 0.1 * self._waiting).astype(float).tolist()
        done = self.current_step >= self.config.simulation_time
        dones = [done] * len(self.traffic_lights)

        info = {
            "step": self.current_step,
            "total_vehicles": int(self._queues.sum()),
            "average_speed": float(self._speeds.mean()),
            "mean_reward": _safe_mean(rewards),
            "mean_waiting_time": float(self._waiting.mean()),
            "mean_queue": float(self._queues.mean()),
        }

        state, edge_index = self.get_state()
        return state, edge_index, rewards, dones, info

    def close(self) -> None:
        """模擬環境沒有外部資源要釋放。"""
        return None


def create_environment(config, force_mock: bool = False) -> BaseTrafficEnvironment:
    """建立交通環境的統一入口。

    有裝 SUMO 就用真的，沒有就依 ``config.fallback_to_mock`` 決定
    要改用模擬環境還是直接報錯。
    """
    if force_mock:
        logger.info("依要求使用模擬交通環境（MockTrafficEnvironment）")
        return MockTrafficEnvironment(config)

    if TRACI_AVAILABLE:
        try:
            return SUMOTrafficEnvironment(config)
        except Exception as exc:
            if not getattr(config, "fallback_to_mock", True):
                raise
            logger.warning("SUMO 啟動失敗（%s），改用模擬交通環境", exc)
            return MockTrafficEnvironment(config)

    if getattr(config, "fallback_to_mock", True):
        logger.warning("找不到 traci，改用模擬交通環境（MockTrafficEnvironment）")
        return MockTrafficEnvironment(config)

    raise RuntimeError("找不到 traci，且 fallback_to_mock 已關閉")
