"""多模態 AI 智慧城市交通優化系統 ── 主程式入口。

用法::

    python main.py --mode web                    # 啟動網頁介面（預設）
    python main.py --mode train --episodes 100   # 訓練
    python main.py --mode inference              # 推理
    python main.py --mode train --config configs/system_config.json

修正的問題：

1. ``logs/`` 與 ``models/`` 資料夾不存在時會直接 FileNotFoundError，
   現在會先自動建立。
2. ``--config`` 參數有定義卻完全沒用到，現在真的會讀設定檔。
3. 訓練中途 Ctrl+C 會留下沒關掉的 SUMO 連線，改用 try/finally。
4. ``torch.load`` 沒有指定 ``map_location``，在沒有 GPU 的機器上載入
   GPU 訓練的模型會失敗。
5. 啟動 Streamlit 用的是相對路徑，從其他資料夾執行就會找不到檔案。
"""

from __future__ import annotations

import argparse
import html
import logging
import random
import subprocess
import sys
from pathlib import Path
from typing import Callable, Optional
from wsgiref.types import StartResponse, WSGIEnvironment

import numpy as np

from config import SystemConfig, load_env_file

# 專案根目錄，用來組出絕對路徑
PROJECT_ROOT = Path(__file__).resolve().parent

logger = logging.getLogger(__name__)


def _wsgi_json_response(
    start_response: Callable[[str, list[tuple[str, str]]], None],
    status: str,
    body: str,
    *,
    extra_headers: Optional[list[tuple[str, str]]] = None,
    send_body: bool = True,
) -> list[bytes]:
    payload = body.encode("utf-8")
    headers = [
        ("Content-Type", "application/json; charset=utf-8"),
        ("Content-Length", str(len(payload))),
        ("Cache-Control", "no-store"),
    ]
    if extra_headers:
        headers.extend(extra_headers)
    start_response(
        status,
        headers,
    )
    return [payload] if send_body else []


def _wsgi_html_response(
    start_response: Callable[[str, list[tuple[str, str]]], None],
    status: str,
    body: str,
    *,
    send_body: bool = True,
) -> list[bytes]:
    payload = body.encode("utf-8")
    start_response(
        status,
        [
            ("Content-Type", "text/html; charset=utf-8"),
            ("Content-Length", str(len(payload))),
            ("Cache-Control", "no-store"),
        ],
    )
    return [payload] if send_body else []


def _wsgi_empty_response(
    start_response: Callable[[str, list[tuple[str, str]]], None],
    status: str,
    headers: list[tuple[str, str]],
) -> list[bytes]:
    start_response(status, headers)
    return []


def _vercel_html_page() -> str:
    readme_path = PROJECT_ROOT / "README.md"
    streamlit_path = PROJECT_ROOT / "web_interface" / "app.py"
    return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>SmartTraffic Multimodal AI</title>
    <style>
      body {{
        font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        margin: 0;
        background: #0f172a;
        color: #e2e8f0;
      }}
      main {{
        max-width: 760px;
        margin: 0 auto;
        padding: 48px 24px;
      }}
      h1 {{
        margin-bottom: 12px;
      }}
      p, li {{
        line-height: 1.6;
      }}
      code {{
        background: rgba(148, 163, 184, 0.18);
        border-radius: 6px;
        padding: 0.1rem 0.35rem;
      }}
      a {{
        color: #93c5fd;
      }}
    </style>
  </head>
  <body>
    <main>
      <h1>SmartTraffic Multimodal AI</h1>
      <p>
        This Vercel deployment exposes a lightweight health endpoint for the repository.
        The interactive dashboard is implemented as a Streamlit app and should be started
        with <code>python main.py --mode web</code> or
        <code>streamlit run {html.escape(str(streamlit_path.relative_to(PROJECT_ROOT)))}</code>.
      </p>
      <ul>
        <li><a href="/health">Health check</a></li>
        <li><a href="https://github.com/yakiniku35/smarttraffic-multimodal-ai">Repository</a></li>
        <li><a href="https://github.com/yakiniku35/smarttraffic-multimodal-ai/blob/main/{html.escape(str(readme_path.relative_to(PROJECT_ROOT)))}">README</a></li>
      </ul>
    </main>
  </body>
</html>
"""


def vercel_app(environ: WSGIEnvironment, start_response: StartResponse):
    """WSGI entrypoint for Vercel Python deployments.

    Routes:
    - ``GET /`` returns a small HTML landing page for the repository.
    - ``GET /health`` returns a JSON health payload.
    - ``HEAD`` mirrors the matching ``GET`` headers with no body.
    - ``OPTIONS`` returns ``204 No Content`` plus the supported methods.
    - Unknown paths return ``404`` JSON.
    - Unsupported methods return ``405`` JSON with an ``Allow`` header.
    """
    path = environ.get("PATH_INFO", "/")
    method = environ.get("REQUEST_METHOD", "GET").upper()
    send_body = method != "HEAD"
    allow_header = [("Allow", "GET, HEAD, OPTIONS")]

    if method == "OPTIONS":
        return _wsgi_empty_response(
            start_response,
            "204 No Content",
            allow_header + [("Cache-Control", "no-store")],
        )

    if method not in {"GET", "HEAD"}:
        return _wsgi_json_response(
            start_response,
            "405 Method Not Allowed",
            '{"error":"method_not_allowed","message":"Use GET or HEAD"}',
            extra_headers=allow_header,
            send_body=send_body,
        )

    if path == "/health":
        return _wsgi_json_response(
            start_response,
            "200 OK",
            '{"status":"ok","service":"smarttraffic-multimodal-ai"}',
            send_body=send_body,
        )

    if path in {"/", ""}:
        return _wsgi_html_response(
            start_response, "200 OK", _vercel_html_page(), send_body=send_body
        )

    return _wsgi_json_response(
        start_response,
        "404 Not Found",
        '{"error":"not_found","message":"Use / or /health"}',
        send_body=send_body,
    )


# Vercel looks for one of: app / application / handler.
app = vercel_app
application = app
handler = app


def setup_logging(config: SystemConfig) -> None:
    """設定日誌，同時輸出到檔案與終端機。"""
    log_path = Path(config.log_dir) / "smarttraffic.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=getattr(logging, config.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,  # 避免重複執行時堆疊出多個 handler
    )


def set_seed(seed: Optional[int]) -> None:
    """固定亂數種子，讓實驗結果可以重現。

    只想跑網頁介面的人不一定會安裝 PyTorch（光是 torch 就好幾 GB），
    所以 torch 的種子是「有裝才設」，沒裝也不會讓 `--mode web` 掛掉。
    """
    if seed is None:
        return

    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
    except ImportError:
        logger.debug("未安裝 PyTorch，略過 torch 的亂數種子設定")
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(prefer_gpu: bool = True):
    """有 GPU 就用 GPU，沒有就用 CPU。"""
    import torch

    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# --------------------------------------------------------------------------- #
# 訓練
# --------------------------------------------------------------------------- #
def train_system(
    config: SystemConfig,
    num_episodes: int = 1000,
    use_mock_env: bool = False,
) -> None:
    """訓練多模態 AI 交通優化系統。"""
    import torch

    from reinforcement_learning.fed_ppo_agent import FederatedPPOAgent
    from traffic_simulation.sumo_interface import NODE_FEATURE_DIM, create_environment

    logger.info("開始訓練多模態 AI 交通優化系統（共 %d 回合）", num_episodes)

    device = get_device()
    logger.info("使用裝置：%s", device)

    env = create_environment(config.traffic, force_mock=use_mock_env)

    try:
        # 狀態維度由環境決定，不要在這裡亂寫死
        config.rl.state_dim = NODE_FEATURE_DIM
        graph_structure = env.create_edge_index()

        agent = FederatedPPOAgent(config.rl, graph_structure).to(device)

        model_path = Path(config.model_dir) / "best_model.pth"
        model_path.parent.mkdir(parents=True, exist_ok=True)

        best_reward = float("-inf")

        for episode in range(num_episodes):
            state, edge_index = env.reset()
            state, edge_index = state.to(device), edge_index.to(device)
            episode_reward = 0.0

            # 收集經驗時要關掉 dropout，否則存下來的 log 機率是「被 dropout
            # 擾動過的策略」算出來的，PPO 更新時重新評估的機率對不上，
            # 重要性取樣比值一開始就不是 1，梯度會被雜訊帶偏。
            agent.eval()

            for _ in range(config.traffic.simulation_time):
                with torch.no_grad():
                    action, action_logprob, state_value = agent.get_action_and_value(
                        state, edge_index
                    )

                next_state, next_edge_index, rewards, dones, _info = env.step(
                    action.cpu().numpy()
                )

                mean_reward = float(np.mean(rewards))
                agent.memory.store(
                    state,
                    edge_index,
                    action,
                    action_logprob,
                    mean_reward,
                    any(dones),
                    state_value,
                )

                episode_reward += mean_reward
                state = next_state.to(device)
                edge_index = next_edge_index.to(device)

                if any(dones):
                    break

            agent.train()  # 更新時才需要 dropout
            loss_info = agent.update()
            if loss_info:
                logger.info(
                    "Episode %d：獎勵=%.2f，損失=%.4f",
                    episode,
                    episode_reward,
                    loss_info["total_loss"],
                )

            if episode_reward > best_reward:
                best_reward = episode_reward
                torch.save(
                    {
                        "agent": agent.state_dict(),
                        "config": config.to_dict(),
                        "episode": episode,
                        "reward": episode_reward,
                    },
                    model_path,
                )
                logger.info("已儲存新的最佳模型，獎勵：%.2f", episode_reward)

    except KeyboardInterrupt:
        logger.warning("使用者中斷訓練")
    finally:
        # 不管正常結束還是出錯，都要把模擬器關掉
        env.close()
        logger.info("訓練結束")


# --------------------------------------------------------------------------- #
# 推理
# --------------------------------------------------------------------------- #
def _rl_config_from_checkpoint(checkpoint: dict, config: SystemConfig):
    """取出 checkpoint 裡存的 RL 設定，用來重建同樣形狀的網路。

    模型的形狀是由 ``hidden_dim`` / ``gnn_output_dim`` / ``action_dim``
    決定的，這些值必須和訓練當時一致，否則 ``load_state_dict`` 會因為
    形狀不符而失敗。舊的 checkpoint 可能沒有存 config，那就退回目前的設定。
    """
    saved = (checkpoint.get("config") or {}).get("rl")
    if not isinstance(saved, dict):
        logger.warning("checkpoint 沒有存 RL 設定，改用目前的設定重建模型")
        return config.rl

    from dataclasses import fields as dataclass_fields

    from config import RLConfig

    allowed = {f.name for f in dataclass_fields(RLConfig)}
    rl_config = RLConfig(**{k: v for k, v in saved.items() if k in allowed})

    changed = [
        name
        for name in ("hidden_dim", "gnn_output_dim", "action_dim", "state_dim")
        if getattr(rl_config, name) != getattr(config.rl, name)
    ]
    if changed:
        logger.info("採用 checkpoint 內的 RL 設定（與目前設定不同：%s）", ", ".join(changed))

    return rl_config


def run_inference(config: SystemConfig, use_mock_env: bool = False) -> None:
    """載入訓練好的模型並執行推理。"""
    import torch

    from reinforcement_learning.fed_ppo_agent import FederatedPPOAgent
    from traffic_simulation.sumo_interface import NODE_FEATURE_DIM, create_environment

    logger.info("開始執行推理模式")

    model_path = Path(config.model_dir) / "best_model.pth"
    if not model_path.exists():
        # 原本找不到檔案只會丟出一長串 traceback，這裡給清楚的提示
        raise FileNotFoundError(
            f"找不到模型檔 {model_path}，請先執行 `python main.py --mode train`"
        )

    device = get_device()
    # weights_only=True 只允許張量與基本型別，不會在反序列化時執行任意程式碼。
    # model_dir 可以被設定檔指定，模型檔也可能是從別處下載的，
    # 用安全模式載入才不會被惡意的 .pth 檔植入程式碼。
    # 我們存的內容（state_dict、dict、int、float）都在允許範圍內。
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)

    # 用訓練當時存下來的 RL 設定來重建模型。
    # 不這樣做的話，只要訓練時改過 hidden_dim / gnn_output_dim / action_dim，
    # 這裡就會用預設值建出形狀不同的網路，load_state_dict 直接報
    # 「size mismatch for critic.0.bias: ... torch.Size([64]) vs torch.Size([128])」。
    rl_config = _rl_config_from_checkpoint(checkpoint, config)

    env = create_environment(config.traffic, force_mock=use_mock_env)

    try:
        rl_config.state_dim = NODE_FEATURE_DIM
        agent = FederatedPPOAgent(rl_config, env.create_edge_index()).to(device)
        agent.load_state_dict(checkpoint["agent"])
        agent.eval()  # 關掉 dropout

        state, edge_index = env.reset()
        state, edge_index = state.to(device), edge_index.to(device)
        total_reward = 0.0

        for step in range(config.traffic.simulation_time):
            with torch.no_grad():
                action, _, _ = agent.get_action_and_value(
                    state, edge_index, deterministic=True
                )

            next_state, next_edge_index, rewards, dones, info = env.step(
                action.cpu().numpy()
            )
            total_reward += float(np.mean(rewards))

            if step % 100 == 0:
                logger.info(
                    "Step %d：平均車速=%.2f，車輛總數=%s",
                    step,
                    info["average_speed"],
                    info["total_vehicles"],
                )

            state = next_state.to(device)
            edge_index = next_edge_index.to(device)

            if any(dones):
                break

        logger.info("推理完成，總獎勵：%.2f", total_reward)
    finally:
        env.close()


# --------------------------------------------------------------------------- #
# 網頁介面
# --------------------------------------------------------------------------- #
def launch_web_interface(port: int = 8501) -> int:
    """啟動 Streamlit 網頁介面。

    用 ``sys.executable -m streamlit`` 而不是直接叫 ``streamlit``，
    這樣一定會用到目前虛擬環境裡的版本；路徑也改成絕對路徑，
    從任何資料夾執行 main.py 都不會找不到檔案。
    """
    app_path = PROJECT_ROOT / "web_interface" / "app.py"
    if not app_path.exists():
        raise FileNotFoundError(f"找不到網頁介面程式：{app_path}")

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.port",
        str(port),
    ]
    logger.info("啟動網頁介面：http://localhost:%d", port)

    try:
        return subprocess.run(cmd, check=False).returncode
    except FileNotFoundError:
        logger.error("找不到 streamlit，請先執行：pip install -r requirements.txt")
        return 1


# --------------------------------------------------------------------------- #
# 進入點
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="多模態 AI 智慧城市交通優化系統",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["train", "inference", "web"],
        default="web",
        help="執行模式：train(訓練) / inference(推理) / web(網頁介面)",
    )
    parser.add_argument("--config", type=str, default=None, help="JSON 設定檔路徑")
    parser.add_argument("--episodes", type=int, default=1000, help="訓練回合數")
    parser.add_argument("--port", type=int, default=8501, help="網頁介面連接埠")
    parser.add_argument("--seed", type=int, default=None, help="亂數種子")
    parser.add_argument(
        "--mock-env",
        action="store_true",
        help="強制使用內建模擬環境（不需要安裝 SUMO）",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default=None,
        help="日誌級別",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    # 先載入 .env，後面讀 API 金鑰才拿得到（README 有寫這個用法）
    load_env_file()

    # 讀檔和驗證要放在同一個 try 裡。
    # 設定檔可能是壞掉的 JSON、最外層不是物件、或某個區塊型別不對，
    # 這些都會丟出例外；放在 try 之外的話使用者看到的是一長串 traceback，
    # 而不是說好的「設定檔有誤 + 結束碼 2」。
    try:
        # --config 現在真的會被使用
        config = SystemConfig.load(args.config) if args.config else SystemConfig()

        if args.log_level:
            config.log_level = args.log_level
        if args.seed is not None:
            config.traffic.random_seed = args.seed

        config.validate()
    except (ValueError, TypeError) as exc:
        # json.JSONDecodeError 是 ValueError 的子類別，所以也涵蓋在內
        print(f"設定檔有誤：{exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"讀取設定檔失敗：{exc}", file=sys.stderr)
        return 2

    config.ensure_directories()
    setup_logging(config)
    set_seed(config.traffic.random_seed)

    try:
        if args.mode == "train":
            train_system(config, num_episodes=args.episodes, use_mock_env=args.mock_env)
        elif args.mode == "inference":
            run_inference(config, use_mock_env=args.mock_env)
        else:
            return launch_web_interface(port=args.port)
    except FileNotFoundError as exc:
        # 這類「檔案不在」是使用者操作問題，不是程式壞掉，
        # 印一行清楚的訊息就好，不用整串 traceback 嚇人。
        print(f"錯誤：{exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
