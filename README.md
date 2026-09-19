# SmartTraffic Multimodal AI

> **An Intelligent Urban Traffic Optimization System Powered by Multimodal AI and Reinforcement Learning**

---

## Project Overview

SmartTraffic Multimodal AI is an advanced smart city solution that leverages multimodal data and cutting-edge AI techniques to optimize urban traffic flow in real time. By integrating traffic camera images, GPS trajectories, weather data, and social media feeds, the system dynamically adjusts traffic signals, predicts congestion, and recommends optimal routes to minimize travel time and reduce emissions.

---

## Key Features

- **Multimodal Data Fusion:** Integrates real-time data from cameras, GPS, IoT sensors, and weather APIs.
- **Federated Multi-Agent Reinforcement Learning:** Decentralized AI agents collaboratively optimize traffic signals using Fed-PPO.
- **Graph Neural Network Prediction:** Models the entire urban road network for accurate, city-scale traffic flow forecasting.
- **Real-Time Dashboard:** Visualizes traffic status, congestion alerts, and system performance with Streamlit and React.
- **Environmental Impact Analysis:** Quantifies emission reductions and energy savings from optimized traffic management.
- **Cloud Native Deployment:** Supports scalable deployment with Docker and Kubernetes.

---

## Tech Stack

- **Python 3.10+**
- **PyTorch, PyTorch Geometric** (GNNs)
- **Transformers (Hugging Face)** (Vision-Language Models)
- **SUMO, TraCI** (Traffic Simulation)
- **Streamlit, React** (Dashboard)
- **Docker, Kubernetes** (Deployment)

---

## Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/yourusername/smarttraffic-multimodal-ai.git
   cd smarttraffic-multimodal-ai
   ```

2. **Create a virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **(Optional) Install SUMO**  
   [SUMO download & setup guide](https://sumo.dlr.de/docs/Downloads.html)

   SUMO is **not required to get started**. If `traci` is unavailable, the
   system automatically falls back to a built-in `MockTrafficEnvironment`,
   so you can run training, the dashboard, and the tests straight away.

5. **Run the main application**

   ```bash
   python main.py --mode web
   # Or for the dashboard only:
   streamlit run web_interface/app.py
   ```

---

## Usage

```bash
python main.py --mode web                      # dashboard (default) at localhost:8501
python main.py --mode train --episodes 100     # train
python main.py --mode train --mock-env         # train without SUMO installed
python main.py --mode inference                # run the saved best model
python main.py --config configs/system_config.json --mode train
python main.py --help                          # all options
```

| Flag | Meaning |
| --- | --- |
| `--mode` | `train` / `inference` / `web` |
| `--config` | Path to a JSON config file |
| `--episodes` | Number of training episodes |
| `--port` | Dashboard port (default `8501`) |
| `--seed` | Random seed for reproducible runs |
| `--mock-env` | Force the built-in simulator, skipping SUMO |
| `--log-level` | `DEBUG` / `INFO` / `WARNING` / `ERROR` |

### Dashboard

Five tabs: **即時監控** (live monitoring), **AI 模型** (model status),
**交通控制** (signal control), **效能分析** (analytics) and **系統設定** (settings).

The settings tab is bound to `SystemConfig`: every control validates its input,
applies atomically via a form, and can be written to `configs/system_config.json`,
reloaded, reset to defaults, or exported/imported as JSON.

### API keys

Keys are read from environment variables only — they are never written to the
config file, so they cannot be committed by accident. Create a `.env`
(already git-ignored) or export them directly:

```bash
export OPENAI_API_KEY=sk-...
export WEATHER_API_KEY=...
export MAPS_API_KEY=...
```

---

## Running the tests

```bash
pip install pytest
pytest -q
```

Tests that need PyTorch or PyTorch Geometric skip automatically when those
packages are not installed, so the suite runs on a minimal environment too.

---

## Repository Structure

```txt
smarttraffic-multimodal-ai/
├── main.py                       # CLI entry point (train / inference / web)
├── config.py                     # dataclass config + JSON load/save + validation
├── requirements.txt
├── .streamlit/config.toml        # Streamlit theme & server options
├── multimodal/
│   └── data_fusion.py            # text + image + sensor attention fusion
├── reinforcement_learning/
│   └── fed_ppo_agent.py          # GNN-based PPO agent + federated averaging
├── traffic_simulation/
│   └── sumo_interface.py         # SUMO env + mock fallback env
├── web_interface/
│   ├── app.py                    # dashboard layout
│   ├── theme.py                  # CSS tokens, light/dark palettes
│   ├── data_source.py            # deterministic demo data
│   └── components/
│       ├── metrics.py            # metric rows & status cards
│       └── settings_panel.py     # the settings tab
├── tests/                        # pytest suite
├── models/                       # saved checkpoints (git-ignored)
├── data/                         # datasets (git-ignored)
└── logs/                         # run logs (git-ignored)
```

---

## Contributing

Contributions are welcome! Please open issues or submit pull requests for improvements, bug fixes, or new features.

---

## License

This project is licensed under the **Apache License 2.0**.  
See [LICENSE](LICENSE) for details.

---

## Acknowledgements

- [PyTorch](https://pytorch.org/)
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [SUMO Traffic Simulator](https://sumo.dlr.de/)
- [Streamlit](https://streamlit.io/)
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)

---

- [AI-enhanced description image](https://pplx-res.cloudinary.com/image/upload/v1749434187/user_uploads/74390550/c9975176-931e-4e83-b716-f0bab76d9a45/image.jpg)
- [2-Synopsis-Format-3-1 (AI traffic management thesis example)](https://www.scribd.com/document/849481935/2-Synopsis-Format-3-1)
- [AI-READI: Software Development Best Practices (GitHub)](https://github.com/AI-READI/software-development-best-practices)
- [How to Write an AI Project README (Logobean)](https://www.logobean.com/blog/ai-readme-generation.html)
- [RoadRanger AI Traffic Optimization System (GitHub)](https://github.com/chahalbaljinder/RoadRanger-AI-Traffic-Optimization-System)
- [Smart-Traffic-System (GitHub)](https://github.com/suvanbanerjee/Smart-Traffic-System)
- [AI for Traffic Management: Trends and Solutions (Xenonstack Blog)](https://www.xenonstack.com/blog/traffic-management)
- [AI-Based Urban Traffic Management Research (MDPI)](https://www.mdpi.com/2071-1050/16/24/11265)
- [AI for Smart Cities and Future Mobility (PTV Group Blog)](https://blog.ptvgroup.com/en/trend-topics/ai-for-smart-cities-and-future-mobility-a-quick-guide/)
- [Multimodal Autoencoder for Networking (GitHub)](https://github.com/SmartData-Polito/multimodal-ae-for-networking)
- [Multimodal AI Examples & Real-World Applications (SmartDev)](https://smartdev.com/multimodal-ai-examples-how-it-works-real-world-applications-and-future-trends/)

---
## Connect with me

[Connect with me on Twitter](https://x.com/Peyerchiu1)

[Connect with me on LinkedIn](www.linkedin.com/in/yen-chia-chiu-a3a8a6212)
