"""多模態 AI 智慧城市交通優化系統 ── Streamlit 網頁介面。

執行方式::

    streamlit run web_interface/app.py
    # 或
    python main.py --mode web

這個檔案只負責「版面」，資料來源在 ``data_source.py``、
樣式在 ``theme.py``、可重複使用的元件在 ``components/``。
拆開之後每個檔案都短很多，也比較好找東西。
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# 讓 `streamlit run web_interface/app.py` 也能 import 到專案根目錄的模組
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import load_env_file  # noqa: E402
from web_interface import data_source as ds  # noqa: E402
from web_interface.components.metrics import (  # noqa: E402
    metric_row,
    signal_card,
    source_badges,
)
from web_interface.components.settings_panel import (  # noqa: E402
    PRESERVED_KEYS,
    get_config,
    render_settings_tab,
)
from web_interface.theme import (  # noqa: E402
    CHART_COLORS,
    hero,
    inject_css,
    plotly_template,
    resolve_theme,
)

# `streamlit run` 不會經過 main.py，所以這裡要自己載入 .env，
# 否則設定頁的金鑰狀態永遠顯示「未設定」
load_env_file()

st.set_page_config(
    page_title="多模態AI智慧城市交通優化系統",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------------------------------- #
# 工作階段狀態
# --------------------------------------------------------------------------- #
def init_session_state() -> None:
    """初始化所有會跨 rerun 保存的狀態。

    這是舊版最缺的東西：以前按下「啟動 AI 優化」只會閃一個綠色訊息，
    下一次重新執行就忘得一乾二淨。現在狀態真的會被記住。
    """
    defaults = {
        "tick": 0,  # 資料刻度，+1 代表抓到一批新資料
        # 這一批資料實際載入的時間。存在 session_state 裡而不是每次
        # 現算，否則「最後更新」會在沒有新資料時也一直往前跳。
        "tick_loaded_at": datetime.now(),
        "optimizer_running": False,
        "started_at": datetime.now(),
        "event_log": [],
        "data_sources": {key: key != "social" for key in ds.DATA_SOURCE_LABELS},
        "operation_mode": "自動模式",
        "control_strategy": "AI自適應控制",
        "optimization_goal": "最小化等待時間",
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def log_event(message: str, level: str = "info") -> None:
    """把操作記錄下來，顯示在側邊欄，讓使用者知道剛剛發生了什麼事。"""
    st.session_state.event_log.insert(
        0, {"time": datetime.now().strftime("%H:%M:%S"), "message": message, "level": level}
    )
    del st.session_state.event_log[12:]  # 只留最近 12 筆


def refresh_data() -> None:
    """抓下一批資料（把刻度 +1，所有圖表就會跟著更新）。"""
    st.session_state.tick += 1
    st.session_state.tick_loaded_at = datetime.now()


def _auto_refresh_ticker(interval_s: int) -> None:
    """定時自動載入下一批資料。

    重點是**不要用 ``time.sleep()``**。在腳本主體裡 sleep 會把這個
    工作階段的執行緒一直佔住：使用者在等待期間點任何東西都不會有反應，
    多人同時使用時更是每個人各佔一條執行緒。

    改用 ``st.fragment(run_every=...)``：Streamlit 會自己安排計時，
    時間到才執行這個小片段，不阻塞腳本。片段裡再用
    ``st.rerun(scope="app")`` 讓整頁跟著更新。

    ``run_every`` 只能在套用裝飾器時指定，而間隔是使用者可調的，
    所以在這裡動態套用（fragment 以函式的 qualname 辨識，每次 rerun
    重新套用仍然是同一個片段）。

    那個時間判斷不能省。片段的內容在「每一次正常的整頁執行」也會跑，
    不是只有計時器到點才跑；少了判斷就會變成
    整頁執行 → 片段 → rerun → 整頁執行 → …的全速迴圈
    （實測 3 秒的間隔會變成每秒更新約 3 次）。
    改成只有真的經過設定的秒數才更新，正常那一趟就直接跳過。
    """

    def _tick() -> None:
        elapsed = (datetime.now() - st.session_state.tick_loaded_at).total_seconds()
        if elapsed < interval_s:
            return
        refresh_data()
        st.rerun(scope="app")

    st.fragment(run_every=interval_s)(_tick)()


# --------------------------------------------------------------------------- #
# 側邊欄
# --------------------------------------------------------------------------- #
def render_sidebar(config) -> None:
    with st.sidebar:
        st.header("🎛️ 控制面板")

        # ---------------- 運作模式 ---------------- #
        st.session_state.operation_mode = st.radio(
            "運作模式",
            ["自動模式", "手動模式", "維護模式"],
            index=["自動模式", "手動模式", "維護模式"].index(
                st.session_state.operation_mode
            ),
            horizontal=True,
        )

        mode_hint = {
            "自動模式": ("AI 正在自動優化交通流量", st.success),
            "手動模式": ("手動控制已啟用，AI 只提供建議", st.warning),
            "維護模式": ("系統維護中，號誌維持固定時制", st.error),
        }[st.session_state.operation_mode]
        mode_hint[1](mode_hint[0])

        st.divider()

        # ---------------- 即時控制 ---------------- #
        st.subheader("即時控制")
        running = st.session_state.optimizer_running
        st.metric("優化引擎", "運行中" if running else "已停止")

        col1, col2 = st.columns(2)
        with col1:
            if st.button(
                "🚀 啟動", width="stretch", disabled=running, type="primary"
            ):
                st.session_state.optimizer_running = True
                log_event("AI 優化已啟動", "success")
                st.rerun()
        with col2:
            if st.button("⏹️ 停止", width="stretch", disabled=not running):
                st.session_state.optimizer_running = False
                log_event("AI 優化已停止", "warning")
                st.rerun()

        if st.button("🔄 更新資料", width="stretch"):
            refresh_data()
            log_event("已載入最新一批資料")
            st.rerun()

        if st.button("♻️ 重置系統", width="stretch"):
            # 重置時把狀態清掉，但保留使用者調好的設定。
            # 存檔快照也要一起保留，否則「系統設定」會一直誤報尚未寫入檔案。
            preserved = {
                key: st.session_state[key]
                for key in PRESERVED_KEYS
                if key in st.session_state
            }
            st.session_state.clear()
            st.session_state.update(preserved)
            init_session_state()
            log_event("系統已重置", "warning")
            st.rerun()

        st.divider()

        # ---------------- 資料源 ---------------- #
        st.subheader("多模態資料源")
        st.caption("取消勾選的資料源，融合權重會自動重新分配。")
        for key, label in ds.DATA_SOURCE_LABELS.items():
            st.session_state.data_sources[key] = st.checkbox(
                label, value=st.session_state.data_sources[key], key=f"src_{key}"
            )

        if not any(st.session_state.data_sources.values()):
            st.error("至少要啟用一個資料源，否則模型沒有輸入。")

        st.divider()

        # ---------------- 操作記錄 ---------------- #
        st.subheader("操作記錄")
        if st.session_state.event_log:
            for entry in st.session_state.event_log:
                st.caption(f"`{entry['time']}` {entry['message']}")
        else:
            st.caption("目前沒有記錄。")

        st.divider()
        st.caption(f"資料刻度 #{st.session_state.tick} · 主題：{config.ui.theme}")


# --------------------------------------------------------------------------- #
# 分頁一：即時監控
# --------------------------------------------------------------------------- #
def render_monitoring_tab(config, template: str) -> None:
    st.subheader("📊 即時交通監控")

    tick = st.session_state.tick
    seed = config.traffic.random_seed or 0
    dp = config.ui.decimal_places

    snapshot = ds.build_live_snapshot(tick, seed, st.session_state.tick_loaded_at)

    # 第一次載入時還沒有「上一批資料」可以比，就不要顯示 +0.0 的假變化量
    def delta(text: str) -> str | None:
        return text if tick > 0 else None

    metric_row(
        [
            (
                "當前車流量",
                f"{snapshot.traffic_volume} 輛/小時",
                delta(f"{snapshot.traffic_delta:+d} 輛/小時"),
            ),
            (
                "平均車速",
                f"{snapshot.average_speed:.{dp}f} km/h",
                delta(f"{snapshot.speed_delta:+.{dp}f} km/h"),
            ),
            (
                "平均等待時間",
                f"{snapshot.waiting_time:.{dp}f} 秒",
                delta(f"{snapshot.waiting_delta:+.{dp}f} 秒"),
                # 等待時間越小越好，所以要反過來：變多顯示紅色
                "inverse",
            ),
            (
                "系統效率",
                f"{snapshot.efficiency:.{dp}f}%",
                delta(f"{snapshot.efficiency_delta:+.{dp}f}%"),
            ),
        ]
    )

    st.caption(
        f"最後更新：{snapshot.timestamp:%Y-%m-%d %H:%M:%S}"
        + ("" if tick else "（按左側「更新資料」可載入下一批）")
    )

    st.markdown("#### 24 小時車流量趨勢")
    history = ds.build_traffic_history(seed)

    fig = go.Figure()
    for i, tl_id in enumerate(ds.INTERSECTION_IDS):
        fig.add_trace(
            go.Scatter(
                x=history["timestamp"],
                y=history[tl_id],
                mode="lines",
                name=f"路口 {tl_id}",
                line=dict(width=2.5, color=CHART_COLORS[i % len(CHART_COLORS)]),
            )
        )
    fig.update_layout(
        template=template,
        xaxis_title="時間",
        yaxis_title="車流量 (輛/小時)",
        hovermode="x unified",
        height=config.ui.chart_height,
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0),
    )
    st.plotly_chart(fig, width="stretch")

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("#### 交通密度熱力圖")
        density = ds.build_density_grid(tick, seed)
        fig_heatmap = px.imshow(
            density,
            color_continuous_scale="Reds",
            labels={"x": "東西向區塊", "y": "南北向區塊", "color": "密度"},
            aspect="auto",
        )
        fig_heatmap.update_layout(
            template=template,
            height=config.ui.chart_height,
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig_heatmap, width="stretch")

    with col2:
        st.markdown("#### 號誌即時狀態")
        signals = ds.build_signal_status(tick, seed)
        for _, row in signals.iterrows():
            signal_card(row.to_dict())

        if config.ui.show_advanced:
            st.dataframe(signals, width="stretch", hide_index=True)


# --------------------------------------------------------------------------- #
# 分頁二：AI 模型狀態
# --------------------------------------------------------------------------- #
def render_model_tab(config, template: str) -> None:
    st.subheader("🤖 AI 模型運行狀態")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 模型效能指標")
        metrics = {
            "預測準確率": 94.2,
            "收斂速度": 87.5,
            "決策效率": 91.8,
            "學習穩定性": 89.3,
        }
        for name, value in metrics.items():
            st.metric(name, f"{value}%")
            st.progress(value / 100)

    with col2:
        st.markdown("#### 多模態融合權重")
        fusion = ds.build_fusion_table(st.session_state.data_sources)
        active = fusion[fusion["啟用"]]

        if active.empty:
            st.info("目前沒有啟用任何資料源，請在左側勾選。")
        else:
            fig_pie = px.pie(
                active,
                values="融合權重",
                names="資料源",
                hole=0.45,
                color_discrete_sequence=CHART_COLORS,
            )
            fig_pie.update_layout(
                template=template,
                height=config.ui.chart_height,
                margin=dict(l=10, r=10, t=10, b=10),
            )
            fig_pie.update_traces(textposition="inside", textinfo="percent+label")
            st.plotly_chart(fig_pie, width="stretch")

    source_badges(ds.DATA_SOURCE_LABELS, st.session_state.data_sources)

    if config.ui.show_advanced:
        st.dataframe(fusion, width="stretch", hide_index=True)

    st.markdown("#### 訓練損失曲線")
    curves = ds.build_training_curves(seed=config.traffic.random_seed or 0)

    fig_training = go.Figure()
    for i, column in enumerate(["Actor Loss", "Critic Loss"]):
        fig_training.add_trace(
            go.Scatter(
                x=curves["epoch"],
                y=curves[column],
                name=column,
                line=dict(color=CHART_COLORS[i], width=2.5),
            )
        )
    fig_training.update_layout(
        template=template,
        xaxis_title="訓練輪數",
        yaxis_title="損失值",
        height=config.ui.chart_height,
        hovermode="x unified",
        margin=dict(l=10, r=10, t=30, b=10),
    )
    st.plotly_chart(fig_training, width="stretch")


# --------------------------------------------------------------------------- #
# 分頁三：交通控制
# --------------------------------------------------------------------------- #
def render_control_tab(config, template: str) -> None:
    st.subheader("🚦 智慧號誌控制")

    manual_allowed = st.session_state.operation_mode == "手動模式"
    if not manual_allowed:
        st.info("手動控制只有在「手動模式」下才能使用，請先到左側切換模式。")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### 控制策略")
        strategies = ["AI自適應控制", "定時控制", "感應控制", "手動控制"]
        st.session_state.control_strategy = st.selectbox(
            "選擇控制策略",
            strategies,
            index=strategies.index(st.session_state.control_strategy),
        )

        goals = ["最小化等待時間", "最大化通過量", "均衡化流量", "減少排放"]
        st.session_state.optimization_goal = st.selectbox(
            "優化目標", goals, index=goals.index(st.session_state.optimization_goal)
        )

    with col2:
        st.markdown("#### 即時調整")
        if st.button("🚨 緊急車輛優先", type="primary", width="stretch"):
            log_event("已開啟緊急車輛綠色通道", "success")
            refresh_data()
            st.rerun()
        if st.button("🔧 重新路徑規劃", width="stretch"):
            log_event("已重新計算最佳路徑")
            refresh_data()
            st.rerun()
        if st.button("📊 流量重分配", width="stretch"):
            log_event("已執行流量重新分配", "warning")
            refresh_data()
            st.rerun()

    with col3:
        st.markdown("#### 手動控制")
        intersection = st.selectbox(
            "選擇路口", ds.INTERSECTION_IDS, disabled=not manual_allowed
        )
        phase = st.selectbox(
            "設定號誌相位",
            ["南北直行", "東西直行", "左轉", "全紅"],
            disabled=not manual_allowed,
        )
        if st.button(
            "✅ 執行手動控制",
            disabled=not manual_allowed,
            width="stretch",
        ):
            log_event(f"{intersection} 已設為「{phase}」", "success")
            refresh_data()
            st.rerun()

    st.markdown("#### 未來 30 分鐘等待時間預測")
    prediction = ds.build_prediction(st.session_state.tick, config.traffic.random_seed or 0)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=prediction["分鐘"],
            y=prediction["現行策略"],
            name="現行策略",
            line=dict(color=CHART_COLORS[4], width=2.5, dash="dash"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=prediction["分鐘"],
            y=prediction["AI 優化策略"],
            name="AI 優化策略",
            line=dict(color=CHART_COLORS[2], width=3),
            fill="tonexty",
            fillcolor="rgba(47, 158, 99, 0.12)",
        )
    )
    fig.update_layout(
        template=template,
        xaxis_title="時間 (分鐘)",
        yaxis_title="平均等待時間 (秒)",
        height=config.ui.chart_height,
        hovermode="x unified",
        margin=dict(l=10, r=10, t=30, b=10),
    )
    st.plotly_chart(fig, width="stretch")

    saving = prediction["現行策略"].mean() - prediction["AI 優化策略"].mean()
    st.success(
        f"預估平均每車可少等 {saving:.1f} 秒"
        f"（{saving / prediction['現行策略'].mean() * 100:.1f}%）"
    )


# --------------------------------------------------------------------------- #
# 分頁四：效能分析
# --------------------------------------------------------------------------- #
def render_analytics_tab(config, template: str) -> None:
    st.subheader("📈 系統效能分析")

    col1, col2, col3 = st.columns(3)

    # 舊版把 delta 寫成 f"+{改善幅度}"，等於把同一個數字顯示兩次
    # （「27.34%」下面再掛一個「+27.34%」）。改成數值 + 與上月相比的變化。
    with col1:
        st.markdown("#### ⏱️ 時間效益")
        metric_row(
            [
                ("平均等待時間減少", "27.3%", "+2.1%"),
                ("通勤時間縮短", "18.7%", "+1.4%"),
                ("紅燈等待減少", "31.2%", "+3.0%"),
            ],
            columns=1,
        )

    with col2:
        st.markdown("#### 🌱 環境效益")
        metric_row(
            [
                ("CO₂ 排放減少", "15.8%", "+0.9%"),
                ("燃油消耗降低", "12.4%", "+0.6%"),
                ("空氣品質改善", "8.9%", "+0.4%"),
            ],
            columns=1,
        )

    with col3:
        st.markdown("#### 💰 經濟效益")
        metric_row(
            [
                ("運輸成本節省", "NT$2.1 億/年", "+8.0%"),
                ("燃料費用降低", "NT$1.3 億/年", "+5.2%"),
                ("維護成本減少", "NT$0.8 億/年", "+3.1%"),
            ],
            columns=1,
        )

    st.caption("※ 以上為與導入前基準期相比的模擬結果，變化量為與上月相比。")

    st.markdown("#### 30 天系統效率趨勢")
    trend = ds.build_efficiency_trend(seed=config.traffic.random_seed or 0)

    fig = px.line(trend, x="日期", y="系統效率", markers=False)
    fig.update_traces(line=dict(width=3, color=CHART_COLORS[0]))
    fig.add_hline(
        y=float(trend["系統效率"].mean()),
        line_dash="dot",
        line_color=CHART_COLORS[1],
        annotation_text=f"期間平均 {trend['系統效率'].mean():.1f}%",
    )
    fig.update_layout(
        template=template,
        height=config.ui.chart_height,
        yaxis_title="系統效率 (%)",
        margin=dict(l=10, r=10, t=30, b=10),
    )
    st.plotly_chart(fig, width="stretch")

    if config.ui.show_advanced:
        st.dataframe(trend, width="stretch", hide_index=True)


# --------------------------------------------------------------------------- #
# 分頁五：系統設定 + 系統資訊
# --------------------------------------------------------------------------- #
def render_settings_page(config) -> None:
    render_settings_tab()

    st.divider()
    st.markdown("##### 💻 系統資訊")
    info = ds.build_system_info(
        st.session_state.started_at, config.traffic.random_seed or 0
    )
    metric_row([(key, value, None) for key, value in info], columns=4)


# --------------------------------------------------------------------------- #
# 主程式
# --------------------------------------------------------------------------- #
def main() -> None:
    init_session_state()

    config = get_config()
    theme = resolve_theme(config.ui.theme)
    inject_css(theme)
    template = plotly_template(theme)

    status = "運行中" if st.session_state.optimizer_running else "待命中"
    hero(
        "🚦 多模態 AI 智慧城市交通優化系統",
        f"{st.session_state.operation_mode} · 優化引擎{status} · "
        f"{config.traffic.num_intersections} 個路口聯合調度",
    )

    render_sidebar(config)

    tabs = st.tabs(
        ["📊 即時監控", "🤖 AI 模型", "🚦 交通控制", "📈 效能分析", "⚙️ 系統設定"]
    )

    with tabs[0]:
        render_monitoring_tab(config, template)
    with tabs[1]:
        render_model_tab(config, template)
    with tabs[2]:
        render_control_tab(config, template)
    with tabs[3]:
        render_analytics_tab(config, template)
    with tabs[4]:
        render_settings_page(config)

    # 「自動更新資料」以前只是個沒有作用的開關，這裡讓它真的會動。
    if config.ui.auto_refresh:
        _auto_refresh_ticker(max(1, int(config.ui.refresh_interval_s)))


if __name__ == "__main__":
    main()
