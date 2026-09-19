"""系統設定分頁。

這是這次改版的重點。舊版的設定頁有幾個問題：

* 每個滑桿 / 輸入框的值抓到之後**完全沒有被使用**，
  按下「儲存」只會跳出一個綠色訊息，重新整理就全部回到預設值。
* API 金鑰用 ``st.text_input`` 收集後放在記憶體裡，既沒存也沒用。
* 融合權重三條滑桿可以各自調到 1.0，加起來變成 3.0 也沒人管。
* 沒有「回復預設值」、「匯出設定」這些基本功能。

新版把設定和 :class:`config.SystemConfig` 真正綁在一起：
用 ``st.form`` 一次套用、存成 JSON 檔、可以匯出匯入、有驗證與
「尚未儲存」提示，金鑰則改成只讀環境變數並顯示狀態。
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Tuple

import streamlit as st

from config import DEFAULT_CONFIG_PATH, SystemConfig

# session_state 用到的鍵，集中管理避免打錯字
CONFIG_KEY = "system_config"
SAVED_SNAPSHOT_KEY = "saved_config_snapshot"
# 記住「哪一個上傳的檔案已經匯入過了」，避免每次 rerun 重複套用
IMPORTED_FILE_KEY = "imported_config_file"
# 開啟頁面時讀取設定檔失敗的警告訊息
LOAD_WARNING_KEY = "config_load_warning"

# 「重置系統」時要保留的鍵（設定本身與存檔快照要一起留，
# 否則重置後會一直誤報「尚未寫入檔案」）
PRESERVED_KEYS = (CONFIG_KEY, SAVED_SNAPSHOT_KEY)


# --------------------------------------------------------------------------- #
# session_state 輔助
# --------------------------------------------------------------------------- #
def _load_validated(path) -> Tuple[SystemConfig, str]:
    """從檔案讀設定並驗證，回傳 (設定, 警告訊息)。

    設定檔是可以手動編輯的，內容壞掉（JSON 語法錯、數值不合法、
    權限不足）都有可能。這些情況一律退回預設值並附上說明，
    而不是讓整個網頁介面白屏。
    """
    try:
        config = SystemConfig.load(path)
    except (OSError, json.JSONDecodeError) as exc:
        return SystemConfig(), f"讀取 {path} 失敗，已改用預設值：{exc}"
    except TypeError as exc:
        return SystemConfig(), f"{path} 的內容格式不對，已改用預設值：{exc}"

    try:
        config.validate()
    except ValueError as exc:
        return SystemConfig(), f"{path} 裡有不合法的設定，已改用預設值：{exc}"

    return config, ""


def get_config() -> SystemConfig:
    """取得目前這個瀏覽器工作階段的設定。"""
    if CONFIG_KEY not in st.session_state:
        config, warning = _load_validated(DEFAULT_CONFIG_PATH)
        st.session_state[CONFIG_KEY] = config
        st.session_state[SAVED_SNAPSHOT_KEY] = json.dumps(
            config.to_dict(), sort_keys=True
        )
        st.session_state[LOAD_WARNING_KEY] = warning
    return st.session_state[CONFIG_KEY]


def has_unsaved_changes() -> bool:
    """目前的設定和上次存檔的內容是否不同。"""
    current = json.dumps(get_config().to_dict(), sort_keys=True)
    return current != st.session_state.get(SAVED_SNAPSHOT_KEY)


def _mark_saved() -> None:
    st.session_state[SAVED_SNAPSHOT_KEY] = json.dumps(
        get_config().to_dict(), sort_keys=True
    )


def _apply(config: SystemConfig) -> Tuple[bool, str]:
    """驗證後套用設定，回傳 (是否成功, 訊息)。"""
    try:
        config.validate()
    except ValueError as exc:
        return False, f"設定不合法：{exc}"

    st.session_state[CONFIG_KEY] = config
    return True, "設定已套用（尚未寫入檔案）"


def _options_including(options: list, current) -> list:
    """確保 ``current`` 一定在選項裡。

    ``st.select_slider`` 的預設值只要不在 options 裡就會直接丟出
    ``ValueError``，整個分頁就打不開了。設定檔是可以手動編輯、也可以
    匯入的，值不在清單裡很正常（例如 MiniLM 原生的 384 維），
    所以把它補進選項而不是讓介面壞掉。
    """
    if current in options:
        return options
    return sorted({*options, current})


# --------------------------------------------------------------------------- #
# 各區塊
# --------------------------------------------------------------------------- #
def _render_rl_section(config: SystemConfig) -> None:
    st.markdown("##### 強化學習（PPO）")

    with st.form("form_rl"):
        col1, col2 = st.columns(2)

        with col1:
            learning_rate = st.number_input(
                "學習率 (learning rate)",
                min_value=1e-6,
                max_value=1e-1,
                value=float(config.rl.learning_rate),
                step=1e-4,
                format="%.6f",
                help="每次更新參數的步伐大小，太大訓練會不穩、太小會學很慢。",
            )
            batch_size = st.number_input(
                "批次大小 (batch size)",
                min_value=8,
                max_value=4096,
                value=int(config.rl.batch_size),
                step=8,
                help="累積幾筆經驗後才做一次更新。",
            )
            n_epochs = st.number_input(
                "每批訓練輪數 (n_epochs)",
                min_value=1,
                max_value=50,
                value=int(config.rl.n_epochs),
                help="同一批資料重複學習幾次。",
            )
            clip_range = st.slider(
                "PPO 裁切範圍 (clip range)",
                0.05,
                0.5,
                float(config.rl.clip_range),
                step=0.01,
                help="限制新舊策略的差距，PPO 穩定訓練的關鍵。",
            )

        with col2:
            gamma = st.slider(
                "折扣因子 (gamma)",
                0.80,
                0.999,
                float(config.rl.gamma),
                step=0.001,
                format="%.3f",
                help="越接近 1，代表越重視長期的回報。",
            )
            gae_lambda = st.slider(
                "GAE lambda",
                0.80,
                1.0,
                float(config.rl.gae_lambda),
                step=0.01,
                help="在偏差與變異之間取捨的參數。",
            )
            entropy_coef = st.slider(
                "熵係數 (探索程度)",
                0.0,
                0.2,
                float(config.rl.entropy_coef),
                step=0.005,
                format="%.3f",
                help="越大越鼓勵智能體嘗試沒試過的動作。",
            )
            federated_sync = st.number_input(
                "聯邦同步間隔（次更新）",
                min_value=0,
                max_value=500,
                value=int(config.rl.federated_sync_interval),
                help="每幾次更新做一次各地區模型的權重平均；0 表示關閉。",
            )

        if st.form_submit_button("套用強化學習設定", type="primary"):
            new_config = replace(
                config,
                rl=replace(
                    config.rl,
                    learning_rate=float(learning_rate),
                    batch_size=int(batch_size),
                    n_epochs=int(n_epochs),
                    clip_range=float(clip_range),
                    gamma=float(gamma),
                    gae_lambda=float(gae_lambda),
                    entropy_coef=float(entropy_coef),
                    federated_sync_interval=int(federated_sync),
                ),
            )
            ok, message = _apply(new_config)
            (st.success if ok else st.error)(message)
            if ok:
                st.rerun()


def _render_multimodal_section(config: SystemConfig) -> None:
    st.markdown("##### 多模態融合")

    with st.form("form_multimodal"):
        col1, col2 = st.columns(2)

        with col1:
            embedding_dim = st.select_slider(
                "嵌入向量維度",
                options=_options_including(
                    [128, 256, 512, 768, 1024], config.multimodal.embedding_dim
                ),
                value=int(config.multimodal.embedding_dim),
                help="三種模態會先被投影到這個共同維度再融合。",
            )
            num_heads = st.select_slider(
                "注意力頭數",
                options=_options_including(
                    [1, 2, 4, 8, 16], config.multimodal.num_attention_heads
                ),
                value=int(config.multimodal.num_attention_heads),
                help="必須能整除嵌入維度，否則 PyTorch 會報錯。",
            )
            dropout = st.slider(
                "Dropout",
                0.0,
                0.6,
                float(config.multimodal.dropout),
                step=0.05,
                help="訓練時隨機丟棄一部分神經元，用來減少過度擬合。",
            )

        with col2:
            sensor_dim = st.number_input(
                "感測器特徵數",
                min_value=1,
                max_value=1024,
                value=int(config.multimodal.sensor_input_dim),
            )
            freeze_text = st.checkbox(
                "凍結文字編碼器",
                value=config.multimodal.freeze_text_encoder,
                help="勾選時只把預訓練模型當特徵抽取器，不會更新它的權重。",
            )
            freeze_image = st.checkbox(
                "凍結影像編碼器",
                value=config.multimodal.freeze_image_encoder,
            )
            pretrained = st.checkbox(
                "下載 ImageNet 預訓練權重",
                value=config.multimodal.pretrained_image_weights,
                help="離線環境請取消勾選，否則建立模型時會卡在下載。",
            )

        # 即時提示：維度與頭數要能整除，先講清楚比按下去才報錯友善
        if embedding_dim % num_heads != 0:
            st.warning(
                f"嵌入維度 {embedding_dim} 無法被 {num_heads} 個注意力頭整除，"
                "套用時會被擋下來。"
            )

        if st.form_submit_button("套用融合設定", type="primary"):
            new_config = replace(
                config,
                multimodal=replace(
                    config.multimodal,
                    embedding_dim=int(embedding_dim),
                    num_attention_heads=int(num_heads),
                    dropout=float(dropout),
                    sensor_input_dim=int(sensor_dim),
                    freeze_text_encoder=bool(freeze_text),
                    freeze_image_encoder=bool(freeze_image),
                    pretrained_image_weights=bool(pretrained),
                ),
            )
            ok, message = _apply(new_config)
            (st.success if ok else st.error)(message)
            if ok:
                st.rerun()


def _render_traffic_section(config: SystemConfig) -> None:
    st.markdown("##### 交通模擬")

    with st.form("form_traffic"):
        col1, col2 = st.columns(2)

        with col1:
            sumo_file = st.text_input(
                "SUMO 設定檔路徑",
                value=config.traffic.sumo_config_file,
                help="副檔名通常是 .sumocfg。",
            )
            num_intersections = st.number_input(
                "路口數量",
                min_value=1,
                max_value=256,
                value=int(config.traffic.num_intersections),
            )
            simulation_time = st.number_input(
                "單次模擬時長（秒）",
                min_value=60,
                max_value=86_400,
                value=int(config.traffic.simulation_time),
                step=60,
            )

        with col2:
            neighbor_distance = st.slider(
                "鄰近路口連線距離（公尺）",
                100.0,
                3000.0,
                float(config.traffic.neighbor_distance_m),
                step=50.0,
                help="兩個路口距離小於這個值，就會在圖神經網路裡連一條邊。",
            )
            use_gui = st.checkbox(
                "啟動 SUMO 圖形介面",
                value=config.traffic.use_gui,
                help="用 sumo-gui 取代 sumo，方便觀察但速度較慢。",
            )
            fallback = st.checkbox(
                "找不到 SUMO 時改用內建模擬環境",
                value=config.traffic.fallback_to_mock,
                help="建議保持勾選，沒裝 SUMO 也能先把整套流程跑起來。",
            )
            seed = st.number_input(
                "亂數種子",
                min_value=0,
                max_value=999_999,
                value=int(config.traffic.random_seed or 0),
                help="固定種子可以讓每次執行的結果一樣，方便比較。",
            )

        # SUMO 設定檔存不存在，直接在這裡就告訴使用者
        if sumo_file and not Path(sumo_file).exists():
            st.info(f"目前找不到 `{sumo_file}`，執行時會改用內建模擬環境。")

        if st.form_submit_button("套用交通設定", type="primary"):
            new_config = replace(
                config,
                traffic=replace(
                    config.traffic,
                    sumo_config_file=sumo_file,
                    num_intersections=int(num_intersections),
                    simulation_time=int(simulation_time),
                    neighbor_distance_m=float(neighbor_distance),
                    use_gui=bool(use_gui),
                    fallback_to_mock=bool(fallback),
                    random_seed=int(seed),
                ),
            )
            ok, message = _apply(new_config)
            (st.success if ok else st.error)(message)
            if ok:
                st.rerun()


def _render_ui_section(config: SystemConfig) -> None:
    st.markdown("##### 介面顯示")

    with st.form("form_ui"):
        col1, col2 = st.columns(2)

        with col1:
            theme_options = ["auto", "light", "dark"]
            theme_labels = {"auto": "跟隨系統", "light": "淺色", "dark": "深色"}
            theme = st.selectbox(
                "配色主題",
                theme_options,
                index=theme_options.index(config.ui.theme),
                format_func=lambda key: theme_labels[key],
            )
            chart_height = st.slider(
                "圖表高度（像素）",
                250,
                700,
                int(config.ui.chart_height),
                step=10,
            )
            decimals = st.slider(
                "數值小數位數", 0, 3, int(config.ui.decimal_places)
            )

        with col2:
            auto_refresh = st.checkbox(
                "自動更新資料",
                value=config.ui.auto_refresh,
                help="開啟後每隔一段時間會自動抓一次新的即時資料。",
            )
            refresh_interval = st.slider(
                "更新間隔（秒）",
                1,
                60,
                int(config.ui.refresh_interval_s),
                disabled=not auto_refresh,
            )
            show_advanced = st.checkbox(
                "顯示進階欄位",
                value=config.ui.show_advanced,
                help="在各分頁顯示原始資料表與除錯資訊。",
            )

        if st.form_submit_button("套用介面設定", type="primary"):
            new_config = replace(
                config,
                ui=replace(
                    config.ui,
                    theme=theme,
                    chart_height=int(chart_height),
                    decimal_places=int(decimals),
                    auto_refresh=bool(auto_refresh),
                    refresh_interval_s=int(refresh_interval),
                    show_advanced=bool(show_advanced),
                ),
            )
            ok, message = _apply(new_config)
            (st.success if ok else st.error)(message)
            if ok:
                st.rerun()


def _render_api_section(config: SystemConfig) -> None:
    st.markdown("##### API 金鑰")
    st.caption(
        "金鑰一律從環境變數讀取，不會存進設定檔，也不會顯示在畫面上 ── "
        "避免不小心把密碼 commit 到 GitHub。"
    )

    status = config.api_key_status()
    for name, is_set in status.items():
        col1, col2 = st.columns([3, 1])
        col1.code(name, language=None)
        col2.write("✅ 已設定" if is_set else "⚠️ 未設定")

    with st.expander("要怎麼設定？"):
        st.markdown(
            """
在專案根目錄建立一個 `.env` 檔（這個檔名已經在 `.gitignore` 裡）：

```bash
OPENAI_API_KEY=sk-xxxxxxxx
WEATHER_API_KEY=xxxxxxxx
MAPS_API_KEY=xxxxxxxx
```

或是在終端機直接匯出環境變數：

```bash
export OPENAI_API_KEY=sk-xxxxxxxx
streamlit run web_interface/app.py
```
"""
        )


def _render_persistence_section(config: SystemConfig) -> None:
    st.markdown("##### 儲存與還原")

    config_path = Path(st.session_state.get("config_path", DEFAULT_CONFIG_PATH))
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("💾 寫入設定檔", width="stretch"):
            try:
                saved_path = config.save(config_path)
                _mark_saved()
                st.success(f"已儲存到 {saved_path}")
            except OSError as exc:
                st.error(f"寫入失敗：{exc}")

    with col2:
        if st.button("↩️ 重新載入檔案", width="stretch"):
            # 和開啟頁面時走同一條「讀取 + 驗證」的路徑，
            # 檔案被手動改壞也只會看到訊息，不會整頁當掉
            loaded, warning = _load_validated(config_path)
            if warning:
                st.error(warning)
            else:
                st.session_state[CONFIG_KEY] = loaded
                _mark_saved()
                st.rerun()

    with col3:
        if st.button("🔄 回復預設值", width="stretch"):
            st.session_state[CONFIG_KEY] = SystemConfig()
            st.rerun()

    st.download_button(
        "⬇️ 匯出設定 JSON",
        data=json.dumps(config.to_dict(), indent=2, ensure_ascii=False),
        file_name="system_config.json",
        mime="application/json",
        width="stretch",
    )

    uploaded = st.file_uploader("⬆️ 匯入設定 JSON", type=["json"])
    if uploaded is None:
        # 使用者把檔案移除了，下次再上傳同一個檔案還是要能匯入
        st.session_state.pop(IMPORTED_FILE_KEY, None)
    else:
        # 上傳的檔案會一直留在 uploader 裡，每次 rerun 都拿得到。
        # 如果不記住「這個檔案已經匯入過了」，使用者之後在別的區塊
        # 調整的設定都會被這裡重新套用的舊內容蓋掉。
        file_id = getattr(uploaded, "file_id", None) or (
            uploaded.name,
            uploaded.size,
        )
        if st.session_state.get(IMPORTED_FILE_KEY) != file_id:
            st.session_state[IMPORTED_FILE_KEY] = file_id
            try:
                data = json.loads(uploaded.getvalue().decode("utf-8"))
                # `[1,2]`、`"abc"`、`null` 都是合法 JSON，但 from_dict 會對它們
                # 呼叫 .get() 而丟出 AttributeError；沒先擋下來整頁就會炸掉
                if not isinstance(data, dict):
                    raise TypeError("設定 JSON 的最外層必須是物件（{...}）")
                imported = SystemConfig.from_dict(data)
                ok, message = _apply(imported)
                if ok:
                    st.success("設定已匯入，記得按「寫入設定檔」才會永久保存。")
                    st.rerun()
                else:
                    st.error(message)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                st.error(f"這不是有效的 JSON 檔：{exc}")
            except TypeError as exc:
                st.error(f"設定內容有問題：{exc}")
        else:
            st.success(
                f"已匯入 `{uploaded.name}`，記得按「寫入設定檔」才會永久保存"
                "（移除上傳的檔案後可以重新匯入）。"
            )

    with st.expander("查看目前的完整設定"):
        st.json(config.to_dict())


# --------------------------------------------------------------------------- #
# 對外入口
# --------------------------------------------------------------------------- #
def render_settings_tab() -> None:
    """畫出整個「系統設定」分頁。"""
    config = get_config()

    header_col, status_col = st.columns([3, 1])
    with header_col:
        st.subheader("⚙️ 系統設定")
        st.caption("調整後請按各區塊的「套用」，再用下方的「寫入設定檔」永久保存。")
    with status_col:
        if has_unsaved_changes():
            st.warning("尚未寫入檔案", icon="✏️")
        else:
            st.success("已與檔案同步", icon="✅")

    # 開啟頁面時如果設定檔讀失敗，要讓使用者知道現在用的是預設值
    load_warning = st.session_state.get(LOAD_WARNING_KEY)
    if load_warning:
        st.warning(load_warning, icon="⚠️")

    sections = st.tabs(["🧠 強化學習", "🔀 多模態", "🚗 交通模擬", "🎨 介面", "🔐 金鑰", "💾 儲存"])

    with sections[0]:
        _render_rl_section(config)
    with sections[1]:
        _render_multimodal_section(config)
    with sections[2]:
        _render_traffic_section(config)
    with sections[3]:
        _render_ui_section(config)
    with sections[4]:
        _render_api_section(config)
    with sections[5]:
        _render_persistence_section(config)
