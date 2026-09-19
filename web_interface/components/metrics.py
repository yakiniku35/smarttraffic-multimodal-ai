"""指標與狀態卡片元件。"""

from __future__ import annotations

from typing import Dict, Iterable, Sequence, Tuple

import streamlit as st

from web_interface.theme import PHASE_COLORS, badge


def metric_row(items: Sequence[Tuple], columns: int = 4) -> None:
    """把一組指標排成整齊的一列。

    每個項目是 ``(標題, 數值, 變化量)``，可以再加第四個元素指定
    變化量的顏色：``"normal"``（預設，增加是綠色）或 ``"inverse"``
    （增加是紅色，給「等待時間」這種越小越好的指標用）。

    原本的寫法是一串 ``if col_idx == 0: ... elif col_idx == 1: ...``，
    欄數一改就要整段重寫；這裡改成用 list 索引，欄數變成參數。
    """
    if not items:
        return

    cols = st.columns(columns)
    for i, item in enumerate(items):
        label, value, delta = item[0], item[1], item[2]
        delta_color = item[3] if len(item) > 3 else "normal"
        with cols[i % columns]:
            # delta=None 時 Streamlit 不會畫箭頭，delta_color 也就不影響
            st.metric(label, value, delta=delta, delta_color=delta_color)


def signal_card(row: Dict[str, object]) -> None:
    """畫一張號誌狀態卡片。"""
    phase = str(row["當前相位"])
    colour_var = PHASE_COLORS.get(phase, "--st-primary")

    modifier = {
        "綠燈": "st-card--success",
        "黃燈": "st-card--warning",
        "紅燈": "st-card--danger",
    }.get(phase, "")

    st.markdown(
        f"""
<div class="st-card {modifier}">
    <div class="st-card__title">
        <span class="st-dot" style="background: var({colour_var});"></span>
        {row['路口ID']} · {phase} {row['剩餘秒數']} 秒
    </div>
    <div class="st-card__row">
        <span>等待車輛 {row['等待車輛']} 輛</span>
        <span>AI 建議：<strong>{row['AI建議']}</strong></span>
    </div>
</div>
""",
        unsafe_allow_html=True,
    )


def source_badges(labels: Dict[str, str], enabled: Dict[str, bool]) -> None:
    """用一排小標籤顯示每個資料源的開關狀態。"""
    chips = " ".join(
        badge(label, enabled.get(key, False)) for key, label in labels.items()
    )
    st.markdown(f"<div>{chips}</div>", unsafe_allow_html=True)
