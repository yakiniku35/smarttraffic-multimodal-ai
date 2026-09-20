"""網頁介面的樣式與配色。

原本的 CSS 把背景色寫死成淺色（``#f8f9fa``），使用者切到深色主題時
卡片就會變成「白底白字」完全看不到。這裡改成用 CSS 變數，
淺色 / 深色各給一組值，再依照設定注入對應的那一組。
"""

from __future__ import annotations

from typing import Dict

import streamlit as st

# --------------------------------------------------------------------------- #
# 配色表
# --------------------------------------------------------------------------- #
LIGHT_TOKENS: Dict[str, str] = {
    "--st-surface": "#ffffff",
    "--st-surface-alt": "#f4f6fa",
    "--st-border": "#e2e8f0",
    "--st-text": "#1a202c",
    "--st-text-muted": "#5a6678",
    "--st-shadow": "0 1px 3px rgba(16, 24, 40, 0.08)",
}

DARK_TOKENS: Dict[str, str] = {
    "--st-surface": "#1c2230",
    "--st-surface-alt": "#232b3b",
    "--st-border": "#333e52",
    "--st-text": "#e8ecf4",
    "--st-text-muted": "#9aa7bd",
    "--st-shadow": "0 1px 3px rgba(0, 0, 0, 0.4)",
}

# 不分主題的語意色（都通過淺色與深色底的對比度檢查）
SEMANTIC_TOKENS: Dict[str, str] = {
    "--st-primary": "#3b7dd8",
    "--st-accent": "#e8833a",
    "--st-success": "#2f9e63",
    "--st-warning": "#d99013",
    "--st-danger": "#d9534f",
}

# 圖表用的分類色，順序固定，讓同一個路口在每張圖裡都是同一個顏色
CHART_COLORS = [
    "#3b7dd8",
    "#e8833a",
    "#2f9e63",
    "#9061d9",
    "#d9534f",
    "#00968f",
]

# 號誌相位對應的顏色
PHASE_COLORS = {
    "綠燈": "--st-success",
    "黃燈": "--st-warning",
    "紅燈": "--st-danger",
}


def resolve_theme(preference: str) -> str:
    """把設定裡的 auto / light / dark 換算成實際要用的主題。

    ``auto`` 會去問 Streamlit 目前的主題是什麼；問不到就當作淺色。
    """
    if preference in ("light", "dark"):
        return preference

    try:
        base = st.get_option("theme.base")
    except Exception:  # pragma: no cover - 不同 Streamlit 版本行為略有差異
        base = None

    return "dark" if base == "dark" else "light"


def plotly_template(theme: str) -> str:
    """回傳對應主題的 Plotly 樣板名稱。"""
    return "plotly_dark" if theme == "dark" else "plotly_white"


def inject_css(theme: str) -> None:
    """把 CSS 變數與元件樣式注入頁面。"""
    tokens = {**(DARK_TOKENS if theme == "dark" else LIGHT_TOKENS), **SEMANTIC_TOKENS}
    variables = "\n".join(f"    {name}: {value};" for name, value in tokens.items())

    st.markdown(
        f"""
<style>
:root {{
{variables}
}}

/* 頁面主標題 */
.st-hero {{
    display: flex;
    flex-direction: column;
    gap: 0.35rem;
    padding: 1.1rem 1.4rem;
    margin-bottom: 1.2rem;
    border-radius: 14px;
    border: 1px solid var(--st-border);
    background: var(--st-surface-alt);
    box-shadow: var(--st-shadow);
}}
.st-hero h1 {{
    margin: 0;
    font-size: 1.85rem;
    line-height: 1.25;
    color: var(--st-text);
}}
.st-hero p {{
    margin: 0;
    font-size: 0.95rem;
    color: var(--st-text-muted);
}}

/* 通用卡片 */
.st-card {{
    background: var(--st-surface);
    border: 1px solid var(--st-border);
    border-left: 4px solid var(--st-primary);
    border-radius: 10px;
    padding: 0.85rem 1rem;
    margin-bottom: 0.7rem;
    color: var(--st-text);
    box-shadow: var(--st-shadow);
}}
.st-card__title {{
    font-weight: 600;
    font-size: 0.95rem;
    margin-bottom: 0.3rem;
}}
.st-card__row {{
    font-size: 0.85rem;
    color: var(--st-text-muted);
    display: flex;
    justify-content: space-between;
    gap: 0.75rem;
}}
.st-card--success {{ border-left-color: var(--st-success); }}
.st-card--warning {{ border-left-color: var(--st-warning); }}
.st-card--danger  {{ border-left-color: var(--st-danger); }}

/* 狀態圓點 */
.st-dot {{
    display: inline-block;
    width: 0.6rem;
    height: 0.6rem;
    border-radius: 50%;
    margin-right: 0.4rem;
    vertical-align: baseline;
}}

/* 小標籤 */
.st-badge {{
    display: inline-block;
    padding: 0.12rem 0.55rem;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 600;
    border: 1px solid transparent;
}}
.st-badge--on {{
    color: var(--st-success);
    border-color: var(--st-success);
    background: color-mix(in srgb, var(--st-success) 12%, transparent);
}}
.st-badge--off {{
    color: var(--st-text-muted);
    border-color: var(--st-border);
    background: var(--st-surface-alt);
}}

/* 讓手機上的分頁標籤可以橫向捲動，不會被擠成兩行 */
.stTabs [data-baseweb="tab-list"] {{
    overflow-x: auto;
    scrollbar-width: thin;
}}
</style>
""",
        unsafe_allow_html=True,
    )


def hero(title: str, subtitle: str) -> None:
    """畫出頁面最上方的標題區塊。

    原本用 ``-webkit-background-clip: text`` 做漸層字，在部分瀏覽器上
    會整行變透明（等於看不見標題），所以改成單純的實心文字。
    """
    st.markdown(
        f"""
<div class="st-hero">
    <h1>{title}</h1>
    <p>{subtitle}</p>
</div>
""",
        unsafe_allow_html=True,
    )


def badge(label: str, active: bool) -> str:
    """回傳一個開 / 關狀態的小標籤 HTML。"""
    css_class = "st-badge--on" if active else "st-badge--off"
    return f'<span class="st-badge {css_class}">{label}</span>'
