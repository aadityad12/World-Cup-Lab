from __future__ import annotations

import html

import streamlit as st


GLOBAL_STYLES = """
<style>
  :root {
    --bg: #f6f8fb;
    --card: #ffffff;
    --card-soft: #f8fafc;
    --ink: #111827;
    --ink-2: #334155;
    --muted: #64748b;
    --border: #e2e8f0;
    --blue: #2563eb;
    --green: #059669;
    --amber: #d97706;
    --red: #dc2626;
    --shadow: 0 10px 24px rgba(15, 23, 42, .06);
  }

  .stApp {background: var(--bg); color: var(--ink);}
  .block-container {padding-top: 1.15rem; padding-bottom: 2.5rem; max-width: 1180px;}
  section[data-testid="stSidebar"] {background: #ffffff; border-right: 1px solid var(--border);}
  section[data-testid="stSidebar"] * {color: var(--ink) !important;}

  h1, h2, h3, h4, h5, h6, p, li, label, span {letter-spacing: -0.01em;}
  div[data-testid="stMarkdownContainer"] p {line-height: 1.55;}

  .hero {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 58%, #2563eb 100%);
    color: #ffffff;
    padding: 1.45rem 1.6rem;
    border-radius: 18px;
    margin-bottom: 1.05rem;
    box-shadow: var(--shadow);
  }
  .hero h1 {margin: 0; font-size: 2rem; color: #ffffff; letter-spacing: -.035em;}
  .hero p {margin: .45rem 0 0 0; color: rgba(255,255,255,.88); max-width: 820px; line-height: 1.5;}

  .section-title {margin: .25rem 0 .75rem 0;}
  .section-title h2 {font-size: 1.18rem; margin: 0; color: var(--ink); letter-spacing: -.025em;}
  .section-title p {margin: .2rem 0 0 0; color: var(--muted); font-size: .94rem;}

  .score-card, .app-card, .match-card {
    background: var(--card);
    color: var(--ink);
    border: 1px solid var(--border);
    border-radius: 16px;
    box-shadow: var(--shadow);
  }
  .score-card {padding: .9rem .95rem; min-height: 86px;}
  .small-label {font-size: .72rem; color: var(--muted); text-transform: uppercase; letter-spacing: .055em; font-weight: 800; margin-bottom: .22rem;}
  .score-card-value {font-size: 1.65rem; font-weight: 850; line-height: 1.1; color: var(--ink);}

  .app-card {padding: 1rem; min-height: 142px;}
  .app-card h3 {margin: 0 0 .4rem 0; color: var(--ink); font-size: 1.06rem;}
  .app-card p {margin: 0 0 .75rem 0; color: var(--ink-2); line-height: 1.45; font-size: .94rem;}

  .match-card {padding: 1.05rem 1.1rem;}
  .match-card h2 {margin: .18rem 0 .35rem 0; color: var(--ink); letter-spacing: -.03em;}
  .match-card strong {color: var(--ink);}
  .match-card .muted, .muted {color: var(--muted);}


  .pill, .status-pill {
    display: inline-block;
    padding: .32rem .62rem;
    border-radius: 999px;
    font-weight: 800;
    font-size: .76rem;
    border: 1px solid #bfdbfe;
    background: #eff6ff;
    color: #1d4ed8;
  }
  .status-pill.good {background:#ecfdf5;color:#047857;border-color:#a7f3d0;}
  .status-pill.warn {background:#fffbeb;color:#b45309;border-color:#fde68a;}
  .status-pill.bad {background:#fef2f2;color:#b91c1c;border-color:#fecaca;}

  div[data-testid="stMetric"] {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: .85rem .95rem;
    box-shadow: var(--shadow);
  }
  div[data-testid="stMetric"] label, div[data-testid="stMetric"] [data-testid="stMetricValue"] {color: var(--ink) !important;}

  button[data-baseweb="tab"] {font-weight: 750; color: var(--ink-2);}
  button[data-baseweb="tab"][aria-selected="true"] {color: var(--blue);}

  .result-list {background: var(--card); border:1px solid var(--border); border-radius:16px; box-shadow:var(--shadow); overflow:hidden;}
  .result-row {display:grid; grid-template-columns: 1.35fr .95fr .95fr .55fr; gap:.7rem; align-items:center; padding:.85rem 1rem; border-bottom:1px solid var(--border); color:var(--ink);}
  .result-row:last-child {border-bottom:0;}
  .result-row .title {font-weight:800; color:var(--ink);}
  .result-row .sub {font-size:.82rem; color:var(--muted); margin-top:.1rem;}
  .result-row .label {font-size:.72rem; color:var(--muted); text-transform:uppercase; font-weight:800; letter-spacing:.05em;}
  .result-row .value {font-weight:800; color:var(--ink);}

  @media (max-width: 800px) {
    .result-row {grid-template-columns: 1fr;}
    .hero h1 {font-size: 1.65rem;}
  }
</style>
"""


def inject_global_styles() -> None:
    st.markdown(GLOBAL_STYLES, unsafe_allow_html=True)


def render_hero(title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="hero">
          <h1>{html.escape(title)}</h1>
          <p>{html.escape(subtitle)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_header(title: str, subtitle: str | None = None) -> None:
    subtitle_html = f"<p>{html.escape(subtitle)}</p>" if subtitle else ""
    st.markdown(
        f"""
        <div class="section-title">
          <h2>{html.escape(title)}</h2>
          {subtitle_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def score_color(value: float) -> str:
    if value >= 75:
        return "#059669"
    if value >= 55:
        return "#2563eb"
    if value >= 40:
        return "#d97706"
    return "#dc2626"


def render_score_card(label: str, value: float | str, suffix: str = "") -> None:
    numeric_value = float(value) if isinstance(value, (int, float)) else None
    color = score_color(numeric_value) if numeric_value is not None else "#111827"
    display = f"{numeric_value:.1f}{suffix}" if numeric_value is not None else str(value)
    st.markdown(
        f"""
        <div class="score-card">
          <div class="small-label">{html.escape(label)}</div>
          <div class="score-card-value" style="color:{color};">{html.escape(display)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_app_card(title: str, description: str, meta: str) -> None:
    st.markdown(
        f"""
        <div class="app-card">
          <h3>{html.escape(title)}</h3>
          <p>{html.escape(description)}</p>
          <span class="pill">{html.escape(meta)}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_status_pill(text: str, tone: str = "good") -> None:
    st.markdown(f'<span class="status-pill {tone}">{html.escape(text)}</span>', unsafe_allow_html=True)
