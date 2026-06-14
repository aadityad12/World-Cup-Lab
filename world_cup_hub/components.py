from __future__ import annotations

import streamlit as st


GLOBAL_STYLES = """
<style>
  .block-container {padding-top: 1.25rem; padding-bottom: 2rem;}
  .hero {
    background: linear-gradient(135deg, #081225 0%, #0f172a 45%, #1d4ed8 100%);
    color: white;
    padding: 1.35rem 1.5rem;
    border-radius: 20px;
    margin-bottom: 1rem;
    box-shadow: 0 10px 30px rgba(0,0,0,.18);
  }
  .hero h1 {margin: 0; font-size: 2rem;}
  .hero p {margin: .35rem 0 0 0; opacity: .92;}
  .small-label {font-size: 0.82rem; opacity: .75; margin-bottom: .2rem;}
  .score-card {
    border: 1px solid rgba(148,163,184,.25);
    border-radius: 16px;
    padding: 1rem;
    background: rgba(255,255,255,.02);
    min-height: 94px;
  }
  .app-card {
    border: 1px solid rgba(148,163,184,.22);
    border-radius: 18px;
    padding: 1rem 1rem .85rem 1rem;
    background: linear-gradient(180deg, rgba(255,255,255,.04), rgba(255,255,255,.02));
    min-height: 175px;
  }
  .app-card h3 {margin-top: 0; margin-bottom: .4rem;}
  .app-card p {opacity: .82; margin-bottom: .65rem;}
  .pill {
    display: inline-block;
    padding: .35rem .65rem;
    border-radius: 999px;
    background: #e0ecff;
    color: #1e3a8a;
    font-weight: 600;
    font-size: .85rem;
  }
</style>
"""


def inject_global_styles() -> None:
    st.markdown(GLOBAL_STYLES, unsafe_allow_html=True)


def render_hero(title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="hero">
          <h1>{title}</h1>
          <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def score_color(value: float) -> str:
    if value >= 75:
        return "#16a34a"
    if value >= 55:
        return "#2563eb"
    if value >= 40:
        return "#d97706"
    return "#dc2626"


def render_score_card(label: str, value: float | str) -> None:
    numeric_value = float(value) if isinstance(value, (int, float)) else None
    color = score_color(numeric_value) if numeric_value is not None else "#f8fafc"
    display = f"{numeric_value:.1f}" if numeric_value is not None else str(value)
    st.markdown(
        f"""
        <div class="score-card">
          <div class="small-label">{label}</div>
          <div style="font-size:1.8rem;font-weight:800;color:{color};">{display}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_app_card(title: str, description: str, meta: str) -> None:
    st.markdown(
        f"""
        <div class="app-card">
          <h3>{title}</h3>
          <p>{description}</p>
          <span class="pill">{meta}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
