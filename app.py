from __future__ import annotations

import streamlit as st

from world_cup_hub.apps import NAV_ITEMS, render_app
from world_cup_hub.components import inject_global_styles

st.set_page_config(page_title="World Cup Fun Lab", page_icon="⚽", layout="wide")
inject_global_styles()

st.sidebar.title("⚽ World Cup Fun Lab")
st.sidebar.caption("A bundle of playful World Cup mini-apps.")
selected_label = st.sidebar.radio("Navigate", list(NAV_ITEMS.keys()))
st.sidebar.markdown("---")
st.sidebar.caption("Current app")
st.sidebar.write(selected_label)

render_app(NAV_ITEMS[selected_label])
