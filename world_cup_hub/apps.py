from __future__ import annotations

from datetime import date

import pandas as pd
import streamlit as st

from .components import render_app_card, render_hero, render_score_card
from .data import (
    AURA_REQUIRED_COLUMNS,
    CHAOS_REQUIRED_COLUMNS,
    UNDERDOG_REQUIRED_COLUMNS,
    WINNER_REQUIRED_COLUMNS,
    build_winner_factor_breakdown,
    get_winner_model_metrics,
    load_sample_chaos_data,
    load_sample_player_data,
    load_sample_underdog_data,
    load_sample_winner_data,
    load_uploaded_csv,
    prepare_aura_data,
    score_chaos_matches,
    score_underdog_scenarios,
    score_winner_matches,
    simulate_fan_pain,
)


NAV_ITEMS = {
    "🏠 Home": "home",
    "🏟️ Match Predictor": "winner",
    "✨ Aura Lab": "aura",
    "🌪️ Chaos Center": "chaos",
    "📡 Underdog Radar": "underdog",
    "💔 Fan Pain Lab": "pain",
}


def render_app(page: str) -> None:
    if page == "home":
        render_home()
    elif page == "winner":
        render_match_predictor()
    elif page == "aura":
        render_aura_lab()
    elif page == "chaos":
        render_chaos_center()
    elif page == "underdog":
        render_underdog_radar()
    elif page == "pain":
        render_fan_pain_lab()
    else:
        render_home()


def render_home() -> None:
    render_hero(
        "⚽ World Cup Fun Lab",
        "A mini site of playful World Cup analytics apps: the flagship match predictor plus aura, chaos, upset danger, and fan suffering.",
    )

    aura_matches, aura_board = prepare_aura_data(load_sample_player_data())
    chaos = score_chaos_matches(load_sample_chaos_data())
    underdogs = score_underdog_scenarios(load_sample_underdog_data())
    predictions = score_winner_matches(load_sample_winner_data())

    top_player = aura_board.iloc[0]
    top_match = chaos.iloc[0]
    top_upset = underdogs.iloc[0]
    next_prediction = predictions.iloc[0]

    cols = st.columns(5)
    with cols[0]:
        render_score_card("Apps", 5.0)
    with cols[1]:
        render_score_card("Next pick confidence", float(next_prediction["confidence"]))
    with cols[2]:
        render_score_card("Aura leader", float(top_player["aura_score"]))
    with cols[3]:
        render_score_card("Top chaos match", float(top_match["chaos_score"]))
    with cols[4]:
        render_score_card("Biggest upset risk", float(top_upset["upset_risk"]))

    st.markdown("### Explore")
    row0 = st.columns(1)
    with row0[0]:
        render_app_card(
            "🏟️ Match Predictor — flagship",
            "Predicts winners and scores for 2026 men's World Cup-style fixtures, then exposes the model factors behind every pick.",
            f"Next: {next_prediction['fixture']} → {next_prediction['prediction_summary']}",
        )

    row1 = st.columns(2)
    with row1[0]:
        render_app_card(
            "✨ Aura Lab",
            "Leaderboard for main-character energy using player performance, broadcast focus, crowd response, and social buzz.",
            f"Current leader: {top_player['player']}",
        )
    with row1[1]:
        render_app_card(
            "🌪️ Chaos Center",
            "Ranks matches by how much nonsense they are likely to produce: cards, late goals, VAR drama, and total bedlam.",
            f"Most chaotic: {top_match['fixture']}",
        )

    row2 = st.columns(2)
    with row2[0]:
        render_app_card(
            "📡 Underdog Radar",
            "Scans favorites for upset danger using transition threat, set-piece edge, goalkeeper form, and composure.",
            f"Spiciest trap: {top_upset['fixture']}",
        )
    with row2[1]:
        render_app_card(
            "💔 Fan Pain Lab",
            "A fake-but-honest emotional risk calculator for fanbases with trauma, penalties, and blown-lead anxiety.",
            "Best used before knockout matches",
        )

    st.markdown("### Featured right now")
    feature_cols = st.columns(3)
    with feature_cols[0]:
        st.markdown(f"**Next predicted winner:** {next_prediction['winner_pick']}")
        st.caption(f"{next_prediction['fixture']} · {next_prediction['predicted_score']}")
    with feature_cols[1]:
        st.markdown(f"**Aura king:** {top_player['player']} ({top_player['team']})")
        st.caption(f"Archetype: {top_player['latest_archetype']}")
    with feature_cols[2]:
        st.markdown(f"**Chaos pick:** {top_match['fixture']}")
        st.caption(f"Tag: {top_match['chaos_tag']}")

    with st.expander("What powers this site?"):
        st.markdown(
            "- Public fixture/rating/weather feeds where available  \n"
            "- Hybrid heuristics inspired by ML feature engineering  \n"
            "- Streamlit frontend with multiple mini-apps in one site"
        )


def render_match_predictor() -> None:
    render_hero(
        "🏟️ Match Predictor",
        "Flagship winner + score predictor using public 2026 fixture data, stadiums, weather, Elo ratings, and transparent factor breakdowns.",
    )

    st.sidebar.subheader("Match Predictor controls")
    uploaded_file = st.sidebar.file_uploader("Upload fixture CSV", type=["csv"], key="winner_upload")
    use_sample = st.sidebar.toggle("Use generated public fixture data", value=uploaded_file is None, key="winner_sample")

    try:
        if uploaded_file is not None and not use_sample:
            source_df = load_uploaded_csv(uploaded_file)
            source_label = uploaded_file.name
        else:
            source_df = load_sample_winner_data()
            source_label = "data/public_2026_match_features.csv"
        predictions = score_winner_matches(source_df)
    except Exception as exc:
        st.error(str(exc))
        show_schema_help(WINNER_REQUIRED_COLUMNS)
        return

    st.sidebar.success(f"Loaded: {source_label}")
    stages = ["All"] + sorted(predictions["stage"].unique().tolist())
    selected_stage = st.sidebar.selectbox("Stage filter", stages, key="winner_stage")
    upcoming_only = st.sidebar.toggle("Upcoming only", value=True, key="winner_upcoming")
    min_confidence = st.sidebar.slider("Minimum confidence", 0.0, 100.0, 0.0, 1.0, key="winner_conf")

    filtered = predictions.copy()
    if upcoming_only:
        filtered = filtered[pd.to_datetime(filtered["date"], errors="coerce").dt.date >= date.today()]
    if selected_stage != "All":
        filtered = filtered[filtered["stage"] == selected_stage]
    filtered = filtered[filtered["confidence"] >= min_confidence]

    if filtered.empty:
        st.warning("No matches match the filters.")
        return

    strongest_pick = filtered.sort_values("confidence", ascending=False).iloc[0]
    fixture_options = filtered["fixture"].tolist()
    if st.session_state.get("winner_browse_fixture") not in fixture_options:
        st.session_state["winner_browse_fixture"] = fixture_options[0]

    current_index = fixture_options.index(st.session_state["winner_browse_fixture"])
    browse_prev, browse_select, browse_next = st.columns([0.7, 3.0, 0.7])
    with browse_prev:
        if st.button("← Previous", disabled=current_index == 0, key="winner_prev_match"):
            st.session_state["winner_browse_fixture"] = fixture_options[current_index - 1]
            st.rerun()
    with browse_next:
        if st.button("Next →", disabled=current_index == len(fixture_options) - 1, key="winner_next_match"):
            st.session_state["winner_browse_fixture"] = fixture_options[current_index + 1]
            st.rerun()
    with browse_select:
        selected_fixture = st.selectbox(
            "Browse detailed prediction",
            fixture_options,
            index=fixture_options.index(st.session_state["winner_browse_fixture"]),
            key="winner_browse_fixture",
        )

    next_match = filtered.loc[filtered["fixture"] == selected_fixture].iloc[0]

    metric_cols = st.columns(4)
    with metric_cols[0]:
        render_score_card("Fixtures", float(len(filtered)))
    with metric_cols[1]:
        render_score_card("Avg confidence", float(filtered["confidence"].mean()))
    with metric_cols[2]:
        render_score_card("Strongest confidence", float(filtered["confidence"].max()))
    with metric_cols[3]:
        render_score_card("Avg total goals", float((filtered["expected_goals_a"] + filtered["expected_goals_b"]).mean()))

    st.markdown("### Selected match prediction")
    next_left, next_right = st.columns([1.25, 1])
    with next_left:
        st.markdown(f"## {next_match['fixture']}")
        st.caption(f"{next_match['date']} · {next_match.get('kickoff_local', '')} · {next_match['stage']} · {next_match['model_tag']}")
        st.caption(f"{next_match.get('stadium', 'Unknown stadium')} · {next_match.get('city', 'Unknown city')}")
        if pd.notna(next_match.get('weather_temperature_c', '')) and str(next_match.get('weather_temperature_c', '')) != '':
            st.caption(
                f"Weather: {next_match.get('weather_temperature_c')}°C, "
                f"precip {next_match.get('weather_precipitation_probability')}%, "
                f"wind {next_match.get('weather_wind_kmh')} km/h"
            )
        st.markdown(f"### Pick: **{next_match['winner_pick']}**")
        st.markdown(f"Predicted score: **{next_match['predicted_score']}**")
        st.progress(min(max(float(next_match["confidence"]) / 100, 0.0), 1.0), text=f"Confidence: {next_match['confidence']:.1f}%")
    with next_right:
        prob_df = pd.DataFrame(
            {
                "outcome": [
                    f"{next_match['team_a']} win",
                    "Draw",
                    f"{next_match['team_b']} win",
                ],
                "probability": [
                    next_match["team_a_win_prob"],
                    next_match["draw_prob"],
                    next_match["team_b_win_prob"],
                ],
            }
        )
        st.dataframe(prob_df, hide_index=True, use_container_width=True)
        st.bar_chart(prob_df, x="outcome", y="probability")

    st.markdown("### Prediction board")
    st.dataframe(
        filtered[[
            "date",
            "kickoff_local",
            "stage",
            "fixture",
            "stadium",
            "city",
            "winner_pick",
            "predicted_score",
            "team_a_win_prob",
            "draw_prob",
            "team_b_win_prob",
            "confidence",
            "model_tag",
            "model_family",
        ]],
        hide_index=True,
        use_container_width=True,
    )

    board_col, inspect_col = st.columns([1.05, 1])
    with board_col:
        st.markdown("### Confidence ranking")
        st.bar_chart(
            filtered.sort_values("confidence", ascending=False).set_index("fixture")[["confidence"]],
            horizontal=True,
        )
        st.caption(f"Strongest lean: {strongest_pick['fixture']} → {strongest_pick['prediction_summary']}")
    with inspect_col:
        st.markdown("### Inspect selected match")
        selected = next_match
        st.markdown(f"**Prediction:** {selected['prediction_summary']}")
        st.markdown(f"**Model tag:** `{selected['model_tag']}`")
        st.markdown(f"**Model family:** `{selected.get('model_family', 'n/a')}`")
        st.markdown(f"**Top exact scores:** {selected.get('top_scorelines', 'n/a')}")
        if float(selected.get('team_a_advances_prob', 0) or 0) > 0 or float(selected.get('team_b_advances_prob', 0) or 0) > 0:
            st.markdown(
                f"**Knockout advancement:** {selected['team_a']} {selected.get('team_a_advances_prob', 0):.1f}% · "
                f"{selected['team_b']} {selected.get('team_b_advances_prob', 0):.1f}%"
            )
            st.markdown(
                f"**Extra-time / penalties:** ET {selected.get('extra_time_prob', 0):.1f}% · "
                f"pens {selected.get('penalty_shootout_prob', 0):.1f}%"
            )
        st.markdown(f"**Venue:** {selected.get('stadium', 'Unknown stadium')} — {selected.get('city', 'Unknown city')}")
        st.markdown(
            f"**Weather:** {selected.get('weather_temperature_c', 'n/a')}°C, "
            f"precip {selected.get('weather_precipitation_probability', 'n/a')}%, "
            f"wind {selected.get('weather_wind_kmh', 'n/a')} km/h"
        )
        goals_df = pd.DataFrame(
            {
                "team": [selected["team_a"], selected["team_b"]],
                "expected_goals": [selected["expected_goals_a"], selected["expected_goals_b"]],
            }
        )
        st.dataframe(goals_df, hide_index=True, use_container_width=True)
        st.bar_chart(goals_df, x="team", y="expected_goals")

    st.markdown("### Why the model picked it")
    selected_row = filtered.loc[filtered["fixture"] == selected_fixture].iloc[0]
    breakdown = build_winner_factor_breakdown(selected_row)
    breakdown["favours"] = breakdown["contribution"].apply(
        lambda value: selected_row["team_a"] if value > 0 else selected_row["team_b"] if value < 0 else "Neutral"
    )

    factor_left, factor_right = st.columns([1.2, 1])
    with factor_left:
        st.dataframe(
            breakdown[["factor", "team_a_signal", "team_b_signal", "edge", "weight", "contribution", "favours"]]
            .sort_values("contribution", key=lambda series: series.abs(), ascending=False),
            hide_index=True,
            use_container_width=True,
        )
    with factor_right:
        chart_df = breakdown[["factor", "contribution"]].set_index("factor")
        st.bar_chart(chart_df)
        st.caption(
            f"Positive contribution favours {selected_row['team_a']}; negative contribution favours {selected_row['team_b']}."
        )

    with st.expander("Data sources and limitations"):
        source_rows = [
            ("Fixture source", selected_row.get("fixture_source", "uploaded/manual CSV")),
            ("Rating source", selected_row.get("rating_source", "uploaded/manual CSV")),
            ("Venue source", selected_row.get("venue_source", "uploaded/manual CSV")),
            ("Weather source", selected_row.get("weather_source", "not provided")),
            ("Unavailable data", selected_row.get("unavailable_data_notes", "")),
        ]
        st.dataframe(pd.DataFrame(source_rows, columns=["item", "value"]), hide_index=True, use_container_width=True)
        st.warning(
            "Fixtures are coming from a public community dataset, not an official FIFA API. "
            "Player status/injuries/lineups are not connected yet because reliable versions usually require paid sports data APIs."
        )

    with st.expander("Model evaluation metrics"):
        metrics = get_winner_model_metrics()
        if not metrics:
            st.info("No trained model artifact found yet. Run `python3 training/train_team_strength_model.py` to create one.")
        else:
            metric_cols = st.columns(5)
            with metric_cols[0]:
                render_score_card("Accuracy", float(metrics.get("accuracy", 0)) * 100)
            with metric_cols[1]:
                render_score_card("Log loss", float(metrics.get("log_loss", 0)))
            with metric_cols[2]:
                render_score_card("Brier", float(metrics.get("brier_score", 0)))
            with metric_cols[3]:
                render_score_card("Exact score %", float(metrics.get("exact_score_accuracy", 0)) * 100)
            with metric_cols[4]:
                render_score_card("Goal MAE", float(metrics.get("mean_goal_absolute_error", 0)))
            calibration = metrics.get("calibration", [])
            if calibration:
                calibration_df = pd.DataFrame(calibration)
                st.markdown("#### Calibration buckets")
                st.dataframe(calibration_df, hide_index=True, use_container_width=True)
                st.line_chart(calibration_df.set_index("bucket")[["avg_confidence", "actual_accuracy"]])

    with st.expander("Winner predictor CSV schema"):
        show_schema_help(WINNER_REQUIRED_COLUMNS)
        st.info("You can upload a richer official/provider CSV at any time. The app recomputes predictions immediately.")



def render_aura_lab() -> None:
    render_hero(
        "✨ Aura Lab",
        "Quantifying player mythology from impact, clutch moments, crowd reaction, broadcast attention, and social buzz.",
    )

    st.sidebar.subheader("Aura Lab controls")
    uploaded_file = st.sidebar.file_uploader("Upload player-match CSV", type=["csv"], key="aura_upload")
    use_sample = st.sidebar.toggle("Use sample player data", value=uploaded_file is None, key="aura_sample")

    try:
        if uploaded_file is not None and not use_sample:
            source_df = load_uploaded_csv(uploaded_file)
            source_label = uploaded_file.name
        else:
            source_df = load_sample_player_data()
            source_label = "data/sample_player_matches.csv"
        match_scores, leaderboard = prepare_aura_data(source_df)
    except Exception as exc:
        st.error(str(exc))
        show_schema_help(AURA_REQUIRED_COLUMNS)
        return

    st.sidebar.success(f"Loaded: {source_label}")
    teams = ["All"] + sorted(leaderboard["team"].unique().tolist())
    selected_team = st.sidebar.selectbox("Team filter", teams, key="aura_team")
    min_aura = st.sidebar.slider("Minimum aura", 0.0, 100.0, 0.0, 1.0, key="aura_min")
    top_n = st.sidebar.slider("Show top N", 3, max(3, len(leaderboard)), min(10, len(leaderboard)), key="aura_topn")

    filtered = leaderboard.copy()
    if selected_team != "All":
        filtered = filtered[filtered["team"] == selected_team]
    filtered = filtered[filtered["aura_score"] >= min_aura].head(top_n)

    if filtered.empty:
        st.warning("No players match the filters.")
        return

    leader = filtered.iloc[0]
    metric_cols = st.columns(4)
    with metric_cols[0]:
        render_score_card("Players shown", float(len(filtered)))
    with metric_cols[1]:
        render_score_card("Top aura", float(filtered["aura_score"].max()))
    with metric_cols[2]:
        render_score_card("Highest peak", float(filtered["peak_aura"].max()))
    with metric_cols[3]:
        render_score_card("Top main-character %", float(filtered["main_character_probability"].max()))

    lead_left, lead_right = st.columns([1.25, 1])
    with lead_left:
        st.markdown("### Current leader")
        st.markdown(f"## {leader['player']}")
        st.markdown(f"**{leader['team']}** · `{leader['latest_archetype']}`")
        st.progress(min(max(float(leader['main_character_probability']) / 100, 0.0), 1.0), text=f"Main-character probability: {leader['main_character_probability']:.1f}%")
    with lead_right:
        st.dataframe(
            pd.DataFrame(
                {
                    "metric": ["Aura", "Peak aura", "Matches", "Social mentions"],
                    "value": [
                        f"{leader['aura_score']:.1f}",
                        f"{leader['peak_aura']:.1f}",
                        int(leader['matches']),
                        f"{int(leader['social_mentions']):,}",
                    ],
                }
            ),
            hide_index=True,
            use_container_width=True,
        )

    st.markdown("### Leaderboard")
    st.dataframe(
        filtered[[
            "player",
            "team",
            "matches",
            "aura_score",
            "peak_aura",
            "main_character_probability",
            "latest_archetype",
        ]],
        hide_index=True,
        use_container_width=True,
    )

    chart_col, inspect_col = st.columns([1.05, 1])
    with chart_col:
        st.markdown("### Aura ranking")
        st.bar_chart(filtered.set_index("player")[["aura_score", "peak_aura"]], horizontal=True)
    with inspect_col:
        selected_player = st.selectbox("Inspect player", filtered["player"].tolist(), key="aura_player")
        player_summary = leaderboard.loc[leaderboard["player"] == selected_player].iloc[0]
        player_matches = match_scores.loc[match_scores["player"] == selected_player].copy()
        st.markdown(f"**Team:** {player_summary['team']}")
        st.markdown(f"**Archetype:** {player_summary['latest_archetype']}")
        st.progress(min(max(float(player_summary['aura_score']) / 100, 0.0), 1.0), text=f"Aura score: {player_summary['aura_score']:.1f}")

    component_col, history_col = st.columns([1, 1.2])
    with component_col:
        st.markdown("### Component profile")
        profile_df = pd.DataFrame(
            {
                "component": ["impact", "clutch", "crowd", "broadcast", "social", "aesthetic"],
                "score": [
                    player_summary["impact_score"],
                    player_summary["clutch_score"],
                    player_summary["crowd_score"],
                    player_summary["broadcast_score"],
                    player_summary["social_score"],
                    player_summary["aesthetic_score"],
                ],
            }
        )
        st.bar_chart(profile_df, x="component", y="score")
    with history_col:
        st.markdown("### Match-by-match aura")
        display_matches = player_matches[[
            "opponent",
            "stage",
            "aura_score",
            "impact_score",
            "clutch_score",
            "crowd_score",
            "broadcast_score",
            "social_score",
        ]].sort_values("aura_score", ascending=False)
        st.dataframe(display_matches, hide_index=True, use_container_width=True)
        st.line_chart(player_matches[["opponent", "aura_score"]].set_index("opponent"))

    with st.expander("Aura CSV schema"):
        show_schema_help(AURA_REQUIRED_COLUMNS)


def render_chaos_center() -> None:
    render_hero(
        "🌪️ Chaos Center",
        "A match-level nonsense detector: cards, late goals, lead changes, penalties, VAR incidents, and crowd-fueled disorder.",
    )

    st.sidebar.subheader("Chaos Center controls")
    uploaded_file = st.sidebar.file_uploader("Upload chaos-match CSV", type=["csv"], key="chaos_upload")
    use_sample = st.sidebar.toggle("Use sample chaos data", value=uploaded_file is None, key="chaos_sample")

    try:
        if uploaded_file is not None and not use_sample:
            source_df = load_uploaded_csv(uploaded_file)
            source_label = uploaded_file.name
        else:
            source_df = load_sample_chaos_data()
            source_label = "data/sample_match_chaos.csv"
        chaos = score_chaos_matches(source_df)
    except Exception as exc:
        st.error(str(exc))
        show_schema_help(CHAOS_REQUIRED_COLUMNS)
        return

    st.sidebar.success(f"Loaded: {source_label}")
    stages = ["All"] + sorted(chaos["stage"].unique().tolist())
    selected_stage = st.sidebar.selectbox("Stage filter", stages, key="chaos_stage")

    filtered = chaos.copy()
    if selected_stage != "All":
        filtered = filtered[filtered["stage"] == selected_stage]

    feature = filtered.iloc[0]
    metric_cols = st.columns(4)
    with metric_cols[0]:
        render_score_card("Matches", float(len(filtered)))
    with metric_cols[1]:
        render_score_card("Top chaos", float(filtered["chaos_score"].max()))
    with metric_cols[2]:
        render_score_card("Meme voltage", float(filtered["meme_voltage"].max()))
    with metric_cols[3]:
        render_score_card("Average chaos", float(filtered["chaos_score"].mean()))

    st.markdown(f"### Featured chaos pick: {feature['fixture']}")
    st.caption(f"{feature['stage']} · {feature['chaos_tag']}")

    top_col, detail_col = st.columns([1.1, 1])
    with top_col:
        st.dataframe(
            filtered[["fixture", "stage", "chaos_score", "meme_voltage", "chaos_tag"]],
            hide_index=True,
            use_container_width=True,
        )
        st.bar_chart(filtered.set_index("fixture")[["chaos_score", "meme_voltage"]], horizontal=True)
    with detail_col:
        selected_match = st.selectbox("Inspect match", filtered["fixture"].tolist(), key="chaos_match")
        match_row = filtered.loc[filtered["fixture"] == selected_match].iloc[0]
        detail_df = pd.DataFrame(
            {
                "signal": [
                    "goals",
                    "yellow cards",
                    "red cards",
                    "penalties",
                    "VAR incidents",
                    "late goals",
                    "lead changes",
                    "crowd noise",
                ],
                "value": [
                    match_row["total_goals"],
                    match_row["yellow_cards"],
                    match_row["red_cards"],
                    match_row["penalties"],
                    match_row["var_incidents"],
                    match_row["late_goals"],
                    match_row["lead_changes"],
                    match_row["crowd_noise"],
                ],
            }
        )
        st.dataframe(detail_df, hide_index=True, use_container_width=True)
        st.progress(min(max(float(match_row["chaos_score"]) / 100, 0.0), 1.0), text=f"Chaos score: {match_row['chaos_score']:.1f}")
        st.progress(min(max(float(match_row["meme_voltage"]) / 100, 0.0), 1.0), text=f"Meme voltage: {match_row['meme_voltage']:.1f}%")

    with st.expander("Chaos CSV schema"):
        show_schema_help(CHAOS_REQUIRED_COLUMNS)


def render_underdog_radar() -> None:
    render_hero(
        "📡 Underdog Radar",
        "An upset-risk scanner for favorites. Good teams still wobble when the underdog can run, defend set pieces, and believe.",
    )

    try:
        scenarios = score_underdog_scenarios(load_sample_underdog_data())
    except Exception as exc:
        st.error(str(exc))
        show_schema_help(UNDERDOG_REQUIRED_COLUMNS)
        return

    metric_cols = st.columns(4)
    with metric_cols[0]:
        render_score_card("Scenarios", float(len(scenarios)))
    with metric_cols[1]:
        render_score_card("Top upset risk", float(scenarios["upset_risk"].max()))
    with metric_cols[2]:
        render_score_card("Average upset risk", float(scenarios["upset_risk"].mean()))
    with metric_cols[3]:
        render_score_card("High-alert fixtures", float((scenarios["upset_risk"] >= 58).sum()))

    left, right = st.columns([1.05, 1])
    with left:
        st.markdown("### Sample upset board")
        st.dataframe(
            scenarios[["fixture", "stage", "upset_risk", "danger_meter"]],
            hide_index=True,
            use_container_width=True,
        )
        st.bar_chart(scenarios.set_index("fixture")[["upset_risk"]], horizontal=True)
    with right:
        st.markdown("### Build a custom trap game")
        favorite_rating = st.slider("Favorite rating", 70, 99, 90)
        underdog_form = st.slider("Underdog recent form", 40, 99, 82)
        transition_threat = st.slider("Transition threat", 40, 99, 84)
        set_piece_edge = st.slider("Set-piece edge", 40, 99, 72)
        goalkeeper_form = st.slider("Goalkeeper form", 40, 99, 78)
        fatigue_gap = st.slider("Fatigue gap", 0, 40, 18)
        crowd_support = st.slider("Crowd support", 20, 99, 70)
        composure = st.slider("Underdog composure", 40, 99, 76)
        injury_disruption = st.slider("Favorite injury disruption", 0, 40, 12)

        custom = pd.DataFrame(
            [{
                "favorite": "Favorite",
                "underdog": "Underdog",
                "stage": "Custom",
                "favorite_rating": favorite_rating,
                "underdog_form": underdog_form,
                "transition_threat": transition_threat,
                "set_piece_edge": set_piece_edge,
                "goalkeeper_form": goalkeeper_form,
                "fatigue_gap": fatigue_gap,
                "crowd_support": crowd_support,
                "composure": composure,
                "injury_disruption": injury_disruption,
            }]
        )
        combined = pd.concat([scenarios[UNDERDOG_REQUIRED_COLUMNS], custom], ignore_index=True)
        custom_scored = score_underdog_scenarios(combined).loc[lambda df: df["stage"] == "Custom"].iloc[0]
        st.progress(min(max(float(custom_scored["upset_risk"]) / 100, 0.0), 1.0), text=f"Upset risk: {custom_scored['upset_risk']:.1f}%")
        st.markdown(f"**Danger meter:** `{custom_scored['danger_meter']}`")

    with st.expander("Underdog CSV schema"):
        show_schema_help(UNDERDOG_REQUIRED_COLUMNS)


def render_fan_pain_lab() -> None:
    render_hero(
        "💔 Fan Pain Lab",
        "A fake but emotionally accurate simulator for tournament dread.",
    )

    left, right = st.columns([1, 1.1])
    with left:
        expectation = st.slider("Expectation pressure", 0, 100, 78)
        trauma = st.slider("Historical trauma", 0, 100, 72)
        lead_fragility = st.slider("Blown-lead anxiety", 0, 100, 65)
        rival_success = st.slider("Rival success irritation", 0, 100, 60)
        penalties = st.slider("Penalty shootout fear", 0, 100, 74)
        referee_anxiety = st.slider("Referee paranoia", 0, 100, 58)
    with right:
        result = simulate_fan_pain(
            expectation=expectation,
            trauma=trauma,
            lead_fragility=lead_fragility,
            rival_success=rival_success,
            penalties=penalties,
            referee_anxiety=referee_anxiety,
        )
        render_score_card("Pain score", float(result["pain_score"]))
        st.progress(min(max(float(result["pain_score"]) / 100, 0.0), 1.0), text=f"Tier: {result['tier']}")
        st.progress(min(max(float(result["meltdown_probability"]) / 100, 0.0), 1.0), text=f"Meltdown probability: {result['meltdown_probability']:.1f}%")

        breakdown = pd.DataFrame(
            {
                "factor": [
                    "Expectation",
                    "Trauma",
                    "Lead fragility",
                    "Rival success",
                    "Penalties",
                    "Referee anxiety",
                ],
                "value": [
                    expectation,
                    trauma,
                    lead_fragility,
                    rival_success,
                    penalties,
                    referee_anxiety,
                ],
            }
        )
        st.bar_chart(breakdown, x="factor", y="value")

    st.markdown("### Suggested reading of the situation")
    if result["pain_score"] >= 80:
        st.error("Your fanbase should not go 1-0 up early. The universe will treat that as bait.")
    elif result["pain_score"] >= 65:
        st.warning("This is a textbook 'dominate xG, lose on penalties' setup.")
    else:
        st.success("You may enjoy this match for up to 63 minutes before things get strange.")


def show_schema_help(columns: list[str]) -> None:
    st.markdown("**Required columns**")
    st.code(", ".join(columns), language="text")
