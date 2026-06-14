from __future__ import annotations

from collections import defaultdict
from datetime import date
import html
import json
import random

import pandas as pd
import streamlit as st

from .components import render_app_card, render_hero, render_score_card, render_section_header, render_status_pill
from .data import (
    AURA_REQUIRED_COLUMNS,
    CHAOS_REQUIRED_COLUMNS,
    UNDERDOG_REQUIRED_COLUMNS,
    WINNER_REQUIRED_COLUMNS,
    build_prediction_result_comparison,
    build_scoreline_heatmap,
    build_winner_factor_breakdown,
    get_winner_model_metrics,
    load_sample_chaos_data,
    load_public_2026_results,
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
    "Home": "home",
    "Match Predictor": "winner",
    "Tournament Simulator": "tournament",
    "Player Impact": "aura",
    "Match Volatility": "chaos",
    "Upset Risk": "underdog",
    "Fan Pressure": "pain",
}


def render_app(page: str) -> None:
    if page == "home":
        render_home()
    elif page == "winner":
        render_match_predictor()
    elif page == "tournament":
        render_tournament_simulator()
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
        "World Cup Lab",
        "A clean workspace for World Cup predictions, tournament simulations, player impact, match volatility, and upset risk.",
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
        render_score_card("Apps", 6.0)
    with cols[1]:
        render_score_card("Next pick confidence", float(next_prediction["confidence"]))
    with cols[2]:
        render_score_card("Impact leader", float(top_player["aura_score"]))
    with cols[3]:
        render_score_card("Top chaos match", float(top_match["chaos_score"]))
    with cols[4]:
        render_score_card("Biggest upset risk", float(top_upset["upset_risk"]))

    st.markdown("### Explore")
    row0 = st.columns(1)
    with row0[0]:
        render_app_card(
            "Match Predictor — flagship",
            "Predicts winners and scores for 2026 men's World Cup-style fixtures, then exposes the model factors behind every pick.",
            f"Next: {next_prediction['fixture']} → {next_prediction['prediction_summary']}",
        )

    row1 = st.columns(3)
    with row1[0]:
        render_app_card(
            "Tournament Simulator",
            "Runs Monte Carlo group-stage simulations and an approximate knockout bracket to estimate advancement and title odds.",
            "New: qualification + champion odds",
        )
    with row1[1]:
        render_app_card(
            "Player Impact",
            "Ranks player influence using match impact, clutch moments, broadcast attention, and public reaction.",
            f"Current leader: {top_player['player']}",
        )
    with row1[2]:
        render_app_card(
            "Match Volatility",
            "Ranks fixtures by high-variance signals such as cards, penalties, late goals, lead changes, and VAR incidents.",
            f"Most volatile: {top_match['fixture']}",
        )

    row2 = st.columns(2)
    with row2[0]:
        render_app_card(
            "Upset Risk",
            "Scans favorites for upset exposure using transition threat, set-piece edge, goalkeeper form, and composure.",
            f"Highest risk: {top_upset['fixture']}",
        )
    with row2[1]:
        render_app_card(
            "Fan Pressure",
            "A lightweight pressure model for expectation, historical stress, penalties, referee anxiety, and blown-lead risk.",
            "Useful before knockout matches",
        )

    st.markdown("### Featured right now")
    feature_cols = st.columns(3)
    with feature_cols[0]:
        st.markdown(f"**Next predicted winner:** {next_prediction['winner_pick']}")
        st.caption(f"{next_prediction['fixture']} · {next_prediction['predicted_score']}")
    with feature_cols[1]:
        st.markdown(f"**Player impact leader:** {top_player['player']} ({top_player['team']})")
        st.caption(f"Archetype: {top_player['latest_archetype']}")
    with feature_cols[2]:
        st.markdown(f"**Volatility pick:** {top_match['fixture']}")
        st.caption(f"Tag: {top_match['chaos_tag']}")

    with st.expander("What powers this site?"):
        st.markdown(
            "- Public fixture/rating/weather feeds where available  \n"
            "- Hybrid heuristics inspired by ML feature engineering  \n"
            "- Streamlit frontend with multiple mini-apps in one site"
        )


def render_match_predictor() -> None:
    render_hero(
        "Match Predictor",
        "Clean match forecasts with probabilities, scorelines, and a post-match audit for completed World Cup fixtures.",
    )

    st.sidebar.subheader("Match Predictor")
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
        actual_results = load_public_2026_results() if uploaded_file is None or use_sample else pd.DataFrame()
        completed = build_prediction_result_comparison(predictions, actual_results)
    except Exception as exc:
        st.error(str(exc))
        show_schema_help(WINNER_REQUIRED_COLUMNS)
        return

    st.sidebar.caption(f"Source: {source_label}")
    fallback_count = int(predictions["model_family"].eq("Heuristic Poisson fallback").sum())
    if fallback_count:
        st.sidebar.warning(f"{fallback_count} fixture(s) used the heuristic fallback.")

    stages = ["All"] + sorted(predictions["stage"].dropna().unique().tolist())
    selected_stage = st.sidebar.selectbox("Stage", stages, key="winner_stage")
    view_mode = st.sidebar.radio("Match set", ["Upcoming", "All", "Completed"], horizontal=True, key="winner_view_mode")
    min_confidence = st.sidebar.slider("Minimum confidence", 0.0, 100.0, 0.0, 1.0, key="winner_conf")

    filtered = predictions.copy()
    if selected_stage != "All":
        filtered = filtered[filtered["stage"] == selected_stage]
    if view_mode == "Upcoming":
        completed_ids = set(completed["match_id"]) if not completed.empty else set()
        filtered = filtered[~filtered["match_id"].isin(completed_ids)]
        filtered = filtered[pd.to_datetime(filtered["date"], errors="coerce").dt.date >= date.today()]
    elif view_mode == "Completed":
        filtered = completed.copy() if not completed.empty else filtered.iloc[0:0]
    filtered = filtered[filtered["confidence"] >= min_confidence]

    if filtered.empty:
        st.warning("No matches match the current filters.")
        return

    strongest_pick = filtered.sort_values("confidence", ascending=False).iloc[0]
    avg_goals = float((filtered["expected_goals_a"] + filtered["expected_goals_b"]).mean())
    audit_rate = float(completed["prediction_correct"].mean() * 100) if not completed.empty else 0.0

    metric_cols = st.columns(4)
    with metric_cols[0]:
        render_score_card("Fixtures in view", float(len(filtered)))
    with metric_cols[1]:
        render_score_card("Avg confidence", float(filtered["confidence"].mean()), "%")
    with metric_cols[2]:
        render_score_card("Avg total goals", avg_goals)
    with metric_cols[3]:
        render_score_card("Completed accuracy", audit_rate, "%")

    tabs = st.tabs(["Overview", "Match detail", "Fixtures", "Model & data"])

    with tabs[0]:
        render_section_header("At a glance", "A compact read on the tournament forecast and the model's finished-match record.")
        left, right = st.columns([1.1, 1])
        with left:
            st.markdown("#### Strongest current lean")
            st.markdown(
                f"""
                <div class="match-card">
                  <div class="muted">{strongest_pick['date']} · {strongest_pick.get('stage', '')} · {strongest_pick.get('city', '')}</div>
                  <h2>{strongest_pick['fixture']}</h2>
                  <div><strong>Pick:</strong> {strongest_pick['winner_pick']} &nbsp; <strong>Score:</strong> {strongest_pick['predicted_score']}</div>
                  <div class="muted">Confidence {strongest_pick['confidence']:.1f}% · {strongest_pick['model_tag']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.progress(min(max(float(strongest_pick["confidence"]) / 100, 0.0), 1.0), text=f"Confidence {strongest_pick['confidence']:.1f}%")
        with right:
            st.markdown("#### Confidence distribution")
            chart_df = filtered.assign(confidence_bucket=pd.cut(filtered["confidence"], bins=[0, 40, 50, 60, 70, 100], include_lowest=True))
            bucket_df = chart_df.groupby("confidence_bucket", observed=False).size().reset_index(name="matches")
            bucket_df["confidence_bucket"] = bucket_df["confidence_bucket"].astype(str)
            st.bar_chart(bucket_df, x="confidence_bucket", y="matches")

        if not completed.empty:
            render_section_header("Predictions vs real results", "Completed fixtures from the public World Cup feed, shown next to this model's original prediction.")
            audit_cols = st.columns(3)
            with audit_cols[0]:
                render_score_card("Completed games", float(len(completed)))
            with audit_cols[1]:
                render_score_card("Winner/draw correct", float(completed["prediction_correct"].mean() * 100), "%")
            with audit_cols[2]:
                render_score_card("Exact score correct", float(completed["exact_score_correct"].mean() * 100), "%")

            _render_completed_result_cards(completed)
            with st.expander("View completed results as a table"):
                completed_view = completed[[
                    "date",
                    "fixture",
                    "winner_pick",
                    "predicted_score",
                    "confidence",
                    "actual_score",
                    "actual_outcome",
                    "result_status",
                ]].rename(columns={
                    "winner_pick": "predicted_winner",
                    "actual_score": "real_score",
                    "actual_outcome": "real_winner",
                })
                st.dataframe(completed_view, hide_index=True, use_container_width=True)
            st.caption("Result source: openfootball/worldcup.json public repository. Treat as a community data source, not an official FIFA API.")
        else:
            st.info("No completed-match result file found yet. Add `data/public_2026_results.csv` or rerun ingestion once results are available.")

    with tabs[1]:
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
                "Select match",
                fixture_options,
                index=fixture_options.index(st.session_state["winner_browse_fixture"]),
                key="winner_browse_fixture",
            )

        selected = filtered.loc[filtered["fixture"] == selected_fixture].iloc[0]
        render_section_header("Match detail", "Prediction, probabilities, exact score grid, and the biggest factor edges.")
        detail_left, detail_right = st.columns([1.15, 1])
        with detail_left:
            st.markdown(
                f"""
                <div class="match-card">
                  <div class="muted">{selected['date']} · {selected.get('kickoff_local', '')} · {selected['stage']}</div>
                  <h2>{selected['fixture']}</h2>
                  <div><strong>Pick:</strong> {selected['winner_pick']} &nbsp; <strong>Projected score:</strong> {selected['predicted_score']}</div>
                  <div class="muted">{selected.get('stadium', 'Unknown stadium')} · {selected.get('city', 'Unknown city')} · {selected['model_family']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.progress(min(max(float(selected["confidence"]) / 100, 0.0), 1.0), text=f"Confidence {selected['confidence']:.1f}%")
            if "actual_score" in selected and pd.notna(selected.get("actual_score")):
                tone = "good" if bool(selected.get("prediction_correct")) else "bad"
                render_status_pill(f"Actual result: {selected.get('actual_score')} · {selected.get('result_status')}", tone)
        with detail_right:
            prob_df = pd.DataFrame({
                "Outcome": [f"{selected['team_a']} win", "Draw", f"{selected['team_b']} win"],
                "Probability": [selected["team_a_win_prob"], selected["draw_prob"], selected["team_b_win_prob"]],
            })
            st.dataframe(prob_df, hide_index=True, use_container_width=True)
            st.bar_chart(prob_df, x="Outcome", y="Probability")

        heatmap = build_scoreline_heatmap(selected)
        if not heatmap.empty:
            with st.expander("Exact score probability grid", expanded=True):
                st.dataframe(heatmap.style.format("{:.2f}%").background_gradient(cmap="Blues", axis=None), use_container_width=True)

        breakdown = build_winner_factor_breakdown(selected)
        breakdown["favours"] = breakdown["contribution"].apply(
            lambda value: selected["team_a"] if value > 0 else selected["team_b"] if value < 0 else "Neutral"
        )
        factor_cols = st.columns([1.1, 1])
        with factor_cols[0]:
            st.markdown("#### Key factor edges")
            st.dataframe(
                breakdown[["factor", "team_a_signal", "team_b_signal", "contribution", "favours"]]
                .sort_values("contribution", key=lambda series: series.abs(), ascending=False)
                .head(8),
                hide_index=True,
                use_container_width=True,
            )
        with factor_cols[1]:
            st.markdown("#### Expected goals")
            goals_df = pd.DataFrame({"Team": [selected["team_a"], selected["team_b"]], "Expected goals": [selected["expected_goals_a"], selected["expected_goals_b"]]})
            st.bar_chart(goals_df, x="Team", y="Expected goals")
            st.caption(f"Top exact scores: {selected.get('top_scorelines', 'n/a')}")

    with tabs[2]:
        render_section_header("Fixture board", "A sortable table for scanning every match in the current filter.")
        board_columns = [
            "date", "kickoff_local", "stage", "fixture", "stadium", "city", "winner_pick", "predicted_score",
            "team_a_win_prob", "draw_prob", "team_b_win_prob", "confidence", "model_tag",
        ]
        optional_actual = [col for col in ["actual_score", "actual_outcome", "result_status"] if col in filtered.columns]
        st.dataframe(filtered[board_columns + optional_actual], hide_index=True, use_container_width=True)
        st.download_button(
            "Download predictions CSV",
            filtered[board_columns + optional_actual].to_csv(index=False).encode("utf-8"),
            file_name="world_cup_predictions.csv",
            mime="text/csv",
        )

    with tabs[3]:
        render_section_header("Model and data", "Evaluation metrics, source notes, and CSV schema are tucked away here instead of crowding the main view.")
        metrics = get_winner_model_metrics()
        if metrics:
            metric_cols = st.columns(5)
            with metric_cols[0]:
                render_score_card("Accuracy", float(metrics.get("accuracy", 0)) * 100, "%")
            with metric_cols[1]:
                render_score_card("Log loss", float(metrics.get("log_loss", 0)))
            with metric_cols[2]:
                render_score_card("Brier", float(metrics.get("brier_score", 0)))
            with metric_cols[3]:
                render_score_card("Exact score", float(metrics.get("exact_score_accuracy", 0)) * 100, "%")
            with metric_cols[4]:
                render_score_card("Goal MAE", float(metrics.get("mean_goal_absolute_error", 0)))
            calibration = metrics.get("calibration", [])
            if calibration:
                st.markdown("#### Calibration")
                calibration_df = pd.DataFrame(calibration)
                st.line_chart(calibration_df.set_index("bucket")[["avg_confidence", "actual_accuracy"]])
                st.dataframe(calibration_df, hide_index=True, use_container_width=True)
        else:
            st.info("No trained model artifact found yet. Run `python3 training/train_team_strength_model.py`.")

        source_rows = [
            ("Fixture source", str(strongest_pick.get("fixture_source", "uploaded/manual CSV"))),
            ("Rating source", str(strongest_pick.get("rating_source", "uploaded/manual CSV"))),
            ("Venue source", str(strongest_pick.get("venue_source", "uploaded/manual CSV"))),
            ("Weather source", str(strongest_pick.get("weather_source", "not provided"))),
            ("Result source", "openfootball/worldcup.json public repository when available"),
            ("Unavailable data", str(strongest_pick.get("unavailable_data_notes", ""))),
        ]
        source_df = pd.DataFrame(source_rows, columns=["item", "value"])
        source_df["value"] = source_df["value"].astype(str)
        st.dataframe(source_df, hide_index=True, use_container_width=True)
        with st.expander("Winner predictor CSV schema"):
            show_schema_help(WINNER_REQUIRED_COLUMNS)
            st.info("You can upload richer official/provider CSVs with lineups, injuries, suspensions, and travel columns.")


def _render_completed_result_cards(completed: pd.DataFrame) -> None:
    rows_html = []
    for row in completed.head(8).itertuples(index=False):
        tone = "good" if bool(row.prediction_correct) else "bad"
        rows_html.append(
            f"""
            <div class="result-row">
              <div>
                <div class="title">{html.escape(str(row.fixture))}</div>
                <div class="sub">{html.escape(str(row.date))} · confidence {float(row.confidence):.1f}%</div>
              </div>
              <div>
                <div class="label">Prediction</div>
                <div class="value">{html.escape(str(row.winner_pick))} · {html.escape(str(row.predicted_score))}</div>
              </div>
              <div>
                <div class="label">Actual</div>
                <div class="value">{html.escape(str(row.actual_outcome))} · {html.escape(str(row.actual_score))}</div>
              </div>
              <div><span class="status-pill {tone}">{html.escape(str(row.result_status))}</span></div>
            </div>
            """
        )
    st.markdown(f"<div class='result-list'>{''.join(rows_html)}</div>", unsafe_allow_html=True)


def render_tournament_simulator() -> None:
    render_hero(
        "Tournament Simulator"
        "Monte Carlo view of group qualification plus an approximate knockout bracket using the Match Predictor probabilities.",
    )

    st.sidebar.subheader("Tournament Simulator controls")
    simulations = st.sidebar.slider("Simulations", 250, 5000, 1000, 250, key="tournament_sims")
    seed = st.sidebar.number_input("Random seed", min_value=1, max_value=999999, value=2026, step=1, key="tournament_seed")

    try:
        predictions = score_winner_matches(load_sample_winner_data())
        sim = simulate_tournament(predictions, simulations, int(seed))
    except Exception as exc:
        st.error(str(exc))
        return

    if sim.empty:
        st.warning("No group-stage fixture data available for simulation.")
        return

    favorite = sim.sort_values("title_odds", ascending=False).iloc[0]
    qual_leader = sim.sort_values("advance_odds", ascending=False).iloc[0]
    dark_horse = sim[(sim["title_odds"] >= 2) & (sim["avg_elo"] < sim["avg_elo"].median())]
    dark_horse_label = dark_horse.sort_values("title_odds", ascending=False).iloc[0]["team"] if not dark_horse.empty else "n/a"

    metric_cols = st.columns(4)
    with metric_cols[0]:
        render_score_card("Teams", float(len(sim)))
    with metric_cols[1]:
        render_score_card("Title favorite %", float(favorite["title_odds"]))
    with metric_cols[2]:
        render_score_card("Safest advance %", float(qual_leader["advance_odds"]))
    with metric_cols[3]:
        render_score_card("Simulations", float(simulations))

    st.markdown(f"### Title favorite: {favorite['team']}")
    st.caption(
        "Knockout bracket is approximate until official post-group pairings are known; group qualification uses the selected public fixtures."
    )
    st.markdown(f"**Dark-horse watch:** {dark_horse_label}")

    left, right = st.columns([1.15, 1])
    with left:
        st.markdown("### Qualification board")
        st.dataframe(
            sim[["group", "team", "avg_points", "top2_odds", "advance_odds", "title_odds", "avg_elo"]],
            hide_index=True,
            use_container_width=True,
        )
    with right:
        st.markdown("### Champion odds")
        st.bar_chart(sim.sort_values("title_odds", ascending=False).head(16).set_index("team")[["title_odds"]], horizontal=True)
        st.markdown("### Advance odds")
        st.bar_chart(sim.sort_values("advance_odds", ascending=False).head(16).set_index("team")[["advance_odds"]], horizontal=True)

    with st.expander("How this simulation works"):
        st.markdown(
            "Each group match samples from the Match Predictor exact-score distribution. "
            "Teams are ranked by points, goal difference, and goals for. The top two in each group advance, plus the eight best third-place teams. "
            "The knockout stage uses an approximate Elo-based bracket because final FIFA pairings depend on the actual third-place qualifiers."
        )


def _sample_scoreline(row: pd.Series, rng: random.Random) -> tuple[int, int]:
    try:
        records = json.loads(row.get("scoreline_grid_json", "[]"))
    except Exception:
        records = []
    if records:
        draw = rng.random() * sum(float(item["probability"]) for item in records)
        running = 0.0
        for item in records:
            running += float(item["probability"])
            if running >= draw:
                return int(item["team_a_goals"]), int(item["team_b_goals"])
    roll = rng.random() * 100
    if roll < float(row["team_a_win_prob"]):
        return 1, 0
    if roll < float(row["team_a_win_prob"]) + float(row["draw_prob"]):
        return 1, 1
    return 0, 1


def _knockout_win_probability(team_a: str, team_b: str, team_elo: dict[str, float]) -> float:
    elo_diff = team_elo.get(team_a, 1700.0) - team_elo.get(team_b, 1700.0)
    return 1 / (1 + 10 ** (-elo_diff / 400))


def simulate_tournament(predictions: pd.DataFrame, simulations: int, seed: int) -> pd.DataFrame:
    group_matches = predictions[predictions["stage"].astype(str).str.contains("Group", case=False, na=False)].copy()
    if group_matches.empty:
        return pd.DataFrame()

    teams = sorted(set(group_matches["team_a"]).union(group_matches["team_b"]))
    team_group: dict[str, str] = {}
    team_elo: dict[str, float] = {}
    for row in group_matches.itertuples(index=False):
        team_group[str(row.team_a)] = str(row.stage)
        team_group[str(row.team_b)] = str(row.stage)
        team_elo.setdefault(str(row.team_a), float(row.elo_a))
        team_elo.setdefault(str(row.team_b), float(row.elo_b))

    counters = defaultdict(lambda: {"top2": 0, "advance": 0, "title": 0, "points": 0.0})
    rng = random.Random(seed)

    for _ in range(simulations):
        table = {team: {"pts": 0, "gd": 0, "gf": 0} for team in teams}
        for _, row in group_matches.iterrows():
            goals_a, goals_b = _sample_scoreline(row, rng)
            team_a, team_b = str(row["team_a"]), str(row["team_b"])
            table[team_a]["gf"] += goals_a
            table[team_b]["gf"] += goals_b
            table[team_a]["gd"] += goals_a - goals_b
            table[team_b]["gd"] += goals_b - goals_a
            if goals_a > goals_b:
                table[team_a]["pts"] += 3
            elif goals_b > goals_a:
                table[team_b]["pts"] += 3
            else:
                table[team_a]["pts"] += 1
                table[team_b]["pts"] += 1

        qualified: list[str] = []
        thirds: list[str] = []
        for group in sorted(set(team_group.values())):
            group_teams = [team for team in teams if team_group[team] == group]
            ranked = sorted(
                group_teams,
                key=lambda team: (table[team]["pts"], table[team]["gd"], table[team]["gf"], rng.random()),
                reverse=True,
            )
            for team in ranked[:2]:
                counters[team]["top2"] += 1
                qualified.append(team)
            if len(ranked) >= 3:
                thirds.append(ranked[2])

        best_thirds = sorted(
            thirds,
            key=lambda team: (table[team]["pts"], table[team]["gd"], table[team]["gf"], rng.random()),
            reverse=True,
        )[:8]
        qualified.extend(best_thirds)
        for team in qualified:
            counters[team]["advance"] += 1
        for team in teams:
            counters[team]["points"] += table[team]["pts"]

        bracket = qualified[:]
        rng.shuffle(bracket)
        while len(bracket) > 1:
            next_round: list[str] = []
            for i in range(0, len(bracket), 2):
                if i + 1 >= len(bracket):
                    next_round.append(bracket[i])
                    continue
                team_a, team_b = bracket[i], bracket[i + 1]
                p_a = _knockout_win_probability(team_a, team_b, team_elo)
                next_round.append(team_a if rng.random() < p_a else team_b)
            bracket = next_round
        if bracket:
            counters[bracket[0]]["title"] += 1

    rows = []
    for team in teams:
        rows.append(
            {
                "group": team_group[team],
                "team": team,
                "avg_points": round(counters[team]["points"] / simulations, 2),
                "top2_odds": round(counters[team]["top2"] / simulations * 100, 1),
                "advance_odds": round(counters[team]["advance"] / simulations * 100, 1),
                "title_odds": round(counters[team]["title"] / simulations * 100, 1),
                "avg_elo": round(team_elo.get(team, 1700.0), 0),
            }
        )
    return pd.DataFrame(rows).sort_values(["title_odds", "advance_odds"], ascending=False).reset_index(drop=True)


def render_aura_lab() -> None:
    render_hero(
        "Player Impact",
        "A focused leaderboard for player influence, built from performance, clutch moments, crowd response, broadcast attention, and public reaction.",
    )

    st.sidebar.subheader("Player Impact")
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

    st.sidebar.caption(f"Source: {source_label}")
    teams = ["All"] + sorted(leaderboard["team"].unique().tolist())
    selected_team = st.sidebar.selectbox("Team", teams, key="aura_team")
    min_aura = st.sidebar.slider("Minimum impact", 0.0, 100.0, 0.0, 1.0, key="aura_min")
    top_n = st.sidebar.slider("Players shown", 3, max(3, len(leaderboard)), min(10, len(leaderboard)), key="aura_topn")

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
        render_score_card("Top impact", float(filtered["aura_score"].max()))
    with metric_cols[2]:
        render_score_card("Peak impact", float(filtered["peak_aura"].max()))
    with metric_cols[3]:
        render_score_card("Spotlight probability", float(filtered["main_character_probability"].max()), "%")

    overview_tab, detail_tab, data_tab = st.tabs(["Overview", "Player detail", "Data"])

    with overview_tab:
        render_section_header("Leaderboard summary", "The main leaderboard keeps the page scannable; deeper match history is one tab away.")
        left, right = st.columns([1.05, 1])
        with left:
            st.markdown(
                f"""
                <div class="match-card">
                  <div class="muted">Current leader</div>
                  <h2>{leader['player']}</h2>
                  <div><strong>{leader['team']}</strong> · {leader['latest_archetype']}</div>
                  <div class="muted">Impact {leader['aura_score']:.1f} · Peak {leader['peak_aura']:.1f}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with right:
            st.bar_chart(filtered.set_index("player")[["aura_score", "peak_aura"]], horizontal=True)
        st.dataframe(
            filtered[["player", "team", "matches", "aura_score", "peak_aura", "main_character_probability", "latest_archetype"]],
            hide_index=True,
            use_container_width=True,
        )

    with detail_tab:
        selected_player = st.selectbox("Inspect player", filtered["player"].tolist(), key="aura_player")
        player_summary = leaderboard.loc[leaderboard["player"] == selected_player].iloc[0]
        player_matches = match_scores.loc[match_scores["player"] == selected_player].copy()
        render_section_header(selected_player, f"{player_summary['team']} · {player_summary['latest_archetype']}")
        left, right = st.columns([1, 1.2])
        with left:
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
        with right:
            st.dataframe(
                player_matches[["opponent", "stage", "aura_score", "impact_score", "clutch_score", "crowd_score", "broadcast_score", "social_score"]]
                .sort_values("aura_score", ascending=False),
                hide_index=True,
                use_container_width=True,
            )

    with data_tab:
        render_section_header("Upload schema", "Use this if you want to replace the sample player-match data.")
        show_schema_help(AURA_REQUIRED_COLUMNS)


def render_chaos_center() -> None:
    render_hero(
        "Match Volatility",
        "A compact view of fixtures with the highest variance signals: cards, penalties, late goals, lead changes, VAR incidents, and crowd intensity.",
    )

    st.sidebar.subheader("Match Volatility")
    uploaded_file = st.sidebar.file_uploader("Upload volatility CSV", type=["csv"], key="chaos_upload")
    use_sample = st.sidebar.toggle("Use sample volatility data", value=uploaded_file is None, key="chaos_sample")

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

    st.sidebar.caption(f"Source: {source_label}")
    stages = ["All"] + sorted(chaos["stage"].unique().tolist())
    selected_stage = st.sidebar.selectbox("Stage", stages, key="chaos_stage")

    filtered = chaos.copy()
    if selected_stage != "All":
        filtered = filtered[filtered["stage"] == selected_stage]

    feature = filtered.iloc[0]
    metric_cols = st.columns(4)
    with metric_cols[0]:
        render_score_card("Matches", float(len(filtered)))
    with metric_cols[1]:
        render_score_card("Top volatility", float(filtered["chaos_score"].max()))
    with metric_cols[2]:
        render_score_card("Viral risk", float(filtered["meme_voltage"].max()), "%")
    with metric_cols[3]:
        render_score_card("Average volatility", float(filtered["chaos_score"].mean()))

    overview_tab, detail_tab, data_tab = st.tabs(["Overview", "Match detail", "Data"])
    with overview_tab:
        render_section_header("Volatility board", "A short ranking first, with detailed signal breakdowns kept separate.")
        left, right = st.columns([1, 1])
        with left:
            st.markdown(
                f"""
                <div class="match-card">
                  <div class="muted">Highest volatility fixture</div>
                  <h2>{feature['fixture']}</h2>
                  <div><strong>{feature['stage']}</strong> · {feature['chaos_tag']}</div>
                  <div class="muted">Score {feature['chaos_score']:.1f} · Viral risk {feature['meme_voltage']:.1f}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with right:
            chart = filtered.rename(columns={"chaos_score": "volatility_score", "meme_voltage": "viral_risk"})
            st.bar_chart(chart.set_index("fixture")[["volatility_score", "viral_risk"]], horizontal=True)
        st.dataframe(
            filtered[["fixture", "stage", "chaos_score", "meme_voltage", "chaos_tag"]]
            .rename(columns={"chaos_score": "volatility_score", "meme_voltage": "viral_risk", "chaos_tag": "tag"}),
            hide_index=True,
            use_container_width=True,
        )

    with detail_tab:
        selected_match = st.selectbox("Inspect match", filtered["fixture"].tolist(), key="chaos_match")
        match_row = filtered.loc[filtered["fixture"] == selected_match].iloc[0]
        render_section_header(selected_match, f"{match_row['stage']} · {match_row['chaos_tag']}")
        detail_df = pd.DataFrame(
            {
                "signal": ["goals", "yellow cards", "red cards", "penalties", "VAR incidents", "late goals", "lead changes", "crowd noise"],
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
        left, right = st.columns([1, 1])
        with left:
            st.dataframe(detail_df, hide_index=True, use_container_width=True)
        with right:
            st.bar_chart(detail_df, x="signal", y="value")
            st.progress(min(max(float(match_row["chaos_score"]) / 100, 0.0), 1.0), text=f"Volatility score: {match_row['chaos_score']:.1f}")
            st.progress(min(max(float(match_row["meme_voltage"]) / 100, 0.0), 1.0), text=f"Viral risk: {match_row['meme_voltage']:.1f}%")

    with data_tab:
        render_section_header("Upload schema", "Use this if you want to replace the sample match-volatility data.")
        show_schema_help(CHAOS_REQUIRED_COLUMNS)


def render_underdog_radar() -> None:
    render_hero(
        "Upset Risk",
        "A focused scanner for favorite vulnerability based on transition threat, set pieces, goalkeeper form, fatigue, crowd support, and composure.",
    )

    st.sidebar.subheader("Upset Risk")
    uploaded_file = st.sidebar.file_uploader("Upload scenario CSV", type=["csv"], key="underdog_upload")
    use_sample = st.sidebar.toggle("Use sample scenario data", value=uploaded_file is None, key="underdog_sample")

    try:
        if uploaded_file is not None and not use_sample:
            source_df = load_uploaded_csv(uploaded_file)
            source_label = uploaded_file.name
        else:
            source_df = load_sample_underdog_data()
            source_label = "data/sample_underdog_scenarios.csv"
        scenarios = score_underdog_scenarios(source_df)
    except Exception as exc:
        st.error(str(exc))
        show_schema_help(UNDERDOG_REQUIRED_COLUMNS)
        return

    st.sidebar.caption(f"Source: {source_label}")
    metric_cols = st.columns(4)
    with metric_cols[0]:
        render_score_card("Scenarios", float(len(scenarios)))
    with metric_cols[1]:
        render_score_card("Top risk", float(scenarios["upset_risk"].max()), "%")
    with metric_cols[2]:
        render_score_card("Average risk", float(scenarios["upset_risk"].mean()), "%")
    with metric_cols[3]:
        render_score_card("High-alert", float((scenarios["upset_risk"] >= 58).sum()))

    board_tab, builder_tab, data_tab = st.tabs(["Risk board", "Scenario builder", "Data"])
    with board_tab:
        render_section_header("Upset board", "The table is intentionally short; use the builder tab to test your own scenario.")
        left, right = st.columns([1, 1])
        with left:
            st.dataframe(scenarios[["fixture", "stage", "upset_risk", "danger_meter"]], hide_index=True, use_container_width=True)
        with right:
            st.bar_chart(scenarios.set_index("fixture")[["upset_risk"]], horizontal=True)

    with builder_tab:
        render_section_header("Build a custom upset scenario", "Adjust the key signals and see how the risk score responds.")
        left, right = st.columns([1, 1])
        with left:
            favorite_rating = st.slider("Favorite rating", 70, 99, 90)
            underdog_form = st.slider("Underdog recent form", 40, 99, 82)
            transition_threat = st.slider("Transition threat", 40, 99, 84)
            set_piece_edge = st.slider("Set-piece edge", 40, 99, 72)
            goalkeeper_form = st.slider("Goalkeeper form", 40, 99, 78)
        with right:
            fatigue_gap = st.slider("Fatigue gap", 0, 40, 18)
            crowd_support = st.slider("Crowd support", 20, 99, 70)
            composure = st.slider("Underdog composure", 40, 99, 76)
            injury_disruption = st.slider("Favorite injury disruption", 0, 40, 12)

        custom = pd.DataFrame([
            {
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
            }
        ])
        combined = pd.concat([scenarios[UNDERDOG_REQUIRED_COLUMNS], custom], ignore_index=True)
        custom_scored = score_underdog_scenarios(combined).loc[lambda df: df["stage"] == "Custom"].iloc[0]
        st.progress(min(max(float(custom_scored["upset_risk"]) / 100, 0.0), 1.0), text=f"Upset risk: {custom_scored['upset_risk']:.1f}%")
        render_status_pill(f"{custom_scored['danger_meter']}", "warn" if custom_scored["upset_risk"] >= 58 else "good")

    with data_tab:
        render_section_header("Upload schema", "Use this if you want to replace the sample upset-risk scenarios.")
        show_schema_help(UNDERDOG_REQUIRED_COLUMNS)


def render_fan_pain_lab() -> None:
    render_hero(
        "Fan Pressure",
        "A lightweight pressure model for tournament expectation, penalty stress, referee anxiety, and blown-lead risk.",
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
        render_score_card("Pressure score", float(result["pain_score"]))
        st.progress(min(max(float(result["pain_score"]) / 100, 0.0), 1.0), text=f"Tier: {result['tier']}")
        st.progress(min(max(float(result["meltdown_probability"]) / 100, 0.0), 1.0), text=f"High-pressure reaction probability: {result['meltdown_probability']:.1f}%")

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

    st.markdown("### Interpretation")
    if result["pain_score"] >= 80:
        st.error("Very high pressure profile. Early leads, referee decisions, and penalties could produce a sharp fan reaction.")
    elif result["pain_score"] >= 65:
        st.warning("Elevated pressure profile. The match is likely to feel stressful if it stays close late.")
    else:
        st.success("Manageable pressure profile. Risk is present, but the inputs do not suggest extreme fan stress.")


def show_schema_help(columns: list[str]) -> None:
    st.markdown("**Required columns**")
    st.code(", ".join(columns), language="text")
