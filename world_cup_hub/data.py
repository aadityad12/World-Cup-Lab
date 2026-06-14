from __future__ import annotations

import json
import math
from io import StringIO
from pathlib import Path

import pandas as pd
import streamlit as st

from aura_model.scoring import REQUIRED_COLUMNS as AURA_REQUIRED_COLUMNS
from aura_model.scoring import build_player_leaderboard, score_matches
from world_cup_hub.normalization import candidate_team_keys, normalize_team_name

CHAOS_REQUIRED_COLUMNS = [
    "match_id",
    "team_a",
    "team_b",
    "stage",
    "total_goals",
    "yellow_cards",
    "red_cards",
    "penalties",
    "var_incidents",
    "late_goals",
    "lead_changes",
    "upset_factor",
    "dribble_duels",
    "goalkeeper_heroics",
    "crowd_noise",
]

UNDERDOG_REQUIRED_COLUMNS = [
    "favorite",
    "underdog",
    "stage",
    "favorite_rating",
    "underdog_form",
    "transition_threat",
    "set_piece_edge",
    "goalkeeper_form",
    "fatigue_gap",
    "crowd_support",
    "composure",
    "injury_disruption",
]

WINNER_REQUIRED_COLUMNS = [
    "match_id",
    "date",
    "stage",
    "team_a",
    "team_b",
    "elo_a",
    "elo_b",
    "fifa_rank_a",
    "fifa_rank_b",
    "recent_form_a",
    "recent_form_b",
    "attack_rating_a",
    "attack_rating_b",
    "defense_rating_a",
    "defense_rating_b",
    "midfield_control_a",
    "midfield_control_b",
    "goalkeeper_form_a",
    "goalkeeper_form_b",
    "set_piece_strength_a",
    "set_piece_strength_b",
    "injuries_impact_a",
    "injuries_impact_b",
    "rest_days_a",
    "rest_days_b",
    "travel_fatigue_a",
    "travel_fatigue_b",
    "home_region_boost_a",
    "home_region_boost_b",
    "big_match_experience_a",
    "big_match_experience_b",
]

WINNER_OPTIONAL_COLUMNS = {
    "kickoff_local": "",
    "team_a_code": "",
    "team_b_code": "",
    "stadium": "Unknown stadium",
    "city": "Unknown city",
    "venue_country": "",
    "latitude": "",
    "longitude": "",
    "roof_notes": "",
    "weather_temperature_c": "",
    "weather_precipitation_probability": "",
    "weather_wind_kmh": "",
    "expected_lineup_strength_a": 100,
    "expected_lineup_strength_b": 100,
    "suspensions_impact_a": 0,
    "suspensions_impact_b": 0,
    "travel_km_a": "",
    "travel_km_b": "",
    "timezone_shift_a": "",
    "timezone_shift_b": "",
    "altitude_change_m_a": "",
    "altitude_change_m_b": "",
    "fixture_source": "uploaded/manual CSV",
    "rating_source": "uploaded/manual CSV",
    "venue_source": "uploaded/manual CSV",
    "weather_source": "not provided",
    "unavailable_data_notes": "",
}


@st.cache_data
def _load_csv_cached(path: str, mtime_ns: int) -> pd.DataFrame:
    return pd.read_csv(path)


def _load_project_csv(path: str) -> pd.DataFrame:
    file_path = Path(path)
    return _load_csv_cached(str(file_path), file_path.stat().st_mtime_ns)


def load_sample_player_data() -> pd.DataFrame:
    return _load_project_csv("data/sample_player_matches.csv")


def load_sample_chaos_data() -> pd.DataFrame:
    return _load_project_csv("data/sample_match_chaos.csv")


def load_sample_underdog_data() -> pd.DataFrame:
    return _load_project_csv("data/sample_underdog_scenarios.csv")


def load_sample_winner_data() -> pd.DataFrame:
    return _load_project_csv("data/public_2026_match_features.csv")


def load_public_2026_results() -> pd.DataFrame:
    path = Path("data/public_2026_results.csv")
    if not path.exists():
        return pd.DataFrame()
    return _load_project_csv(str(path))


def load_uploaded_csv(uploaded_file) -> pd.DataFrame:
    return pd.read_csv(StringIO(uploaded_file.getvalue().decode("utf-8")))


@st.cache_data
def _load_team_strength_model_cached(path: str, mtime_ns: int) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_team_strength_model() -> dict:
    path = Path("models/artifacts/team_strength_model.json")
    if not path.exists():
        return {}
    return _load_team_strength_model_cached(str(path), path.stat().st_mtime_ns)


def get_winner_model_metrics() -> dict:
    return load_team_strength_model().get("metrics", {})


def _safe_norm(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce").fillna(0)
    minimum = numeric.min()
    maximum = numeric.max()
    if math.isclose(float(minimum), float(maximum)):
        return pd.Series([0.5] * len(series), index=series.index)
    return (numeric - minimum) / (maximum - minimum)


def _sigmoid(value: float) -> float:
    return 1 / (1 + math.exp(-value))


def _validate_columns(df: pd.DataFrame, required: list[str]) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")


def prepare_aura_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    _validate_columns(df, AURA_REQUIRED_COLUMNS)
    match_scores = score_matches(df)
    leaderboard = build_player_leaderboard(match_scores)
    return match_scores, leaderboard


def score_chaos_matches(df: pd.DataFrame) -> pd.DataFrame:
    _validate_columns(df, CHAOS_REQUIRED_COLUMNS)
    scored = df.copy()

    volatility = (
        0.18 * _safe_norm(scored["total_goals"])
        + 0.12 * _safe_norm(scored["yellow_cards"])
        + 0.12 * _safe_norm(scored["red_cards"])
        + 0.10 * _safe_norm(scored["penalties"])
        + 0.12 * _safe_norm(scored["var_incidents"])
        + 0.12 * _safe_norm(scored["late_goals"])
        + 0.10 * _safe_norm(scored["lead_changes"])
        + 0.06 * _safe_norm(scored["dribble_duels"])
        + 0.04 * _safe_norm(scored["goalkeeper_heroics"])
        + 0.04 * _safe_norm(scored["crowd_noise"])
    )
    chaos_score = (volatility * (1 + 0.08 * scored["upset_factor"]) * 100).clip(0, 100)

    scored["chaos_score"] = chaos_score.round(1)
    scored["volatility_score"] = (volatility * 100).round(1)
    scored["meme_voltage"] = chaos_score.apply(lambda value: _sigmoid((value - 55) / 8) * 100).round(1)
    scored["chaos_tag"] = scored.apply(_chaos_tag, axis=1)
    scored["fixture"] = scored["team_a"] + " vs " + scored["team_b"]
    return scored.sort_values("chaos_score", ascending=False).reset_index(drop=True)


def _chaos_tag(row: pd.Series) -> str:
    if row["chaos_score"] >= 82:
        return "Pure cinema"
    if row["var_incidents"] >= 2 or row["penalties"] >= 2:
        return "VAR meltdown"
    if row["late_goals"] >= 2:
        return "Late-game bedlam"
    if row["red_cards"] >= 1:
        return "Spicy knockout energy"
    return "Controlled mayhem"


def score_underdog_scenarios(df: pd.DataFrame) -> pd.DataFrame:
    _validate_columns(df, UNDERDOG_REQUIRED_COLUMNS)
    scored = df.copy()

    upset_raw = (
        0.18 * _safe_norm(scored["underdog_form"])
        + 0.16 * _safe_norm(scored["transition_threat"])
        + 0.14 * _safe_norm(scored["set_piece_edge"])
        + 0.15 * _safe_norm(scored["goalkeeper_form"])
        + 0.10 * _safe_norm(scored["fatigue_gap"])
        + 0.09 * _safe_norm(scored["crowd_support"])
        + 0.10 * _safe_norm(scored["composure"])
        + 0.08 * _safe_norm(scored["injury_disruption"])
        - 0.14 * _safe_norm(scored["favorite_rating"])
    )

    upset_risk = upset_raw.apply(lambda value: _sigmoid((value - 0.34) * 5.5) * 100)
    scored["upset_risk"] = upset_risk.round(1)
    scored["danger_meter"] = scored["upset_risk"].apply(_danger_label)
    scored["fixture"] = scored["favorite"] + " vs " + scored["underdog"]
    return scored.sort_values("upset_risk", ascending=False).reset_index(drop=True)


def _danger_label(value: float) -> str:
    if value >= 72:
        return "Sirens on"
    if value >= 58:
        return "Very live"
    if value >= 45:
        return "Tricky"
    return "Manageable"


def score_winner_matches(df: pd.DataFrame) -> pd.DataFrame:
    df = _coerce_winner_schema_aliases(df.copy())
    _validate_columns(df, WINNER_REQUIRED_COLUMNS)
    scored = _ensure_winner_optional_columns(df.copy())

    edge = scored.apply(_winner_edge_score, axis=1)
    predictions = [_poisson_prediction_for_row(row) for _, row in scored.iterrows()]
    prediction_df = pd.DataFrame(predictions, index=scored.index)

    scored["fixture"] = scored["team_a"] + " vs " + scored["team_b"]
    scored["team_a_win_prob"] = prediction_df["team_a_win_prob"].round(1)
    scored["draw_prob"] = prediction_df["draw_prob"].round(1)
    scored["team_b_win_prob"] = prediction_df["team_b_win_prob"].round(1)
    scored["winner_pick"] = prediction_df["winner_pick"]
    scored["confidence"] = prediction_df["confidence"].round(1)
    scored["expected_goals_a"] = prediction_df["expected_goals_a"].round(2)
    scored["expected_goals_b"] = prediction_df["expected_goals_b"].round(2)
    scored["predicted_score"] = prediction_df["predicted_score"]
    scored["prediction_summary"] = prediction_df["prediction_summary"]
    scored["top_scorelines"] = prediction_df["top_scorelines"]
    scored["scoreline_grid_json"] = prediction_df["scoreline_grid_json"]
    scored["model_family"] = prediction_df["model_family"]
    scored["team_a_advances_prob"] = prediction_df["team_a_advances_prob"].round(1)
    scored["team_b_advances_prob"] = prediction_df["team_b_advances_prob"].round(1)
    scored["extra_time_prob"] = prediction_df["extra_time_prob"].round(1)
    scored["penalty_shootout_prob"] = prediction_df["penalty_shootout_prob"].round(1)
    scored["edge_score"] = edge.round(1)
    scored["model_tag"] = scored.apply(_winner_model_tag, axis=1)
    return scored.sort_values(["date", "match_id"]).reset_index(drop=True)


def _find_team_profile(profiles: dict, team_name: str) -> dict | None:
    for key in candidate_team_keys(team_name):
        if key in profiles:
            return profiles[key]
    return None


def _numeric_value(row: pd.Series, column: str, default: float = 0.0) -> float:
    value = pd.to_numeric(pd.Series([row.get(column, default)]), errors="coerce").iloc[0]
    if pd.isna(value):
        return default
    return float(value)


def _poisson_pmf(k: int, lam: float) -> float:
    return math.exp(-lam) * (lam ** k) / math.factorial(k)


def _dixon_coles_factor(goals_a: int, goals_b: int, lam_a: float, lam_b: float, rho: float) -> float:
    if goals_a == 0 and goals_b == 0:
        return max(0.05, 1 - lam_a * lam_b * rho)
    if goals_a == 0 and goals_b == 1:
        return max(0.05, 1 + lam_a * rho)
    if goals_a == 1 and goals_b == 0:
        return max(0.05, 1 + lam_b * rho)
    if goals_a == 1 and goals_b == 1:
        return max(0.05, 1 - rho)
    return 1.0


def _score_distribution(lam_a: float, lam_b: float, rho: float, max_goals: int = 8) -> list[tuple[int, int, float]]:
    cells = []
    total = 0.0
    for goals_a in range(max_goals + 1):
        for goals_b in range(max_goals + 1):
            probability = (
                _poisson_pmf(goals_a, lam_a)
                * _poisson_pmf(goals_b, lam_b)
                * _dixon_coles_factor(goals_a, goals_b, lam_a, lam_b, rho)
            )
            cells.append((goals_a, goals_b, probability))
            total += probability
    return [(goals_a, goals_b, probability / max(total, 1e-12)) for goals_a, goals_b, probability in cells]


def _is_knockout_stage(stage: str) -> bool:
    stage_lower = str(stage).lower()
    return any(token in stage_lower for token in ["round", "r16", "quarter", "semi", "final", "third"])


def _availability_multiplier(row: pd.Series, side: str) -> float:
    lineup_strength = _numeric_value(row, f"expected_lineup_strength_{side}", 100.0)
    injuries = _numeric_value(row, f"injuries_impact_{side}", 0.0)
    suspensions = _numeric_value(row, f"suspensions_impact_{side}", 0.0)
    return max(0.72, min(1.16, 1 + (lineup_strength - 100) / 350 - injuries / 180 - suspensions / 160))


def _travel_rest_multiplier(row: pd.Series, side: str) -> float:
    opponent = "b" if side == "a" else "a"
    rest_edge = _numeric_value(row, f"rest_days_{side}", 5.0) - _numeric_value(row, f"rest_days_{opponent}", 5.0)
    travel_fatigue = _numeric_value(row, f"travel_fatigue_{side}", 0.0)
    travel_km = _numeric_value(row, f"travel_km_{side}", 0.0)
    timezone_shift = abs(_numeric_value(row, f"timezone_shift_{side}", 0.0))
    altitude_change = abs(_numeric_value(row, f"altitude_change_m_{side}", 0.0))
    fatigue_penalty = travel_fatigue / 700 + travel_km / 18000 + timezone_shift / 120 + altitude_change / 45000
    return max(0.82, min(1.12, 1 + rest_edge * 0.018 - fatigue_penalty))


def _goal_lambdas_from_model(row: pd.Series) -> tuple[float, float, str]:
    model = load_team_strength_model()
    profiles = model.get("team_profiles", {})
    team_a_profile = _find_team_profile(profiles, str(row["team_a"]))
    team_b_profile = _find_team_profile(profiles, str(row["team_b"]))

    if not team_a_profile or not team_b_profile:
        return _expected_goals(row, "a"), _expected_goals(row, "b"), "Heuristic Poisson fallback"

    base = float(model.get("global_avg_goals_per_team", 1.32))
    lam_a = base * float(team_a_profile["attack_factor"]) * float(team_b_profile["defense_concede_factor"])
    lam_b = base * float(team_b_profile["attack_factor"]) * float(team_a_profile["defense_concede_factor"])

    elo_edge = (_numeric_value(row, "elo_a", 1700.0) - _numeric_value(row, "elo_b", 1700.0)) / 400.0
    lam_a *= math.exp(0.075 * elo_edge)
    lam_b *= math.exp(-0.075 * elo_edge)

    lam_a *= 1 + _numeric_value(row, "home_region_boost_a", 0.0) / 180
    lam_b *= 1 + _numeric_value(row, "home_region_boost_b", 0.0) / 180
    lam_a *= _availability_multiplier(row, "a") * _travel_rest_multiplier(row, "a")
    lam_b *= _availability_multiplier(row, "b") * _travel_rest_multiplier(row, "b")

    weather_penalty = _weather_goal_penalty(row)
    lam_a *= max(0.72, 1 - weather_penalty)
    lam_b *= max(0.72, 1 - weather_penalty)
    return max(0.15, min(4.75, lam_a)), max(0.15, min(4.75, lam_b)), "Trained Poisson/Dixon-Coles"


def _poisson_prediction_for_row(row: pd.Series) -> dict[str, float | str]:
    model = load_team_strength_model()
    rho = float(model.get("dixon_coles_rho", -0.05))
    max_goals = int(model.get("max_goals_modeled", 8))
    lam_a, lam_b, family = _goal_lambdas_from_model(row)
    distribution = _score_distribution(lam_a, lam_b, rho, max_goals=max_goals)

    team_a_win = sum(prob for goals_a, goals_b, prob in distribution if goals_a > goals_b)
    draw = sum(prob for goals_a, goals_b, prob in distribution if goals_a == goals_b)
    team_b_win = sum(prob for goals_a, goals_b, prob in distribution if goals_b > goals_a)
    top_scores = sorted(distribution, key=lambda item: item[2], reverse=True)[:5]
    predicted_a, predicted_b, score_prob = top_scores[0]

    outcome_probs = {
        str(row["team_a"]): team_a_win,
        "Draw": draw,
        str(row["team_b"]): team_b_win,
    }
    winner_pick = max(outcome_probs, key=outcome_probs.get)
    confidence = outcome_probs[winner_pick] * 100
    # Group-stage soccer often has a draw as the single most likely exact score even
    # when one side has a small aggregate win edge. In near-tossups, surface that draw
    # instead of forcing a team pick.
    if not _is_knockout_stage(str(row.get("stage", ""))) and predicted_a == predicted_b:
        if outcome_probs[winner_pick] - draw <= 0.05:
            winner_pick = "Draw"
            confidence = draw * 100

    no_draw_share = team_a_win / max(team_a_win + team_b_win, 1e-12)
    team_a_advances = team_a_win + draw * no_draw_share
    team_b_advances = team_b_win + draw * (1 - no_draw_share)
    extra_time_prob = 0.0
    penalty_prob = 0.0
    if _is_knockout_stage(str(row.get("stage", ""))):
        extra_time_prob = draw * 0.65 * 100
        penalty_prob = draw * 0.35 * 100
        winner_pick = row["team_a"] if team_a_advances >= team_b_advances else row["team_b"]
        confidence = max(team_a_advances, team_b_advances) * 100

    predicted_score = f"{predicted_a}-{predicted_b}"
    top_scorelines = "; ".join(f"{a}-{b} ({p * 100:.1f}%)" for a, b, p in top_scores)
    scoreline_grid = [
        {"team_a_goals": a, "team_b_goals": b, "probability": round(p * 100, 3)}
        for a, b, p in distribution
        if a <= 5 and b <= 5
    ]
    return {
        "team_a_win_prob": team_a_win * 100,
        "draw_prob": draw * 100,
        "team_b_win_prob": team_b_win * 100,
        "winner_pick": winner_pick,
        "confidence": confidence,
        "expected_goals_a": lam_a,
        "expected_goals_b": lam_b,
        "predicted_score": predicted_score,
        "prediction_summary": f"{winner_pick} {predicted_score}",
        "top_scorelines": top_scorelines,
        "scoreline_grid_json": json.dumps(scoreline_grid),
        "model_family": family,
        "team_a_advances_prob": team_a_advances * 100 if _is_knockout_stage(str(row.get("stage", ""))) else 0.0,
        "team_b_advances_prob": team_b_advances * 100 if _is_knockout_stage(str(row.get("stage", ""))) else 0.0,
        "extra_time_prob": extra_time_prob,
        "penalty_shootout_prob": penalty_prob,
    }


def _coerce_winner_schema_aliases(df: pd.DataFrame) -> pd.DataFrame:
    # Backwards compatibility: older project files used fifa_rank_* for Elo ranking.
    # New uploads may use the clearer elo_rank_* names.
    if "fifa_rank_a" not in df.columns and "elo_rank_a" in df.columns:
        df["fifa_rank_a"] = df["elo_rank_a"]
    if "fifa_rank_b" not in df.columns and "elo_rank_b" in df.columns:
        df["fifa_rank_b"] = df["elo_rank_b"]
    if "elo_rank_a" not in df.columns and "fifa_rank_a" in df.columns:
        df["elo_rank_a"] = df["fifa_rank_a"]
    if "elo_rank_b" not in df.columns and "fifa_rank_b" in df.columns:
        df["elo_rank_b"] = df["fifa_rank_b"]
    return df


def _ensure_winner_optional_columns(df: pd.DataFrame) -> pd.DataFrame:
    for column, default in WINNER_OPTIONAL_COLUMNS.items():
        if column not in df.columns:
            df[column] = default
    return df


def _winner_edge_score(row: pd.Series) -> float:
    breakdown = build_winner_factor_breakdown(row)
    return float(breakdown["contribution"].sum())


def _weather_goal_penalty(row: pd.Series) -> float:
    temperature = pd.to_numeric(pd.Series([row.get("weather_temperature_c", "")]), errors="coerce").iloc[0]
    precipitation = pd.to_numeric(pd.Series([row.get("weather_precipitation_probability", "")]), errors="coerce").iloc[0]
    wind = pd.to_numeric(pd.Series([row.get("weather_wind_kmh", "")]), errors="coerce").iloc[0]
    penalty = 0.0
    if pd.notna(temperature):
        if temperature > 30:
            penalty += (temperature - 30) * 0.012
        if temperature < 5:
            penalty += (5 - temperature) * 0.015
    if pd.notna(precipitation):
        penalty += min(0.12, precipitation / 100 * 0.10)
    if pd.notna(wind) and wind > 20:
        penalty += min(0.10, (wind - 20) * 0.006)
    roof_notes = str(row.get("roof_notes", "")).lower()
    if "roof" in roof_notes:
        penalty *= 0.45
    return float(penalty)


def _expected_goals(row: pd.Series, side: str) -> float:
    if side == "a":
        own = "a"
        opp = "b"
    else:
        own = "b"
        opp = "a"

    attack_vs_defense = row[f"attack_rating_{own}"] - row[f"defense_rating_{opp}"]
    form_edge = row[f"recent_form_{own}"] - 75
    midfield_edge = row[f"midfield_control_{own}"] - row[f"midfield_control_{opp}"]
    set_piece_edge = row[f"set_piece_strength_{own}"] - row[f"set_piece_strength_{opp}"]
    keeper_suppression = row[f"goalkeeper_form_{opp}"] - 75
    rest_edge = row[f"rest_days_{own}"] - row[f"rest_days_{opp}"]

    goals = (
        1.18
        + 0.018 * attack_vs_defense
        + 0.010 * form_edge
        + 0.006 * midfield_edge
        + 0.006 * set_piece_edge
        - 0.007 * keeper_suppression
        + 0.025 * rest_edge
        - 0.012 * row[f"injuries_impact_{own}"]
        - 0.006 * row[f"travel_fatigue_{own}"]
        + 0.010 * row[f"home_region_boost_{own}"]
        - _weather_goal_penalty(row)
    )
    return max(0.25, min(4.25, float(goals)))


def build_prediction_result_comparison(predictions: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame()
    pred = predictions.copy()
    actual = results.copy()
    pred["_join_key"] = pred.apply(
        lambda row: f"{row['date']}|{normalize_team_name(row['team_a'])}|{normalize_team_name(row['team_b'])}",
        axis=1,
    )
    actual["_join_key"] = actual.apply(
        lambda row: f"{row['date']}|{normalize_team_name(row['team_a'])}|{normalize_team_name(row['team_b'])}",
        axis=1,
    )
    merged = pred.merge(
        actual[["_join_key", "actual_score", "actual_goals_a", "actual_goals_b", "actual_outcome", "result_source"]],
        on="_join_key",
        how="inner",
    )
    if merged.empty:
        return merged
    merged["prediction_correct"] = merged["winner_pick"].astype(str).str.lower() == merged["actual_outcome"].astype(str).str.lower()
    merged["exact_score_correct"] = merged["predicted_score"].astype(str) == merged["actual_score"].astype(str)
    merged["result_status"] = merged["prediction_correct"].map({True: "Correct", False: "Missed"})
    return merged.sort_values(["date", "match_id"]).reset_index(drop=True)


def build_scoreline_heatmap(row: pd.Series) -> pd.DataFrame:
    payload = row.get("scoreline_grid_json", "[]")
    try:
        records = json.loads(payload) if isinstance(payload, str) else payload
    except Exception:
        records = []
    if not records:
        return pd.DataFrame()
    grid = pd.DataFrame(records)
    heatmap = grid.pivot(index="team_b_goals", columns="team_a_goals", values="probability").sort_index(ascending=False)
    heatmap.index = [f"{row['team_b']} {goals}" for goals in heatmap.index]
    heatmap.columns = [f"{row['team_a']} {goals}" for goals in heatmap.columns]
    return heatmap.fillna(0.0).round(2)


def build_winner_factor_breakdown(row: pd.Series) -> pd.DataFrame:
    factors = [
        {
            "factor": "Elo strength",
            "team_a_signal": row["elo_a"],
            "team_b_signal": row["elo_b"],
            "edge": (row["elo_a"] - row["elo_b"]) / 400,
            "weight": 24.0,
        },
        {
            "factor": "Team ranking (Elo rank)",
            "team_a_signal": row["fifa_rank_a"],
            "team_b_signal": row["fifa_rank_b"],
            "edge": (row["fifa_rank_b"] - row["fifa_rank_a"]) / 50,
            "weight": 12.0,
        },
        {
            "factor": "Recent form",
            "team_a_signal": row["recent_form_a"],
            "team_b_signal": row["recent_form_b"],
            "edge": (row["recent_form_a"] - row["recent_form_b"]) / 10,
            "weight": 1.8,
        },
        {
            "factor": "Attack vs defense matchup",
            "team_a_signal": row["attack_rating_a"] - row["defense_rating_b"],
            "team_b_signal": row["attack_rating_b"] - row["defense_rating_a"],
            "edge": ((row["attack_rating_a"] - row["defense_rating_b"]) - (row["attack_rating_b"] - row["defense_rating_a"])) / 10,
            "weight": 1.4,
        },
        {
            "factor": "Midfield control",
            "team_a_signal": row["midfield_control_a"],
            "team_b_signal": row["midfield_control_b"],
            "edge": (row["midfield_control_a"] - row["midfield_control_b"]) / 10,
            "weight": 1.2,
        },
        {
            "factor": "Goalkeeper form",
            "team_a_signal": row["goalkeeper_form_a"],
            "team_b_signal": row["goalkeeper_form_b"],
            "edge": (row["goalkeeper_form_a"] - row["goalkeeper_form_b"]) / 10,
            "weight": 0.8,
        },
        {
            "factor": "Set pieces",
            "team_a_signal": row["set_piece_strength_a"],
            "team_b_signal": row["set_piece_strength_b"],
            "edge": (row["set_piece_strength_a"] - row["set_piece_strength_b"]) / 10,
            "weight": 0.7,
        },
        {
            "factor": "Injury availability",
            "team_a_signal": 100 - row["injuries_impact_a"],
            "team_b_signal": 100 - row["injuries_impact_b"],
            "edge": (row["injuries_impact_b"] - row["injuries_impact_a"]) / 10,
            "weight": 2.0,
        },
        {
            "factor": "Rest / travel",
            "team_a_signal": row["rest_days_a"] - row["travel_fatigue_a"] / 10,
            "team_b_signal": row["rest_days_b"] - row["travel_fatigue_b"] / 10,
            "edge": ((row["rest_days_a"] - row["rest_days_b"]) * 1.0 + (row["travel_fatigue_b"] - row["travel_fatigue_a"]) / 10),
            "weight": 1.1,
        },
        {
            "factor": "Regional / host boost",
            "team_a_signal": row["home_region_boost_a"],
            "team_b_signal": row["home_region_boost_b"],
            "edge": (row["home_region_boost_a"] - row["home_region_boost_b"]) / 10,
            "weight": 1.0,
        },
        {
            "factor": "Big-match experience",
            "team_a_signal": row["big_match_experience_a"],
            "team_b_signal": row["big_match_experience_b"],
            "edge": (row["big_match_experience_a"] - row["big_match_experience_b"]) / 10,
            "weight": 0.8,
        },
        {
            "factor": "Weather goal suppression",
            "team_a_signal": _weather_goal_penalty(row),
            "team_b_signal": _weather_goal_penalty(row),
            "edge": 0.0,
            "weight": 0.0,
        },
    ]
    breakdown = pd.DataFrame(factors)
    breakdown["contribution"] = (breakdown["edge"] * breakdown["weight"]).round(2)
    breakdown["team_a_signal"] = breakdown["team_a_signal"].round(2)
    breakdown["team_b_signal"] = breakdown["team_b_signal"].round(2)
    breakdown["edge"] = breakdown["edge"].round(2)
    breakdown["weight"] = breakdown["weight"].round(2)
    return breakdown


def _winner_model_tag(row: pd.Series) -> str:
    confidence = row["confidence"]
    if confidence >= 62:
        return "Strong lean"
    if confidence >= 50:
        return "Clear edge"
    if confidence >= 40:
        return "Slight edge"
    return "Toss-up"


def simulate_fan_pain(
    expectation: float,
    trauma: float,
    lead_fragility: float,
    rival_success: float,
    penalties: float,
    referee_anxiety: float,
) -> dict[str, float | str]:
    weighted = (
        0.24 * expectation
        + 0.24 * trauma
        + 0.16 * lead_fragility
        + 0.14 * rival_success
        + 0.12 * penalties
        + 0.10 * referee_anxiety
    )
    score = max(0.0, min(100.0, weighted))

    if score >= 82:
        tier = "Catastrophic"
    elif score >= 68:
        tier = "High stress"
    elif score >= 50:
        tier = "Emotionally dangerous"
    else:
        tier = "Mostly survivable"

    return {
        "pain_score": round(score, 1),
        "tier": tier,
        "meltdown_probability": round(_sigmoid((score - 58) / 7) * 100, 1),
    }
