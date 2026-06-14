from __future__ import annotations

import ast
import json
from pathlib import Path

import pandas as pd

from world_cup_hub.apps import simulate_tournament
from world_cup_hub.data import build_prediction_result_comparison, build_scoreline_heatmap, load_public_2026_results, score_winner_matches
from world_cup_hub.normalization import normalize_team_name


def test_team_normalization_handles_accents_and_aliases() -> None:
    assert normalize_team_name("Curaçao") == "curacao"
    assert normalize_team_name("Côte d’Ivoire") == "cote d ivoire"
    assert normalize_team_name("USA") == "united states"
    assert normalize_team_name("Bosnia & Herzegovina") == "bosnia and herzegovina"


def test_public_fixture_predictions_sum_to_roughly_100_and_use_trained_model_for_curacao() -> None:
    predictions = score_winner_matches(pd.read_csv("data/public_2026_match_features.csv"))
    totals = predictions["team_a_win_prob"] + predictions["draw_prob"] + predictions["team_b_win_prob"]
    assert totals.between(99.8, 100.2).all()
    assert (predictions["expected_goals_a"] > 0).all()
    assert (predictions["expected_goals_b"] > 0).all()

    curacao_rows = predictions[predictions["fixture"].str.contains("Cura", case=False, na=False)]
    assert not curacao_rows.empty
    assert not curacao_rows["model_family"].eq("Heuristic Poisson fallback").any()


def test_winner_schema_accepts_elo_rank_aliases() -> None:
    df = pd.read_csv("data/public_2026_match_features.csv").head(1).copy()
    df["elo_rank_a"] = df["fifa_rank_a"]
    df["elo_rank_b"] = df["fifa_rank_b"]
    df = df.drop(columns=["fifa_rank_a", "fifa_rank_b"])
    predictions = score_winner_matches(df)
    assert len(predictions) == 1
    assert "winner_pick" in predictions.columns


def test_scoreline_heatmap_has_probability_cells() -> None:
    row = score_winner_matches(pd.read_csv("data/public_2026_match_features.csv").head(1)).iloc[0]
    payload = json.loads(row["scoreline_grid_json"])
    assert payload
    heatmap = build_scoreline_heatmap(row)
    assert not heatmap.empty
    assert heatmap.max().max() > 0


def test_tournament_simulator_outputs_teams_and_odds() -> None:
    predictions = score_winner_matches(pd.read_csv("data/public_2026_match_features.csv"))
    sim = simulate_tournament(predictions, 25, 2026)
    assert len(sim) == 48
    assert round(sim["advance_odds"].sum(), 1) == 3200.0
    assert round(sim["title_odds"].sum(), 1) == 100.0


def test_completed_results_join_to_predictions() -> None:
    predictions = score_winner_matches(pd.read_csv("data/public_2026_match_features.csv"))
    comparison = build_prediction_result_comparison(predictions, load_public_2026_results())
    assert len(comparison) >= 4
    assert {"actual_score", "actual_outcome", "prediction_correct"}.issubset(comparison.columns)


def test_completed_results_join_handles_reversed_team_order() -> None:
    predictions = score_winner_matches(pd.read_csv("data/public_2026_match_features.csv")).head(1)
    row = predictions.iloc[0]
    reversed_result = pd.DataFrame([
        {
            "date": row["date"],
            "team_a": row["team_b"],
            "team_b": row["team_a"],
            "actual_score": "0-2",
            "actual_goals_a": 0,
            "actual_goals_b": 2,
            "actual_outcome": row["team_a"],
            "result_source": "test",
        }
    ])
    comparison = build_prediction_result_comparison(predictions, reversed_result)
    assert len(comparison) == 1
    assert comparison.iloc[0]["actual_score"] == "2-0"


def test_render_hero_calls_have_title_and_subtitle() -> None:
    tree = ast.parse(Path("world_cup_hub/apps.py").read_text())
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "render_hero"]
    assert calls
    assert all(len(call.args) == 2 for call in calls)
