from __future__ import annotations

import math
from typing import Iterable

import pandas as pd

REQUIRED_COLUMNS = [
    "player",
    "team",
    "opponent",
    "stage",
    "minutes",
    "goals",
    "assists",
    "key_passes",
    "successful_dribbles",
    "progressive_carries",
    "duels_won",
    "fouls_drawn",
    "recoveries",
    "tackles_won",
    "interceptions",
    "shots_on_target",
    "decisive_actions",
    "key_moment_minute",
    "winning_goal",
    "equalizer_goal",
    "motm",
    "crowd_reaction",
    "screen_time_share",
    "social_mentions",
    "sentiment",
    "celebration_rating",
    "underdog_flag",
    "pre_tournament_star_power",
]

STAGE_WEIGHTS = {
    "group": 1.00,
    "r16": 1.10,
    "round of 16": 1.10,
    "qf": 1.22,
    "quarterfinal": 1.22,
    "quarter-final": 1.22,
    "sf": 1.36,
    "semifinal": 1.36,
    "semi-final": 1.36,
    "final": 1.55,
}


def _validate_columns(df: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")


def _safe_norm(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce").fillna(0)
    minimum = numeric.min()
    maximum = numeric.max()
    if math.isclose(minimum, maximum):
        return pd.Series([0.5] * len(series), index=series.index)
    return (numeric - minimum) / (maximum - minimum)


def _sigmoid(value: float) -> float:
    return 1 / (1 + math.exp(-value))


def _stage_series(stages: Iterable[str]) -> pd.Series:
    return pd.Series(
        [STAGE_WEIGHTS.get(str(stage).strip().lower(), 1.0) for stage in stages]
    )


def score_matches(df: pd.DataFrame) -> pd.DataFrame:
    _validate_columns(df)
    scored = df.copy()

    stage_weight = _stage_series(scored["stage"])
    stage_norm = _safe_norm(stage_weight)
    minute_norm = _safe_norm(scored["key_moment_minute"])
    sentiment_scaled = scored["sentiment"].clip(-1, 1).add(1).div(2)

    impact = (
        0.24 * _safe_norm(scored["goals"])
        + 0.14 * _safe_norm(scored["assists"])
        + 0.10 * _safe_norm(scored["key_passes"])
        + 0.11 * _safe_norm(scored["successful_dribbles"])
        + 0.10 * _safe_norm(scored["progressive_carries"])
        + 0.08 * _safe_norm(scored["duels_won"])
        + 0.07 * _safe_norm(scored["recoveries"])
        + 0.08 * _safe_norm(scored["shots_on_target"])
        + 0.08 * _safe_norm(scored["decisive_actions"])
    )

    clutch = (
        0.30 * minute_norm
        + 0.25 * _safe_norm(scored["winning_goal"] + scored["equalizer_goal"])
        + 0.20 * stage_norm
        + 0.15 * _safe_norm(scored["goals"] + scored["assists"])
        + 0.10 * _safe_norm(scored["decisive_actions"])
    )

    crowd = (
        0.60 * _safe_norm(scored["crowd_reaction"])
        + 0.20 * _safe_norm(scored["fouls_drawn"])
        + 0.20 * _safe_norm(scored["successful_dribbles"])
    )

    broadcast = (
        0.70 * _safe_norm(scored["screen_time_share"])
        + 0.30 * _safe_norm(scored["pre_tournament_star_power"])
    )

    social = (
        0.55 * _safe_norm(scored["social_mentions"])
        + 0.25 * sentiment_scaled
        + 0.20 * _safe_norm(scored["motm"])
    )

    aesthetic = (
        0.65 * _safe_norm(scored["celebration_rating"])
        + 0.35 * _safe_norm(scored["successful_dribbles"])
    )

    raw_aura = (
        0.35 * impact
        + 0.20 * clutch
        + 0.15 * crowd
        + 0.15 * broadcast
        + 0.10 * social
        + 0.05 * aesthetic
    )

    aura_multiplier = 1 + 0.04 * scored["underdog_flag"].clip(lower=0, upper=1) + 0.03 * (stage_weight - 1)
    aura_score = (raw_aura * aura_multiplier * 100).clip(0, 100)

    scored["impact_score"] = (impact * 100).round(1)
    scored["clutch_score"] = (clutch * 100).round(1)
    scored["crowd_score"] = (crowd * 100).round(1)
    scored["broadcast_score"] = (broadcast * 100).round(1)
    scored["social_score"] = (social * 100).round(1)
    scored["aesthetic_score"] = (aesthetic * 100).round(1)
    scored["aura_score"] = aura_score.round(1)
    scored["main_character_probability"] = (
        aura_score.apply(lambda value: _sigmoid((value - 65) / 8) * 100).round(1)
    )
    return scored


def _archetype(row: pd.Series) -> str:
    if row["social_score"] > 78 and row["sentiment"] < 0.15:
        return "Villain aura"
    if row["clutch_score"] > 72 and row["impact_score"] > 68:
        return "Cold-blooded finisher"
    if row["aesthetic_score"] > 82 and row["crowd_score"] > 70:
        return "Main character energy"
    if row["crowd_score"] > 74 and row["successful_dribbles"] > 5:
        return "Chaos merchant"
    if row["broadcast_score"] > 78 and row["pre_tournament_star_power"] > 0.9:
        return "Captain aura"
    return "Silent killer"


def build_player_leaderboard(scored_matches: pd.DataFrame) -> pd.DataFrame:
    scored = scored_matches.copy()
    scored["archetype"] = scored.apply(_archetype, axis=1)

    leaderboard = (
        scored.groupby(["player", "team"], as_index=False)
        .agg(
            matches=("player", "size"),
            aura_score=("aura_score", "mean"),
            peak_aura=("aura_score", "max"),
            main_character_probability=("main_character_probability", "mean"),
            impact_score=("impact_score", "mean"),
            clutch_score=("clutch_score", "mean"),
            crowd_score=("crowd_score", "mean"),
            broadcast_score=("broadcast_score", "mean"),
            social_score=("social_score", "mean"),
            aesthetic_score=("aesthetic_score", "mean"),
            social_mentions=("social_mentions", "sum"),
            latest_archetype=("archetype", "last"),
        )
        .sort_values(["aura_score", "peak_aura"], ascending=False)
        .reset_index(drop=True)
    )

    score_columns = [
        "aura_score",
        "peak_aura",
        "main_character_probability",
        "impact_score",
        "clutch_score",
        "crowd_score",
        "broadcast_score",
        "social_score",
        "aesthetic_score",
    ]
    leaderboard[score_columns] = leaderboard[score_columns].round(1)
    return leaderboard
