from __future__ import annotations

import json
import math
import sys
import urllib.request
from collections import defaultdict
from io import StringIO
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from world_cup_hub.normalization import normalize_team_name

ARTIFACT_DIR = ROOT / "models" / "artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "team_strength_model.json"
RESULTS_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
HEADERS = {"User-Agent": "WorldCupFunLab/0.2 educational model training"}
HALF_LIFE_DAYS = 365 * 3.0
PROFILE_SHRINKAGE_MATCHES = 18.0
MAX_GOALS = 8


def fetch_text(url: str, timeout: int = 30) -> str:
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return response.read().decode("utf-8", "replace")


def load_results() -> pd.DataFrame:
    text = fetch_text(RESULTS_URL)
    df = pd.read_csv(StringIO(text))
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "home_team", "away_team", "home_score", "away_score"])
    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")
    df = df.dropna(subset=["home_score", "away_score"])
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    df["neutral"] = df["neutral"].astype(str).str.upper().eq("TRUE")
    df["home_norm"] = df["home_team"].map(normalize_team_name)
    df["away_norm"] = df["away_team"].map(normalize_team_name)
    # Modern football only: older eras have very different scoring environments.
    return df[df["date"] >= "2004-01-01"].sort_values("date").reset_index(drop=True)


def _weight(match_date: pd.Timestamp, as_of: pd.Timestamp, tournament: str) -> float:
    age_days = max(0, (as_of - match_date).days)
    recency = math.exp(-math.log(2) * age_days / HALF_LIFE_DAYS)
    tournament_lower = str(tournament).lower()
    competitive = 0.82 if "friendly" in tournament_lower else 1.15
    if "world cup" in tournament_lower:
        competitive *= 1.12
    return recency * competitive


def build_profiles(df: pd.DataFrame, as_of: pd.Timestamp) -> dict[str, Any]:
    totals: dict[str, dict[str, float]] = defaultdict(lambda: {"w": 0.0, "gf": 0.0, "ga": 0.0, "pts": 0.0, "matches": 0.0})
    total_goal_weight = 0.0
    total_appearance_weight = 0.0
    home_goal_w = 0.0
    away_goal_w = 0.0
    non_neutral_w = 0.0

    display_names: dict[str, str] = {}
    for row in df.itertuples(index=False):
        w = _weight(row.date, as_of, row.tournament)
        h, a = row.home_norm, row.away_norm
        display_names.setdefault(h, row.home_team)
        display_names.setdefault(a, row.away_team)
        hs, aw = int(row.home_score), int(row.away_score)
        hp = 3 if hs > aw else 1 if hs == aw else 0
        ap = 3 if aw > hs else 1 if hs == aw else 0
        totals[h]["w"] += w
        totals[h]["gf"] += w * hs
        totals[h]["ga"] += w * aw
        totals[h]["pts"] += w * hp
        totals[h]["matches"] += 1
        totals[a]["w"] += w
        totals[a]["gf"] += w * aw
        totals[a]["ga"] += w * hs
        totals[a]["pts"] += w * ap
        totals[a]["matches"] += 1
        total_goal_weight += w * (hs + aw)
        total_appearance_weight += 2 * w
        if not bool(row.neutral):
            home_goal_w += w * hs
            away_goal_w += w * aw
            non_neutral_w += w

    global_avg = total_goal_weight / max(1e-9, total_appearance_weight)
    raw_home_avg = home_goal_w / max(1e-9, non_neutral_w)
    raw_away_avg = away_goal_w / max(1e-9, non_neutral_w)
    home_advantage = math.sqrt(max(0.7, min(1.35, raw_home_avg / max(0.1, raw_away_avg))))

    profiles: dict[str, dict[str, float | str]] = {}
    for team, stats in totals.items():
        if stats["matches"] < 4:
            continue
        w = stats["w"]
        gf_avg = stats["gf"] / max(1e-9, w)
        ga_avg = stats["ga"] / max(1e-9, w)
        pts_avg = stats["pts"] / max(1e-9, 3 * w)
        n_eff = min(w, 60.0)
        attack_raw = gf_avg / max(0.1, global_avg)
        defense_raw = ga_avg / max(0.1, global_avg)  # higher means concedes more
        attack = (attack_raw * n_eff + PROFILE_SHRINKAGE_MATCHES) / (n_eff + PROFILE_SHRINKAGE_MATCHES)
        defense = (defense_raw * n_eff + PROFILE_SHRINKAGE_MATCHES) / (n_eff + PROFILE_SHRINKAGE_MATCHES)
        profiles[team] = {
            "display_name": display_names.get(team, team.title()),
            "weighted_matches": round(w, 2),
            "raw_matches": int(stats["matches"]),
            "attack_factor": round(max(0.45, min(1.85, attack)), 4),
            "defense_concede_factor": round(max(0.45, min(1.85, defense)), 4),
            "weighted_goals_for_per_match": round(gf_avg, 3),
            "weighted_goals_against_per_match": round(ga_avg, 3),
            "weighted_points_rate": round(pts_avg, 4),
        }

    return {
        "global_avg_goals_per_team": round(global_avg, 5),
        "home_advantage_multiplier": round(home_advantage, 5),
        "team_profiles": profiles,
    }


def poisson_pmf(k: int, lam: float) -> float:
    return math.exp(-lam) * (lam ** k) / math.factorial(k)


def dixon_coles_factor(home_goals: int, away_goals: int, lam_h: float, lam_a: float, rho: float) -> float:
    if home_goals == 0 and away_goals == 0:
        return max(0.05, 1 - lam_h * lam_a * rho)
    if home_goals == 0 and away_goals == 1:
        return max(0.05, 1 + lam_h * rho)
    if home_goals == 1 and away_goals == 0:
        return max(0.05, 1 + lam_a * rho)
    if home_goals == 1 and away_goals == 1:
        return max(0.05, 1 - rho)
    return 1.0


def score_matrix(lam_h: float, lam_a: float, rho: float) -> list[tuple[int, int, float]]:
    cells = []
    total = 0.0
    for h in range(MAX_GOALS + 1):
        for a in range(MAX_GOALS + 1):
            p = poisson_pmf(h, lam_h) * poisson_pmf(a, lam_a) * dixon_coles_factor(h, a, lam_h, lam_a, rho)
            cells.append((h, a, p))
            total += p
    return [(h, a, p / max(total, 1e-12)) for h, a, p in cells]


def predict_lambdas(row: Any, profiles: dict[str, Any], home_advantage: float, global_avg: float) -> tuple[float, float]:
    home = profiles.get(row.home_norm)
    away = profiles.get(row.away_norm)
    if not home or not away:
        return global_avg, global_avg
    lam_h = global_avg * float(home["attack_factor"]) * float(away["defense_concede_factor"])
    lam_a = global_avg * float(away["attack_factor"]) * float(home["defense_concede_factor"])
    if not bool(row.neutral):
        lam_h *= home_advantage
        lam_a /= home_advantage
    return max(0.15, min(5.0, lam_h)), max(0.15, min(5.0, lam_a))


def outcome_probs(lam_h: float, lam_a: float, rho: float) -> dict[str, float]:
    hwin = draw = awin = 0.0
    for h, a, p in score_matrix(lam_h, lam_a, rho):
        if h > a:
            hwin += p
        elif h == a:
            draw += p
        else:
            awin += p
    return {"home_win": hwin, "draw": draw, "away_win": awin}


def fit_rho(validation: pd.DataFrame, profiles_blob: dict[str, Any]) -> float:
    profiles = profiles_blob["team_profiles"]
    home_advantage = float(profiles_blob["home_advantage_multiplier"])
    global_avg = float(profiles_blob["global_avg_goals_per_team"])
    best_rho = -0.05
    best_loss = float("inf")
    for rho_int in range(-15, 11):
        rho = rho_int / 100.0
        loss = 0.0
        count = 0
        for row in validation.itertuples(index=False):
            lam_h, lam_a = predict_lambdas(row, profiles, home_advantage, global_avg)
            actual_h = min(MAX_GOALS, int(row.home_score))
            actual_a = min(MAX_GOALS, int(row.away_score))
            prob = next(p for h, a, p in score_matrix(lam_h, lam_a, rho) if h == actual_h and a == actual_a)
            loss += -math.log(max(prob, 1e-12))
            count += 1
        if count and loss / count < best_loss:
            best_loss = loss / count
            best_rho = rho
    return best_rho


def evaluate(test: pd.DataFrame, profiles_blob: dict[str, Any], rho: float) -> dict[str, Any]:
    profiles = profiles_blob["team_profiles"]
    home_advantage = float(profiles_blob["home_advantage_multiplier"])
    global_avg = float(profiles_blob["global_avg_goals_per_team"])
    log_loss = 0.0
    brier = 0.0
    correct = 0
    exact = 0
    goal_abs_error = 0.0
    rows = 0
    buckets: dict[str, dict[str, float]] = defaultdict(lambda: {"n": 0, "correct": 0, "confidence_sum": 0.0})

    for row in test.itertuples(index=False):
        lam_h, lam_a = predict_lambdas(row, profiles, home_advantage, global_avg)
        probs = outcome_probs(lam_h, lam_a, rho)
        actual = "home_win" if row.home_score > row.away_score else "draw" if row.home_score == row.away_score else "away_win"
        pred = max(probs, key=probs.get)
        p_actual = max(probs[actual], 1e-12)
        y = {"home_win": 0.0, "draw": 0.0, "away_win": 0.0}
        y[actual] = 1.0
        log_loss += -math.log(p_actual)
        brier += sum((probs[k] - y[k]) ** 2 for k in y) / 3.0
        correct += int(pred == actual)
        top_score = max(score_matrix(lam_h, lam_a, rho), key=lambda item: item[2])
        exact += int(top_score[0] == min(MAX_GOALS, int(row.home_score)) and top_score[1] == min(MAX_GOALS, int(row.away_score)))
        goal_abs_error += (abs(lam_h - row.home_score) + abs(lam_a - row.away_score)) / 2.0
        conf = probs[pred]
        lower = int(conf * 10) * 10
        label = f"{lower}-{lower + 10}%"
        buckets[label]["n"] += 1
        buckets[label]["correct"] += int(pred == actual)
        buckets[label]["confidence_sum"] += conf
        rows += 1

    calibration = []
    for label, stats in sorted(buckets.items()):
        n = int(stats["n"])
        calibration.append({
            "bucket": label,
            "matches": n,
            "avg_confidence": round(stats["confidence_sum"] / n * 100, 2),
            "actual_accuracy": round(stats["correct"] / n * 100, 2),
        })

    return {
        "test_matches": rows,
        "accuracy": round(correct / max(1, rows), 4),
        "log_loss": round(log_loss / max(1, rows), 4),
        "brier_score": round(brier / max(1, rows), 4),
        "exact_score_accuracy": round(exact / max(1, rows), 4),
        "mean_goal_absolute_error": round(goal_abs_error / max(1, rows), 4),
        "calibration": calibration,
    }


def main() -> None:
    df = load_results()
    cutoff = max(pd.Timestamp("2023-01-01"), df["date"].quantile(0.82))
    train = df[df["date"] < cutoff].copy()
    holdout = df[df["date"] >= cutoff].copy()
    validation_cutoff = train["date"].quantile(0.88)
    train_core = train[train["date"] < validation_cutoff].copy()
    validation = train[train["date"] >= validation_cutoff].copy()

    core_blob = build_profiles(train_core, train_core["date"].max())
    rho = fit_rho(validation, core_blob)
    final_blob = build_profiles(train, train["date"].max())
    metrics = evaluate(holdout, final_blob, rho)

    artifact = {
        "model_name": "recency_weighted_international_poisson_dixon_coles",
        "model_version": "0.1.0",
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_url": RESULTS_URL,
        "training_rows": int(len(train)),
        "holdout_rows": int(len(holdout)),
        "first_training_date": str(train["date"].min().date()),
        "last_training_date": str(train["date"].max().date()),
        "holdout_start_date": str(holdout["date"].min().date()) if len(holdout) else None,
        "half_life_days": HALF_LIFE_DAYS,
        "max_goals_modeled": MAX_GOALS,
        "dixon_coles_rho": rho,
        **final_blob,
        "metrics": metrics,
        "limitations": [
            "Uses public historical international results only; it does not include paid xG/event feeds.",
            "Team strengths are aggregate recency-weighted factors, not player-level lineup ratings.",
            "Injuries, suspensions, and lineups are supported in the app schema but require an external provider or manual CSV.",
        ],
    }
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_PATH.write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote {ARTIFACT_PATH}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
