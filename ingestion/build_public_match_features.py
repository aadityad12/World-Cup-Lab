from __future__ import annotations

import csv
import json
import math
import re
import sys
import urllib.request
from collections import defaultdict, deque
from datetime import date, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from world_cup_hub.normalization import normalize_team_name

DATA = ROOT / "data"
OUTPUT = DATA / "public_2026_match_features.csv"
GENERATED_DIR = DATA / "generated"
RAW_CACHE_DIR = DATA / "raw_cache"
SOURCE_NOTES = DATA / "public_2026_data_sources.md"

OPENFOOTBALL_WORLD_CUP_JSON = "https://raw.githubusercontent.com/openfootball/worldcup.json/master/2026/worldcup.json"
ELO_TEAMS_TSV = "https://www.eloratings.net/en.teams.tsv"
ELO_FIXTURES_TSV = "https://www.eloratings.net/fixtures.tsv"
ELO_LATEST_TSV = "https://www.eloratings.net/latest.tsv"
OPEN_METEO_FORECAST = "https://api.open-meteo.com/v1/forecast"

HEADERS = {"User-Agent": "WorldCupFunLab/0.1 educational project"}


def _cache_path_for_url(url: str) -> Path:
    safe = re.sub(r"[^a-zA-Z0-9]+", "_", url).strip("_")[:160]
    return RAW_CACHE_DIR / f"{safe}.txt"


def fetch_text(url: str, timeout: int = 25, retries: int = 2) -> str:
    RAW_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = _cache_path_for_url(url)
    last_error: Exception | None = None
    for _ in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=timeout) as response:
                text = response.read().decode("utf-8", "replace")
            cache_path.write_text(text, encoding="utf-8")
            return text
        except Exception as exc:
            last_error = exc
    if cache_path.exists():
        return cache_path.read_text(encoding="utf-8")
    raise RuntimeError(f"Could not fetch {url}: {last_error}")


def fetch_json(url: str) -> Any:
    return json.loads(fetch_text(url))


def read_venues() -> dict[str, dict[str, str]]:
    with (DATA / "host_venues_2026.csv").open(newline="") as handle:
        return {row["ground"]: row for row in csv.DictReader(handle)}


def normalize_name(value: str) -> str:
    return normalize_team_name(value)


def build_team_code_map() -> dict[str, str]:
    mapping: dict[str, str] = {}
    rows = [line.split("\t") for line in fetch_text(ELO_TEAMS_TSV).splitlines() if line.strip()]
    for fields in rows:
        code, *names = fields
        for name in names:
            mapping[normalize_name(name)] = code
    manual = {
        "usa": "US",
        "united states": "US",
        "south korea": "KR",
        "czech republic": "CZ",
        "czechia": "CZ",
        "turkey": "TR",
        "cote d ivoire": "CI",
        "ivory coast": "CI",
        "curacao": "CW",
        "bosnia and herzegovina": "BA",
        "haiti": "HT",
        "scotland": "SQ",
    }
    mapping.update(manual)
    return mapping


def parse_tsv_rows(text: str) -> list[list[str]]:
    return [line.split("\t") for line in text.splitlines() if line.strip()]


def build_elo_fixture_lookup() -> dict[tuple[str, str, str], dict[str, float | int | str]]:
    lookup = {}
    for fields in parse_tsv_rows(fetch_text(ELO_FIXTURES_TSV)):
        if len(fields) < 10:
            continue
        yyyy, mm, dd, code_a, code_b = fields[:5]
        match_date = f"{yyyy}-{int(mm):02d}-{int(dd):02d}"
        try:
            # World Football Elo fixtures.tsv fields:
            # year, month, day, team_a, team_b, tournament, venue_country,
            # rank_a, rank_b, elo_a, elo_b, followed by win-expectancy columns.
            lookup[(match_date, code_a, code_b)] = {
                "elo_rank_a": int(fields[7]),
                "elo_rank_b": int(fields[8]),
                "elo_a": int(fields[9]),
                "elo_b": int(fields[10]),
            }
        except ValueError:
            continue
    return lookup


def build_current_elo_map(latest_rows: list[list[str]] | None = None) -> dict[str, dict[str, int]]:
    current: dict[str, dict[str, int]] = {}
    for fields in latest_rows or parse_tsv_rows(fetch_text(ELO_LATEST_TSV)):
        if len(fields) < 16:
            continue
        code_a, code_b = fields[3], fields[4]
        try:
            rating_a, rating_b = int(fields[10]), int(fields[11])
            rank_a, rank_b = int(fields[14]), int(fields[15])
        except ValueError:
            continue
        current.setdefault(code_a, {"elo": rating_a, "rank": rank_a})
        current.setdefault(code_b, {"elo": rating_b, "rank": rank_b})
    return current


def build_recent_profiles(latest_rows: list[list[str]] | None = None) -> dict[str, dict[str, float]]:
    recent: dict[str, deque[dict[str, float]]] = defaultdict(lambda: deque(maxlen=10))
    latest = latest_rows or parse_tsv_rows(fetch_text(ELO_LATEST_TSV))
    for fields in latest:
        if len(fields) < 7:
            continue
        code_a, code_b = fields[3], fields[4]
        try:
            goals_a, goals_b = int(fields[5]), int(fields[6])
        except ValueError:
            continue
        points_a = 3 if goals_a > goals_b else 1 if goals_a == goals_b else 0
        points_b = 3 if goals_b > goals_a else 1 if goals_a == goals_b else 0
        recent[code_a].append({"points": points_a, "gf": goals_a, "ga": goals_b})
        recent[code_b].append({"points": points_b, "gf": goals_b, "ga": goals_a})

    profiles = {}
    for code, matches in recent.items():
        if not matches:
            continue
        n = len(matches)
        profiles[code] = {
            "recent_form": round(sum(m["points"] for m in matches) / (3 * n) * 100, 1),
            "goals_for_avg": round(sum(m["gf"] for m in matches) / n, 2),
            "goals_against_avg": round(sum(m["ga"] for m in matches) / n, 2),
        }
    return profiles


def clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    return radius * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def derived_team_features(code: str, elo: float, rank: float, profiles: dict[str, dict[str, float]]) -> dict[str, float]:
    profile = profiles.get(code, {"recent_form": 50.0, "goals_for_avg": 1.2, "goals_against_avg": 1.2})
    form = profile["recent_form"]
    gf = profile["goals_for_avg"]
    ga = profile["goals_against_avg"]
    attack = clip(60 + (elo - 1500) / 18 + gf * 7 + form / 12, 40, 98)
    defense = clip(62 + (elo - 1500) / 22 - ga * 5 + (100 - rank) / 25, 40, 98)
    midfield = clip(58 + (elo - 1500) / 20 + form / 18, 40, 98)
    keeper = clip(defense + (1.3 - ga) * 6, 40, 98)
    set_piece = clip(62 + gf * 4 + (100 - rank) / 35, 40, 95)
    experience = clip(55 + (elo - 1500) / 12 + (100 - rank) / 5, 35, 98)
    return {
        "recent_form": round(form, 1),
        "attack_rating": round(attack, 1),
        "defense_rating": round(defense, 1),
        "midfield_control": round(midfield, 1),
        "goalkeeper_form": round(keeper, 1),
        "set_piece_strength": round(set_piece, 1),
        "big_match_experience": round(experience, 1),
    }


def parse_kickoff_hour(time_value: str) -> int:
    match = re.search(r"(\d{1,2}):(\d{2})", time_value or "")
    return int(match.group(1)) if match else 12


def weather_for_match(lat: float, lon: float, match_date: str, kickoff_hour: int) -> dict[str, str | float]:
    url = (
        f"{OPEN_METEO_FORECAST}?latitude={lat}&longitude={lon}"
        "&hourly=temperature_2m,precipitation_probability,wind_speed_10m"
        f"&start_date={match_date}&end_date={match_date}&timezone=auto"
    )
    try:
        payload = fetch_json(url)
        hourly = payload.get("hourly", {})
        times = hourly.get("time", [])
        if not times:
            raise ValueError("No hourly weather returned")
        target_prefix = f"{match_date}T{kickoff_hour:02d}"
        index = next((i for i, value in enumerate(times) if value.startswith(target_prefix)), min(len(times) - 1, 12))
        return {
            "weather_temperature_c": hourly.get("temperature_2m", [None])[index],
            "weather_precipitation_probability": hourly.get("precipitation_probability", [None])[index],
            "weather_wind_kmh": hourly.get("wind_speed_10m", [None])[index],
            "weather_source": "Open-Meteo forecast API",
        }
    except Exception as exc:
        return {
            "weather_temperature_c": "",
            "weather_precipitation_probability": "",
            "weather_wind_kmh": "",
            "weather_source": f"Unavailable: {exc}",
        }


def rest_days_for(code: str, match_date: str, previous_dates: dict[str, date]) -> int:
    current = date.fromisoformat(match_date)
    previous = previous_dates.get(code)
    previous_dates[code] = current
    if not previous:
        return 5
    return max(2, min(10, (current - previous).days))


def host_boost(team: str, venue_country: str) -> int:
    normalized = normalize_name(team)
    if normalized == "mexico" and venue_country == "Mexico":
        return 10
    if normalized == "canada" and venue_country == "Canada":
        return 10
    if normalized in {"usa", "united states"} and venue_country == "United States":
        return 10
    return 0


def build_features() -> list[dict[str, Any]]:
    fixtures = fetch_json(OPENFOOTBALL_WORLD_CUP_JSON)["matches"]
    venues = read_venues()
    code_map = build_team_code_map()
    elo_lookup = build_elo_fixture_lookup()
    latest_rows = parse_tsv_rows(fetch_text(ELO_LATEST_TSV))
    current_elo = build_current_elo_map(latest_rows)
    profiles = build_recent_profiles(latest_rows)
    previous_dates: dict[str, date] = {}
    previous_locations: dict[str, tuple[float, float]] = {}
    weather_cache: dict[tuple[str, str, int], dict[str, str | float]] = {}
    rows = []

    for index, match in enumerate(fixtures, start=1):
        team_a = match.get("team1", "")
        team_b = match.get("team2", "")
        if not team_a or not team_b or team_a.startswith("Winner") or team_b.startswith("Winner"):
            continue

        match_date = match["date"]
        code_a = code_map.get(normalize_name(team_a), "")
        code_b = code_map.get(normalize_name(team_b), "")
        # Knockout slots like "1A" or "3A/B/C/D/F" are real fixture slots,
        # but not actual team matchups yet, so they are not useful for winner prediction.
        if not code_a or not code_b:
            continue
        elo_info = elo_lookup.get((match_date, code_a, code_b)) or elo_lookup.get((match_date, code_b, code_a))
        swapped = elo_lookup.get((match_date, code_b, code_a)) is not None and elo_lookup.get((match_date, code_a, code_b)) is None

        if elo_info and swapped:
            elo_a, elo_b = elo_info["elo_b"], elo_info["elo_a"]
            rank_a, rank_b = elo_info["elo_rank_b"], elo_info["elo_rank_a"]
        elif elo_info:
            elo_a, elo_b = elo_info["elo_a"], elo_info["elo_b"]
            rank_a, rank_b = elo_info["elo_rank_a"], elo_info["elo_rank_b"]
        else:
            fallback_a = current_elo.get(code_a, {"elo": 1700, "rank": 50})
            fallback_b = current_elo.get(code_b, {"elo": 1700, "rank": 50})
            elo_a, elo_b = fallback_a["elo"], fallback_b["elo"]
            rank_a, rank_b = fallback_a["rank"], fallback_b["rank"]

        feat_a = derived_team_features(code_a, float(elo_a), float(rank_a), profiles)
        feat_b = derived_team_features(code_b, float(elo_b), float(rank_b), profiles)
        ground = match.get("ground", "")
        venue = venues.get(ground, {})
        kickoff_hour = parse_kickoff_hour(match.get("time", ""))
        weather_key = (ground, match_date, kickoff_hour)
        if weather_key not in weather_cache and venue:
            weather_cache[weather_key] = weather_for_match(float(venue["latitude"]), float(venue["longitude"]), match_date, kickoff_hour)
        weather = weather_cache.get(weather_key, {})

        country = venue.get("country", "")
        current_location = None
        if venue.get("latitude") and venue.get("longitude"):
            current_location = (float(venue["latitude"]), float(venue["longitude"]))

        def team_travel_km(code: str) -> float:
            if not current_location or code not in previous_locations:
                return 0.0
            previous_lat, previous_lon = previous_locations[code]
            return round(haversine_km(previous_lat, previous_lon, current_location[0], current_location[1]), 1)

        travel_km_a = team_travel_km(code_a)
        travel_km_b = team_travel_km(code_b)

        row = {
            "match_id": f"WC2026-{index:03d}",
            "date": match_date,
            "kickoff_local": match.get("time", ""),
            "stage": match.get("group") or match.get("round", ""),
            "team_a": team_a,
            "team_b": team_b,
            "team_a_code": code_a,
            "team_b_code": code_b,
            "stadium": venue.get("stadium", ground),
            "city": venue.get("city", ground),
            "venue_country": country,
            "latitude": venue.get("latitude", ""),
            "longitude": venue.get("longitude", ""),
            "roof_notes": venue.get("roof_notes", ""),
            "elo_a": elo_a,
            "elo_b": elo_b,
            "fifa_rank_a": rank_a,
            "fifa_rank_b": rank_b,
            "elo_rank_a": rank_a,
            "elo_rank_b": rank_b,
            "recent_form_a": feat_a["recent_form"],
            "recent_form_b": feat_b["recent_form"],
            "attack_rating_a": feat_a["attack_rating"],
            "attack_rating_b": feat_b["attack_rating"],
            "defense_rating_a": feat_a["defense_rating"],
            "defense_rating_b": feat_b["defense_rating"],
            "midfield_control_a": feat_a["midfield_control"],
            "midfield_control_b": feat_b["midfield_control"],
            "goalkeeper_form_a": feat_a["goalkeeper_form"],
            "goalkeeper_form_b": feat_b["goalkeeper_form"],
            "set_piece_strength_a": feat_a["set_piece_strength"],
            "set_piece_strength_b": feat_b["set_piece_strength"],
            "injuries_impact_a": 0,
            "injuries_impact_b": 0,
            "rest_days_a": rest_days_for(code_a, match_date, previous_dates),
            "rest_days_b": rest_days_for(code_b, match_date, previous_dates),
            "travel_fatigue_a": round((0 if host_boost(team_a, country) else 10) + travel_km_a / 350, 1),
            "travel_fatigue_b": round((0 if host_boost(team_b, country) else 10) + travel_km_b / 350, 1),
            "travel_km_a": travel_km_a,
            "travel_km_b": travel_km_b,
            "home_region_boost_a": host_boost(team_a, country),
            "home_region_boost_b": host_boost(team_b, country),
            "big_match_experience_a": feat_a["big_match_experience"],
            "big_match_experience_b": feat_b["big_match_experience"],
            "weather_temperature_c": weather.get("weather_temperature_c", ""),
            "weather_precipitation_probability": weather.get("weather_precipitation_probability", ""),
            "weather_wind_kmh": weather.get("weather_wind_kmh", ""),
            "fixture_source": "openfootball/worldcup.json public repository",
            "rating_source": "World Football Elo Ratings public TSV",
            "venue_source": "FIFA 2026 host city/stadium public reference mapping",
            "weather_source": weather.get("weather_source", "Open-Meteo forecast API"),
            "unavailable_data_notes": "Player injuries/status/suspensions and confirmed lineups not available from no-key public sources; injuries_impact is set to 0 until a provider is connected.",
        }
        rows.append(row)
        if current_location:
            previous_locations[code_a] = current_location
            previous_locations[code_b] = current_location
    return rows


def main() -> None:
    rows = build_features()
    if not rows:
        raise SystemExit("No fixture rows produced")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    snapshot_output = GENERATED_DIR / f"public_2026_match_features_{generated_at}.csv"
    for output_path in (OUTPUT, snapshot_output):
        with output_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    (GENERATED_DIR / "latest_manifest.json").write_text(
        json.dumps(
            {
                "generated_at_utc": generated_at,
                "latest_csv": str(OUTPUT.relative_to(ROOT)),
                "snapshot_csv": str(snapshot_output.relative_to(ROOT)),
                "row_count": len(rows),
                "fixture_source": OPENFOOTBALL_WORLD_CUP_JSON,
                "rating_sources": [ELO_FIXTURES_TSV, ELO_LATEST_TSV, ELO_TEAMS_TSV],
                "weather_source": OPEN_METEO_FORECAST,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    SOURCE_NOTES.write_text(
        "# Public 2026 Match Feature Data Sources\n\n"
        "Generated by `ingestion/build_public_match_features.py`.\n\n"
        "## Connected public/no-key sources\n"
        f"- Fixtures: `{OPENFOOTBALL_WORLD_CUP_JSON}`. This is a public community dataset, not an official FIFA API.\n"
        f"- Team ratings / Elo ranks / fixtures cross-check: `{ELO_FIXTURES_TSV}`, `{ELO_LATEST_TSV}`, `{ELO_TEAMS_TSV}`. Columns named `fifa_rank_*` are kept for backwards compatibility; `elo_rank_*` is the clearer alias.\n"
        "- Venues/stadium coordinates: `data/host_venues_2026.csv` static host venue reference.\n"
        "- Weather: Open-Meteo forecast API at stadium coordinates and kickoff date/hour.\n\n"
        "## Important limitations\n"
        "- Official FIFA fixtures API was not connected; if you have an official schedule feed, use it instead.\n"
        "- Player injuries, player status, suspensions, and expected lineups were not available from reliable no-key public APIs.\n"
        "- Attack/defense/midfield/set-piece features are derived proxies from Elo + recent results, not provider-grade event data.\n"
        "- Travel fatigue includes rough venue-to-venue great-circle distance once a team has already played in the tournament.\n"
        "- Weather availability depends on Open-Meteo coverage for the requested dates.\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(rows)} rows to {OUTPUT}")
    print(f"Wrote snapshot to {snapshot_output}")
    print(f"Wrote source notes to {SOURCE_NOTES}")


if __name__ == "__main__":
    main()
