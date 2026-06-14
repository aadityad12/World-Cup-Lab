# World Cup Fun Lab

A Streamlit mini-site full of playful World Cup analytics apps.

## Included apps

- **Match Predictor** — flagship section that predicts match winners and scores, then shows the factors behind every pick
- **Tournament Simulator** — Monte Carlo group qualification and approximate title-odds simulator powered by Match Predictor probabilities
- **Aura Lab** — scores player aura from match impact, crowd reaction, broadcast attention, and social buzz
- **Chaos Center** — ranks matches by cards, late goals, penalties, VAR drama, and overall nonsense
- **Underdog Radar** — estimates upset danger for favorites
- **Fan Pain Lab** — simulates the emotional suffering of a fanbase

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 training/train_team_strength_model.py
python3 ingestion/build_public_match_features.py
python3 -m pytest -q
streamlit run app.py --server.port 8503
```

## Match Predictor data files

- `data/public_2026_match_features.csv` — generated public-data fixture/features file used by the app
- `data/public_2026_results.csv` — completed-match final scores from the public fixture source for prediction-vs-result audits
- `data/public_2026_data_sources.md` — exact source notes and limitations
- `data/host_venues_2026.csv` — host stadium/city/coordinate reference
- `data/sample_fake_2026_match_features.csv` — old fake/demo fixture features, kept only as an explicitly labeled sample

## Other sample data

- `data/sample_player_matches.csv`
- `data/sample_match_chaos.csv`
- `data/sample_underdog_scenarios.csv`

## Project structure

- `app.py` — main site entrypoint and navigation
- `world_cup_hub/apps.py` — mini-app pages
- `world_cup_hub/data.py` — data loading + scoring helpers
- `world_cup_hub/normalization.py` — shared team-name canonicalization used by training, ingestion, and prediction
- `world_cup_hub/components.py` — shared UI components/styles
- `aura_model/scoring.py` — aura scoring logic
- `ingestion/build_public_match_features.py` — public data ingestion/generation script
- `training/train_team_strength_model.py` — trains the recency-weighted Poisson/Dixon-Coles match model from public historical international results
- `models/artifacts/team_strength_model.json` — trained model artifact + evaluation metrics

## Connected public/no-key sources

- Fixtures: `openfootball/worldcup.json` public repository
- Elo ratings/ranks/recent results: World Football Elo public TSV files
- Stadiums/cities/coordinates: local 2026 host venue reference CSV
- Weather: Open-Meteo forecast API by stadium coordinate and kickoff date/hour

## Match Predictor factors

- Trained recency-weighted team attack/defense factors from public historical international results
- Poisson/Dixon-Coles exact score probabilities
- Draw predictions for group-stage near-tossups
- Knockout advancement, extra-time, and penalty-shootout estimates when knockout fixtures are provided
- Elo strength
- Team ranking / Elo rank (`elo_rank_*` preferred; legacy `fifa_rank_*` still accepted)
- Recent form
- Attack vs opponent defense proxy
- Midfield control proxy
- Goalkeeper form proxy
- Set-piece strength proxy
- Injury impact placeholder
- Rest/travel load, including rough venue-to-venue travel distance where available
- Regional/host boost
- Big-match experience proxy
- Weather goal suppression
- Optional expected lineup strength, injury impact, and suspension impact columns for richer uploaded/provider CSVs
- Exact-score probability grid for heatmap display
- Completed-match prediction audit that shows predicted winner/score beside the real result when available
- Model evaluation metrics: accuracy, log loss, Brier score, exact-score accuracy, goal MAE, and calibration buckets

## Known limitations

- The fixture source is public/community data, not an official FIFA API feed.
- Player injuries/status, suspensions, and expected lineups are not connected yet.
- Knockout title odds in the Tournament Simulator use an approximate Elo-based bracket until official post-group pairings are known.
- Attack/defense/midfield/set-piece metrics are derived proxies from public Elo/recent-result data, not provider-grade event data.
- For production-grade accuracy, connect a paid provider such as Sportradar, SportMonks, Opta/Stats Perform, or API-Football.
