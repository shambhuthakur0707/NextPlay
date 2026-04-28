# -*- coding: utf-8 -*-
"""
NextPlay — NBA AI Score Predictor Dashboard
=============================================
Streamlit dashboard for the current NextPlay prediction system.
Run with: streamlit run app.py
"""
import os
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from nba_api.stats.endpoints import scoreboardv3

from config import (
    DATA_DIR, MODEL_READY_PATH, SHOT_PROFILES_PATH,
    PLAYER_IMPACT_PATH, BACKTEST_RESULTS_PATH, ELO_RATINGS_PATH,
    NBA_TEAMS, FEATURE_COLS_FINAL,
    USE_PLAYOFF_MODELS,
    PLAYOFF_SEASON_START_MONTH, PLAYOFF_SEASON_END_MONTH,
)
from ingestion.odds import pull_closing_market_lines
from models.train import load_models, load_playoff_models
from prediction.predict import predict_game
from utils.metadata import (
    feature_category_counts,
    project_metadata_snapshot,
)


NBA_TZ = ZoneInfo("America/New_York")


def nba_today_date():
    """Return current NBA calendar date in US Eastern timezone."""
    return datetime.now(NBA_TZ).date()

# ════════════════════════════════════════════════════════════
# PAGE CONFIG
# ════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="NextPlay — NBA AI Predictor",
    page_icon="🏀",
    layout="wide",
)

# ════════════════════════════════════════════════════════════
# DATA LOADING
# ════════════════════════════════════════════════════════════

@st.cache_data
def load_data():
    """Load all required data files."""
    data = {}

    if os.path.exists(MODEL_READY_PATH):
        data["model_df"] = pd.read_csv(MODEL_READY_PATH)
        data["model_df"]["GAME_DATE"] = pd.to_datetime(
            data["model_df"]["GAME_DATE"])
    else:
        data["model_df"] = None

    if os.path.exists(SHOT_PROFILES_PATH):
        data["shot_df"] = pd.read_csv(SHOT_PROFILES_PATH)
    else:
        data["shot_df"] = None

    if os.path.exists(PLAYER_IMPACT_PATH):
        data["player_impact_df"] = pd.read_csv(PLAYER_IMPACT_PATH)
    else:
        data["player_impact_df"] = None

    if os.path.exists(BACKTEST_RESULTS_PATH):
        data["backtest"] = pd.read_csv(BACKTEST_RESULTS_PATH)
        if "game_date" in data["backtest"].columns:
            data["backtest"]["game_date"] = pd.to_datetime(
                data["backtest"]["game_date"])
    else:
        data["backtest"] = None

    if os.path.exists(ELO_RATINGS_PATH):
        data["elo_df"] = pd.read_csv(ELO_RATINGS_PATH).sort_values(
            "elo", ascending=False).reset_index(drop=True)
    else:
        data["elo_df"] = None

    return data


@st.cache_resource
def load_cached_models():
    """Load trained models (cached as resource)."""
    try:
        regular_models = load_models()
        playoff_models = None
        if USE_PLAYOFF_MODELS:
            try:
                playoff_models = load_playoff_models()
            except FileNotFoundError:
                playoff_models = None
        return {
            "regular": regular_models,
            "playoff": playoff_models,
        }
    except FileNotFoundError:
        return None


def is_playoff_season(now=None):
    """Return True during the playoff window configured in config.py."""
    now = now or datetime.now()
    month = now.month
    if PLAYOFF_SEASON_START_MONTH <= PLAYOFF_SEASON_END_MONTH:
        return PLAYOFF_SEASON_START_MONTH <= month <= PLAYOFF_SEASON_END_MONTH
    return month >= PLAYOFF_SEASON_START_MONTH or month <= PLAYOFF_SEASON_END_MONTH


def active_model_bundle(now=None):
    """Return the currently active model bundle with fallback handling."""
    if not models:
        return None, "none"

    if USE_PLAYOFF_MODELS and is_playoff_season(now) and models.get("playoff") is not None:
        return models["playoff"], "playoff"

    return models["regular"], "regular"


def _suggest_player_names(player_impact_df, team_abbr, raw_text, latest_season=None, limit=5):
    """Return likely player names for a team's injury text input."""
    if player_impact_df is None or not raw_text:
        return []

    frame = player_impact_df.copy()
    if latest_season is not None and "SEASON" in frame.columns:
        frame = frame[frame["SEASON"].astype(str) == str(latest_season)]

    if "TEAM_ABBR" in frame.columns:
        frame = frame[frame["TEAM_ABBR"] == team_abbr]

    if len(frame) == 0 or "PLAYER_NAME" not in frame.columns:
        return []

    parts = [part.strip() for part in raw_text.split(",") if part.strip()]
    if not parts:
        return []

    suggestions = []
    for part in parts:
        matches = frame[frame["PLAYER_NAME"].astype(str).str.contains(part, case=False, na=False)]
        suggestions.extend(matches["PLAYER_NAME"].astype(str).head(limit).tolist())

    seen = set()
    ordered = []
    for name in suggestions:
        if name not in seen:
            ordered.append(name)
            seen.add(name)
    return ordered[:limit]


@st.cache_data(ttl=600)
def get_games_for_date(game_date):
    """Return scheduled NBA games for a given date."""
    game_date = pd.Timestamp(game_date).strftime("%Y-%m-%d")

    try:
        sb = scoreboardv3.ScoreboardV3(game_date=game_date, league_id="00")
        frames = sb.get_data_frames()
        if len(frames) < 3:
            return []

        game_header = frames[1].copy()
        team_lines = frames[2][["gameId", "teamId", "teamTricode"]].copy()
        game_header["gameTimeUTC"] = pd.to_datetime(game_header["gameTimeUTC"], utc=True)

        selected_games = game_header.copy()
        if game_date == nba_today_date().strftime("%Y-%m-%d"):
            now_utc = pd.Timestamp.now(tz=timezone.utc)
            selected_games = selected_games[selected_games["gameTimeUTC"] > now_utc]

        schedule = []
        for _, row in selected_games.iterrows():
            game_teams = team_lines[team_lines["gameId"] == row["gameId"]].reset_index(drop=True)
            if len(game_teams) < 2:
                continue

            schedule.append({
                "game_id": row["gameId"],
                "start_time": row.get("gameEt") or row.get("gameTimeUTC"),
                "home_team": game_teams.iloc[0]["teamTricode"],
                "away_team": game_teams.iloc[1]["teamTricode"],
            })

        return schedule
    except Exception as exc:
        st.warning(f"Could not load games for {game_date}: {exc}")
        return []


data = load_data()
models = load_cached_models()
metadata = project_metadata_snapshot()

# ════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════

st.sidebar.title("🏀 NextPlay")
st.sidebar.caption(f"NBA AI Score Predictor — {metadata['model_version']}")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["📅 Today’s Games", "🏀 Game Predictor", "📊 Model Dashboard",
     "🏆 Team Rankings", "ℹ️ About"],
)

if data["model_df"] is not None:
    active_label = "Playoff" if is_playoff_season() and USE_PLAYOFF_MODELS else "Regular Season"
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        f"**Dataset:** {len(data['model_df']):,} games  \n"
        f"**Features:** {metadata['base_feature_count']}  \n"
        f"**Model Stack:** Home + Away + Stacked Total ({metadata['model_version']})  \n"
        f"**Active Model:** {active_label}"
    )

# ════════════════════════════════════════════════════════════
# PAGE 1 — TODAY'S GAMES
# ════════════════════════════════════════════════════════════

if page == "📅 Today’s Games":
    st.title("📅 Today's NBA Games")
    st.markdown("Browse the slate for any date and see the matchup list for that day.")

    selected_date = st.date_input(
        "Select game date",
        value=nba_today_date(),
        help="Defaults to today's slate.",
    )

    games = get_games_for_date(selected_date)
    st.caption(f"{len(games)} game(s) found for {pd.Timestamp(selected_date).strftime('%Y-%m-%d')}")

    if len(games) == 0:
        st.info("No games found for this date.")
    else:
        market_lines = None
        try:
            market_lines = pull_closing_market_lines(date_from=selected_date, date_to=selected_date)
        except Exception:
            market_lines = None

        for idx, game in enumerate(games, start=1):
            home_team = game["home_team"]
            away_team = game["away_team"]
            market_total_line = None
            if market_lines is not None and len(market_lines) > 0:
                match = market_lines[
                    (market_lines["HOME_TEAM"] == home_team) &
                    (market_lines["AWAY_TEAM"] == away_team)
                ]
                if len(match) > 0:
                    market_total_line = match.iloc[0].get("CLOSE_TOTAL")

            with st.container():
                left, right = st.columns([3, 1])
                with left:
                    st.subheader(f"{idx}. {away_team} at {home_team}")
                    st.write(f"Start time: {game['start_time']}")
                    if market_total_line is not None and pd.notna(market_total_line):
                        st.write(f"Sportsbook total: {float(market_total_line):.1f}")
                with right:
                    active_models, _ = active_model_bundle(pd.Timestamp(selected_date))
                    if active_models is not None and data["model_df"] is not None:
                        try:
                            feature_cols = list(active_models["model_C"].feature_names_in_)
                        except Exception:
                            feature_cols = [c for c in FEATURE_COLS_FINAL if c in data["model_df"].columns]

                        result = predict_game(
                            home_team, away_team,
                            data["model_df"], active_models,
                            shot_df=data.get("shot_df"),
                            player_impact_df=data.get("player_impact_df"),
                            feature_cols=feature_cols,
                            game_date=pd.Timestamp(selected_date),
                            market_total_line=market_total_line,
                            verbose=False,
                        )
                        if result is not None:
                            st.metric("Model total", f"{result['pred_total']} pts")
                            st.write(f"{home_team} {result['pred_home']} - {result['pred_away']} {away_team}")
                            st.caption(f"{result['confidence']} | Win prob: {result['win_prob']:.1f}%")


# ════════════════════════════════════════════════════════════
# PAGE 2 — GAME PREDICTOR
# ════════════════════════════════════════════════════════════

elif page == "🏀 Game Predictor":
    st.title("🏀 NBA Game Score Predictor")
    st.markdown(
        "Select two teams to get an AI-powered score prediction "
        "with injury adjustments."
    )

    if models is None or data["model_df"] is None:
        st.error(
            "⚠️ Models or data not found. Please copy your data files "
            "to the `data/` directory and restart."
        )
        st.stop()

    active_models, active_name = active_model_bundle()

    col1, col2 = st.columns(2)
    with col1:
        home_idx = NBA_TEAMS.index("BOS") if "BOS" in NBA_TEAMS else 0
        home_team = st.selectbox("🏠 Home Team", NBA_TEAMS, index=home_idx)
    with col2:
        away_idx = NBA_TEAMS.index("LAL") if "LAL" in NBA_TEAMS else 1
        away_team = st.selectbox("✈️ Away Team", NBA_TEAMS, index=away_idx)

    if home_team == away_team:
        st.warning("⚠️ Please select two different teams!")
        st.stop()

    # Optional injury inputs
    with st.expander("🏥 Flag Injured Players (Optional)"):
        col_inj1, col_inj2 = st.columns(2)
        with col_inj1:
            home_inj_str = st.text_input(
                f"{home_team} players OUT (comma-separated)",
                placeholder="e.g. Tatum, Brown")
            if home_inj_str:
                latest_season = None
                if data.get("model_df") is not None and "SEASON" in data["model_df"].columns:
                    latest_season = data["model_df"]["SEASON"].dropna().astype(str).max()
                home_suggestions = _suggest_player_names(
                    data.get("player_impact_df"), home_team, home_inj_str,
                    latest_season=latest_season,
                )
                if home_suggestions:
                    st.caption("Matches: " + ", ".join(home_suggestions))
        with col_inj2:
            away_inj_str = st.text_input(
                f"{away_team} players OUT (comma-separated)",
                placeholder="e.g. LeBron")
            if away_inj_str:
                latest_season = None
                if data.get("model_df") is not None and "SEASON" in data["model_df"].columns:
                    latest_season = data["model_df"]["SEASON"].dropna().astype(str).max()
                away_suggestions = _suggest_player_names(
                    data.get("player_impact_df"), away_team, away_inj_str,
                    latest_season=latest_season,
                )
                if away_suggestions:
                    st.caption("Matches: " + ", ".join(away_suggestions))

    home_out = [x.strip() for x in home_inj_str.split(",") if x.strip()] \
        if home_inj_str else []
    away_out = [x.strip() for x in away_inj_str.split(",") if x.strip()] \
        if away_inj_str else []

    if st.button("🔮 Predict Matchup", type="primary",
                 use_container_width=True):
        with st.spinner("Analyzing matchup..."):
            feature_cols = None
            try:
                feature_cols = list(active_models["model_C"].feature_names_in_)
            except Exception:
                feature_cols = [c for c in FEATURE_COLS_FINAL
                                if c in data["model_df"].columns]

            result = predict_game(
                home_team, away_team,
                data["model_df"], active_models,
                shot_df=data.get("shot_df"),
                player_impact_df=data.get("player_impact_df"),
                feature_cols=feature_cols,
                home_out=home_out, away_out=away_out,
                verbose=False,
            )

        if result is None:
            st.error("❌ Could not generate prediction. Check team data.")
            st.stop()

        # Prediction banner
        winner = home_team if result["win_prob"] > 50 else away_team
        color = "#2A9D8F" if result["win_prob"] > 50 else "#E63946"

        st.markdown(f"""
        <div style="background:{color};padding:20px;border-radius:12px;
                    text-align:center;margin:16px 0">
            <h2 style="color:white;margin:0">
                {home_team} {result['pred_home']}  —  {result['pred_away']} {away_team}
            </h2>
            <p style="color:white;margin:4px 0;font-size:18px">
                Total: {result['pred_total']} pts |
                {winner} wins {max(result['win_prob'], 100-result['win_prob']):.0f}% |
                {result['confidence']}
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.caption(f"Active model: {active_name}")

        # Injury log
        if result.get("injury_log"):
            st.warning("🏥 **Injury Adjustments Applied:**\n\n"
                       + "\n".join(result["injury_log"]))

        # Score ranges
        st.subheader("📊 Score Ranges")
        c1, c2, c3 = st.columns(3)
        c1.metric(f"{home_team} Score",
                  f"{result['pred_home']} pts",
                  f"Range: {result['home_range'][0]}–{result['home_range'][1]}")
        c2.metric(f"{away_team} Score",
                  f"{result['pred_away']} pts",
                  f"Range: {result['away_range'][0]}–{result['away_range'][1]}")
        c3.metric("Total Points",
                  f"{result['pred_total']} pts",
                  f"Range: {result['total_range'][0]}–{result['total_range'][1]}")

        # Shot styles
        st.subheader("🎯 Shot Style Comparison")
        c4, c5 = st.columns(2)
        c4.metric(f"{home_team} Style", result["home_style"],
                  f"Momentum: {result['home_momentum']:+.1f}")
        c5.metric(f"{away_team} Style", result["away_style"],
                  f"Momentum: {result['away_momentum']:+.1f}")


# ════════════════════════════════════════════════════════════
# PAGE 2 — MODEL DASHBOARD
# ════════════════════════════════════════════════════════════

elif page == "📊 Model Dashboard":
    st.title("📊 Model Performance Dashboard")

    if models is None:
        st.error("⚠️ Models not loaded.")
        st.stop()

    st.subheader("📉 MAE Journey (Historical Milestones)")
    mae_data = pd.DataFrame({
        "Version": ["V1 Rolling", "V2 Defense", "V3 Shots",
                     "V4 EWM", "V5 Players", "V6 SoS+Filter"],
        "Total MAE": [15.27, 14.99, 14.89, 14.82, 14.80, 14.52],
        "Features": [37, 41, 58, 70, 82, 84],
    })
    fig = px.bar(mae_data, x="Version", y="Total MAE",
                 text="Total MAE", color="Total MAE",
                 color_continuous_scale=["#2A9D8F", "#E9C46A", "#E63946"][::-1])
    fig.add_hline(y=14.85, line_dash="dash", line_color="red",
                  annotation_text="Old project baseline")
    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig.update_layout(height=350, showlegend=False,
                      coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

    # Feature importance
    st.subheader("🔧 Top Feature Importances")
    try:
        model_c = models["model_C"]
        feat_names = list(model_c.feature_names_in_)
        importances = pd.Series(
            model_c.feature_importances_, index=feat_names
        ).sort_values(ascending=True).tail(15)

        fig2 = px.bar(x=importances.values, y=importances.index,
                      orientation="h", color=importances.values,
                      color_continuous_scale="teal")
        fig2.update_layout(height=450, coloraxis_showscale=False,
                           yaxis_title="", xaxis_title="Importance")
        st.plotly_chart(fig2, use_container_width=True)
    except Exception as e:
        st.info(f"Feature importance not available: {e}")

    # Key metrics
    st.subheader("📈 Key Metrics")
    model_c_name = "Stacked Total Model"
    try:
        model_c_name = type(models["model_C"]).__name__
    except Exception:
        pass

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Base Features", str(metadata["base_feature_count"]), metadata["model_version"])
    m2.metric("Training Seasons", "3", "2023–2026")
    m3.metric("Model C Type", model_c_name, "stacked total")
    m4.metric("Target MAE", "< 14.5 pts", "Total points")


# ════════════════════════════════════════════════════════════
# PAGE 3 — TEAM RANKINGS
# ════════════════════════════════════════════════════════════

elif page == "🏆 Team Rankings":
    st.title("🏆 Team Rankings")

    if data["elo_df"] is not None:
        elo_df = data["elo_df"]
        fig = px.bar(elo_df, x="elo", y="team", orientation="h",
                     color="elo", color_continuous_scale="RdYlGn",
                     text="elo")
        fig.add_vline(x=1500, line_dash="dash", line_color="white",
                      annotation_text="League avg 1500")
        fig.update_traces(texttemplate="%{text:.0f}",
                          textposition="outside")
        fig.update_layout(
            height=700,
            yaxis={"categoryorder": "total ascending"},
            coloraxis_showscale=False,
            xaxis_title="ELO Rating", yaxis_title="",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("ELO ratings data not found in data/ directory.")

    # Team form table from model_df
    if data["model_df"] is not None:
        st.subheader("📊 Current Season Form")
        mdf = data["model_df"]
        latest = mdf["SEASON"].max()
        season_data = mdf[mdf["SEASON"] == latest]

        team_form = season_data.groupby("HOME_TEAM").agg(
            Avg_Scored=("HOME_ROLL10_PTS", "last"),
            Avg_Allowed=("HOME_DEF_ROLL10", "last"),
            Home_Strength=("HOME_COURT_STRENGTH", "last"),
        ).round(1).sort_values("Avg_Scored", ascending=False)

        st.dataframe(team_form, use_container_width=True)


# ════════════════════════════════════════════════════════════
# PAGE 4 — ABOUT
# ════════════════════════════════════════════════════════════

elif page == "ℹ️ About":
    st.title("ℹ️ About NextPlay")

    st.markdown("""
    ## NBA AI Score Predictor

    NextPlay uses machine learning to predict NBA game scores
    using a three-model stack (home, away, and total).
    """)

    feature_table = pd.DataFrame(
        feature_category_counts(),
        columns=["Category", "Features", "Description"],
    )
    st.dataframe(feature_table, use_container_width=True, hide_index=True)

    st.markdown(f"""
    **Current model version:** {metadata['model_version']}  
    **Base feature count:** {metadata['base_feature_count']}  
    **Stacked total features:** {metadata['stacked_total_feature_count']}

    ### How It Works
    1. **Three models** predict home score, away score, and total
    2. **Injury adjustments** modify predictions based on player impact
    3. **Confidence tiers** flag how reliable each prediction is
    4. **Nightly updates** keep the model current

    ### Tech Stack
    - **ML**: scikit-learn + LightGBM ensemble stack
    - **Data**: nba_api (official NBA stats)
    - **Dashboard**: Streamlit + Plotly
    - **Features**: configuration-driven feature registry from config.py
    """)
