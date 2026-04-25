# -*- coding: utf-8 -*-
"""
NextPlay — NBA AI Score Predictor Dashboard
=============================================
Streamlit dashboard for the V7 NBA prediction system.
Run with: streamlit run app.py
"""
import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from config import (
    DATA_DIR, MODEL_READY_PATH, SHOT_PROFILES_PATH,
    PLAYER_IMPACT_PATH, BACKTEST_RESULTS_PATH, ELO_RATINGS_PATH,
    NBA_TEAMS, FEATURE_COLS_FINAL,
)
from models.train import load_models
from prediction.predict import predict_game

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
        return load_models()
    except FileNotFoundError:
        return None


data = load_data()
models = load_cached_models()

# ════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════

st.sidebar.title("🏀 NextPlay")
st.sidebar.caption("NBA AI Score Predictor — V7")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["🏀 Game Predictor", "📊 Model Dashboard",
     "🏆 Team Rankings", "ℹ️ About"],
)

if data["model_df"] is not None:
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        f"**Dataset:** {len(data['model_df']):,} games  \n"
        f"**Features:** {len(FEATURE_COLS_FINAL)}  \n"
        f"**Model:** RandomForest V7"
    )

# ════════════════════════════════════════════════════════════
# PAGE 1 — GAME PREDICTOR
# ════════════════════════════════════════════════════════════

if page == "🏀 Game Predictor":
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
        with col_inj2:
            away_inj_str = st.text_input(
                f"{away_team} players OUT (comma-separated)",
                placeholder="e.g. LeBron")

    home_out = [x.strip() for x in home_inj_str.split(",") if x.strip()] \
        if home_inj_str else []
    away_out = [x.strip() for x in away_inj_str.split(",") if x.strip()] \
        if away_inj_str else []

    if st.button("🔮 Predict Matchup", type="primary",
                 use_container_width=True):
        with st.spinner("Analyzing matchup..."):
            feature_cols = None
            try:
                feature_cols = list(models["model_C"].feature_names_in_)
            except Exception:
                feature_cols = [c for c in FEATURE_COLS_FINAL
                                if c in data["model_df"].columns]

            result = predict_game(
                home_team, away_team,
                data["model_df"], models,
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

    st.subheader("📉 MAE Journey")
    mae_data = pd.DataFrame({
        "Version": ["V1 Rolling", "V2 Defense", "V3 Shots",
                     "V4 EWM", "V5 Players", "V7 SoS+Filter"],
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
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Features", "84", "V7")
    m2.metric("Training Seasons", "3", "2023–2026")
    m3.metric("Model Type", "RandomForest", "400 trees")
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
    ## NBA AI Score Predictor — V7

    NextPlay uses machine learning to predict NBA game scores
    using **84 engineered features** across 6 categories:

    | Category | Features | Examples |
    |---|---|---|
    | Rolling Form | 23 | Last 10 game averages |
    | Rest & Momentum | 6 | Rest days, win streaks |
    | Context | 5 | Season stage, home court |
    | Defensive | 4 | Points allowed rolling |
    | Shot Profiles | 17 | 3PT rate, paint rate, PPS |
    | Advanced | 29 | EWM, SoS, player impact |

    ### How It Works
    1. **Three models** predict home score, away score, and total
    2. **Injury adjustments** modify predictions based on player impact
    3. **Confidence tiers** flag how reliable each prediction is
    4. **Nightly updates** keep the model current

    ### Tech Stack
    - **ML**: scikit-learn RandomForest (400 trees)
    - **Data**: nba_api (official NBA stats)
    - **Dashboard**: Streamlit + Plotly
    - **Features**: 84 engineered features (V7)
    """)
