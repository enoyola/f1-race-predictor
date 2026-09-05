"""
F1 Race Predictor web app.

    streamlit run app.py
"""

import json
import os
from typing import Dict, List, Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from f1_predictor import __version__
from f1_predictor.engine import PredictionEngine
from f1_predictor.features import FACTOR_LABELS
from f1_predictor.formatter import result_to_dict
from f1_predictor.models import PredictionError

BACKTEST_PATH = "models/backtest_results.json"

# Categorical slots from the validated reference palette (blue, orange, aqua, yellow).
COLORS = {
    "statistical": "#2a78d6",
    "ml": "#eb6834",
    "ai": "#e87ba4",
    "verdict": "#4a3aa7",
    "podium": "#1baf7a",
    "pole": "#eda100",
    "leader": "#e87ba4",
    "factor": "#2a78d6",
}
MODEL_LABELS = {"statistical": "Statistical", "ml": "Machine learning", "ai": "AI analyst", "verdict": "AI verdict",
                "pole": "Pole sitter wins", "leader": "Points leader wins"}
MODEL_CHOICES = ["Statistical", "Machine learning", "AI analyst", "AI verdict", "Compare all"]
MODEL_KEYS = {"Statistical": "statistical", "Machine learning": "ml", "AI analyst": "ai"}

st.set_page_config(page_title="F1 Race Predictor", page_icon="🏎️", layout="wide")


# ----------------------------------------------------------------------
# Cached helpers
# ----------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def get_engine() -> PredictionEngine:
    return PredictionEngine(top_n=None, use_weather=True)


@st.cache_data(ttl=3600, show_spinner=False)
def load_calendar(season: int) -> List[dict]:
    engine = get_engine()
    completed = set(engine.completed_rounds(season))
    return [
        {
            "round": r.round,
            "name": r.race_name,
            "date": r.date.strftime("%d %b"),
            "is_sprint": r.is_sprint,
            "done": r.round in completed,
        }
        for r in engine.get_schedule(season)
    ]


@st.cache_data(ttl=900, show_spinner=False)
def run_prediction(season: int, round_num: int, model: str) -> dict:
    engine = get_engine()
    return result_to_dict(engine.predict_race(season, round_num, model))


@st.cache_data(ttl=900, show_spinner=False)
def run_verdict(season: int, round_num: int) -> Dict[str, dict]:
    engine = get_engine()
    return {name: result_to_dict(r) for name, r in engine.verdict(season, round_num).items()}


@st.cache_data(ttl=900, show_spinner=False)
def run_compare(season: int, round_num: int) -> Dict[str, dict]:
    engine = get_engine()
    return {name: result_to_dict(r) for name, r in engine.compare(season, round_num).items()}


def ai_status() -> Optional[str]:
    """None when the AI analyst is usable, otherwise the reason it is not."""
    engine = get_engine()
    return None if engine.ai_available else engine.availability_error("ai")


@st.cache_data(ttl=600, show_spinner=False)
def load_backtest() -> Optional[dict]:
    if not os.path.exists(BACKTEST_PATH):
        return None
    with open(BACKTEST_PATH, encoding="utf-8") as f:
        return json.load(f)


def model_info() -> dict:
    engine = get_engine()
    analyzer = engine.get_analyzer("ml")
    bundle = getattr(analyzer, "bundle", None) or {}
    return {
        "loaded": getattr(analyzer, "model_loaded", False),
        "error": getattr(analyzer, "load_error", None),
        "trained_at": bundle.get("trained_at"),
        "train_seasons": bundle.get("train_seasons"),
        "n_samples": bundle.get("n_samples"),
        "algorithm": bundle.get("algorithm"),
        "metrics": bundle.get("metrics") or {},
        "feature_importances": bundle.get("feature_importances") or {},
        "sklearn_version": bundle.get("sklearn_version"),
    }


# ----------------------------------------------------------------------
# Charts
# ----------------------------------------------------------------------

def _base_layout(fig: go.Figure, height: int) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=40, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        bargap=0.25,
        bargroupgap=0.08,
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(128,128,128,0.2)", zeroline=False, tickformat=".0%", range=[0, 1.08])
    fig.update_yaxes(showgrid=False, autorange="reversed")
    return fig


def probability_chart(predictions: List[dict], n: int) -> go.Figure:
    rows = predictions[:n]
    labels = [f"{p['driver']} ({p['constructor']})" for p in rows]
    fig = go.Figure()
    fig.add_bar(
        y=labels, x=[p["win_probability"] for p in rows], orientation="h", name="Win",
        marker_color=COLORS["statistical"], marker_line_width=0,
        text=[f"{p['win_probability']:.1%}" for p in rows], textposition="outside", cliponaxis=False,
        hovertemplate="%{y}<br>Win %{x:.1%}<extra></extra>",
    )
    fig.add_bar(
        y=labels, x=[p["podium_probability"] for p in rows], orientation="h", name="Podium",
        marker_color=COLORS["podium"], marker_line_width=0,
        text=[f"{p['podium_probability']:.0%}" for p in rows], textposition="outside", cliponaxis=False,
        hovertemplate="%{y}<br>Podium %{x:.1%}<extra></extra>",
    )
    fig.update_layout(barmode="group")
    return _base_layout(fig, height=60 + 44 * len(rows))


def comparison_chart(results: Dict[str, dict], n: int) -> go.Figure:
    by_driver: Dict[str, Dict[str, float]] = {}
    names: Dict[str, str] = {}
    for model, result in results.items():
        for p in result["predictions"]:
            by_driver.setdefault(p["driver_id"], {})[model] = p["win_probability"]
            names[p["driver_id"]] = f"{p['driver']} ({p['constructor']})"
    ordered = sorted(by_driver, key=lambda d: max(by_driver[d].values()), reverse=True)[:n]
    fig = go.Figure()
    for model in results:
        fig.add_bar(
            y=[names[d] for d in ordered], x=[by_driver[d].get(model, 0.0) for d in ordered], orientation="h",
            name=MODEL_LABELS.get(model, model), marker_color=COLORS.get(model, "#888"), marker_line_width=0,
            text=[f"{by_driver[d].get(model, 0.0):.1%}" for d in ordered], textposition="outside", cliponaxis=False,
            hovertemplate="%{y}<br>" + MODEL_LABELS.get(model, model) + " win %{x:.1%}<extra></extra>",
        )
    fig.update_layout(barmode="group")
    return _base_layout(fig, height=60 + 44 * len(ordered))


def factor_chart(factors: Dict[str, float]) -> go.Figure:
    names = [FACTOR_LABELS.get(k, k) for k in factors]
    values = list(factors.values())
    fig = go.Figure(go.Bar(
        y=names, x=[v / 100 for v in values], orientation="h", marker_color=COLORS["factor"], marker_line_width=0,
        text=[f"{v:.0f}" for v in values], textposition="outside", cliponaxis=False,
        hovertemplate="%{y}: %{text}/100<extra></extra>",
    ))
    fig = _base_layout(fig, height=40 + 30 * len(names))
    fig.update_xaxes(tickformat=".0%", range=[0, 1.12], title=None)
    return fig


def accuracy_chart(report: dict, metric: str, title: str) -> go.Figure:
    seasons = [str(s) for s in report["seasons"]]
    fig = go.Figure()
    for model in report["models"]:
        values = [report["summary"].get(model, {}).get(s, {}).get(metric) or 0 for s in seasons]
        fig.add_bar(
            x=seasons, y=values, name=MODEL_LABELS.get(model, model), marker_color=COLORS.get(model, "#888"),
            marker_line_width=0, text=[f"{v:.0%}" for v in values], textposition="outside", cliponaxis=False,
            hovertemplate="%{x} " + MODEL_LABELS.get(model, model) + ": %{y:.1%}<extra></extra>",
        )
    fig.update_layout(
        barmode="group", height=340, margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    fig.update_yaxes(tickformat=".0%", range=[0, 1.15], showgrid=True, gridcolor="rgba(128,128,128,0.2)", zeroline=False)
    fig.update_xaxes(showgrid=False)
    return fig


# ----------------------------------------------------------------------
# Page pieces
# ----------------------------------------------------------------------

SHORT_FACTOR_LABELS = {
    "championship": "Champ.",
    "form": "Form",
    "team": "Team",
    "qualifying": "Quali",
    "circuit": "Circuit",
    "reliability": "Reliab.",
    "teammate": "Teammate",
    "sprint": "Sprint",
}


def predictions_table(predictions: List[dict], actual: Optional[List[dict]] = None, factors: bool = True) -> pd.DataFrame:
    """Rows for a static table; probabilities pre-formatted as strings."""
    actual_pos = {a["driver_id"]: a["position"] for a in (actual or [])}
    rows = []
    for p in predictions:
        row = {
            "#": p["rank"],
            "Driver": p["driver"],
            "Team": p["constructor"],
            "Grid": f"P{p['grid_position']}" if p["grid_position"] else "-",
            "Win": f"{p['win_probability']:.1%}",
            "Podium": f"{p['podium_probability']:.0%}",
        }
        if actual_pos:
            fin = actual_pos.get(p["driver_id"])
            row["Finished"] = f"P{fin}" if fin else "-"
        if factors:
            for key, label in SHORT_FACTOR_LABELS.items():
                if key in p["factors"]:
                    row[label] = f"{p['factors'][key]:.0f}"
        rows.append(row)
    return pd.DataFrame(rows).set_index("#")


def show_table(df: pd.DataFrame) -> None:
    st.table(df)


def race_header(result: dict) -> None:
    race = result["race"]
    sprint = "  ·  Sprint weekend" if race["is_sprint"] else ""
    st.subheader(f"{race['name']}  ·  Round {race['round']} of {race['total_rounds']}{sprint}")
    circuit = race["circuit"]
    st.caption(f"{circuit['circuit_name']}, {circuit['location']}, {circuit['country']}  ·  {race['date'][:10]}")

    pole = next((p for p in result["predictions"] if p["grid_position"] == 1), None)
    weather = result["weather"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Qualifying", "Available" if result["qualifying_available"] else "Not yet")
    if pole:
        c1.caption(f"Pole: {pole['driver']}")
    if weather:
        rain = weather.get("precipitation_mm")
        prob = weather.get("precipitation_probability")
        temp = weather.get("temperature_max")
        parts = []
        if rain is not None:
            parts.append(f"{rain:.1f} mm rain")
        if prob is not None:
            parts.append(f"{prob:.0f}% chance")
        if temp is not None:
            parts.append(f"{temp:.0f}°C")
        c2.metric("Weather", "Wet" if (rain or 0) >= 1 or (prob or 0) >= 60 else "Dry")
        c2.caption(", ".join(parts) + f" ({weather.get('source')})")
    else:
        c2.metric("Weather", "Unknown")
    c3.metric("Model", MODEL_LABELS.get(result["model"], result["model"]))
    c4.metric("Data completeness", f"{result['data_completeness'] * 100:.0f}%")
    c4.caption("standings, results, qualifying, circuit history, weather")
    for note in result["notes"]:
        st.info(note)


def actual_banner(result: dict) -> None:
    actual = result.get("actual_results")
    if not actual:
        return
    top3 = ", ".join(f"P{a['position']} {a['driver']}" for a in actual[:3])
    pick = result["predictions"][0]
    hit = pick["driver_id"] == actual[0]["driver_id"]
    (st.success if hit else st.warning)(
        f"Actual result: {top3}.  Predicted winner {pick['driver']} was {'right' if hit else 'wrong'}."
    )


def driver_details(predictions: List[dict], n: int = 5) -> None:
    for p in predictions[:n]:
        with st.expander(f"{p['rank']}. {p['driver']} — win {p['win_probability']:.1%}, podium {p['podium_probability']:.0%}"):
            left, right = st.columns([1, 1])
            with left:
                for line in p["reasoning"]:
                    st.markdown(f"- {line}")
            with right:
                st.plotly_chart(factor_chart(p["factors"]), width="stretch", config={"displayModeBar": False})


# ----------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------

def sidebar() -> dict:
    with st.sidebar:
        st.title("🏎️ F1 Race Predictor")
        st.caption(f"v{__version__}")
        engine = get_engine()
        try:
            current = engine.current_season()
        except Exception:
            current = 2026
        season = st.selectbox("Season", list(range(current, current - 4, -1)))
        try:
            calendar = load_calendar(season)
        except Exception as e:
            st.error(f"Could not load the {season} calendar: {e}")
            st.stop()
        if not calendar:
            st.error(f"No races found for {season}")
            st.stop()

        labels = [
            f"R{c['round']:02d}  {c['name']}  ({c['date']}){' ⚡' if c['is_sprint'] else ''}{'  ✓' if c['done'] else ''}"
            for c in calendar
        ]
        upcoming = [i for i, c in enumerate(calendar) if not c["done"]]
        done = [i for i, c in enumerate(calendar) if c["done"]]
        default = upcoming[0] if (season == current and upcoming) else (done[-1] if done else 0)
        choice = st.selectbox("Race", range(len(calendar)), index=default, format_func=lambda i: labels[i])
        st.caption("⚡ sprint weekend · ✓ completed (prediction uses only pre-race data)")

        model_choice = st.radio("Model", MODEL_CHOICES, index=0)
        show_n = st.slider("Drivers in chart", 3, 22, 10)

        if st.button("Refresh data", help="Drop cached data for this season and refetch"):
            engine.refresh_season(season)
            st.cache_data.clear()
            st.rerun()

        st.divider()
        info = model_info()
        if info["loaded"]:
            st.caption(f"ML model: {info['algorithm']} trained {str(info['trained_at'])[:10]} on {info['n_samples']} rows")
        else:
            st.caption("ML model not trained. Run `python train_model.py`.")
        ai_error = ai_status()
        if ai_error:
            st.caption(f"AI analyst unavailable: {ai_error}")
        else:
            st.caption("AI analyst: Claude, one call per race (cached)")
    return {"season": season, "round": calendar[choice]["round"], "model": model_choice, "n": show_n}


# ----------------------------------------------------------------------
# Tabs
# ----------------------------------------------------------------------

def analysis_block(result: dict) -> None:
    if result.get("analysis"):
        st.markdown("**AI analysis**")
        st.markdown(result["analysis"])


def components_table(verdict: dict, results: Dict[str, dict], n: int) -> pd.DataFrame:
    names = {p["driver_id"]: (p["driver"], p["constructor"]) for r in results.values() for p in r["predictions"]}
    rows = []
    for p in verdict["predictions"][:n]:
        comp = (verdict.get("components") or {}).get(p["driver_id"], {})
        row = {"#": p["rank"], "Driver": names.get(p["driver_id"], (p["driver"], ""))[0], "Team": p["constructor"], "Grid": f"P{p['grid_position']}" if p["grid_position"] else "-"}
        for model in ("statistical", "ml", "ai"):
            if model in results:
                row[MODEL_LABELS[model]] = f"{comp.get(model, 0):.1%}"
        row["Verdict win"] = f"{p['win_probability']:.1%}"
        row["Verdict podium"] = f"{p['podium_probability']:.0%}"
        rows.append(row)
    return pd.DataFrame(rows).set_index("#")


def prediction_tab(sel: dict) -> None:
    season, rnd, n = sel["season"], sel["round"], sel["n"]
    choice = sel["model"]
    try:
        if choice == "AI verdict":
            with st.spinner("Running every model and asking the AI analyst..."):
                results = run_verdict(season, rnd)
            verdict = results["verdict"]
            race_header(verdict)
            actual_banner(verdict)
            if "ai" not in results:
                st.warning(f"AI analyst not included: {ai_status()}")
            analysis_block(verdict)
            st.plotly_chart(probability_chart(verdict["predictions"], n), width="stretch", config={"displayModeBar": False})
            st.markdown("**How each model voted**")
            show_table(components_table(verdict, {k: v for k, v in results.items() if k != "verdict"}, n))
            st.markdown("**Why these picks**")
            driver_details(verdict["predictions"])
        elif choice == "Compare all":
            with st.spinner("Building pre-race context and running every model..."):
                results = run_compare(season, rnd)
            first = results["statistical"]
            race_header(first)
            actual_banner(first)
            if "ml" in results and results["ml"]["model"] != "ml":
                st.warning("The ML model is not available, so its column shows the statistical model. Run `python train_model.py`.")
            if "ai" not in results:
                st.info(f"AI analyst not shown: {ai_status()}")
            st.plotly_chart(comparison_chart(results, n), width="stretch", config={"displayModeBar": False})
            analysis_block(results.get("ai", {}))
            cols = st.columns(len(results))
            for col, (name, result) in zip(cols, results.items()):
                with col:
                    st.markdown(f"**{MODEL_LABELS[name]}**")
                    show_table(predictions_table(result["predictions"], result.get("actual_results"), factors=False))
            st.markdown("**Why these picks**")
            cols = st.columns(len(results))
            for col, (name, result) in zip(cols, results.items()):
                with col:
                    st.caption(MODEL_LABELS[name])
                    driver_details(result["predictions"], 3)
        else:
            model = MODEL_KEYS[choice]
            with st.spinner("Asking the AI analyst..." if model == "ai" else "Building pre-race context and running the model..."):
                result = run_prediction(season, rnd, model)
            race_header(result)
            actual_banner(result)
            if result["model"] != model:
                st.warning(f"{MODEL_LABELS[model]} is not available; showing the statistical model instead. "
                           + (ai_status() or "") if model == "ai" else f"{MODEL_LABELS[model]} is not available; showing the statistical model instead.")
            analysis_block(result)
            st.plotly_chart(probability_chart(result["predictions"], n), width="stretch", config={"displayModeBar": False})
            st.markdown("**Full field**")
            st.caption("Factor scores are 0-100: Champ. championship, Quali qualifying, Reliab. reliability")
            show_table(predictions_table(result["predictions"], result.get("actual_results")))
            st.markdown("**Why these picks**")
            driver_details(result["predictions"])
    except PredictionError as e:
        st.error(f"{e.error_type}: {e.message}")
        for s_ in e.suggestions:
            st.caption(f"• {s_}")


def accuracy_tab() -> None:
    report = load_backtest()
    if not report:
        st.info("No backtest results yet. Run `python backtest.py` to evaluate the models against past seasons.")
        return
    st.caption(
        f"Walk-forward backtest over {report['seasons'][0]}-{report['seasons'][-1]} generated {report['generated_at'][:10]}. "
        "For each race the models only see data available before that race; the ML model is retrained per season on earlier seasons only."
    )
    rows = []
    for model in report["models"]:
        m = report["summary"].get(model, {}).get("overall", {})
        rows.append({
            "Model": MODEL_LABELS.get(model, model),
            "Races": m.get("races", 0),
            "Winner picked": f"{m.get('top1_accuracy', 0):.1%}",
            "Winner in top 3": f"{m.get('winner_in_top3', 0):.1%}",
            "Podium precision": f"{m.get('podium_precision', 0):.1%}",
            "Log loss": f"{m['log_loss']:.3f}" if m.get("log_loss") is not None else "–",
            "Brier": f"{m['brier']:.3f}" if m.get("brier") is not None else "–",
        })
    st.table(pd.DataFrame(rows).set_index("Model"))
    st.caption("Winner picked: top pick won · Winner in top 3: real winner among the predicted top three · "
               "Podium precision: predicted podium drivers who finished top three · Log loss and Brier: lower is better, baselines give no probabilities")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Winner picked correctly, by season**")
        st.plotly_chart(accuracy_chart(report, "top1_accuracy", ""), width="stretch", config={"displayModeBar": False})
    with c2:
        st.markdown("**Podium precision, by season**")
        st.plotly_chart(accuracy_chart(report, "podium_precision", ""), width="stretch", config={"displayModeBar": False})
    if report.get("calibration"):
        st.caption(f"Softmax temperature with the lowest log loss for the statistical model: {report['calibration'].get('best_temperature')}")

    st.markdown("**Race by race**")
    season_filter = st.selectbox("Season", ["All"] + [str(s) for s in report["seasons"]], key="bt_season")
    race_rows = []
    for r in report["races"]:
        if season_filter != "All" and str(r["season"]) != season_filter:
            continue
        row = {"Season": r["season"], "Round": r["round"], "Race": r["race_name"], "Winner": r["actual_winner"], "Wet": "🌧" if r.get("wet") else ""}
        for model in ("statistical", "ml"):
            pred = r["predictions"].get(model)
            if pred:
                top = pred["top3"][0]
                row[MODEL_LABELS[model]] = f"{top['code']} {top['win_probability']:.0%}{' ✓' if pred['winner_hit'] else ''}"
        pole = r["predictions"].get("pole")
        if pole:
            row["Pole"] = f"{pole['top3'][0]['code']}{' ✓' if pole['winner_hit'] else ''}"
        race_rows.append(row)
    st.caption("✓ marks a correct winner pick; percentages are the win probability given to the top pick")
    st.table(pd.DataFrame(race_rows).set_index("Round") if race_rows else pd.DataFrame())


def about_tab() -> None:
    st.markdown(
        """
**How it works.** For any race the app builds a *pre-race context*: standings computed from the
results so far, each driver's last five results (rolling over from the previous season), the
qualifying grid, this weekend's sprint result, past results at the circuit, retirements in the last
ten races, the gap to the teammate, and race-day weather from Open-Meteo. Both models work from that
same context, so the accuracy measured in backtests is what you should expect before a real race.

**Statistical model.** A weighted average of 0-100 factor scores (championship 22%, form 18%, team 17%,
qualifying 20%, circuit history 10%, reliability 6%, teammate 7%), turned into win probabilities with a
softmax and podium probabilities with a Plackett-Luce model. In the wet, grid position is weighted less.

**Machine-learning model.** Two logistic-regression classifiers (win and podium) trained on every race
since 2020 with the same features; random forest and gradient boosting are available with `--algorithm`. Each race is also added with qualifying hidden, so the model
can predict before Saturday. Probabilities are normalized across the field.

**AI analyst.** Claude receives the same briefing (grid, standings, form, circuit history, reliability,
teammate gap, weather) plus the two models' picks, and returns its own probabilities with a written
analysis. Needs an Anthropic API key; one call per race, cached for six hours.

**AI verdict.** The average of the available models' probabilities, renormalized, with the AI analysis attached.

**Data.** Jolpica F1 API (results, qualifying, sprints, schedule), Open-Meteo (weather) and the Anthropic API
(AI analyst). Responses are cached locally; completed seasons are cached for 30 days.
        """
    )
    info = model_info()
    if info["loaded"]:
        st.markdown("**Trained model**")
        st.caption(
            f"{info['algorithm']} · trained {str(info['trained_at'])[:16]} · seasons {info['train_seasons'][0]}-{info['train_seasons'][-1]} · "
            f"{info['n_samples']} rows · scikit-learn {info['sklearn_version']}"
        )
        metrics = info["metrics"]
        if metrics:
            st.caption(
                f"Holdout {metrics.get('holdout_season')}: winner picked {metrics.get('top1_accuracy', 0):.1%}, "
                f"winner in top 3 {metrics.get('winner_in_top3', 0):.1%}, podium precision {metrics.get('podium_precision', 0):.1%}"
            )
        if info["feature_importances"]:
            fi = dict(sorted(info["feature_importances"].items(), key=lambda kv: kv[1], reverse=True))
            fig = go.Figure(go.Bar(
                y=list(fi.keys()), x=list(fi.values()), orientation="h", marker_color=COLORS["ml"], marker_line_width=0,
                text=[f"{v:.1%}" for v in fi.values()], textposition="outside", cliponaxis=False,
                hovertemplate="%{y}: %{x:.1%}<extra></extra>",
            ))
            fig = _base_layout(fig, height=40 + 24 * len(fi))
            fig.update_xaxes(range=[0, max(fi.values()) * 1.25], tickformat=".0%")
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    else:
        st.caption(f"ML model not loaded: {info['error']}")


def main() -> None:
    sel = sidebar()
    tab_predict, tab_accuracy, tab_about = st.tabs(["Prediction", "Model accuracy", "About"])
    with tab_predict:
        prediction_tab(sel)
    with tab_accuracy:
        accuracy_tab()
    with tab_about:
        about_tab()


main()
