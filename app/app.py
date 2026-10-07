"""Streamlit UI for the Music Recommendation System.

Run with:
    streamlit run app/app.py

The user types how they feel, the app detects the mood (TF-IDF + Logistic
Regression), maps it to an audio-feature cluster (K-Means) and shows the
top-ranked tracks for that mood.
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv  # noqa: E402

from src.recommendation import MusicRecommender  # noqa: E402

load_dotenv()

st.set_page_config(page_title="Music Recommendation System", page_icon="🎵", layout="centered")

MOOD_EMOJI = {
    "joy": "😊", "love": "❤️", "sadness": "😔",
    "anger": "😠", "fear": "😨", "surprise": "😲",
}

st.markdown(
    """
    <style>
      .block-container {max-width: 760px;}
      .mood-badge {font-size: 1.4rem; font-weight: 700; padding: .35rem .9rem;
                   border-radius: 999px; background: #6C5CE7; color: #fff;
                   display: inline-block;}
      .track-card {border: 1px solid rgba(128,128,128,.25); border-radius: 12px;
                   padding: .8rem 1rem; margin-bottom: .6rem;}
      .track-title {font-weight: 600; font-size: 1.05rem;}
      .track-meta {opacity: .75; font-size: .9rem;}
      .sim-pill {float: right; font-size: .8rem; opacity: .7;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading models...")
def get_recommender() -> MusicRecommender:
    return MusicRecommender.load()


st.title("🎵 Music Recommendation System")
st.caption(
    "Mood-aware recommendations: NLP emotion classification (TF-IDF + Logistic "
    "Regression) meets K-Means clustering of Spotify audio features."
)

try:
    recommender = get_recommender()
except FileNotFoundError:
    st.error(
        "Models are not trained yet. Run:\n\n"
        "```\npython scripts/prepare_data.py\n"
        "python scripts/train_mood.py\n"
        "python scripts/train_clusters.py\n```"
    )
    st.stop()

with st.form("mood_form"):
    text = st.text_area(
        "How are you feeling today?",
        placeholder="e.g. I've had a rough day and just want something calm...",
        height=110,
    )
    n = st.slider("Number of recommendations", 3, 10, 5)
    submitted = st.form_submit_button("Recommend songs 🎧")

if submitted:
    if not text or not text.strip():
        st.warning("Please tell me how you're feeling first.")
        st.stop()
    try:
        result = recommender.recommend(text, n=n)
    except ValueError:
        st.warning("That input didn't contain any usable words. Try again.")
        st.stop()
    except Exception as exc:  # defensive: never leak a raw stack trace
        st.error(f"Something went wrong while generating recommendations: {exc}")
        st.stop()

    emoji = MOOD_EMOJI.get(result.predicted_mood, "🎵")
    st.markdown(
        f"Detected mood &nbsp; "
        f"<span class='mood-badge'>{emoji} {result.predicted_mood.title()}</span> "
        f"&nbsp; <span class='track-meta'>confidence {result.confidence:.0%} · "
        f"cluster {result.cluster}</span>",
        unsafe_allow_html=True,
    )
    st.subheader("Recommended songs")
    for i, rec in enumerate(result.recommendations, 1):
        title = rec.track_name
        if rec.spotify_url:
            title = f"<a href='{rec.spotify_url}' target='_blank'>{title}</a>"
        meta_bits = [b for b in [rec.artists, rec.genre] if b]
        st.markdown(
            f"<div class='track-card'>"
            f"<span class='sim-pill'>match {rec.similarity:.2f}</span>"
            f"<div class='track-title'>{i}. {title}</div>"
            f"<div class='track-meta'>{' · '.join(meta_bits)}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    with st.expander("Mood → cluster mapping (how this works)"):
        st.dataframe(recommender.mapping_report(), hide_index=True, use_container_width=True)
