# Models

Model artifacts are generated at runtime by `app.py` using `@st.cache_resource`.

| Model | R² | Notes |
|---|---|---|
| Linear Regression | 0.854 | Interpretable baseline |
| Random Forest | 0.838 | 100 trees, depth-optimized |

No `.pkl` files are committed here — the app trains on startup to keep the repo lightweight and Streamlit Cloud deploy instant.
