"""Streamlit interface for the MFC Power_density model package."""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="MFC Power Density Predictor", page_icon=None, layout="wide")
MODEL_PATH = Path(__file__).with_name("mfc_power_density_model.pkl")

st.title("MFC Power Density Predictor")
st.write("Estimate literature-based Power_density from operating, system, electrode, electrolyte and biological inputs.")
st.warning("This is a research baseline, not an experimentally validated catalyst optimiser. Performance on entirely unseen studies was weak in grouped validation.")

if not MODEL_PATH.exists():
    st.error("Model file not found. Place mfc_power_density_model.pkl in the same folder as app.py. If it was lost, run train_model.py with processed_v1.xlsx first.")
    st.stop()

try:
    package = joblib.load(MODEL_PATH)
    pipeline = package["pipeline"]
    features = package["features"]
except Exception as exc:
    st.error(f"Could not load model package: {exc}")
    st.stop()

NUMERIC_DEFAULTS = {
    "HRT": 24.0, "COD": 1000.0, "Temp": 25.0, "Light_intensity": 1000.0,
    "Light_cycle": 12.0, "Flowrate": 1.0, "Resis": 1000.0, "M_DO": 5.0,
}
CATEGORICAL_OPTIONS = {
    "Reactor type": ["two chamber", "single chamber", "Others"],
    "Carbon source": ["acetate", "glucose", "wastewater", "Others"],
    "Light_source": ["natural light", "artificial light", "dark", "Others"],
    "Anode": ["CC", "GP", "GF", "Metal", "CP", "Others"],
    "Membrane": ["porous membrane", "PEM", "CEM", "AEM", "no membrane", "Others"],
    "Anolyte": ["wastewater", "buffer", "synthetic medium", "Others"],
    "Anode_Inoc": ["wastewater", "activated sludge", "mixed culture", "Others"],
    "Catholyte": ["wastewater", "buffer", "synthetic medium", "Others"],
    "Cathode_Inoc": ["Chlorella vulgaris", "microalgae", "mixed culture", "None", "Others"],
    "Algae": ["Chlorella vulgaris", "Chlorella sp.", "Scenedesmus sp.", "mixed algae", "None", "Others"],
}

st.subheader("Enter system characteristics")
with st.form("mfc_input_form"):
    left, right = st.columns(2)
    values = {}
    with left:
        st.markdown("**Operating conditions**")
        for col in package.get("numeric_features", features[:8]):
            values[col] = st.number_input(col, value=float(NUMERIC_DEFAULTS.get(col, 0.0)), format="%.6g")
    with right:
        st.markdown("**System and biological characteristics**")
        for col in package.get("categorical_features", features[8:]):
            opts = CATEGORICAL_OPTIONS.get(col, ["Unknown"])
            values[col] = st.selectbox(col, opts, index=0)
    submitted = st.form_submit_button("Predict Power Density")

if submitted:
    row = pd.DataFrame([{col: values[col] for col in features}], columns=features)
    try:
        prediction = float(np.expm1(pipeline.predict(row)[0]))
        prediction = max(0.0, prediction)
        st.metric("Predicted Power Density", f"{prediction:.6g}")
        st.caption("Prediction from the trained model; it is not a measured value. Confirm units and input-category spellings against the source dataset before interpreting the estimate.")
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")

with st.expander("Model information"):
    st.write(f"Training rows recorded in package: {package.get('training_rows', 'not recorded')}")
    st.write(f"Study groups recorded: {package.get('study_groups', 'not recorded')}")
    st.write("Input fields: " + ", ".join(features))
    st.write("Random holdout metrics:")
    if package.get("random_holdout_metrics"):
        st.dataframe(pd.DataFrame(package["random_holdout_metrics"]).T)
    st.write("Grouped cross-validation metrics:")
    if package.get("grouped_cv_metrics"):
        st.dataframe(pd.DataFrame(package["grouped_cv_metrics"]).T)
