# MFC Power Density ML Project — Rebuilt Source

This folder reconstructs the core source code for the MFC Power_density regression project. It contains a training/validation script and a Streamlit inference app.

## Files

- `train_model.py`: loads the processed dataset, validates required columns, compares Random Forest / Gradient Boosting / SVR on a random holdout, evaluates grouped cross-validation by `Study_ID`, trains the final Random Forest pipeline, and saves a reloadable model package.
- `app.py`: Streamlit UI that uses the saved model package.
- `requirements.txt`: Python dependencies.

## Dataset required

Place `processed_v1.xlsx` in this folder, or pass its path to the training script. The dataset needs these columns:

`Power_density`, `Study_ID` (recommended for grouped validation), `HRT`, `COD`, `Temp`, `Light_intensity`, `Light_cycle`, `Flowrate`, `Resis`, `M_DO`, `Reactor type`, `Carbon source`, `Light_source`, `Anode`, `Membrane`, `Anolyte`, `Anode_Inoc`, `Catholyte`, `Cathode_Inoc`, `Algae`.

The training script intentionally excludes target-derived columns and `Study_ID` from model inputs. It uses `log1p(Power_density)` during fitting and converts predictions back with `expm1`.

## Setup on Windows

Open Command Prompt or PowerShell in this folder:

```powershell
py -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python train_model.py --data processed_v1.xlsx --output mfc_power_density_model.pkl
streamlit run app.py
```

If the Excel file is stored elsewhere, use its full path after `--data`.

## Interpretation caveat

The project dataset has 136 observations from 44 source studies. Prior recorded random-row test performance was much stronger than study-grouped validation (pooled GroupKFold R² was approximately -0.20). Therefore the model should be treated as a research baseline and not as a validated predictor for unseen studies or a single-chamber catalyst optimiser. The current dataset contains only one explicitly labelled single-chamber observation.

The code is reconstructed from the project workflow and stored results, not recovered from the deleted original folder. Exact numerical results may differ slightly if the original data file, package versions, split, or preprocessing differs.
