# Machine Learning Capstone

DDoS attack detection on CICDDoS2019: preprocessing, baselines, a CNN-BiLSTM-Attention model, and a web app
that classifies flows with the trained models.

## Project Structure

```
Machine_Learning-Capstone/
│
├── data/
│   ├── raw/
│   ├── processed/
│
├── notebooks/        01-07, one per milestone (see below)
├── src/
│   ├── preprocessing/
│   ├── models/       hybrid.py (network), predictor.py (serving)
│   ├── training/
│   ├── evaluation/
│   ├── explainability/
│   └── utils/
│
├── app/
│   ├── api/          FastAPI backend (models, classify, explain, stream, results)
│   ├── web/          Vite + React + TypeScript frontend
│   └── data/         held-out sample flows for the demo
├── saved_models/
├── reports/
├── requirements.txt
├── README.md
└── .gitignore
```

## Notebooks

1. `01_dataset_exploration` - streaming profile of the 18 raw CSVs
2. `02_data_preprocessing` - clean, standardize labels, merge to Parquet
3. `03_eda` - class balance, missing values, redundancy
4. `04_baseline_models` (+ `04b_base_paper_reproduction`) - baselines, behavior-only ceiling, base-paper check
5. `05_hybrid_model` - CNN-BiLSTM-Attention
6. `06_final_evaluation` - every result against its ceiling
7. `07_serving_prep` - saves what the app needs and checks the served models reproduce the reported scores

## Running the app

Python dependencies: `pip install -r requirements.txt`. The frontend needs Node.js.

```powershell
# once: build the frontend
cd app\web
npm install
npm run build
cd ..\..

# start the API (it also serves the built frontend)
.venv\Scripts\python -m uvicorn app.api.main:app --port 8000
```

Open http://127.0.0.1:8000. The neural network loads in the background after startup, so its first use can
take a few seconds.

For frontend development, run the API as above (add `--reload` if you edit Python) and, in a second terminal,
`cd app\web; npm run dev`, then open http://localhost:5173 (it proxies `/api` to port 8000).

The app opens on Traffic Replay: each second it replays flows from the held-out test side, scores them with the
selected model and shows the detection status, a traffic chart, alerts and response guidance. It is a simulation, not
captured traffic. "Inject attack traffic" raises the share and volume of attack flows; the attack itself is a
scenario setting, the verdicts are the model's. The other pages are Classification (score a sample or your own CSV),
Model Evaluation (every model against the behavior-only ceiling) and Dataset and Method. Pages have their own URLs,
for example `#/replay`, `#/classification`, `#/evaluation` and `#/dataset`.

The app serves the models trained on the group split. Demo mode draws flows from that split's test side, so the
scores it shows are for behaviors the models never saw. To classify your own data, upload a CICFlowMeter CSV
with the 60 model features (a `Label` column is optional).
