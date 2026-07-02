# 🌸 FemCare — Early Endometriosis Risk Screening

> An AI-powered web application that screens for endometriosis risk using machine learning and explainable AI.

🔗 **Live App:** [https://femcare-k8ty1rx7y-kaviyas-projects-cb95cf97.vercel.app](https://femcare-k8ty1rx7y-kaviyas-projects-cb95cf97.vercel.app)  
⚙️ **API Docs:** [https://kaviyarose-femcare-backend.hf.space/docs](https://kaviyarose-femcare-backend.hf.space/docs)

---

## What is FemCare?

Endometriosis affects **1 in 10 women** worldwide, yet the average time from first symptom to diagnosis is **7 to 10 years**. Pain is frequently dismissed as normal period cramps, leading to years of unnecessary suffering.

FemCare is a free, private, 2-minute screening tool that analyses a patient's symptom pattern against data from 886 confirmed patient records. It gives a personalised risk score and identifies which symptoms are most significant — helping users have an informed conversation with their doctor.

> ⚠️ FemCare is a screening tool only — not a medical diagnosis. Always consult a qualified gynaecologist.

---

## Features

- 🩺 **14-symptom questionnaire** — covers menstrual, pelvic, bowel, and fatigue symptoms
- 📊 **ML risk prediction** — AdaBoost classifier trained on 886 patient records
- 🔍 **SHAP explainability** — shows which symptoms drove the result
- 🏷️ **Risk tiering** — Low / Moderate / High / Urgent with data-derived boundaries
- 👤 **Auth system** — signup, login, guest mode
- 📁 **History tracking** — logged-in users can view all past assessments
- 📱 **Responsive UI** — works on mobile and desktop

---

## Model Details

| Property | Value |
|---|---|
| Model | AdaBoostClassifier (Platt Scaling calibrated) |
| Training data | 886 patient records |
| Features | 14 clinical symptoms |
| Test AUC | 0.9535 |
| Cross-validated AUC | 0.9693 |
| Accuracy | 89.2% |
| Sensitivity | 90.8% |
| Specificity | 87.4% |
| F1 Score | 0.9000 |
| Explainability | SHAP (permutation-based) |

### Confusion Matrix — Test Set (222 patients)

|  | Predicted Endo | Predicted Healthy |
|---|---|---|
| **Actual Endo** | TP = 108 | FN = 11 |
| **Actual Healthy** | FP = 13 | TN = 90 |

---

## Risk Tiers

| Tier | Probability | What it means |
|---|---|---|
| 🟢 Low | < 6.5% | Symptoms do not strongly align with endometriosis. Continue monitoring. |
| 🟡 Moderate | 6.5% – 81.5% | Some symptoms present. Consider speaking with a doctor if they persist. |
| 🟠 High | 81.5% – 99.5% | Several key markers present. A gynaecological consultation is recommended. |
| 🔴 Urgent | ≥ 99.5% | Strong symptom pattern. Please seek a gynaecological assessment soon. |

> Tier boundaries are data-derived from the calibrated probability distribution of 886 patients.  
> The Low tier has a Negative Predictive Value of **96.8%** — clinically safe to recommend monitoring.

### Tier Justification
The thresholds were validated against our test set using three criteria:
1. Probability distributions of endo vs non-endo patients naturally separate at these boundaries
2. Endo rate increases monotonically across tiers: 3.6% → 39.0% → 96.4% → 100%
3. NPV of Low tier = 96.8%, exceeding the 94% clinical safety benchmark used in the SAFE score paper

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | React.js (Vite), single-file component |
| Backend | FastAPI (Python 3.11) |
| ML Model | scikit-learn — AdaBoostClassifier |
| Explainability | SHAP (permutation explainer) |
| Database | PostgreSQL (Supabase) |
| Auth | JWT + bcrypt (passlib) |
| Frontend hosting | Vercel |
| Backend hosting | Hugging Face Spaces (Docker) |

---

## Project Structure

```
FemCare/
│
├── main.py                          # FastAPI app — routes, auth, prediction, risk tiering
├── fix__tiers.py                    # Calibration + data-driven tier boundary generator
├── regenerate_explainer.py          # Regenerates SHAP explainer if model changes
├── femcare_evaluate.py              # Standalone script to test model metrics
│
├── femcare_model_final.pkl          # Production model — Calibrated AdaBoost
├── femcare_threshold_final.pkl      # Youden-optimal binary decision threshold
├── femcare_tiers_final.pkl          # Data-driven tier boundaries
├── Femcares_explainer_final.pkl     # SHAP permutation explainer
│
├── femcare_features.txt             # Single source of truth — 14 production feature names
├── femcare_db_query.sql             # Database schema
├── considerable dataset.xlsx        # Training dataset (886 patient records)
├── requirements.txt                 # Python dependencies
├── Dockerfile                       # Docker config for Hugging Face deployment
├── .env                             # Environment variables — never commit this
│
├── femcare-frontend/
│   ├── src/
│   │   ├── App.jsx                  # Entire frontend (single-file React app)
│   │   ├── main.jsx                 # React entry point
│   │   └── index.css                # Base styles
│   ├── index.html
│   ├── vite.config.js
│   └── package.json
│
└── plots/                           # Model evaluation charts
    ├── plot_calibration.png
    ├── plot_tier_fixed.png
    ├── plot_A_model_comparison.png
    ├── plot_B_roc_curves.png
    └── ...
```

---

## API Endpoints

| Method | Endpoint | Auth | Description |
|---|---|---|---|
| POST | `/signup` | None | Create a new account |
| POST | `/login` | None | Login and receive JWT token |
| POST | `/predict` | Optional | Run symptom prediction (works for guests too) |
| GET | `/history` | Required | Get all past assessments for logged-in user |

---

## Running Locally

### Backend
```bash
cd FemCare
pip install -r requirements.txt
uvicorn main:app --reload
```

### Frontend
```bash
cd femcare-frontend
npm install
npm run dev
```

Make sure your `.env` file has:
```
DB_HOST=your_host
DB_PORT=5432
DB_NAME=postgres
DB_USER=postgres
DB_PASSWORD=your_password
```

---

## Deployment

| Service | Platform |
|---|---|
| Frontend | Vercel (drag and drop `dist`) |
| Backend | Hugging Face Spaces (Docker) |
| Database | Supabase (free PostgreSQL) |

---

## Disclaimer

FemCare is a research and screening tool built for educational purposes. It is **not a substitute for medical advice, diagnosis, or treatment**. Always seek the guidance of a qualified gynaecologist or healthcare provider with any questions you may have regarding a medical condition.

---

## License

This project is for Educational and Research Purposes only.
