# ─────────────────────────────────────────────
# FemCare — FastAPI Backend (clean version)
#
# DESIGN RULE: femcare_features.txt is the single
# source of truth. PredictRequest field names are
# snake_case versions of those exact feature names.
# No mapping layer needed — they line up 1-to-1.
# ─────────────────────────────────────────────

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from typing import Optional
import psycopg2
import psycopg2.extras
import joblib
import pandas as pd
import numpy as np
import shap
import os
import uuid
import json
from datetime import datetime, timedelta
from dotenv import load_dotenv
from passlib.context import CryptContext
import jwt
import uvicorn

load_dotenv()

# ─────────────────────────────────────────────
# APP SETUP
# ─────────────────────────────────────────────
app = FastAPI(title="FemCare API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# LOAD MODEL
# femcare_features.txt is the source of truth.
# FEATURES list order must be preserved exactly.
# ─────────────────────────────────────────────
model     = joblib.load("femcare_model_final.pkl")
THRESHOLD = joblib.load("femcare_threshold_final.pkl")
tiers     = joblib.load("femcare_tiers_final.pkl")

try:
    explainer = joblib.load("Femcares_explainer_final.pkl")
except Exception as e:
    print("Explainer load failed:", e)
    explainer = None

with open("femcare_features.txt", "r") as f:
    FEATURES = [line.strip() for line in f if line.strip()]

# femcare_features.txt contains exactly:
#   Heavy / Extreme menstrual bleeding
#   Menstrual pain (Dysmenorrhea)
#   Pelvic pain
#   Painful bowel movements
#   Infertility
#   Painful cramps during period
#   Fatigue / Chronic fatigue
#   IBS-like symptoms
#   Excessive bleeding
#   Bowel pain
#   Cysts (unspecified)
#   Abnormal uterine bleeding
#   Fever
#   Loss of appetite

tiers = joblib.load("femcare_tiers_final.pkl")

def risk_tier(prob):
    if prob >= tiers['high_threshold']:        return "Urgent"
    elif prob >= tiers['moderate_threshold']:  return "High"
    elif prob >= tiers['low_threshold']:       return "Moderate"
    else:                                      return "Low"

# ─────────────────────────────────────────────
# DATABASE
# ─────────────────────────────────────────────
def get_db():
    conn = psycopg2.connect(
        host     = os.getenv("DB_HOST"),
        port     = os.getenv("DB_PORT"),
        dbname   = os.getenv("DB_NAME"),
        user     = os.getenv("DB_USER"),
        password = os.getenv("DB_PASSWORD")
    )
    conn.autocommit = True
    return conn

# ─────────────────────────────────────────────
# AUTH
# ─────────────────────────────────────────────
SECRET_KEY = "femcare-secret-key-change-this-in-production"
ALGORITHM  = "HS256"

pwd_context   = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")
# auto_error=False -> doesn't raise if no Authorization header is sent.
# This is what lets guests (no token) still hit /predict.
oauth2_scheme_optional = OAuth2PasswordBearer(tokenUrl="login", auto_error=False)

def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)

def create_token(user_id: str) -> str:
    payload = {
        "sub" : user_id,
        "exp" : datetime.utcnow() + timedelta(hours=24)
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload["sub"]
    except:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

def get_current_user_optional(token: str = Depends(oauth2_scheme_optional)):
    # Returns the user_id if a valid token was sent, otherwise None (guest).
    # Never raises — /predict stays usable for guests, but real model still runs.
    if not token:
        return None
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload["sub"]
    except Exception:
        return None

# ─────────────────────────────────────────────
# REQUEST MODELS
#
# Field names are snake_case of femcare_features.txt.
# No mapping dict needed in /predict — they match directly.
#
# femcare_features.txt line  →  PredictRequest field
# ─────────────────────────────────────────────────
# Heavy / Extreme menstrual bleeding  →  heavy_bleeding
# Menstrual pain (Dysmenorrhea)       →  menstrual_pain
# Pelvic pain                         →  pelvic_pain
# Painful bowel movements             →  painful_bowel
# Infertility                         →  infertility        (optional)
# Painful cramps during period        →  painful_cramps
# Fatigue / Chronic fatigue           →  fatigue
# IBS-like symptoms                   →  ibs_symptoms
# Excessive bleeding                  →  excessive_bleeding
# Bowel pain                          →  bowel_pain
# Cysts (unspecified)                 →  cysts              (optional)
# Abnormal uterine bleeding           →  abnormal_bleeding
# Fever                               →  fever
# Loss of appetite                    →  loss_appetite
# ─────────────────────────────────────────────
class SignupRequest(BaseModel):
    username : str
    email    : str
    password : str

class PredictRequest(BaseModel):
    # Required — 12 self-reported symptoms
    heavy_bleeding     : int   # Heavy / Extreme menstrual bleeding
    menstrual_pain     : int   # Menstrual pain (Dysmenorrhea)
    pelvic_pain        : int   # Pelvic pain
    painful_bowel      : int   # Painful bowel movements
    painful_cramps     : int   # Painful cramps during period
    fatigue            : int   # Fatigue / Chronic fatigue
    ibs_symptoms       : int   # IBS-like symptoms
    excessive_bleeding : int   # Excessive bleeding
    bowel_pain         : int   # Bowel pain
    abnormal_bleeding  : int   # Abnormal uterine bleeding
    fever              : int   # Fever
    loss_appetite      : int   # Loss of appetite
    # Optional — 2 clinical fields (default 0 if not answered)
    infertility        : Optional[int] = 0   # Infertility
    cysts              : Optional[int] = 0   # Cysts (unspecified)

# ─────────────────────────────────────────────
# ENDPOINT 1 — SIGNUP
# ─────────────────────────────────────────────
@app.post("/signup")
def signup(data: SignupRequest):
    db  = get_db()
    cur = db.cursor()

    cur.execute("SELECT id FROM users WHERE email = %s", (data.email,))
    if cur.fetchone():
        raise HTTPException(status_code=400, detail="Email already registered")

    cur.execute("SELECT id FROM users WHERE username = %s", (data.username,))
    if cur.fetchone():
        raise HTTPException(status_code=400, detail="Username already taken")

    user_id = str(uuid.uuid4())
    cur.execute(
        "INSERT INTO users (id, username, email, password_hash) VALUES (%s, %s, %s, %s)",
        (user_id, data.username, data.email, hash_password(data.password))
    )
    db.close()
    return {"message": "Account created successfully", "user_id": user_id}

# ─────────────────────────────────────────────
# ENDPOINT 2 — LOGIN
# ─────────────────────────────────────────────
@app.post("/login")
def login(form: OAuth2PasswordRequestForm = Depends()):
    db  = get_db()
    cur = db.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    cur.execute("SELECT * FROM users WHERE username = %s", (form.username,))
    user = cur.fetchone()
    db.close()

    if not user or not verify_password(form.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Incorrect username or password")

    token = create_token(str(user["id"]))
    return {
        "access_token" : token,
        "token_type"   : "bearer",
        "username"     : user["username"]
    }

# ─────────────────────────────────────────────
# ENDPOINT 3 — PREDICT
#
# No confusing mapping layer.
# PredictRequest fields line up directly with
# femcare_features.txt — just build the dict.
# ─────────────────────────────────────────────
@app.post("/predict")
def predict(data: PredictRequest, user_id: str = Depends(get_current_user_optional)):
    # user_id is None for guests (no/invalid token) — the real model and
    # SHAP explanation still run for everyone; only the DB save is skipped.
    is_guest = user_id is None

    # Build input using exact feature names from femcare_features.txt
    # Field name on left = feature name the model knows
    # data.x on right    = what the frontend sent, same meaning
    symptom_inputs = {
        "Heavy / Extreme menstrual bleeding" : data.heavy_bleeding,
        "Menstrual pain (Dysmenorrhea)"      : data.menstrual_pain,
        "Pelvic pain"                        : data.pelvic_pain,
        "Painful bowel movements"            : data.painful_bowel,
        "Infertility"                        : data.infertility,
        "Painful cramps during period"       : data.painful_cramps,
        "Fatigue / Chronic fatigue"          : data.fatigue,
        "IBS-like symptoms"                  : data.ibs_symptoms,
        "Excessive bleeding"                 : data.excessive_bleeding,
        "Bowel pain"                         : data.bowel_pain,
        "Cysts (unspecified)"                : data.cysts,
        "Abnormal uterine bleeding"          : data.abnormal_bleeding,
        "Fever"                              : data.fever,
        "Loss of appetite"                   : data.loss_appetite,
    }

    # Run model
    input_df = pd.DataFrame([symptom_inputs])[FEATURES]
    prob     = float(model.predict_proba(input_df)[0][1])
    prediction = int(prob >= THRESHOLD) 
    tier     = risk_tier(prob)

    # Run SHAP
    try:
        raw = explainer(input_df).values
    except Exception:
    # CalibratedClassifierCV wraps the base estimator
    # get feature importances from the inner AdaBoost
        base = model.calibrated_classifiers_[0].estimator
        raw = np.array(base.feature_importances_).reshape(1, -1)
    if len(raw.shape) == 3:
        shap_vals = raw[0, :, 1]   # class 1 = endometriosis
    else:
        shap_vals = raw[0]
    shap_dict = {feat: round(float(val), 4) for feat, val in zip(FEATURES, shap_vals)}
    top5      = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:5]

    # Save to database — only for logged-in users. Guests get the
    # real model + SHAP result, but nothing is persisted for them.
    if is_guest:
        assessment_id = None
    else:
        db  = get_db()
        cur = db.cursor()

        assessment_id = str(uuid.uuid4())
        cur.execute(
            """INSERT INTO assessments (id, user_id, risk_pct, risk_tier, symptom_inputs)
               VALUES (%s, %s, %s, %s, %s)""",
            (assessment_id, user_id, round(prob * 100, 1), tier, json.dumps(symptom_inputs))
        )
        for feat, val in shap_dict.items():
            cur.execute(
                """INSERT INTO shap_explanations (assessment_id, feature_name, shap_value)
                   VALUES (%s, %s, %s)""",
                (assessment_id, feat, val)
            )
        db.close()

    return {
        "assessment_id" : assessment_id,
        "risk_pct"      : round(prob * 100, 1),
        "risk_tier"     : tier,
        "top_features"  : [{"feature": f, "shap": round(v * 100, 1)} for f, v in top5],
        "guest"         : is_guest,
        "advice"        : {
            "Urgent"   : "Your symptom pattern suggests a high likelihood of endometriosis. Please consult a gynaecologist as soon as possible.",
            "High"     : "Several key endometriosis markers are present. We strongly recommend scheduling a gynaecological consultation soon.",
            "Moderate" : "Some symptoms align with endometriosis. Consider speaking with a doctor if they persist or worsen.",
            "Low"      : "Your symptoms don't strongly indicate endometriosis right now. Continue monitoring your cycle."
        }[tier]
    }

# ─────────────────────────────────────────────
# ENDPOINT 4 — HISTORY
# ─────────────────────────────────────────────
@app.get("/history")
def history(user_id: str = Depends(get_current_user)):
    db  = get_db()
    cur = db.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    cur.execute(
        """SELECT id, taken_at, risk_pct, risk_tier, symptom_inputs
           FROM assessments WHERE user_id = %s ORDER BY taken_at DESC""",
        (user_id,)
    )
    rows = cur.fetchall()

    assessments = []
    for r in rows:
        # Pull this assessment's top 5 SHAP drivers (same shape as /predict's top_features)
        cur.execute(
            """SELECT feature_name, shap_value FROM shap_explanations
               WHERE assessment_id = %s
               ORDER BY ABS(shap_value) DESC LIMIT 5""",
            (r["id"],)
        )
        shap_rows = cur.fetchall()
        assessments.append({
            "assessment_id" : str(r["id"]),
            "taken_at"      : r["taken_at"].strftime("%Y-%m-%d %H:%M"),
            "risk_pct"      : r["risk_pct"],
            "risk_tier"     : r["risk_tier"],
            "symptoms"      : r["symptom_inputs"],
            "top_features"  : [
                {"feature": s["feature_name"], "shap": round(s["shap_value"] * 100, 1)}
                for s in shap_rows
            ],
        })

    db.close()

    return {
        "total_assessments" : len(assessments),
        "assessments" : assessments
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)