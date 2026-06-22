# ─────────────────────────────────────────────
# FemCare — FastAPI Backend
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

# Allow React frontend to talk to this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────
model    = joblib.load("femcare_model.pkl")
explainer = joblib.load("femcare_explainer.pkl")

with open("femcare_features.txt", "r") as f:
    FEATURES = [line.strip() for line in f if line.strip()]

def risk_tier(prob):
    if prob >= 0.80:   return "Urgent"
    elif prob >= 0.60: return "High"
    elif prob >= 0.40: return "Moderate"
    else:              return "Low"

# ─────────────────────────────────────────────
# DATABASE CONNECTION
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
# AUTH SETUP
# ─────────────────────────────────────────────
SECRET_KEY = "femcare-secret-key-change-this-in-production"
ALGORITHM  = "HS256"

pwd_context   = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

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
        return payload["sub"]   # returns user_id
    except:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

# ─────────────────────────────────────────────
# REQUEST / RESPONSE MODELS
# ─────────────────────────────────────────────
class SignupRequest(BaseModel):
    username : str
    email    : str
    password : str

class PredictRequest(BaseModel):
    # Required — self-reported symptoms
    menstrual_pain      : int
    abnormal_bleeding   : int
    pelvic_pain         : int
    fever               : int
    nausea              : int
    constipation        : int
    abdominal_cramps    : int
    irregular_periods   : int
    lower_back_pain     : int
    bloating            : int
    decreased_energy    : int
    diarrhea            : int
    # Optional — clinical
    infertility         : Optional[int] = 0
    cysts               : Optional[int] = 0

# ─────────────────────────────────────────────
# ENDPOINT 1 — SIGNUP
# POST /signup
# ─────────────────────────────────────────────
@app.post("/signup")
def signup(data: SignupRequest):
    db = get_db()
    cur = db.cursor()

    # Check if email already exists
    cur.execute("SELECT id FROM users WHERE email = %s", (data.email,))
    if cur.fetchone():
        raise HTTPException(status_code=400, detail="Email already registered")

    # Check if username already exists
    cur.execute("SELECT id FROM users WHERE username = %s", (data.username,))
    if cur.fetchone():
        raise HTTPException(status_code=400, detail="Username already taken")

    # Save new user
    user_id = str(uuid.uuid4())
    cur.execute(
        "INSERT INTO users (id, username, email, password_hash) VALUES (%s, %s, %s, %s)",
        (user_id, data.username, data.email, hash_password(data.password))
    )

    db.close()
    return {"message": "Account created successfully", "user_id": user_id}

# ─────────────────────────────────────────────
# ENDPOINT 2 — LOGIN
# POST /login
# ─────────────────────────────────────────────
@app.post("/login")
def login(form: OAuth2PasswordRequestForm = Depends()):
    db = get_db()
    cur = db.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    # Find user by username
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
# POST /predict  (requires login)
# ─────────────────────────────────────────────
@app.post("/predict")
def predict(data: PredictRequest, user_id: str = Depends(get_current_user)):

    # Map request fields to feature names
    symptom_inputs = {
        "Menstrual pain (Dysmenorrhea)"       : data.menstrual_pain,
        "Abnormal uterine bleeding"           : data.abnormal_bleeding,
        "Pelvic pain"                         : data.pelvic_pain,
        "Fever"                               : data.fever,
        "Infertility"                         : data.infertility,
        "Nausea"                              : data.nausea,
        "Constipation / Chronic constipation" : data.constipation,
        "Abdominal Cramps during Intercourse" : data.abdominal_cramps,
        "Irregular / Missed periods"          : data.irregular_periods,
        "Lower back pain"                     : data.lower_back_pain,
        "Bloating"                            : data.bloating,
        "Decreased energy / Exhaustion"       : data.decreased_energy,
        "Diarrhea"                            : data.diarrhea,
        "Cysts (unspecified)"                 : data.cysts,
    }

    # Run model
    input_df   = pd.DataFrame([symptom_inputs])[FEATURES]
    prob       = float(model.predict_proba(input_df)[0][1])
    tier       = risk_tier(prob)

    # Run SHAP
    raw = explainer(input_df).values
    if len(raw.shape) == 3:
        shap_vals = raw[0, :, 1]   # class 1 (endometriosis)
    else:
        shap_vals = raw[0]
    shap_dict  = {feat: round(float(val), 4) for feat, val in zip(FEATURES, shap_vals)}
    top5       = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:5]

    # Save to database
    db = get_db()
    cur = db.cursor()

    assessment_id = str(uuid.uuid4())
    cur.execute(
        """INSERT INTO assessments (id, user_id, risk_pct, risk_tier, symptom_inputs)
           VALUES (%s, %s, %s, %s, %s)""",
        (assessment_id, user_id, round(prob * 100, 1), tier, json.dumps(symptom_inputs))
    )

    # Save SHAP values
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
        "advice"        : {
            "Urgent"   : "Please consult a gynaecologist immediately.",
            "High"     : "We strongly recommend a gynaecological consultation.",
            "Moderate" : "Consider scheduling a check-up with your doctor.",
            "Low"      : "Low risk detected. Monitor your symptoms over time."
        }[tier]
    }

# ─────────────────────────────────────────────
# ENDPOINT 4 — HISTORY
# GET /history  (requires login)
# ─────────────────────────────────────────────
@app.get("/history")
def history(user_id: str = Depends(get_current_user)):
    db = get_db()
    cur = db.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    cur.execute(
        """SELECT id, taken_at, risk_pct, risk_tier, symptom_inputs
           FROM assessments
           WHERE user_id = %s
           ORDER BY taken_at DESC""",
        (user_id,)
    )
    rows = cur.fetchall()
    db.close()

    return {
        "total_assessments" : len(rows),
        "assessments"       : [
            {
                "assessment_id" : str(r["id"]),
                "taken_at"      : r["taken_at"].strftime("%Y-%m-%d %H:%M"),
                "risk_pct"      : r["risk_pct"],
                "risk_tier"     : r["risk_tier"],
                "symptoms"      : r["symptom_inputs"]
            }
            for r in rows
        ]
    }

# ─────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)