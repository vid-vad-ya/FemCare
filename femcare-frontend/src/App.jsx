import { useState, useEffect, useRef } from "react";

const API_BASE = "http://localhost:8000";

const STANDARD_QS = [
  { id: "heavy_bleeding",     feature: "Heavy / Extreme menstrual bleeding", text: "Do you experience heavy or extreme menstrual bleeding?" },
  { id: "menstrual_pain",     feature: "Menstrual pain (Dysmenorrhea)",      text: "Do you experience painful periods (dysmenorrhea)?" },
  { id: "pelvic_pain",        feature: "Pelvic pain",                        text: "Do you have pelvic pain outside of your period?" },
  { id: "painful_bowel",      feature: "Painful bowel movements",            text: "Do you have painful bowel movements, especially during your period?" },
  { id: "painful_cramps",     feature: "Painful cramps during period",       text: "Do you have painful cramps during your period?" },
  { id: "fatigue",            feature: "Fatigue / Chronic fatigue",          text: "Do you experience fatigue or chronic fatigue regularly?" },
  { id: "ibs_symptoms",       feature: "IBS-like symptoms",                  text: "Do you have IBS-like symptoms (bloating, cramps, irregular bowel)?" },
  { id: "excessive_bleeding", feature: "Excessive bleeding",                 text: "Do you experience excessive bleeding beyond your normal period?" },
  { id: "bowel_pain",         feature: "Bowel pain",                        text: "Do you have bowel pain unrelated to food?" },
  { id: "abnormal_bleeding",  feature: "Abnormal uterine bleeding",          text: "Do you have abnormal uterine bleeding (spotting between periods)?" },
  { id: "fever",              feature: "Fever",                              text: "Do you get unexplained low-grade fevers?" },
  { id: "loss_appetite",      feature: "Loss of appetite",                  text: "Do you experience unexplained loss of appetite?" },
];

const OPTIONAL_QS = [
  { id: "cysts",       feature: "Cysts (unspecified)", text: "Have you ever been told you have cysts? (requires prior ultrasound scan)" },
  { id: "infertility", feature: "Infertility",          text: "Have you been diagnosed with infertility issues? (requires clinical diagnosis)" },
];

async function apiSignup(username, email, password) {
  const res = await fetch(`${API_BASE}/signup`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username, email, password }),
  });
  if (!res.ok) { const e = await res.json(); throw new Error(e.detail || "Signup failed"); }
  return res.json();
}

async function apiLogin(username, password) {
  const form = new URLSearchParams();
  form.append("username", username);
  form.append("password", password);
  const res = await fetch(`${API_BASE}/login`, {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body: form.toString(),
  });
  if (!res.ok) { const e = await res.json(); throw new Error(e.detail || "Login failed"); }
  return res.json();
}

async function apiHistory(token) {
  const res = await fetch(`${API_BASE}/history`, {
    headers: { "Authorization": `Bearer ${token}` },
  });
  if (!res.ok) { const e = await res.json(); throw new Error(e.detail || "Failed to load history"); }
  return res.json();
}

async function apiPredict(answers, token) {
  const body = {
    heavy_bleeding:     answers.heavy_bleeding,
    menstrual_pain:     answers.menstrual_pain,
    pelvic_pain:        answers.pelvic_pain,
    painful_bowel:      answers.painful_bowel,
    painful_cramps:     answers.painful_cramps,
    fatigue:            answers.fatigue,
    ibs_symptoms:       answers.ibs_symptoms,
    excessive_bleeding: answers.excessive_bleeding,
    bowel_pain:         answers.bowel_pain,
    abnormal_bleeding:  answers.abnormal_bleeding,
    fever:              answers.fever,
    loss_appetite:      answers.loss_appetite,
    infertility:        answers.infertility ?? 0,
    cysts:              answers.cysts        ?? 0,
  };
  const headers = { "Content-Type": "application/json" };
  if (token) headers["Authorization"] = `Bearer ${token}`;
  const res = await fetch(`${API_BASE}/predict`, {
    method: "POST",
    headers,
    body: JSON.stringify(body),
  });
  if (!res.ok) { const e = await res.json(); throw new Error(e.detail || "Prediction failed"); }
  return res.json();
}

const TIER = {
  Urgent:   { color: "#be185d", glow: "#be185d40", bg: "#fff1f6", border: "#fda4af", label: "Urgent" },
  High:     { color: "#9333ea", glow: "#9333ea40", bg: "#faf5ff", border: "#d8b4fe", label: "High" },
  Moderate: { color: "#7c3aed", glow: "#7c3aed40", bg: "#f5f3ff", border: "#c4b5fd", label: "Moderate" },
  Low:      { color: "#059669", glow: "#05966940", bg: "#ecfdf5", border: "#6ee7b7", label: "Low" },
};

const css = `
  @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,400&family=DM+Serif+Display:ital@0;1&display=swap');

  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --rose: #e11d48;
    --plum: #7c3aed;
    --soft-plum: #a855f7;
    --cream: #fdf8f5;
    --ink: #1c0a2e;
    --muted: #7c6d8a;
    --border: #ede5f5;
    --card: rgba(255,255,255,0.90);
    --grad: linear-gradient(135deg, #7c3aed 0%, #e11d48 100%);
  }

  body {
    font-family: 'DM Sans', sans-serif;
    background: var(--cream);
    color: var(--ink);
    min-height: 100vh;
    -webkit-font-smoothing: antialiased;
  }

  /* ── TOP NAV ── */
  .nav {
    position: sticky; top: 0; z-index: 100;
    display: flex; align-items: center; justify-content: space-between;
    padding: 0 40px; height: 64px;
    background: rgba(253,248,245,0.88);
    backdrop-filter: blur(18px);
    border-bottom: 1px solid var(--border);
  }
  .nav-logo {
    font-family: 'DM Serif Display', serif;
    font-size: 22px;
    background: var(--grad);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    cursor: pointer;
    display: flex; align-items: center; gap: 8px;
  }
  .nav-right { display: flex; align-items: center; gap: 10px; }
  .nav-btn {
    padding: 7px 18px; border-radius: 100px;
    font-size: 13px; font-weight: 600; font-family: 'DM Sans', sans-serif;
    cursor: pointer; transition: all 0.18s; border: none;
  }
  .nav-btn-outline {
    background: transparent; border: 1.5px solid var(--border);
    color: var(--plum);
  }
  .nav-btn-outline:hover { background: #f3eeff; border-color: #c4b5fd; }
  .nav-btn-fill {
    background: var(--grad); color: white;
    box-shadow: 0 2px 12px #7c3aed30;
  }
  .nav-btn-fill:hover { opacity: 0.9; transform: translateY(-1px); box-shadow: 0 4px 18px #7c3aed40; }
  .user-chip {
    display: flex; align-items: center; gap: 8px;
    padding: 5px 14px 5px 5px;
    background: #f5f0ff; border: 1px solid var(--border);
    border-radius: 100px; font-size: 13px; color: var(--plum); font-weight: 500;
  }
  .avatar {
    width: 28px; height: 28px;
    background: var(--grad); border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 11px; font-weight: 800; color: white;
  }
  .nav-link {
    background: none; border: none; color: var(--plum);
    font-size: 13px; font-weight: 600; cursor: pointer;
    font-family: 'DM Sans', sans-serif;
    padding: 7px 14px; border-radius: 8px; transition: background 0.15s;
  }
  .nav-link:hover { background: #f3eeff; }

  /* ── HOME HERO ── */
  .home-hero {
    background: linear-gradient(160deg, #fdf8f5 0%, #f5f0ff 45%, #fff1f6 100%);
    padding: 80px 40px 64px;
    text-align: center;
    position: relative; overflow: hidden;
  }
  .home-hero::before {
    content: '';
    position: absolute; inset: 0;
    background:
      radial-gradient(ellipse 60% 50% at 20% 20%, #ddd6fe40, transparent),
      radial-gradient(ellipse 50% 40% at 80% 80%, #fce7f340, transparent);
    pointer-events: none;
  }
  .hero-eyebrow {
    display: inline-flex; align-items: center; gap: 7px;
    background: #f5f0ff; border: 1px solid #ddd6fe;
    color: var(--plum); font-size: 12px; font-weight: 700;
    letter-spacing: 0.8px; text-transform: uppercase;
    padding: 5px 14px; border-radius: 100px; margin-bottom: 28px;
  }
  .hero-dot {
    width: 6px; height: 6px; border-radius: 50%;
    background: var(--grad); display: inline-block;
    animation: pulse-dot 2s ease-in-out infinite;
  }
  @keyframes pulse-dot { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:0.5;transform:scale(0.7)} }
  .hero-h1 {
    font-family: 'DM Serif Display', serif;
    font-size: clamp(38px, 6vw, 68px);
    line-height: 1.1; color: var(--ink);
    margin-bottom: 22px; position: relative;
  }
  .hero-h1 em {
    font-style: italic;
    background: var(--grad);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  }
  .hero-p {
    font-size: 17px; color: var(--muted); line-height: 1.75;
    max-width: 560px; margin: 0 auto 36px;
  }
  .hero-cta-row {
    display: flex; align-items: center; justify-content: center; gap: 12px;
    flex-wrap: wrap; margin-bottom: 48px;
  }
  .cta-primary {
    display: inline-flex; align-items: center; gap: 8px;
    padding: 14px 32px; border-radius: 100px;
    background: var(--grad); color: white;
    font-size: 16px; font-weight: 700; font-family: 'DM Sans', sans-serif;
    border: none; cursor: pointer; transition: all 0.2s;
    box-shadow: 0 4px 20px #7c3aed30;
  }
  .cta-primary:hover { transform: translateY(-2px); box-shadow: 0 8px 30px #7c3aed40; }
  .cta-secondary {
    padding: 14px 28px; border-radius: 100px;
    background: transparent; border: 1.5px solid #c4b5fd;
    color: var(--plum); font-size: 15px; font-weight: 600;
    font-family: 'DM Sans', sans-serif; cursor: pointer; transition: all 0.18s;
  }
  .cta-secondary:hover { background: #f3eeff; }
  .hero-stats {
    display: flex; align-items: center; justify-content: center; gap: 40px;
    flex-wrap: wrap;
  }
  .hero-stat { text-align: center; }
  .hero-stat-num {
    font-family: 'DM Serif Display', serif;
    font-size: 28px; color: var(--ink);
  }
  .hero-stat-num span { background: var(--grad); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
  .hero-stat-lbl { font-size: 12px; color: var(--muted); margin-top: 2px; font-weight: 500; }
  .stat-divider { width: 1px; height: 32px; background: var(--border); }

  /* ── HOW IT WORKS ── */
  .how-section {
    padding: 80px 40px;
    max-width: 960px; margin: 0 auto;
  }
  .section-eyebrow {
    font-size: 11px; font-weight: 800; letter-spacing: 1.5px;
    text-transform: uppercase; color: var(--soft-plum);
    margin-bottom: 12px;
  }
  .section-title {
    font-family: 'DM Serif Display', serif;
    font-size: clamp(26px, 4vw, 40px);
    line-height: 1.2; color: var(--ink); margin-bottom: 14px;
  }
  .section-sub {
    font-size: 15px; color: var(--muted); line-height: 1.7;
    max-width: 500px; margin-bottom: 52px;
  }
  .steps-grid {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
    gap: 24px;
  }
  .step-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 20px; padding: 28px 24px;
    position: relative; overflow: hidden;
    transition: all 0.2s;
  }
  .step-card::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 3px;
    background: var(--grad);
    opacity: 0; transition: opacity 0.2s;
  }
  .step-card:hover { transform: translateY(-3px); box-shadow: 0 12px 32px #7c3aed14; }
  .step-card:hover::before { opacity: 1; }
  .step-num {
    font-family: 'DM Serif Display', serif;
    font-size: 48px; line-height: 1;
    background: var(--grad); -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 16px; opacity: 0.25;
  }
  .step-icon { font-size: 28px; margin-bottom: 14px; display: block; }
  .step-h { font-size: 16px; font-weight: 700; color: var(--ink); margin-bottom: 8px; }
  .step-p { font-size: 13.5px; color: var(--muted); line-height: 1.65; }

  /* ── QUESTIONNAIRE SECTION (home inline) ── */
  .q-section {
    background: linear-gradient(180deg, #f5f0ff 0%, var(--cream) 100%);
    padding: 72px 40px 80px;
  }
  .q-inner { max-width: 680px; margin: 0 auto; }
  .q-header { text-align: center; margin-bottom: 48px; }
  .q-header .section-sub { margin: 0 auto; }
  .prog-row {
    display: flex; align-items: center; justify-content: space-between;
    margin-bottom: 8px;
  }
  .prog-lbl { font-size: 12px; font-weight: 600; color: var(--muted); }
  .prog-count { font-size: 12px; font-weight: 700; color: var(--plum); }
  .prog-track {
    background: #e9d5ff; border-radius: 100px; height: 5px;
    overflow: hidden; margin-bottom: 36px;
  }
  .prog-fill {
    height: 100%; border-radius: 100px;
    background: var(--grad); transition: width 0.35s ease;
  }
  .q-group-label {
    font-size: 10px; font-weight: 800; letter-spacing: 1.8px;
    text-transform: uppercase; color: #a78bca;
    margin-bottom: 14px; margin-top: 32px;
    display: flex; align-items: center; gap: 10px;
  }
  .q-group-label::after { content: ''; flex: 1; height: 1px; background: #e9d5ff; }
  .q-card {
    background: white;
    border: 1.5px solid #ede9fe;
    border-radius: 14px; padding: 16px 18px;
    margin-bottom: 10px;
    display: flex; align-items: center; gap: 14px;
    transition: all 0.15s;
    cursor: default;
  }
  .q-card:hover { border-color: #c4b5fd; box-shadow: 0 2px 12px #c084fc15; }
  .q-card.answered-yes { border-color: #f9a8d4; background: #fff5f9; }
  .q-card.answered-no  { border-color: #6ee7b7; background: #f0fdf9; }
  .q-text { flex: 1; font-size: 14px; color: #1c0a2e; line-height: 1.5; font-weight: 450; }
  .opt-badge {
    font-size: 9px; font-weight: 800; letter-spacing: 0.5px; text-transform: uppercase;
    background: #d1fae5; color: #065f46; border: 1px solid #a7f3d0;
    border-radius: 100px; padding: 2px 8px; white-space: nowrap; flex-shrink: 0;
  }
  .yn-wrap { display: flex; gap: 5px; flex-shrink: 0; }
  .yn {
    width: 40px; height: 32px; border-radius: 9px;
    border: 1.5px solid #ddd6fe; background: #faf5ff;
    color: #a78bca; font-size: 12px; font-weight: 800;
    cursor: pointer; font-family: 'DM Sans', sans-serif;
    transition: all 0.15s;
    display: flex; align-items: center; justify-content: center;
  }
  .yn:hover { border-color: #c084fc; background: #f3e8ff; }
  .yn.y-active { background: #e11d48; border-color: #e11d48; color: white; box-shadow: 0 2px 8px #e11d4840; }
  .yn.n-active { background: #059669; border-color: #059669; color: white; box-shadow: 0 2px 8px #05966940; }
  .q-optional-note {
    font-size: 12px; color: var(--muted); line-height: 1.65;
    margin-bottom: 14px; margin-top: -10px;
    padding: 10px 14px; background: #f5f0ff;
    border-radius: 10px; border: 1px solid #e9d5ff;
  }
  .submit-area { margin-top: 36px; text-align: center; }
  .submit-note { font-size: 11.5px; color: #a78bca; margin-top: 12px; line-height: 1.6; }
  .err-box {
    background: #fff1f5; border: 1px solid #fda4af;
    border-radius: 10px; padding: 12px 16px;
    font-size: 13px; color: #be185d; margin-bottom: 18px;
  }

  /* ── LOADING ── */
  .loading-page {
    min-height: 60vh; display: flex; flex-direction: column;
    align-items: center; justify-content: center; gap: 18px; padding: 60px 20px;
  }
  .spinner {
    width: 48px; height: 48px; border-radius: 50%;
    border: 3px solid #ddd6fe; border-top-color: var(--plum);
    animation: spin 0.75s linear infinite;
  }
  @keyframes spin { to { transform: rotate(360deg); } }
  .loading-txt { font-size: 14px; color: var(--muted); }

  /* ── RESULTS PAGE ── */
  .results-page { max-width: 680px; margin: 0 auto; padding: 52px 28px 100px; }
  .results-hero { text-align: center; margin-bottom: 40px; }
  .score-ring {
    width: 180px; height: 180px; border-radius: 50%;
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    margin: 0 auto 20px; position: relative;
    transition: box-shadow 0.3s;
  }
  .ring-orbit {
    position: absolute; inset: -12px; border-radius: 50%;
    border: 1.5px dashed; opacity: 0.3;
    animation: spin 12s linear infinite;
  }
  .score-num {
    font-family: 'DM Serif Display', serif;
    font-size: 52px; line-height: 1;
  }
  .score-unit { font-size: 11px; font-weight: 700; letter-spacing: 1.5px; text-transform: uppercase; opacity: 0.6; margin-top: 4px; }
  .tier-badge {
    display: inline-flex; align-items: center; gap: 8px;
    padding: 8px 22px; border-radius: 100px;
    font-size: 13px; font-weight: 700; letter-spacing: 0.5px;
    text-transform: uppercase;
  }
  .tier-dot { width: 8px; height: 8px; border-radius: 50%; }
  .results-guest-note {
    font-size: 12px; color: var(--muted); margin-top: 10px;
    padding: 8px 16px; background: #f5f0ff; border-radius: 8px;
    border: 1px solid #ddd6fe; display: inline-block; margin-top: 12px;
  }
  .advice-card {
    border-radius: 16px; padding: 22px 24px;
    margin-bottom: 28px; border-left: 3px solid;
  }
  .advice-eyebrow {
    font-size: 10px; font-weight: 800; letter-spacing: 1px;
    text-transform: uppercase; opacity: 0.55; margin-bottom: 8px;
  }
  .advice-text { font-size: 14.5px; line-height: 1.75; }
  .drivers-section { margin-bottom: 28px; }
  .drivers-label {
    font-size: 11px; font-weight: 800; letter-spacing: 1px;
    text-transform: uppercase; color: var(--muted); margin-bottom: 14px;
  }
  .driver-row { display: flex; align-items: center; gap: 10px; margin-bottom: 10px; }
  .d-rank {
    width: 24px; height: 24px; border-radius: 7px;
    background: #ede9fe; color: var(--plum);
    font-size: 10px; font-weight: 800;
    display: flex; align-items: center; justify-content: center; flex-shrink: 0;
  }
  .d-name { flex: 1; font-size: 13px; color: var(--ink); font-weight: 500; }
  .d-bar-bg { width: 100px; background: #ede9fe; border-radius: 100px; height: 5px; overflow: hidden; flex-shrink: 0; }
  .d-bar-fill { height: 100%; background: var(--grad); border-radius: 100px; }
  .d-val { font-size: 11px; font-family: monospace; color: #db2777; min-width: 40px; text-align: right; flex-shrink: 0; }
  .metrics-row {
    display: grid; grid-template-columns: repeat(2, 1fr); gap: 12px; margin-bottom: 28px;
  }
  .metric-tile {
    background: white; border: 1px solid var(--border);
    border-radius: 14px; padding: 18px 20px;
  }
  .m-val {
    font-family: 'DM Serif Display', serif; font-size: 26px;
    background: var(--grad); -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    line-height: 1;
  }
  .m-lbl { font-size: 11px; color: var(--muted); margin-top: 5px; text-transform: uppercase; letter-spacing: 0.5px; font-weight: 600; }
  .model-card {
    background: #f0fdf9; border: 1px solid #a7f3d0;
    border-radius: 14px; padding: 14px 18px;
    font-size: 12.5px; color: #065f46; line-height: 1.75; margin-bottom: 18px;
  }
  .disclaimer-card {
    background: white; border: 1px solid var(--border);
    border-radius: 14px; padding: 14px 18px;
    font-size: 12.5px; color: var(--muted); line-height: 1.75; margin-bottom: 28px;
  }

  /* ── RESULTS ACTIONS ── */
  .results-actions {
    background: white; border: 1px solid var(--border);
    border-radius: 20px; padding: 28px;
    margin-bottom: 16px; text-align: center;
  }
  .results-actions-title {
    font-family: 'DM Serif Display', serif; font-size: 20px;
    color: var(--ink); margin-bottom: 8px;
  }
  .results-actions-sub {
    font-size: 13px; color: var(--muted); margin-bottom: 22px; line-height: 1.6;
  }
  .results-actions-btns { display: flex; gap: 10px; flex-wrap: wrap; justify-content: center; }
  .btn-save {
    flex: 1; min-width: 160px;
    padding: 13px 24px; border-radius: 100px;
    background: var(--grad); color: white;
    font-size: 14px; font-weight: 700; font-family: 'DM Sans', sans-serif;
    border: none; cursor: pointer; transition: all 0.2s;
    box-shadow: 0 2px 14px #7c3aed30;
  }
  .btn-save:hover { opacity: 0.9; transform: translateY(-1px); }
  .btn-retake {
    flex: 1; min-width: 160px;
    padding: 13px 24px; border-radius: 100px;
    background: transparent; border: 1.5px solid #c4b5fd;
    color: var(--plum); font-size: 14px; font-weight: 600;
    font-family: 'DM Sans', sans-serif; cursor: pointer; transition: all 0.18s;
  }
  .btn-retake:hover { background: #f3eeff; }
  .btn-exit {
    flex: 1; min-width: 160px;
    padding: 13px 24px; border-radius: 100px;
    background: transparent; border: 1.5px solid #fda4af;
    color: #be185d; font-size: 14px; font-weight: 600;
    font-family: 'DM Sans', sans-serif; cursor: pointer; transition: all 0.18s;
  }
  .btn-exit:hover { background: #fff1f5; }

  /* ── AUTH MODAL ── */
  .modal-backdrop {
    position: fixed; inset: 0; z-index: 200;
    background: rgba(0,0,0,0.45); backdrop-filter: blur(6px);
    display: flex; align-items: center; justify-content: center; padding: 20px;
    animation: fadeIn 0.18s ease;
  }
  @keyframes fadeIn { from{opacity:0} to{opacity:1} }
  .modal-card {
    width: 100%; max-width: 420px;
    background: white; border-radius: 24px;
    padding: 40px 36px; box-shadow: 0 20px 60px #0000002a;
    animation: slideUp 0.22s ease;
  }
  @keyframes slideUp { from{opacity:0;transform:translateY(20px)} to{opacity:1;transform:translateY(0)} }
  .modal-close {
    position: absolute; right: 0; top: 0; /* handled by parent relative */
    /* use a floating X button */
  }
  .modal-brand { text-align: center; margin-bottom: 28px; }
  .modal-icon {
    width: 56px; height: 56px;
    background: var(--grad); border-radius: 18px;
    display: flex; align-items: center; justify-content: center;
    font-size: 28px; margin: 0 auto 14px;
    box-shadow: 0 4px 20px #7c3aed30;
  }
  .modal-title {
    font-family: 'DM Serif Display', serif; font-size: 26px;
    color: var(--ink);
  }
  .modal-sub { font-size: 13px; color: var(--muted); margin-top: 4px; }
  .auth-tabs {
    display: flex; background: #f3eeff; border-radius: 12px;
    padding: 4px; gap: 4px; margin-bottom: 24px;
  }
  .auth-tab {
    flex: 1; padding: 9px; border: none; border-radius: 9px;
    font-size: 13px; font-weight: 600; cursor: pointer;
    font-family: 'DM Sans', sans-serif; transition: all 0.18s;
    color: var(--plum); background: transparent;
  }
  .auth-tab.active {
    background: var(--grad); color: white;
    box-shadow: 0 2px 10px #7c3aed30;
  }
  .field-lbl {
    display: block; font-size: 11px; font-weight: 800;
    letter-spacing: 0.8px; text-transform: uppercase;
    color: var(--muted); margin-bottom: 6px;
  }
  .field-inp {
    width: 100%; background: #faf5ff; border: 1.5px solid #ede9fe;
    border-radius: 10px; padding: 11px 14px; color: var(--ink);
    font-size: 14px; font-family: 'DM Sans', sans-serif;
    outline: none; transition: all 0.2s; margin-bottom: 14px;
  }
  .field-inp:focus { border-color: var(--plum); box-shadow: 0 0 0 3px #7c3aed18; }
  .field-inp::placeholder { color: #c4b5d4; }
  .btn-modal-main {
    width: 100%; padding: 13px; border-radius: 12px;
    background: var(--grad); color: white;
    font-size: 15px; font-weight: 700; font-family: 'DM Sans', sans-serif;
    border: none; cursor: pointer; transition: all 0.2s;
    box-shadow: 0 4px 18px #7c3aed30; margin-bottom: 12px;
  }
  .btn-modal-main:hover { opacity: 0.9; transform: translateY(-1px); }
  .btn-modal-main:disabled { opacity: 0.4; cursor: not-allowed; transform: none; }
  .btn-modal-ghost {
    width: 100%; padding: 12px; border-radius: 12px;
    background: transparent; border: 1.5px solid #ddd6fe;
    color: var(--plum); font-size: 13px; font-weight: 600;
    font-family: 'DM Sans', sans-serif; cursor: pointer; transition: all 0.18s;
  }
  .btn-modal-ghost:hover { background: #f3eeff; }
  .modal-or {
    display: flex; align-items: center; gap: 10px;
    color: #c4b5d4; font-size: 12px; margin: 10px 0;
  }
  .modal-or::before,.modal-or::after { content:''; flex:1; height:1px; background:#ede9fe; }
  .modal-note { font-size: 11px; color: var(--muted); text-align: center; line-height: 1.65; margin-top: 16px; }

  /* ── HISTORY PAGE ── */
  .hist-page { max-width: 680px; margin: 0 auto; padding: 52px 28px 100px; }
  .hist-back {
    display: inline-flex; align-items: center; gap: 6px;
    background: none; border: none; color: var(--plum);
    font-size: 13px; font-weight: 600; cursor: pointer;
    font-family: 'DM Sans', sans-serif; margin-bottom: 28px;
    padding: 6px 0;
  }
  .hist-title { font-family: 'DM Serif Display', serif; font-size: 32px; color: var(--ink); margin-bottom: 6px; }
  .hist-sub { font-size: 13px; color: var(--muted); margin-bottom: 32px; }
  .hist-empty { text-align: center; padding: 60px 20px; color: var(--muted); font-size: 14px; }
  .hist-card {
    background: white; border: 1.5px solid var(--border);
    border-radius: 18px; padding: 20px 22px; margin-bottom: 14px;
    cursor: pointer; transition: all 0.18s;
    display: flex; align-items: center; gap: 16px;
  }
  .hist-card:hover { border-color: #c4b5fd; box-shadow: 0 6px 20px #c084fc18; transform: translateY(-2px); }
  .hist-score-block { text-align: center; min-width: 64px; }
  .hist-pct { font-family: 'DM Serif Display', serif; font-size: 30px; line-height: 1; }
  .hist-info { flex: 1; }
  .hist-date { font-size: 12px; color: var(--muted); font-weight: 600; margin-bottom: 6px; }
  .hist-tier-pill {
    display: inline-flex; align-items: center; gap: 5px;
    padding: 3px 12px; border-radius: 100px;
    font-size: 11px; font-weight: 700; letter-spacing: 0.3px;
  }
  .hist-chev { color: #c4b5d4; font-size: 20px; }

  /* ── FADE ANIM ── */
  .fade-up { animation: fadeUpAnim 0.4s ease forwards; }
  @keyframes fadeUpAnim { from{opacity:0;transform:translateY(16px)} to{opacity:1;transform:translateY(0)} }

  /* ── ENDOMETRIOSIS INFO SECTION ── */
  .endo-section {
    background: white;
    border-top: 1px solid var(--border);
    border-bottom: 1px solid var(--border);
    padding: 80px 40px;
  }
  .endo-inner {
    max-width: 1080px; margin: 0 auto;
    display: grid; grid-template-columns: 1fr 1fr; gap: 64px; align-items: start;
  }
  .endo-body {
    font-size: 15px; color: #3d2a52; line-height: 1.8;
    margin-bottom: 18px;
  }
  .endo-body strong { color: var(--ink); }
  .endo-symptoms-title {
    font-size: 12px; font-weight: 800; letter-spacing: 0.8px;
    text-transform: uppercase; color: var(--muted);
    margin-bottom: 12px; margin-top: 8px;
  }
  .endo-symptoms-grid {
    display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-bottom: 22px;
  }
  .endo-symptom-chip {
    display: flex; align-items: flex-start; gap: 7px;
    font-size: 13px; color: #3d2a52; line-height: 1.4;
    padding: 8px 12px; background: #faf5ff;
    border: 1px solid #ede9fe; border-radius: 10px;
  }
  .endo-chip-dot { color: var(--plum); font-size: 8px; margin-top: 4px; flex-shrink: 0; }
  .endo-disclaimer {
    font-size: 12.5px; color: var(--muted); line-height: 1.7;
    padding: 14px 16px; background: #fff8e7;
    border: 1px solid #fde68a; border-radius: 12px;
    border-left: 3px solid #f59e0b;
  }
  .endo-right { display: flex; flex-direction: column; gap: 16px; }
  .endo-fact-card {
    background: #faf5ff; border: 1px solid #ede9fe;
    border-radius: 16px; padding: 22px 22px 20px;
  }
  .endo-fact-highlight {
    background: linear-gradient(135deg, #f5f0ff, #fff1f6);
    border-color: #ddd6fe;
  }
  .endo-fact-icon { font-size: 22px; margin-bottom: 10px; }
  .endo-fact-title { font-size: 14px; font-weight: 700; color: var(--ink); margin-bottom: 8px; }
  .endo-fact-body { font-size: 13.5px; color: #4a3560; line-height: 1.7; }

  /* ── FOOTER ── */
  .footer {
    text-align: center; padding: 32px 20px;
    border-top: 1px solid var(--border);
    font-size: 12px; color: var(--muted); line-height: 1.8;
  }

  @media (max-width: 640px) {
    .endo-section { padding: 56px 20px; }
    .endo-inner { grid-template-columns: 1fr; gap: 36px; }
    .endo-symptoms-grid { grid-template-columns: 1fr; }
    .nav { padding: 0 16px; }
    .home-hero { padding: 56px 20px 48px; }
    .how-section { padding: 56px 20px; }
    .q-section { padding: 56px 20px 60px; }
    .hero-stats { gap: 24px; }
    .stat-divider { display: none; }
    .modal-card { padding: 28px 22px; }
    .results-actions-btns { flex-direction: column; }
  }
`;

// ── AUTH MODAL ──────────────────────────────────────────────
function AuthModal({ mode = "login", onLogin, onClose }) {
  const [tab, setTab]           = useState(mode);
  const [username, setUsername] = useState("");
  const [email, setEmail]       = useState("");
  const [password, setPassword] = useState("");
  const [loading, setLoading]   = useState(false);
  const [error, setError]       = useState("");

  const handleSubmit = async () => {
    setError("");
    if (!username || !password) { setError("Please fill in all fields."); return; }
    setLoading(true);
    try {
      if (tab === "signup") {
        if (!email) { setError("Email is required for signup."); setLoading(false); return; }
        await apiSignup(username, email, password);
      }
      const data = await apiLogin(username, password);
      onLogin({ username: data.username, token: data.access_token, guest: false });
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="modal-backdrop" onClick={e => e.target === e.currentTarget && onClose()}>
      <div className="modal-card fade-up">
        <div className="modal-brand">
          <div className="modal-icon">🌸</div>
          <div className="modal-title">Welcome to FemCare</div>
          <div className="modal-sub">Sign in to save your results and view history</div>
        </div>
        <div className="auth-tabs">
          {[["login","Sign In"],["signup","Sign Up"]].map(([t,l]) => (
            <button key={t} className={`auth-tab ${tab===t?"active":""}`}
              onClick={() => { setTab(t); setError(""); }}>{l}</button>
          ))}
        </div>
        {error && <div className="err-box">⚠ {error}</div>}
        <label className="field-lbl">Username</label>
        <input className="field-inp" placeholder="your_username" value={username}
          onChange={e => setUsername(e.target.value)}
          onKeyDown={e => e.key === "Enter" && handleSubmit()} />
        {tab === "signup" && (
          <>
            <label className="field-lbl">Email</label>
            <input className="field-inp" type="email" placeholder="you@email.com"
              value={email} onChange={e => setEmail(e.target.value)} />
          </>
        )}
        <label className="field-lbl">Password</label>
        <input className="field-inp" type="password" placeholder="••••••••"
          value={password} onChange={e => setPassword(e.target.value)}
          onKeyDown={e => e.key === "Enter" && handleSubmit()} />
        <button className="btn-modal-main" onClick={handleSubmit} disabled={loading}>
          {loading ? "Please wait…" : tab === "login" ? "Sign In →" : "Create Account →"}
        </button>
        <div className="modal-or">or</div>
        <button className="btn-modal-ghost" onClick={onClose}>Continue without signing in</button>
        <div className="modal-note">
          Screening tool only — not a medical diagnosis.<br/>
          Always consult a qualified healthcare professional.
        </div>
      </div>
    </div>
  );
}

// ── TOP NAV ────────────────────────────────────────────────
function Nav({ user, onSignOut, onShowAuth, onHistoryClick, onLogoClick }) {
  return (
    <nav className="nav">
      <div className="nav-logo" onClick={onLogoClick}>
        <span>🌸</span> FemCare
      </div>
      <div className="nav-right">
        {user ? (
          <>
            {!user.guest && (
              <button className="nav-link" onClick={onHistoryClick}>Past Results</button>
            )}
            <div className="user-chip">
              <div className="avatar">{user.username[0].toUpperCase()}</div>
              {user.username}
              {user.guest && (
                <span style={{ fontSize:9, background:"#dcfce7", color:"#15803d", padding:"1px 6px", borderRadius:100, fontWeight:800, marginLeft:2 }}>GUEST</span>
              )}
            </div>
            <button className="nav-btn nav-btn-outline" onClick={onSignOut} style={{ padding:"7px 14px" }}>
              Sign Out
            </button>
          </>
        ) : (
          <>
            <button className="nav-btn nav-btn-outline" onClick={() => onShowAuth("login")}>Sign In</button>
            <button className="nav-btn nav-btn-fill" onClick={() => onShowAuth("signup")}>Sign Up</button>
          </>
        )}
      </div>
    </nav>
  );
}

// ── HOME PAGE ──────────────────────────────────────────────
function HomePage({ user, onShowAuth, onStartQuiz }) {
  const qRef = useRef(null);
  const scrollToQ = () => qRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });

  return (
    <>
      {/* Hero */}
      <section className="home-hero">
        <div className="hero-eyebrow">
          <span className="hero-dot" /> Free · 2-minute screening
        </div>
        <h1 className="hero-h1">
          Could your pain be<br /><em>endometriosis?</em>
        </h1>
        <p className="hero-p">
          FemCare is an early endometriosis risk screening tool. Answer a short set of symptom questions and receive a personalised risk report — free, and private.
        </p>
        <div className="hero-cta-row">
          <button className="cta-primary" onClick={scrollToQ}>
            Take the Free Screening ↓
          </button>
          {!user && (
            <button className="cta-secondary" onClick={() => onShowAuth("signup")}>
              Sign Up to Save Results
            </button>
          )}
        </div>
        <div className="hero-stats">
          <div className="hero-stat">
            <div className="hero-stat-num"><span>886</span></div>
            <div className="hero-stat-lbl">Patient records trained on</div>
          </div>
          <div className="stat-divider" />
          <div className="hero-stat">
            <div className="hero-stat-num"><span>96.95%</span></div>
            <div className="hero-stat-lbl">Cross-validated AUC</div>
          </div>
          <div className="stat-divider" />
          <div className="hero-stat">
            <div className="hero-stat-num"><span>2 min</span></div>
            <div className="hero-stat-lbl">Average completion time</div>
          </div>
          <div className="stat-divider" />
          <div className="hero-stat">
            <div className="hero-stat-num"><span>14</span></div>
            <div className="hero-stat-lbl">Symptom features analysed</div>
          </div>
        </div>
      </section>

      {/* What is Endometriosis */}
      <section className="endo-section">
        <div className="endo-inner">
          <div className="endo-left">
            <div className="section-eyebrow">Understanding Endometriosis</div>
            <h2 className="section-title">A condition that affects 1 in 10 women — often undiagnosed for years</h2>
            <p className="endo-body">
              Endometriosis is a chronic condition where tissue similar to the lining of the uterus grows outside it — on the ovaries, fallopian tubes, and other pelvic organs. This tissue behaves like the uterine lining: it thickens, breaks down, and bleeds with each menstrual cycle. But because it has nowhere to go, it becomes trapped, causing inflammation, scar tissue, and sometimes severe pain.
            </p>
            <p className="endo-body">
              It affects roughly <strong>190 million women and girls</strong> worldwide, yet the average time from first symptom to diagnosis is <strong>7 to 10 years</strong>. Pain is frequently dismissed as "normal period cramps," leading to years of unnecessary suffering.
            </p>
            <div className="endo-symptoms-title">Common symptoms include:</div>
            <div className="endo-symptoms-grid">
              {[
                "Severe or worsening period pain",
                "Chronic pelvic pain",
                "Heavy or prolonged menstrual bleeding",
                "Pain during or after sex",
                "Painful bowel movements or urination",
                "Bloating and IBS-like symptoms",
                "Unexplained fatigue",
                "Difficulty conceiving",
              ].map(s => (
                <div className="endo-symptom-chip" key={s}>
                  <span className="endo-chip-dot">●</span> {s}
                </div>
              ))}
            </div>
            <div className="endo-disclaimer">
              Endometriosis can only be confirmed through laparoscopy (a minor surgical procedure). This screening tool cannot diagnose you — it identifies patterns in your symptoms that may warrant further investigation by a gynaecologist.
            </div>
          </div>
          <div className="endo-right">
            <div className="endo-fact-card">
              <div className="endo-fact-icon">🔬</div>
              <div className="endo-fact-title">What causes it?</div>
              <p className="endo-fact-body">The exact cause is unknown. Leading theories include retrograde menstruation, immune system dysfunction, and genetic factors. It is not caused by anything you did.</p>
            </div>
            <div className="endo-fact-card">
              <div className="endo-fact-icon">⏳</div>
              <div className="endo-fact-title">Why does it take so long to diagnose?</div>
              <p className="endo-fact-body">Symptoms overlap with other conditions like IBS or ovarian cysts. Period pain is often normalised by patients and clinicians alike. Many women see multiple doctors before receiving a referral.</p>
            </div>
            <div className="endo-fact-card">
              <div className="endo-fact-icon">💊</div>
              <div className="endo-fact-title">Can it be treated?</div>
              <p className="endo-fact-body">Yes. While there is no cure, endometriosis is very manageable. Treatment options include hormonal therapy, pain management, laparoscopic surgery to remove lesions, and lifestyle adjustments. Early detection leads to better outcomes.</p>
            </div>
            <div className="endo-fact-card endo-fact-highlight">
              <div className="endo-fact-icon">🌸</div>
              <div className="endo-fact-title">How FemCare helps</div>
              <p className="endo-fact-body">FemCare analyses your symptom pattern against data from 886 confirmed patient records. It gives you a risk score and identifies which symptoms are most significant — so you can have an informed conversation with your doctor.</p>
            </div>
          </div>
        </div>
      </section>

      {/* How it works */}
      <section className="how-section">
        <div className="section-eyebrow">How it works</div>
        <h2 className="section-title">Three steps to your personalised risk report</h2>
        <p className="section-sub">No blood tests, no appointments — just honest answers about how you feel.</p>
        <div className="steps-grid">
          {[
            { icon:"📋", h:"Answer 12 symptom questions", p:"Tell us about your pain, bleeding, fatigue, and bowel symptoms. It takes about 2 minutes and there are no wrong answers." },
            { icon:"📐", h:"Your pattern is scored", p:"A model trained on 886 patient records scores your symptom profile and identifies which factors are most significant." },
            { icon:"📊", h:"Receive your risk report", p:"See your risk percentage, risk tier (Low to Urgent), and the specific symptoms contributing most to your score." },
          ].map((s,i) => (
            <div className="step-card" key={i}>
              <div className="step-num">0{i+1}</div>
              <span className="step-icon">{s.icon}</span>
              <div className="step-h">{s.h}</div>
              <p className="step-p">{s.p}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Questionnaire */}
      <section className="q-section" ref={qRef} id="questionnaire">
        <div className="q-inner">
          <div className="q-header">
            <div className="section-eyebrow" style={{ textAlign:"center", marginBottom:12 }}>Free Screening</div>
            <h2 className="section-title" style={{ textAlign:"center" }}>
              How are you <em style={{ fontStyle:"italic", background:"var(--grad)", WebkitBackgroundClip:"text", WebkitTextFillColor:"transparent" }}>really</em> feeling?
            </h2>
            <p className="section-sub">Answer each question honestly. All 12 symptom questions are required. Optional clinical questions default to No if skipped.</p>
          </div>
          <QuestionnaireForm user={user} onShowAuth={onShowAuth} onResult={onStartQuiz} />
        </div>
      </section>

      <footer className="footer">
        🌸 FemCare — Early Endometriosis Risk Screening<br />
        Trained on 886 patient records · 14 symptom features analysed<br />
        <strong>Not a medical diagnosis. Always consult a qualified gynaecologist.</strong>
      </footer>
    </>
  );
}

// ── QUESTIONNAIRE FORM (used on home page) ──────────────────
function QuestionnaireForm({ user, onShowAuth, onResult }) {
  const [answers, setAnswers] = useState({});
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState("");

  const answered = STANDARD_QS.filter(q => answers[q.id] !== undefined).length;
  const ready    = answered === STANDARD_QS.length;

  const toggle = (id, val) =>
    setAnswers(prev => ({ ...prev, [id]: prev[id] === val ? undefined : val }));

  const handleSubmit = async () => {
    if (!ready) return;
    setError(""); setLoading(true);
    try {
      const result = await apiPredict(answers, user?.token ?? null);
      onResult({ ...result, answers, guest: !user || user.guest });
    } catch (e) {
      setError(e.message || "Something went wrong. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  if (loading) return (
    <div className="loading-page">
      <div className="spinner" />
      <div className="loading-txt">Analysing your symptom pattern…</div>
    </div>
  );

  return (
    <div className="fade-up">
      <div className="prog-row">
        <span className="prog-lbl">Progress</span>
        <span className="prog-count">{answered} / {STANDARD_QS.length}</span>
      </div>
      <div className="prog-track">
        <div className="prog-fill" style={{ width:`${(answered/STANDARD_QS.length)*100}%` }} />
      </div>
      {error && <div className="err-box">⚠ {error}</div>}

      <div className="q-group-label">Symptom Questions — required</div>
      {STANDARD_QS.map(q => (
        <div key={q.id} className={`q-card ${answers[q.id]===1?"answered-yes":answers[q.id]===0?"answered-no":""}`}>
          <div className="q-text">{q.text}</div>
          <div className="yn-wrap">
            <button className={`yn ${answers[q.id]===1?"y-active":""}`} onClick={() => toggle(q.id,1)}>Y</button>
            <button className={`yn ${answers[q.id]===0?"n-active":""}`} onClick={() => toggle(q.id,0)}>N</button>
          </div>
        </div>
      ))}

      <div className="q-group-label" style={{ marginTop:32 }}>Clinical History — optional</div>
      <div className="q-optional-note">
        Only answer if you have a confirmed clinical diagnosis. Skipping defaults to No.
      </div>
      {OPTIONAL_QS.map(q => (
        <div key={q.id} className={`q-card ${answers[q.id]===1?"answered-yes":answers[q.id]===0?"answered-no":""}`}>
          <div className="q-text">{q.text}</div>
          <span className="opt-badge">Optional</span>
          <div className="yn-wrap">
            <button className={`yn ${answers[q.id]===1?"y-active":""}`} onClick={() => toggle(q.id,1)}>Y</button>
            <button className={`yn ${answers[q.id]===0?"n-active":""}`} onClick={() => toggle(q.id,0)}>N</button>
          </div>
        </div>
      ))}

      <div className="submit-area">
        <button className="cta-primary"
          style={{ opacity:ready?1:0.45, cursor:ready?"pointer":"not-allowed", pointerEvents:ready?"auto":"none" }}
          onClick={handleSubmit}>
          {ready ? "See My Risk Report →" : `Answer all 12 questions (${answered}/12 done)`}
        </button>
        <div className="submit-note">
          Trained on 886 patient records · 14 symptom features · Free & private<br/>
          {!user && <span>💡 <button style={{ background:"none",border:"none",color:"var(--plum)",fontWeight:700,cursor:"pointer",fontSize:11,fontFamily:"'DM Sans',sans-serif" }} onClick={() => onShowAuth("login")}>Sign in</button> to save your results after submitting</span>}
        </div>
      </div>
    </div>
  );
}

// ── RESULTS PAGE ────────────────────────────────────────────
function ResultsPage({ user, result, onRetake, onExit, onShowAuth, onSaved }) {
  const { risk_pct, risk_tier, top_features, advice, answers } = result;
  const tier    = TIER[risk_tier] || TIER.Moderate;
  const maxShap = top_features?.[0] ? Math.abs(top_features[0].shap) : 1;
  const yesCount = Object.values(answers||{}).filter(v=>v===1).length;

  return (
    <div className="results-page fade-up">
      {/* Score hero */}
      <div className="results-hero">
        <div className="score-ring" style={{
          background: tier.bg, border: `2px solid ${tier.border}`, color: tier.color,
          boxShadow: `0 0 40px ${tier.glow}`,
          animation: "orbPulse 2.5s ease-in-out infinite"
        }}>
          <style>{`@keyframes orbPulse { 0%,100%{box-shadow:0 0 30px ${tier.glow}} 50%{box-shadow:0 0 60px ${tier.glow}} }`}</style>
          <div className="ring-orbit" style={{ borderColor: tier.color }} />
          <div className="score-num" style={{ color: tier.color }}>{risk_pct}%</div>
          <div className="score-unit" style={{ color: tier.color }}>Risk Score</div>
        </div>
        <div className="tier-badge" style={{ background: tier.bg, color: tier.color, border: `1px solid ${tier.border}` }}>
          <span className="tier-dot" style={{ background: tier.color }} />
          {risk_tier} Risk
        </div>
        {result.guest && (
          <div className="results-guest-note">
            Guest result — <button style={{ background:"none",border:"none",color:"var(--plum)",fontWeight:700,cursor:"pointer",fontFamily:"'DM Sans',sans-serif",fontSize:12 }} onClick={() => onShowAuth("login")}>sign in</button> to save this to your history.
          </div>
        )}
      </div>

      {/* Advice */}
      <div className="advice-card" style={{ background: tier.bg, borderLeftColor: tier.color }}>
        <div className="advice-eyebrow" style={{ color: tier.color }}>Our Assessment</div>
        <div className="advice-text">{advice}</div>
      </div>

      {/* Top drivers */}
      {top_features?.length > 0 && (
        <div className="drivers-section">
          <div className="drivers-label">Top symptoms influencing your result</div>
<div style={{fontSize:'12px', color:'#888', marginBottom:'8px'}}>
  Red bars increase risk · Green bars decrease risk · Longer bar = stronger influence
</div>
          {top_features.map(({ feature, shap }, i) => (
            <div className="driver-row" key={feature}>
            <div className="d-rank">{i+1}</div>
            <div className="d-name">{feature}</div>
            <div className="d-bar-bg">
            <div className="d-bar-fill" style={{ 
              width:`${(Math.abs(shap)/maxShap)*100}%`,
              background: shap > 0 ? '#e11d48' : '#16a34a'
            }} />
        </div>
        <div className="d-val" style={{ color: shap > 0 ? '#e11d48' : '#16a34a', fontSize: '12px' }}>
    {shap > 0 ? '↑ Increases risk' : '↓ Decreases risk'}
  </div>
</div>
          ))}
        </div>
      )}

      {/* Metrics */}
      <div className="metrics-row">
        {[
          { val:`${yesCount} / 14`, lbl:"Symptoms reported" },
          { val:`${risk_pct}%`,     lbl:"Risk probability" },
          { val: risk_tier,         lbl:"Risk tier" },
          { val: top_features?.[0]?.feature?.split(" ").slice(0,2).join(" ") || "—", lbl:"Top driver" },
        ].map(m => (
          <div className="metric-tile" key={m.lbl}>
            <div className="m-val">{m.val}</div>
            <div className="m-lbl">{m.lbl}</div>
          </div>
        ))}
      </div>

      {/* Model info */}
      <div className="model-card">
        🌿 <strong>About this screening:</strong> Trained on 886 confirmed patient records · 14 symptom features · Cross-validated AUC 0.9695<br/>
         
      </div>

      <div className="disclaimer-card">
        ⚠️ <strong>Screening tool only — not a medical diagnosis.</strong> FemCare uses a machine learning model trained on 886 patient records. A risk score is not a confirmed diagnosis of endometriosis. Please share your results with a qualified gynaecologist for clinical evaluation.
      </div>

      {/* Actions */}
      <div className="results-actions">
        <div className="results-actions-title">What would you like to do?</div>
        <div className="results-actions-sub">
          {user && !user.guest
            ? "Your result has been saved to your history automatically."
            : "Sign in to save this result to your history and track changes over time."}
        </div>
        <div className="results-actions-btns">
          {(!user || user.guest) && (
            <button className="btn-save" onClick={() => onShowAuth("login")}>
              🔒 Sign In to Save Result
            </button>
          )}
          <button className="btn-retake" onClick={onRetake}>↺ Retake Assessment</button>
          <button className="btn-exit" onClick={onExit}>✕ Exit</button>
        </div>
      </div>
    </div>
  );
}

// ── HISTORY PAGE ────────────────────────────────────────────
function HistoryPage({ user, onBack }) {
  const [items, setItems]       = useState(null);
  const [error, setError]       = useState("");
  const [selected, setSelected] = useState(null);

  useEffect(() => {
    apiHistory(user.token)
      .then(data => setItems(data.assessments))
      .catch(e => setError(e.message || "Failed to load history"));
  }, []);

  if (selected) {
    const fakeResult = {
      risk_pct: selected.risk_pct,
      risk_tier: selected.risk_tier,
      top_features: selected.top_features,
      advice: {
        Urgent:   "Your symptom pattern suggested a high likelihood of endometriosis. Please consult a gynaecologist as soon as possible.",
        High:     "Several key endometriosis markers were present. We strongly recommended scheduling a gynaecological consultation soon.",
        Moderate: "Some symptoms aligned with endometriosis. Consider speaking with a doctor if they persist or worsen.",
        Low:      "Your symptoms didn't strongly indicate endometriosis at the time. Continue monitoring your cycle.",
      }[selected.risk_tier],
      answers: selected.symptoms,
      guest: false,
    };
    return (
      <div className="hist-page fade-up">
        <button className="hist-back" onClick={() => setSelected(null)}>← Back to history</button>
        <ResultsPage
          user={user}
          result={fakeResult}
          onRetake={() => setSelected(null)}
          onExit={() => setSelected(null)}
          onShowAuth={() => {}}
        />
      </div>
    );
  }

  return (
    <div className="hist-page fade-up">
      <button className="hist-back" onClick={onBack}>← Back to Screening</button>
      <h1 className="hist-title">Your Past Assessments</h1>
      <p className="hist-sub">Every screening you've taken while signed in, most recent first.</p>
      {error && <div className="err-box">⚠ {error}</div>}
      {items === null && !error && <div className="hist-empty">Loading your history…</div>}
      {items?.length === 0 && <div className="hist-empty">No assessments yet — take your first screening to see it here.</div>}
      {items?.map(item => {
        const tier = TIER[item.risk_tier] || TIER.Moderate;
        return (
          <div className="hist-card" key={item.assessment_id} onClick={() => setSelected(item)}>
            <div className="hist-score-block">
              <div className="hist-pct" style={{ color: tier.color }}>{item.risk_pct}%</div>
            </div>
            <div className="hist-info">
              <div className="hist-date">{item.taken_at}</div>
              <div className="hist-tier-pill" style={{ background: tier.bg, color: tier.color, border: `1px solid ${tier.border}` }}>
                <span style={{ width:6, height:6, borderRadius:"50%", background:tier.color, display:"inline-block" }} />
                {item.risk_tier} Risk
              </div>
            </div>
            <div className="hist-chev">›</div>
          </div>
        );
      })}
    </div>
  );
}

// ── APP ROOT ────────────────────────────────────────────────
export default function App() {
  const [user,      setUser]      = useState(null);
  const [result,    setResult]    = useState(null);
  const [page,      setPage]      = useState("home"); // "home" | "results" | "history"
  const [authModal, setAuthModal] = useState(null);   // null | "login" | "signup"

  const handleLogin = async (u) => {
  setUser(u);
  setAuthModal(null);

  if (result && result.guest && result.answers) {
    try {
      const saved = await apiPredict(result.answers, u.token);
      setResult({ ...saved, answers: result.answers, guest: false });
    } catch (e) {
      console.error("Failed to save result after login:", e);
    }
  }
};
  const handleSignOut = () => {
    setUser(null); setResult(null); setPage("home");
  };
  const handleResult = (r) => {
    setResult(r); setPage("results");
  };
  const handleRetake = () => {
    setResult(null); setPage("home");
    // small delay then scroll to questionnaire
    setTimeout(() => {
      document.getElementById("questionnaire")?.scrollIntoView({ behavior:"smooth", block:"start" });
    }, 100);
  };
  const handleExit = () => {
    setResult(null); setPage("home");
    window.scrollTo({ top:0, behavior:"smooth" });
  };
  const handleShowAuth = (mode) => setAuthModal(mode);

  return (
    <>
      <style>{css}</style>
      <div>
        <Nav
          user={user}
          onSignOut={handleSignOut}
          onShowAuth={handleShowAuth}
          onHistoryClick={() => setPage(page === "history" ? "home" : "history")}
          onLogoClick={() => { setPage("home"); setResult(null); window.scrollTo({top:0,behavior:"smooth"}); }}
        />

        {page === "history" ? (
          <HistoryPage user={user} onBack={() => setPage("home")} />
        ) : page === "results" && result ? (
          <ResultsPage
            user={user}
            result={result}
            onRetake={handleRetake}
            onExit={handleExit}
            onShowAuth={handleShowAuth}
          />
        ) : (
          <HomePage user={user} onShowAuth={handleShowAuth} onStartQuiz={handleResult} />
        )}

        {authModal && (
          <AuthModal
            mode={authModal}
            onLogin={handleLogin}
            onClose={() => setAuthModal(null)}
          />
        )}
      </div>
    </>
  );
}