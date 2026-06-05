import { useState } from "react";

const data = {
  dataset: { total: 886, endo: 474, noEndo: 412 },
  testSet: { total: 222, endo: 119, noEndo: 103 },
  models: [
    { name: "Logistic Regression", tp: 103, fn: 16, tn: 91, fp: 12, auc: 0.9570, acc: 0.8874, sens: 0.8655, spec: 0.9126, f1: 0.8918 },
    { name: "Random Forest",       tp:  99, fn: 20, tn: 91, fp: 12, auc: 0.9522, acc: 0.8559, sens: 0.8319, spec: 0.8835, f1: 0.8609 },
    { name: "AdaBoost",            tp: 106, fn: 13, tn: 90, fp: 13, auc: 0.9585, acc: 0.8829, sens: 0.8908, spec: 0.8738, f1: 0.8908 },
    { name: "Gradient Boosting",   tp: 100, fn: 19, tn: 91, fp: 12, auc: 0.9579, acc: 0.8604, sens: 0.8403, spec: 0.8835, f1: 0.8658 },
    { name: "SVM (RBF)",           tp: 100, fn: 19, tn: 89, fp: 14, auc: 0.9494, acc: 0.8514, sens: 0.8403, spec: 0.8641, f1: 0.8584 },
    { name: "XGBoost (GBM)",       tp: 102, fn: 17, tn: 91, fp: 12, auc: 0.9577, acc: 0.8694, sens: 0.8571, spec: 0.8835, f1: 0.8755 },
    { name: "Hybrid Ensemble",     tp: 100, fn: 19, tn: 93, fp: 10, auc: 0.9583, acc: 0.8694, sens: 0.8403, spec: 0.9029, f1: 0.8734, isHybrid: true },
  ]
};

const pct = (v) => `${(v * 100).toFixed(1)}%`;
const bar = (v, color) => (
  <div className="flex items-center gap-2">
    <div className="flex-1 bg-gray-100 rounded-full h-2">
      <div className="h-2 rounded-full" style={{ width: `${v * 100}%`, backgroundColor: color }} />
    </div>
    <span className="text-xs font-mono w-10 text-right">{pct(v)}</span>
  </div>
);

export default function App() {
  const [sort, setSort] = useState("auc");
  const sorted = [...data.models].sort((a, b) => b[sort] - a[sort]);
  const best = sorted[0];

  const cols = [
    { key: "auc",  label: "AUC",         color: "#6366f1" },
    { key: "acc",  label: "Accuracy",    color: "#0ea5e9" },
    { key: "sens", label: "Recall",      color: "#10b981" },
    { key: "spec", label: "Specificity", color: "#f59e0b" },
    { key: "f1",   label: "F1",          color: "#ec4899" },
  ];

  return (
    <div style={{ fontFamily: "system-ui, sans-serif", background: "#f8fafc", minHeight: "100vh", padding: "24px" }}>
      <div style={{ maxWidth: 900, margin: "0 auto" }}>

        {/* Header */}
        <div style={{ background: "linear-gradient(135deg,#6366f1,#ec4899)", borderRadius: 14, padding: "20px 24px", color: "white", marginBottom: 20 }}>
          <div style={{ fontSize: 22, fontWeight: 700 }}>FemCare — Model Performance Results</div>
          <div style={{ fontSize: 13, opacity: 0.85, marginTop: 4 }}>Endometriosis Risk Screening · Step 2 · 15 Selected Features</div>
        </div>

        {/* Dataset cards */}
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 12, marginBottom: 20 }}>
          {[
            { label: "Total Patients", val: data.dataset.total, sub: "full dataset", color: "#6366f1" },
            { label: "Endometriosis +ve", val: data.dataset.endo, sub: `${pct(data.dataset.endo/data.dataset.total)} of dataset`, color: "#ec4899" },
            { label: "Test Set Size", val: data.testSet.total, sub: `${data.testSet.endo} endo · ${data.testSet.noEndo} healthy`, color: "#0ea5e9" },
          ].map(c => (
            <div key={c.label} style={{ background: "white", borderRadius: 10, padding: "14px 18px", borderLeft: `4px solid ${c.color}`, boxShadow: "0 1px 4px #0001" }}>
              <div style={{ fontSize: 26, fontWeight: 800, color: c.color }}>{c.val}</div>
              <div style={{ fontSize: 13, fontWeight: 600, color: "#374151" }}>{c.label}</div>
              <div style={{ fontSize: 11, color: "#9ca3af", marginTop: 2 }}>{c.sub}</div>
            </div>
          ))}
        </div>

        {/* Recall explainer */}
        <div style={{ background: "#ecfdf5", border: "1px solid #6ee7b7", borderRadius: 10, padding: "12px 18px", marginBottom: 20 }}>
          <div style={{ fontWeight: 700, color: "#065f46", fontSize: 13, marginBottom: 6 }}>📌 How to read Recall (Sensitivity)</div>
          <div style={{ fontSize: 13, color: "#047857", lineHeight: 1.6 }}>
            In the <b>test set of 222 patients</b>, exactly <b>119 have endometriosis</b>. 
            Recall tells you: <i>"Out of those 119, how many did the model correctly identify?"</i><br/>
            Best result → <b>AdaBoost found 106 out of 119</b> (Recall 89.1%) — missed only 13 patients. 
            A missed patient (False Negative) is more costly in medical screening than a false alarm.
          </div>
        </div>

        {/* Sort buttons */}
        <div style={{ display: "flex", gap: 8, marginBottom: 12, flexWrap: "wrap" }}>
          <span style={{ fontSize: 12, color: "#6b7280", alignSelf: "center" }}>Sort by:</span>
          {cols.map(c => (
            <button key={c.key} onClick={() => setSort(c.key)}
              style={{ padding: "4px 12px", borderRadius: 20, fontSize: 12, fontWeight: 600, cursor: "pointer", border: "none",
                background: sort === c.key ? c.color : "#e5e7eb", color: sort === c.key ? "white" : "#374151" }}>
              {c.label}
            </button>
          ))}
        </div>

        {/* Main table */}
        <div style={{ background: "white", borderRadius: 12, overflow: "hidden", boxShadow: "0 1px 6px #0001" }}>
          <table style={{ width: "100%", borderCollapse: "collapse" }}>
            <thead>
              <tr style={{ background: "#f1f5f9" }}>
                <th style={{ padding: "10px 14px", textAlign: "left", fontSize: 12, color: "#475569", fontWeight: 700 }}>Model</th>
                <th style={{ padding: "10px 8px", textAlign: "center", fontSize: 11, color: "#475569" }}>Endo in Test</th>
                <th style={{ padding: "10px 8px", textAlign: "center", fontSize: 11, color: "#10b981", fontWeight: 700 }}>Found ✓ (TP)</th>
                <th style={{ padding: "10px 8px", textAlign: "center", fontSize: 11, color: "#ef4444", fontWeight: 700 }}>Missed ✗ (FN)</th>
                <th style={{ padding: "10px 14px", textAlign: "left", fontSize: 11, color: "#6366f1", fontWeight: 700, minWidth: 110 }}>AUC</th>
                <th style={{ padding: "10px 14px", textAlign: "left", fontSize: 11, color: "#0ea5e9", fontWeight: 700, minWidth: 110 }}>Accuracy</th>
                <th style={{ padding: "10px 14px", textAlign: "left", fontSize: 11, color: "#10b981", fontWeight: 700, minWidth: 110 }}>Recall</th>
                <th style={{ padding: "10px 14px", textAlign: "left", fontSize: 11, color: "#f59e0b", fontWeight: 700, minWidth: 110 }}>Specificity</th>
              </tr>
            </thead>
            <tbody>
              {sorted.map((m, i) => {
                const isBest = m.name === best.name;
                return (
                  <tr key={m.name} style={{ background: m.isHybrid ? "#fdf4ff" : isBest ? "#f0fdf4" : i%2===0 ? "white" : "#fafafa",
                    borderTop: "1px solid #f1f5f9" }}>
                    <td style={{ padding: "10px 14px" }}>
                      <div style={{ fontWeight: 700, fontSize: 13, color: "#111827" }}>
                        {m.isHybrid ? "⚡ " : ""}{m.name}
                        {isBest && !m.isHybrid && <span style={{ marginLeft: 6, fontSize: 10, background: "#dcfce7", color: "#16a34a", borderRadius: 10, padding: "1px 7px", fontWeight: 600 }}>BEST RECALL</span>}
                        {m.isHybrid && <span style={{ marginLeft: 6, fontSize: 10, background: "#f3e8ff", color: "#9333ea", borderRadius: 10, padding: "1px 7px", fontWeight: 600 }}>HYBRID</span>}
                      </div>
                    </td>
                    <td style={{ textAlign: "center", fontSize: 13, color: "#6b7280" }}>{m.tp + m.fn}</td>
                    <td style={{ textAlign: "center" }}>
                      <span style={{ fontSize: 15, fontWeight: 800, color: "#059669" }}>{m.tp}</span>
                      <span style={{ fontSize: 11, color: "#6b7280", marginLeft: 2 }}>/ {m.tp+m.fn}</span>
                    </td>
                    <td style={{ textAlign: "center" }}>
                      <span style={{ fontSize: 14, fontWeight: 700, color: m.fn <= 13 ? "#dc2626" : "#f87171" }}>{m.fn}</span>
                    </td>
                    <td style={{ padding: "8px 14px" }}>{bar(m.auc, "#6366f1")}</td>
                    <td style={{ padding: "8px 14px" }}>{bar(m.acc, "#0ea5e9")}</td>
                    <td style={{ padding: "8px 14px" }}>{bar(m.sens, "#10b981")}</td>
                    <td style={{ padding: "8px 14px" }}>{bar(m.spec, "#f59e0b")}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>

        {/* Key takeaway */}
        <div style={{ marginTop: 16, display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
          <div style={{ background: "#eff6ff", border: "1px solid #bfdbfe", borderRadius: 10, padding: "12px 16px" }}>
            <div style={{ fontWeight: 700, fontSize: 13, color: "#1d4ed8", marginBottom: 6 }}>🎯 Best for Recall (Medical Priority)</div>
            <div style={{ fontSize: 13, color: "#1e40af" }}>
              <b>AdaBoost</b> — found <b>106 / 119</b> endo patients (89.1%)<br/>
              Only <b>13 patients missed</b> out of 119 in test set.
            </div>
          </div>
          <div style={{ background: "#fdf4ff", border: "1px solid #e9d5ff", borderRadius: 10, padding: "12px 16px" }}>
            <div style={{ fontWeight: 700, fontSize: 13, color: "#7c3aed", marginBottom: 6 }}>⚡ Best for Fewer False Alarms</div>
            <div style={{ fontSize: 13, color: "#6d28d9" }}>
              <b>Hybrid Ensemble</b> — Specificity <b>90.3%</b><br/>
              Correctly clears <b>93 / 103</b> healthy patients — fewest false alarms.
            </div>
          </div>
        </div>

        <div style={{ fontSize: 11, color: "#9ca3af", textAlign: "center", marginTop: 14 }}>
          FemCare · NIIT Ideathon · Team HLT11 · 15 features from AdaBoost+RFE pipeline
        </div>
      </div>
    </div>
  );
}
