# ============================================================
# HAFTA 2 — Noise Üret + Karşılaştırmalı Analiz (Kişi 1)
# Proje: Veri Gürültüsü Pattern Öğrenme ve Model Dayanıklılık Sistemi
# ============================================================

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix
)
from sklearn.preprocessing import LabelEncoder

np.random.seed(42)

# ── RENK AYARLARI ───────────────────────────────────────────
plt.rcParams.update({
    "font.family"       : "Segoe UI",
    "axes.spines.top"   : False,
    "axes.spines.right" : False,
    "axes.spines.left"  : False,
    "axes.spines.bottom": False,
    "axes.grid"         : True,
    "grid.color"        : "#F0F0F0",
    "grid.linewidth"    : 0.8,
    "figure.facecolor"  : "white",
    "axes.facecolor"    : "white",
})
ACCENT = "#2563EB"
LIGHT  = "#93C5FD"
GRAY   = "#6B7280"
DARK   = "#111827"
RED    = "#EF4444"
GREEN  = "#22C55E"

# ── 1. VERİYİ YÜKLE ─────────────────────────────────────────
print("=" * 65)
print("  HAFTA 2 — Noise Üretimi + Karşılaştırmalı Analiz")
print("=" * 65)

print("\n[1/4] Orijinal veri yükleniyor...")
df_original = pd.read_csv("weatherAUS.csv")
print(f"  Boyut: {df_original.shape[0]:,} satır x {df_original.shape[1]} sütun")

# ── 2. NOISE FONKSİYONLARI ──────────────────────────────────
print("\n[2/4] Noise datasetleri üretiliyor...")

SAYISAL = ["MinTemp","MaxTemp","Rainfall","Evaporation","Sunshine",
           "WindGustSpeed","WindSpeed9am","WindSpeed3pm",
           "Humidity9am","Humidity3pm","Pressure9am","Pressure3pm",
           "Cloud9am","Cloud3pm","Temp9am","Temp3pm"]

def random_missing(df, oran=0.10):
    df2 = df.copy()
    for col in SAYISAL:
        if col in df2.columns:
            mask = np.random.rand(len(df2)) < oran
            df2.loc[mask, col] = np.nan
    return df2

def random_outlier(df, oran=0.05):
    df2 = df.copy()
    for col in SAYISAL:
        if col in df2.columns:
            std = df2[col].std()
            mask = np.random.rand(len(df2)) < oran
            df2.loc[mask, col] = df2.loc[mask, col] + np.random.choice([-1,1]) * std * 5
    return df2

def pattern_noise(df, seviye="low"):
    df2 = df.copy()
    oranlar = {"low": 0.15, "medium": 0.35, "high": 0.60}
    oran = oranlar[seviye]

    if "Humidity3pm" in df2.columns:
        esik = df2["Humidity3pm"].quantile(0.75)
        mask = (df2["Humidity3pm"] > esik) & (np.random.rand(len(df2)) < oran)
        df2.loc[mask, "Humidity3pm"] = np.nan

    if "Temp3pm" in df2.columns:
        esik = df2["Temp3pm"].quantile(0.25)
        std  = df2["Temp3pm"].std()
        mask = (df2["Temp3pm"] < esik) & (np.random.rand(len(df2)) < oran)
        df2.loc[mask, "Temp3pm"] = df2.loc[mask, "Temp3pm"] + std * 4

    if "WindSpeed3pm" in df2.columns and "Pressure9am" in df2.columns:
        esik = df2["WindSpeed3pm"].quantile(0.75)
        std  = df2["Pressure9am"].std()
        mask = (df2["WindSpeed3pm"] > esik) & (np.random.rand(len(df2)) < oran)
        df2.loc[mask, "Pressure9am"] = df2.loc[mask, "Pressure9am"] + std * 4

    return df2

datasets_raw = {
    "Baseline"              : df_original.copy(),
    "Random Missing"        : random_missing(df_original, 0.10),
    "Random Outlier"        : random_outlier(df_original, 0.05),
    "Random Missing+Outlier": random_outlier(random_missing(df_original, 0.10), 0.05),
    "Pattern Low"           : pattern_noise(df_original, "low"),
    "Pattern Medium"        : pattern_noise(df_original, "medium"),
    "Pattern High"          : pattern_noise(df_original, "high"),
}

for isim, df_noise in datasets_raw.items():
    dosya = isim.lower().replace(" ", "_").replace("+", "_") + ".csv"
    df_noise.to_csv(dosya, index=False)

print("  7 dataset uretildi ve kaydedildi")

# ── 3. TEMİZLE + TEST ET ────────────────────────────────────
def preprocess(df):
    df = df.copy()
    drop_cols = [c for c in ["Date", "Location"] if c in df.columns]
    df = df.drop(columns=drop_cols)
    df = df.dropna(subset=["RainTomorrow"])
    df["RainTomorrow"] = (df["RainTomorrow"] == "Yes").astype(int)
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])
        df[col] = le.fit_transform(df[col].astype(str))
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    return df

def evaluate(df, label):
    X = df.drop(columns=["RainTomorrow"])
    y = df["RainTomorrow"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = RandomForestClassifier(n_estimators=100, max_depth=15,
                                   random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return {
        "label"    : label,
        "accuracy" : round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall"   : round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1"       : round(f1_score(y_test, y_pred, zero_division=0), 4),
    }

print("\n[3/4] Modeller test ediliyor (biraz sürebilir)...")
results = []
for label, df_raw in datasets_raw.items():
    print(f"  {label:<28} ", end="", flush=True)
    df_clean = preprocess(df_raw)
    res = evaluate(df_clean, label)
    results.append(res)
    print(f"Acc={res['accuracy']:.4f}  F1={res['f1']:.4f}")

with open("hafta2_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

# ── 4. GRAFİKLER ────────────────────────────────────────────
print("\n[4/4] Grafikler olusturuluyor...")

df_res   = pd.DataFrame(results)
baseline = df_res[df_res["label"] == "Baseline"].iloc[0]
df_res["f1_drop"] = (baseline["f1"] - df_res["f1"]).round(4)

labels = df_res["label"].tolist()
f1s    = df_res["f1"].tolist()
accs   = df_res["accuracy"].tolist()

# Grafik 1: F1 + Accuracy
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle("Hafta 2  -  Noise Senaryolari Karsilastirmasi",
             fontsize=15, fontweight="bold", color=DARK, y=1.02)

bc_f1  = [GREEN if l=="Baseline" else RED if f<baseline["f1"]-0.005 else LIGHT
          for l,f in zip(labels,f1s)]
bc_acc = [GREEN if l=="Baseline" else RED if a<baseline["accuracy"]-0.002 else LIGHT
          for l,a in zip(labels,accs)]

b1 = axes[0].barh(labels[::-1], f1s[::-1], color=bc_f1[::-1], height=0.5, zorder=2)
axes[0].set_title("F1 Skoru", fontsize=12, fontweight="bold", color=DARK, pad=12)
axes[0].set_xlim(0, 1.0)
axes[0].axvline(x=baseline["f1"], color=GREEN, linewidth=1.5, linestyle="--")
axes[0].tick_params(axis="y", colors=DARK, labelsize=9)
axes[0].tick_params(axis="x", colors=GRAY, labelsize=9)
for bar, val in zip(b1, f1s[::-1]):
    axes[0].text(val+0.005, bar.get_y()+bar.get_height()/2,
                 f"{val:.4f}", va="center", fontsize=9, color=DARK)

b2 = axes[1].barh(labels[::-1], accs[::-1], color=bc_acc[::-1], height=0.5, zorder=2)
axes[1].set_title("Accuracy", fontsize=12, fontweight="bold", color=DARK, pad=12)
axes[1].set_xlim(0.82, 0.88)
axes[1].axvline(x=baseline["accuracy"], color=GREEN, linewidth=1.5, linestyle="--")
axes[1].tick_params(axis="y", colors=DARK, labelsize=9)
axes[1].tick_params(axis="x", colors=GRAY, labelsize=9)
for bar, val in zip(b2, accs[::-1]):
    axes[1].text(val+0.0003, bar.get_y()+bar.get_height()/2,
                 f"{val:.4f}", va="center", fontsize=9, color=DARK)

plt.tight_layout(pad=3)
plt.savefig("hafta2_karsilastirma.png", dpi=180, bbox_inches="tight", facecolor="white")
plt.show()

# Grafik 2: F1 Düşüş
df_noise = df_res[df_res["label"] != "Baseline"].copy()
fig2, ax = plt.subplots(figsize=(10, 5))
dc = [RED if d > 0.005 else LIGHT for d in df_noise["f1_drop"]]
b3 = ax.bar(df_noise["label"], df_noise["f1_drop"], color=dc, width=0.5, zorder=2)
ax.set_title("F1 Dusus Miktari (Baseline'a Gore)",
             fontsize=13, fontweight="bold", color=DARK, pad=16)
ax.set_ylabel("F1 Dususu", fontsize=10, color=GRAY)
ax.tick_params(axis="x", colors=DARK, labelsize=9, rotation=20)
ax.tick_params(axis="y", colors=GRAY, labelsize=9)
ax.spines[:].set_visible(False)
ax.grid(axis="y", color="#F0F0F0", linewidth=0.8, zorder=1)
for bar, val in zip(b3, df_noise["f1_drop"]):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.0002,
            f"-{val:.4f}", ha="center", fontsize=9, fontweight="bold", color=DARK)
plt.tight_layout()
plt.savefig("hafta2_f1_dusus.png", dpi=180, bbox_inches="tight", facecolor="white")
plt.show()

# ── ÖZET TABLO ──────────────────────────────────────────────
print("\n" + "=" * 65)
print(f"  {'Senaryo':<28} {'Acc':>7} {'F1':>7} {'F1 Dusus':>10}")
print("  " + "-" * 55)
for _, row in df_res.iterrows():
    d = f"-{row['f1_drop']:.4f}" if row['f1_drop'] > 0 else "  -"
    print(f"  {row['label']:<28} {row['accuracy']:>7.4f} {row['f1']:>7.4f} {d:>10}")

en_kotu = df_noise.loc[df_noise["f1_drop"].idxmax()]
print(f"\n  En cok etkileyen: {en_kotu['label']} (F1 -{en_kotu['f1_drop']:.4f})")
print("\n  HAFTA 2 TAMAMLANDI")
print("  Ciktilar: hafta2_results.json | hafta2_karsilastirma.png | hafta2_f1_dusus.png")
print("=" * 65)
