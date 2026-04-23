# ============================================================
# HAFTA 1 — Model & Evaluation (Kişi 1)
# Proje: Veri Gürültüsü Pattern Öğrenme ve Model Dayanıklılık Sistemi
# Dataset: Rain in Australia (Kaggle)
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
    recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.preprocessing import LabelEncoder

# ── 0. VERİYİ İNDİR ─────────────────────────────────────────
# Kaggle hesabın yoksa alternatif direkt link:
# https://raw.githubusercontent.com/dsrscientist/dataset1/master/weatherAUS.csv

CSV_PATH = "weatherAUS.csv"

print("=" * 60)
print("HAFTA 1 — Baseline Model")
print("=" * 60)

print("\n[1/5] Veri yükleniyor...")
df = pd.read_csv(CSV_PATH)
print(f"  Boyut: {df.shape[0]:,} satır × {df.shape[1]} sütun")
print(f"  Hedef dağılımı:\n{df['RainTomorrow'].value_counts()}")

# ── 1. TEMEL KEŞİF ───────────────────────────────────────────
print("\n[2/5] Veri keşfi...")
print(f"\n  Eksik değer oranları (ilk 10 sütun):")
missing = df.isnull().mean().sort_values(ascending=False).head(10)
for col, rate in missing.items():
    print(f"    {col:<25} %{rate*100:.1f}")

# ── 2. TEMİZLEME ─────────────────────────────────────────────
print("\n[3/5] Veri temizleniyor...")

# Tarih ve yüksek-eksikli sütunları çıkar
df = df.drop(columns=["Date", "Location"])

# Sayısal sütunlarda medyan ile doldur
num_cols = df.select_dtypes(include=[np.number]).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

# Kategorik sütunlarda mod ile doldur
cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
cat_cols = [c for c in cat_cols if c != "RainTomorrow"]
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# Hedef sütunu temizle
df = df.dropna(subset=["RainTomorrow"])

# Label encode
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

df["RainTomorrow"] = (df["RainTomorrow"] == "Yes").astype(int)

print(f"  Temizleme sonrası: {df.shape[0]:,} satır")

# ── 3. TRAIN / TEST AYIRIMI ──────────────────────────────────
X = df.drop(columns=["RainTomorrow"])
y = df["RainTomorrow"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n  Eğitim: {X_train.shape[0]:,} | Test: {X_test.shape[0]:,}")

# ── 4. MODEL EĞİTİMİ ────────────────────────────────────────
print("\n[4/5] Model eğitiliyor (Random Forest)...")
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# ── 5. BASELINE METRİKLER ───────────────────────────────────
print("\n[5/5] Performans ölçülüyor...")
y_pred = model.predict(X_test)

acc   = accuracy_score(y_test, y_pred)
prec  = precision_score(y_test, y_pred)
rec   = recall_score(y_test, y_pred)
f1    = f1_score(y_test, y_pred)

print("\n" + "=" * 40)
print("  BASELINE SONUÇLARI")
print("=" * 40)
print(f"  Accuracy  : {acc:.4f}  ({acc*100:.2f}%)")
print(f"  Precision : {prec:.4f}")
print(f"  Recall    : {rec:.4f}")
print(f"  F1 Score  : {f1:.4f}")
print("=" * 40)

print("\n  Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(f"  TN={cm[0,0]}  FP={cm[0,1]}")
print(f"  FN={cm[1,0]}  TP={cm[1,1]}")

# ── 6. KAYDET (Kişi 2 ile paylaşmak için) ────────────────────
baseline = {
    "model"     : "RandomForestClassifier",
    "dataset"   : "Rain in Australia",
    "n_train"   : int(X_train.shape[0]),
    "n_test"    : int(X_test.shape[0]),
    "features"  : list(X.columns),
    "metrics": {
        "accuracy"  : round(acc,  4),
        "precision" : round(prec, 4),
        "recall"    : round(rec,  4),
        "f1"        : round(f1,   4)
    },
    "confusion_matrix": cm.tolist()
}

with open("baseline_results.json", "w", encoding="utf-8") as f:
    json.dump(baseline, f, indent=2, ensure_ascii=False)

print("\n  'baseline_results.json' kaydedildi.")
print("  Bu dosyayı Kişi 2 ile paylaş — noise eklenmiş")
print("  veriyi aynı modelden geçirip karşılaştıracağız.\n")

# ── 7. GRAFİKLER (Modern & Minimal) ─────────────────────────
plt.rcParams.update({
    "font.family"      : "Segoe UI",
    "axes.spines.top"  : False,
    "axes.spines.right": False,
    "axes.spines.left" : False,
    "axes.spines.bottom": False,
    "axes.grid"        : True,
    "grid.color"       : "#F0F0F0",
    "grid.linewidth"   : 0.8,
    "figure.facecolor" : "white",
    "axes.facecolor"   : "white",
})

ACCENT   = "#2563EB"   # ana mavi
SOFT     = "#DBEAFE"   # açık mavi (arka plan vurgu)
GRAY     = "#6B7280"   # metin ikincil
DARK     = "#111827"   # başlık

fig = plt.figure(figsize=(14, 6))
fig.patch.set_facecolor("white")

# ── Sol: Confusion Matrix ─────────────────────────────
ax1 = fig.add_subplot(1, 2, 1)
cmap = sns.light_palette(ACCENT, as_cmap=True)
sns.heatmap(
    cm, annot=True, fmt="d", cmap=cmap,
    xticklabels=["No Rain", "Rain"],
    yticklabels=["No Rain", "Rain"],
    linewidths=3, linecolor="white",
    annot_kws={"size": 14, "weight": "bold", "color": "white"},
    cbar=False, ax=ax1
)
ax1.set_title("Confusion Matrix", fontsize=13, fontweight="bold",
              color=DARK, pad=16)
ax1.set_ylabel("Gerçek",  fontsize=10, color=GRAY, labelpad=10)
ax1.set_xlabel("Tahmin",  fontsize=10, color=GRAY, labelpad=10)
ax1.tick_params(colors=GRAY, labelsize=10)

# ── Sağ: Metrik Barları ───────────────────────────────
ax2 = fig.add_subplot(1, 2, 2)
metrics = {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1}
labels  = list(metrics.keys())
values  = list(metrics.values())
colors  = [ACCENT if v >= 0.75 else "#93C5FD" for v in values]

bars = ax2.barh(labels[::-1], values[::-1], color=colors[::-1],
                height=0.45, zorder=2)
ax2.set_xlim(0, 1.15)
ax2.set_title("Performans Metrikleri", fontsize=13, fontweight="bold",
              color=DARK, pad=16)
ax2.tick_params(axis="y", colors=DARK, labelsize=11)
ax2.tick_params(axis="x", colors=GRAY, labelsize=9)
ax2.xaxis.set_ticklabels([])

for bar, val in zip(bars, values[::-1]):
    ax2.text(val + 0.02, bar.get_y() + bar.get_height() / 2,
             f"{val:.3f}", va="center", fontsize=12,
             fontweight="bold", color=DARK)

# Referans çizgisi
ax2.axvline(x=0.75, color="#E5E7EB", linewidth=1.2,
            linestyle="--", zorder=1)
ax2.text(0.755, -0.55, "0.75", fontsize=8, color=GRAY)

fig.suptitle("Hafta 1  ·  Baseline Model  ·  Rain in Australia",
             fontsize=15, fontweight="bold", color=DARK, y=1.02)

plt.tight_layout(pad=3)
plt.savefig("hafta1_baseline_grafik.png", dpi=180,
            bbox_inches="tight", facecolor="white")
plt.show()
print("  'hafta1_baseline_grafik.png' kaydedildi.")

# ── 8. EN ÖNEMLİ FEATURE'LAR ────────────────────────────────
fi = pd.Series(model.feature_importances_, index=X.columns)
fi = fi.sort_values(ascending=False).head(10)

fig2, ax3 = plt.subplots(figsize=(10, 5))
fig2.patch.set_facecolor("white")
bar_colors = [ACCENT if i < 3 else "#93C5FD" for i in range(len(fi))]
bars2 = ax3.barh(fi.index[::-1], fi.values[::-1],
                 color=bar_colors[::-1], height=0.5, zorder=2)

ax3.set_title("En Önemli 10 Feature  ·  Baseline Model",
              fontsize=13, fontweight="bold", color=DARK, pad=16)
ax3.set_xlabel("Importance", fontsize=10, color=GRAY)
ax3.tick_params(axis="y", colors=DARK, labelsize=10)
ax3.tick_params(axis="x", colors=GRAY, labelsize=9)
ax3.spines[:].set_visible(False)
ax3.grid(axis="x", color="#F0F0F0", linewidth=0.8, zorder=1)

for bar, val in zip(bars2, fi.values[::-1]):
    ax3.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
             f"{val:.3f}", va="center", fontsize=9, color=GRAY)

plt.tight_layout()
plt.savefig("hafta1_feature_importance.png", dpi=180,
            bbox_inches="tight", facecolor="white")
plt.show()
print("  'hafta1_feature_importance.png' kaydedildi.")

print("\n HAFTA 1 TAMAMLANDI.")
print("  Çıktılar:")
print("    baseline_results.json         — Kişi 2 ile paylaş")
print("    hafta1_baseline_grafik.png    — Confusion matrix + metrikler")
print("    hafta1_feature_importance.png — Hangi feature önemli?")
