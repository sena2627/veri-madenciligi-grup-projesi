# Veri Gürültüsü Pattern Öğrenme ve Model Dayanıklılık Sistemi
| Dosya                                  | Açıklama                                                |
| -------------------------------------- | ------------------------------------------------------- |
| `weatherAUS.csv`                       | Ana veri seti                                           |
| `hafta1_model.py`                      | Baseline Random Forest model kodu                       |
| `baseline_results.json`                | Temiz veri baseline sonuçları                           |
| `hafta1_baseline_grafik.png`           | Baseline model performans grafiği                       |
| `hafta1_feature_importance.png`        | Feature importance grafiği                              |
| `hafta2_model.py`                      | Noise üretimi ve karşılaştırma kodları                  |
| `random_missing.csv`                   | Random missing noise uygulanmış veri                    |
| `random_outlier.csv`                   | Random outlier noise uygulanmış veri                    |
| `random_missing_outlier.csv`           | Missing + outlier noise uygulanmış veri                 |
| `pattern_low.csv`                      | Düşük seviye pattern-based noise veri seti              |
| `pattern_medium.csv`                   | Orta seviye pattern-based noise veri seti               |
| `pattern_high.csv`                     | Yüksek seviye pattern-based noise veri seti             |
| `final_results.csv`                    | WeatherAUS üzerindeki ana model karşılaştırma sonuçları |
| `hafta2_karsilastirma.png`             | Noise senaryoları karşılaştırma grafiği                 |
| `hafta2_f1_dusus.png`                  | F1 düşüş grafiği                                        |
| `Ek_Veri_Kod.ipynb`                    | Ek veri seti doğrulama notebook’u                       |
| `extra_dataset_validation_results.csv` | Breast Cancer ve Wine veri seti doğrulama sonuçları     |

## Proje Amacı

Bu projenin amacı, makine öğrenmesi modellerinin farklı veri bozulma senaryoları altında ne kadar dayanıklı olduğunu analiz etmektir.

Gerçek hayatta kullanılan veri setleri her zaman temiz değildir. Verilerde eksik değerler, hatalı ölçümler, uç değerler ve belirli koşullara bağlı bozulmalar görülebilir. Bu nedenle bu projede yalnızca temiz veri üzerinde model performansı ölçülmemiş; aynı zamanda farklı noise senaryoları oluşturularak modelin bu bozulmalara karşı davranışı incelenmiştir.

Proje kapsamında Random Forest modeli kullanılarak hem temiz veri hem de bozulmuş veri setleri üzerinde performans karşılaştırmaları yapılmıştır.

---

## Kullanılan Ana Veri Seti

Ana veri seti olarak WeatherAUS veri seti kullanılmıştır.

Bu veri setinde amaç, hava durumu değişkenlerinden yararlanarak ertesi gün yağmur yağıp yağmayacağını tahmin etmektir.

Hedef değişken:

- `RainTomorrow`

Problem türü:

- Classification

---

## Kullanılan Ek Veri Setleri

Sistemin yalnızca WeatherAUS veri setine özel olmadığını göstermek amacıyla iki ek classification veri seti üzerinde doğrulama yapılmıştır:

- Breast Cancer Wisconsin
- Wine Recognition

Bu ek veri setleri üzerinde temel random noise senaryoları uygulanmış ve model performansındaki değişim ölçülmüştür.

---

## Model

Projede temel model olarak Random Forest Classifier kullanılmıştır.

Random Forest seçilme nedenleri:

- Classification problemleri için güçlü bir baseline modeldir.
- Sayısal ve kategorik özelliklerle çalışmaya uygundur.
- Outlier ve veri bozulmalarına karşı belirli ölçüde dayanıklı davranabilir.
- Feature importance analizi yapılmasına imkan verir.

---

## Noise Senaryoları

Projede iki ana noise yaklaşımı kullanılmıştır.

### 1. Random Noise

Bu aşamada veri setine rastgele bozulmalar eklenmiştir:

- Random Missing Noise
- Random Outlier Noise
- Missing + Outlier Noise

Bu senaryolar, modelin temel veri bozulmalarına karşı nasıl tepki verdiğini görmek için kullanılmıştır.

### 2. Pattern-Based Noise

İkinci aşamada veri bozulmaları rastgele değil, belirli koşullara bağlı olarak oluşturulmuştur.

WeatherAUS veri seti için kullanılan pattern-based noise kuralları:

- `Humidity3pm` yüksekse → `Humidity3pm` değeri missing yapılmıştır.
- `Temp3pm` düşükse → `Temp3pm` üzerinde ölçüm sapması oluşturulmuştur.
- `WindSpeed3pm` yüksekse → `Pressure9am` üzerinde sapma oluşturulmuştur.

Bu kurallar üç farklı seviyede uygulanmıştır:

- Pattern Low
- Pattern Medium
- Pattern High

Amaç, bozulma şiddeti arttığında model performansının nasıl değiştiğini incelemektir.

---

## Değerlendirme Metrikleri

Model performansı aşağıdaki metriklerle değerlendirilmiştir:

- Accuracy
- Precision
- Recall
- F1 Score

Ayrıca model dayanıklılığını daha net yorumlamak için ek metrikler kullanılmıştır:

### Performance Drop

Modelin baseline performansına göre ne kadar kayıp yaşadığını gösterir.

F1 Drop:

```text
F1 Drop = Original F1 - Scenario F1
