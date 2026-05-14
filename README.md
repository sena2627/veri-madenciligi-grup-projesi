# Veri Gürültüsü Pattern Öğrenme ve Model Dayanıklılık Sistemi

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
