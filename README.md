# 🧠 Sosyal Medya Bağımlılık Dedektörü

Yapay zeka destekli sosyal medya bağımlılığı sınıflandırıcı.  
Dense sinir ağı ile 5 seviyeli analiz: **Sağlıklı → Ciddi Bağımlılık**

---

## 📁 Dosya Yapısı

```
social_media_detector/
├── train_model.py       # Model eğitimi (burayı önce çalıştır!)
├── app_gradio.py        # Web demo arayüzü (Gradio)
├── api.py               # REST API (Flask)
├── requirements.txt     # Bağımlılıklar
└── README.md
```

**Eğitim sonrası otomatik oluşur:**
```
├── social_media_usage.csv   # Sentetik veri seti
├── addiction_model.keras    # Eğitilmiş model
├── scaler.pkl               # StandardScaler
├── label_encoders.pkl       # LabelEncoder
├── feature_cols.json        # Özellik isimleri
└── training_results.png     # Eğitim grafikleri
```

---

## 🚀 Kurulum & Çalıştırma

### 1. Gereksinimleri yükle
```bash
pip install -r requirements.txt
```

### 2. Modeli eğit
```bash
python train_model.py
```
Çıktı: Test accuracy ~%75-80, `training_results.png` grafik

### 3. Web Demo (Gradio)
```bash
python app_gradio.py
# Tarayıcıda aç: http://localhost:7860
```

### 4. REST API (Flask)  
```bash
python api.py
# Endpoint: http://localhost:5000/predict
```

---

## 📊 Model Mimarisi

```
Giriş (19 özellik)
    ↓
Dense(128, ReLU) → BatchNorm → Dropout(0.3)
    ↓
Dense(64, ReLU)  → BatchNorm → Dropout(0.3)
    ↓
Dense(32, ReLU)  → Dropout(0.2)
    ↓
Dense(5, Softmax)  ← 5 bağımlılık seviyesi
```

---

## 📋 API Kullanımı

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 22,
    "gender": "male",
    "relationship": "single",
    "occupation": "student",
    "daily_hours": 6,
    "platforms_count": 5,
    "checks_per_day": 40,
    "night_usage": 4,
    "fomo_score": 4,
    "distraction": 4,
    "restlessness": 3,
    "anxiety": 3,
    "depression": 2,
    "self_comparison": 4,
    "validation_seek": 4,
    "sleep_issues": 3,
    "productivity_loss": 4,
    "relationship_harm": 2,
    "purpose_less": 4
  }'
```

**Yanıt:**
```json
{
  "level": 4,
  "label": "Bağımlılık Başlıyor",
  "emoji": "🔴",
  "color": "#ef4444",
  "advice": "Ciddi uyarı! Uzman desteği faydalı olabilir.",
  "confidence": 0.72,
  "probabilities": { ... }
}
```

---

## 🎯 Bağımlılık Seviyeleri

| Seviye | Etiket | Açıklama |
|--------|--------|----------|
| 1 | ✅ Sağlıklı | Dengeli kullanım |
| 2 | 🟡 Dikkatli | Küçük riskler var |
| 3 | 🟠 Risk Altında | Belirgin bağımlılık işaretleri |
| 4 | 🔴 Bağımlılık Başlıyor | Ciddi uyarı seviyesi |
| 5 | 🚨 Ciddi Bağımlılık | Profesyonel destek gerekli |

---

## 📚 Gerçek Veri Seti (Kaggle)

[Social Media & Mental Health Dataset](https://www.kaggle.com/datasets/souvikahmed071/social-media-and-mental-health)  
481 kullanıcı · 20 özellik · Etiketli

İndirdikten sonra:
```python
# train_model.py → load_and_preprocess() fonksiyonunda:
df = pd.read_csv('your_kaggle_file.csv')
# Sütun isimlerini eşleştir
```

---

## 🏆 Sunum İpuçları

1. **Demo anı**: Hocana canlı anketi doldurt → anında sonuç
2. **Neden bu proje?** "Bağımlılık yaşadım ve çözmek istedim"
3. **Akademik bağ**: Güncel araştırmalar ile ilişkilendir
4. **Teknik derinlik**: Model mimarisi & karar sürecini anlat
