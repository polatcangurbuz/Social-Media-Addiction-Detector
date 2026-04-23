"""
Sosyal Medya Bağımlılık Dedektörü — Flask REST API
====================================================
Gradio yerine saf web arayüzü kullanmak isteyenler için.
Çalıştırma: python api.py
Endpoint:   POST /predict  {json}
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pickle
import json
import os
import tensorflow as tf

app = Flask(__name__)
CORS(app)

# Model yükle
model, scaler, feature_cols = None, None, None

def load_model():
    global model, scaler, feature_cols
    model  = tf.keras.models.load_model('addiction_model.keras')
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('feature_cols.json') as f:
        feature_cols = json.load(f)
    print("✅ Model yüklendi.")

GENDER_MAP = {'male':0, 'female':1, 'other':2}
REL_MAP    = {'single':0, 'relationship':1, 'married':2}
OCC_MAP    = {'student':0, 'employee':1, 'freelancer':2, 'unemployed':3}

LEVEL_INFO = {
    1: {"label": "Sağlıklı",             "emoji": "✅", "color": "#22c55e",
        "advice": "Sosyal medya kullanımın dengeli. Böyle devam et!"},
    2: {"label": "Dikkat Gerektiriyor",  "emoji": "🟡", "color": "#84cc16",
        "advice": "Küçük riskler var. Ekran süresini takip etmeye başla."},
    3: {"label": "Risk Altında",         "emoji": "🟠", "color": "#f97316",
        "advice": "Belirgin bağımlılık işaretleri var. Dijital detoks dene."},
    4: {"label": "Bağımlılık Başlıyor",  "emoji": "🔴", "color": "#ef4444",
        "advice": "Ciddi uyarı! Uzman desteği faydalı olabilir."},
    5: {"label": "Ciddi Bağımlılık",     "emoji": "🚨", "color": "#dc2626",
        "advice": "Profesyonel destek almanı şiddetle tavsiye ederiz."}
}


@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None})


@app.route('/predict', methods=['POST'])
def predict():
    """
    Girdi JSON örneği:
    {
        "age": 22,
        "gender": "male",
        "relationship": "single",
        "occupation": "student",
        "daily_hours": 5,
        "platforms_count": 4,
        "checks_per_day": 30,
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
    }
    """
    if not request.is_json:
        return jsonify({"error": "JSON bekleniyor"}), 400
    
    data = request.get_json()
    
    try:
        user_vector = [
            float(data.get('age', 22)),
            float(GENDER_MAP.get(str(data.get('gender','')).lower(), 0)),
            float(REL_MAP.get(str(data.get('relationship','')).lower(), 0)),
            float(OCC_MAP.get(str(data.get('occupation','')).lower(), 0)),
            float(data.get('daily_hours', 3)),
            float(data.get('platforms_count', 3)),
            float(data.get('checks_per_day', 10)),
            float(data.get('night_usage', 2)),
            float(data.get('fomo_score', 2)),
            float(data.get('distraction', 2)),
            float(data.get('restlessness', 2)),
            float(data.get('anxiety', 2)),
            float(data.get('depression', 2)),
            float(data.get('self_comparison', 2)),
            float(data.get('validation_seek', 2)),
            float(data.get('sleep_issues', 2)),
            float(data.get('productivity_loss', 2)),
            float(data.get('relationship_harm', 2)),
            float(data.get('purpose_less', 2)),
        ]
    except (KeyError, ValueError) as e:
        return jsonify({"error": f"Geçersiz veri: {str(e)}"}), 422
    
    # Tahmin
    scaled = scaler.transform([user_vector])
    probs  = model.predict(scaled, verbose=0)[0]
    level  = int(np.argmax(probs)) + 1
    info   = LEVEL_INFO[level]
    
    response = {
        "level": level,
        "label": info["label"],
        "emoji": info["emoji"],
        "color": info["color"],
        "advice": info["advice"],
        "confidence": float(probs[level-1]),
        "probabilities": {
            "Sağlıklı":            round(float(probs[0]), 4),
            "DikkatGerektiriyor":  round(float(probs[1]), 4),
            "RiskAltında":         round(float(probs[2]), 4),
            "BağımlılıkBaşlıyor":  round(float(probs[3]), 4),
            "CiddiBağımlılık":     round(float(probs[4]), 4),
        }
    }
    return jsonify(response)


if __name__ == '__main__':
    load_model()
    print("\n🌐 API çalışıyor → http://localhost:5000")
    print("📋 Endpoint: POST /predict")
    print("💡 Sağlık kontrolü: GET /health\n")
    app.run(debug=True, port=5000, host='0.0.0.0')
