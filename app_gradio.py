"""
Sosyal Medya Bağımlılık Dedektörü — Gradio Web Demo
=====================================================
Çalıştırma: python app_gradio.py
"""

import numpy as np
import json
import pickle
import os
import tensorflow as tf
import gradio as gr

# ─────────────────────────────────────────
# Artifacts Yükle
# ─────────────────────────────────────────
def find_artifacts_dir():
    """Model dosyalarının bulunduğu dizini bul."""
    for path in ['/kaggle/working', '.', './output']:
        if os.path.exists(f'{path}/addiction_model.keras'):
            return path
    raise FileNotFoundError(
        "❌ Model bulunamadı!\nÖnce: python train_model.py"
    )

BASE = find_artifacts_dir()
model          = tf.keras.models.load_model(f'{BASE}/addiction_model.keras')
with open(f'{BASE}/scaler.pkl', 'rb') as f:          scaler         = pickle.load(f)
with open(f'{BASE}/label_encoders.pkl', 'rb') as f:  label_encoders = pickle.load(f)
with open(f'{BASE}/feature_cols.json') as f:         feature_cols   = json.load(f)

print(f"✅ Model yüklendi. {len(feature_cols)} özellik bekleniyor:")
for c in feature_cols: print(f"   • {c}")

# ─────────────────────────────────────────
# Türkçe Etiket Sözlüğü (olanı çevirir, olmayan İngilizce kalır)
# ─────────────────────────────────────────
TR_LABELS = {
    'Age':                     '👤 Yaş',
    'Gender':                  '⚧️ Cinsiyet',
    'Daily_Screen_Time_Hours': '📱 Günlük Ekran Süresi (saat)',
    'Late_Night_Usage':        '🌙 Gece Kullanımı (1-5)',
    'GAD_7_Score':             '😰 Anksiyete Skoru (GAD-7)',
    'PHQ_9_Score':             '😔 Depresyon Skoru (PHQ-9)',
    'Social_Media_Usage':      '📲 Sosyal Medya Kullanımı',
    'Platforms_Used':          '🔢 Kullanılan Platform Sayısı',
    'Sleep_Hours':             '😴 Günlük Uyku (saat)',
    'Relationship_Status':     '💞 İlişki Durumu',
    'Occupation':              '💼 Meslek',
    'FOMO':                    '🎭 FOMO Seviyesi',
    'Self_Esteem':             '💪 Özgüven',
    'Productivity':            '📈 Verimlilik',
}

def tr(col):
    return TR_LABELS.get(col, f"📊 {col.replace('_', ' ')}")

# ─────────────────────────────────────────
# Dinamik Form Alanı Üreteci
# ─────────────────────────────────────────
def build_input_for(col):
    """Kolon adına/tipine göre uygun Gradio bileşeni üret."""
    lbl = tr(col)

    # Kategorik → Dropdown (orijinal etiketlerle)
    if col in label_encoders:
        choices = list(label_encoders[col].classes_)
        return gr.Dropdown(choices=choices, value=choices[0], label=lbl)

    # İsimden aralık tahmin et
    low = col.lower()
    if 'age' in low:
        return gr.Slider(13, 80, value=22, step=1, label=lbl)
    if 'hour' in low or 'time' in low or 'sleep' in low:
        return gr.Slider(0, 14, value=4, step=0.5, label=lbl)
    if 'gad' in low:   # GAD-7: 0-21
        return gr.Slider(0, 21, value=5, step=1, label=lbl)
    if 'phq' in low:   # PHQ-9: 0-27
        return gr.Slider(0, 27, value=5, step=1, label=lbl)
    if 'count' in low or 'platforms' in low or 'number' in low:
        return gr.Slider(1, 10, value=3, step=1, label=lbl)
    if 'check' in low:
        return gr.Slider(1, 80, value=15, step=1, label=lbl)
    # Varsayılan: 1-5 likert ölçeği
    return gr.Slider(1, 5, value=2, step=1, label=lbl)

# ─────────────────────────────────────────
# Tahmin Fonksiyonu
# ─────────────────────────────────────────
LEVELS = {
    1: ("#10b981", "✅ Sağlıklı Kullanıcı",     "Sosyal medya kullanımın dengeli. Mevcut alışkanlıklarını korumaya devam et!"),
    2: ("#84cc16", "🟡 Dikkat Gerekli",         "Küçük risk işaretleri var. Ekran süreni takip etmeye başla."),
    3: ("#f59e0b", "🟠 Risk Altında",           "Belirgin bağımlılık örüntüleri var. Haftada 1 gün dijital detoks dene."),
    4: ("#ef4444", "🔴 Bağımlılık Başlıyor",    "Ciddi uyarı! Uzman desteği almayı değerlendirmelisin."),
    5: ("#dc2626", "🚨 Ciddi Bağımlılık",       "Profesyonel destek şiddetle tavsiye edilir."),
}
LEVEL_NAMES  = ['Sağlıklı', 'Dikkatli', 'Risk', 'Bağımlılık Başlıyor', 'Ciddi Bağımlılık']
LEVEL_COLORS = ['#10b981', '#84cc16', '#f59e0b', '#ef4444', '#dc2626']

def predict(*args):
    # Gelen değerleri feature_cols sırasıyla eşleştir, kategorikleri encode et
    kullanici = []
    for col, val in zip(feature_cols, args):
        if col in label_encoders:
            kullanici.append(float(label_encoders[col].transform([val])[0]))
        else:
            kullanici.append(float(val))

    veri = scaler.transform([kullanici])
    probs = model.predict(veri, verbose=0)[0]
    seviye = int(np.argmax(probs)) + 1
    color, label, advice = LEVELS[seviye]

    # Olasılık çubukları
    bars = ""
    for i, (name, c) in enumerate(zip(LEVEL_NAMES, LEVEL_COLORS)):
        pct = probs[i] * 100
        active = (i + 1) == seviye
        weight = "700" if active else "500"
        opacity = "1" if active else "0.65"
        bars += f"""
        <div style="margin-bottom:10px;opacity:{opacity};">
          <div style="display:flex;justify-content:space-between;margin-bottom:4px;
                      font-size:13px;font-weight:{weight};color:#f1f5f9;">
            <span>{name}</span><span>{pct:.1f}%</span>
          </div>
          <div style="background:rgba(255,255,255,0.08);border-radius:999px;height:8px;overflow:hidden;">
            <div style="width:{pct:.1f}%;background:{c};height:100%;border-radius:999px;
                        box-shadow:0 0 10px {c}66;transition:width 0.6s ease;"></div>
          </div>
        </div>"""

    return f"""
    <div style="background:linear-gradient(145deg,#1e293b,#0f172a);
                border:1px solid {color}55;border-radius:20px;padding:32px;
                font-family:system-ui,-apple-system,'Segoe UI',sans-serif;color:#f1f5f9;
                box-shadow:0 20px 60px -20px {color}44, inset 0 1px 0 rgba(255,255,255,0.05);">
      
      <div style="text-align:center;margin-bottom:28px;">
        <div style="display:inline-block;padding:6px 16px;border-radius:999px;
                    background:{color}22;color:{color};font-size:12px;font-weight:700;
                    letter-spacing:1.5px;text-transform:uppercase;margin-bottom:16px;">
          Seviye {seviye} / 5
        </div>
        <div style="font-size:28px;font-weight:800;color:{color};margin-bottom:8px;
                    text-shadow:0 0 30px {color}66;">{label}</div>
      </div>
      
      <div style="background:{color}15;border-left:3px solid {color};
                  border-radius:10px;padding:16px 18px;margin-bottom:28px;
                  font-size:14px;line-height:1.6;color:#e2e8f0;">
        💡 {advice}
      </div>
      
      <div style="font-size:11px;font-weight:700;color:#94a3b8;
                  text-transform:uppercase;letter-spacing:1.5px;margin-bottom:14px;">
        Bağımlılık Olasılıkları
      </div>
      {bars}
    </div>
    """

# ─────────────────────────────────────────
# CSS — Tüm Gradio bileşenlerine uyumlu
# ─────────────────────────────────────────
css = """
.gradio-container {
    background: linear-gradient(135deg, #0a0e1a 0%, #0f172a 100%) !important;
    font-family: system-ui, -apple-system, 'Segoe UI', sans-serif !important;
}
.gradio-container * { color: #e2e8f0 !important; }
.gradio-container h1, .gradio-container h2, .gradio-container h3 {
    color: #f8fafc !important;
}

/* Bloklar / kartlar */
.gr-block, .gr-form, .gr-panel, .gr-box {
    background: rgba(30, 41, 59, 0.6) !important;
    border: 1px solid rgba(148, 163, 184, 0.15) !important;
    border-radius: 14px !important;
    backdrop-filter: blur(10px);
}

/* Inputs: slider, dropdown, textbox */
.gr-input, .gr-dropdown, input, select, textarea {
    background: rgba(15, 23, 42, 0.8) !important;
    color: #f1f5f9 !important;
    border: 1px solid rgba(148, 163, 184, 0.2) !important;
    border-radius: 8px !important;
}

/* Labels */
label, .gr-checkbox label, .gr-radio label {
    color: #cbd5e1 !important;
    font-size: 13px !important;
    font-weight: 500 !important;
}

/* Slider track/thumb */
input[type=range] { accent-color: #8b5cf6 !important; }

/* Butonlar */
.gr-button {
    background: linear-gradient(135deg, #8b5cf6, #6366f1) !important;
    color: white !important;
    border: none !important;
    font-weight: 600 !important;
    border-radius: 10px !important;
    box-shadow: 0 10px 25px -10px rgba(139, 92, 246, 0.5) !important;
    transition: transform 0.15s ease !important;
}
.gr-button:hover { transform: translateY(-1px) !important; }

.gr-button-secondary {
    background: rgba(148, 163, 184, 0.1) !important;
    border: 1px solid rgba(148, 163, 184, 0.25) !important;
    color: #e2e8f0 !important;
}

/* Başlık markdown alanı */
.prose, .prose * { color: #e2e8f0 !important; }
.prose strong { color: #f8fafc !important; }

footer { display: none !important; }
"""

# ─────────────────────────────────────────
# Arayüz — form dinamik olarak feature_cols'dan üretilir
# ─────────────────────────────────────────
with gr.Blocks(title="Sosyal Medya Bağımlılık Dedektörü", css=css,
               theme=gr.themes.Base(primary_hue="violet", neutral_hue="slate")) as demo:

    gr.HTML("""
    <div style="text-align:center;padding:24px 0 12px;">
      <div style="font-size:14px;color:#8b5cf6;font-weight:700;letter-spacing:2px;
                  text-transform:uppercase;margin-bottom:8px;">
        Yapay Zeka Destekli Analiz
      </div>
      <div style="font-size:36px;font-weight:800;color:#f8fafc;margin-bottom:8px;">
        🧠 Sosyal Medya Bağımlılık Dedektörü
      </div>
      <div style="font-size:15px;color:#94a3b8;max-width:600px;margin:0 auto;">
        Soruları dürüstçe yanıtla — sonuçların sana kalır, hiçbir veri kaydedilmez.
      </div>
    </div>
    """)

    with gr.Row():
        # ── Sol: dinamik form ──
        with gr.Column(scale=3):
            gr.Markdown("### 📋 Anket")
            input_components = []
            # 2 sütunlu grid
            for i in range(0, len(feature_cols), 2):
                with gr.Row():
                    for col in feature_cols[i:i+2]:
                        input_components.append(build_input_for(col))

            btn = gr.Button("🔍 Analiz Et", variant="primary", size="lg")

        # ── Sağ: sonuç ──
        with gr.Column(scale=2):
            gr.Markdown("### 📊 Sonuç")
            result = gr.HTML(value="""
            <div style="background:rgba(30,41,59,0.4);border:1px dashed rgba(148,163,184,0.3);
                        border-radius:20px;padding:60px 30px;text-align:center;
                        font-family:system-ui,sans-serif;">
              <div style="font-size:64px;margin-bottom:20px;opacity:0.5;">🔮</div>
              <div style="color:#cbd5e1;font-size:16px;font-weight:500;">
                Soruları yanıtla ve<br>
                <strong style="color:#8b5cf6;font-size:18px;">Analiz Et</strong>
                butonuna bas
              </div>
            </div>
            """)

    btn.click(fn=predict, inputs=input_components, outputs=result)

if __name__ == '__main__':
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True, show_error=True)