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
    for path in ['/kaggle/working', '.', './output']:
        if os.path.exists(f'{path}/addiction_model.keras'):
            return path
    raise FileNotFoundError("❌ Model bulunamadı! Önce: python train_model.py")

BASE = find_artifacts_dir()
model          = tf.keras.models.load_model(f'{BASE}/addiction_model.keras')
with open(f'{BASE}/scaler.pkl', 'rb') as f:          scaler         = pickle.load(f)
with open(f'{BASE}/label_encoders.pkl', 'rb') as f:  label_encoders = pickle.load(f)
with open(f'{BASE}/feature_cols.json') as f:         feature_cols   = json.load(f)

print(f"✅ Model yüklendi. {len(feature_cols)} özellik:")
for c in feature_cols: print(f"   • {c}")

# ─────────────────────────────────────────
# Türkçe Etiket Sözlüğü
# ─────────────────────────────────────────
TR_LABELS = {
    'Age':                     'Yaş',
    'Gender':                  'Cinsiyet',
    'Daily_Screen_Time_Hours': 'Günlük Ekran Süresi (saat)',
    'Late_Night_Usage':        'Gece Kullanım Sıklığı',
    'GAD_7_Score':             'Anksiyete Skoru (GAD-7)',
    'PHQ_9_Score':             'Depresyon Skoru (PHQ-9)',
    'Social_Media_Usage':      'Sosyal Medya Kullanımı',
    'Platforms_Used':          'Kullanılan Platform Sayısı',
    'Sleep_Duration_Hours':    'Günlük Uyku (saat)',
    'Sleep_Hours':             'Günlük Uyku (saat)',
    'Relationship_Status':     'İlişki Durumu',
    'Occupation':              'Meslek',
    'FOMO':                    'FOMO Seviyesi',
    'Self_Esteem':             'Özgüven',
    'Productivity':            'Verimlilik',
    'User_Archetype':          'Kullanıcı Profili',
    'Primary_Platform':        'Ana Platform',
    'Dominant_Content_Type':   'Tüketilen İçerik Tipi',
    'Activity_Type':           'Aktivite Türü',
    'Social_Comparison_Trigger': 'Karşılaştırma Tetiği',
}

def tr(col):
    return TR_LABELS.get(col, col.replace('_', ' '))

# ─────────────────────────────────────────
# Dinamik Form Alanı Üreteci (DÜZELTİLDİ)
# ─────────────────────────────────────────
def build_input_for(col):
    lbl = tr(col)
    low = col.lower()

    # 1) Kategorik → Dropdown
    if col in label_encoders:
        choices = list(label_encoders[col].classes_)
        return gr.Dropdown(choices=choices, value=choices[0], label=lbl)

    # 2) YAŞ — "age" kelimesi geçiyorsa
    if 'age' in low and 'usage' not in low:
        return gr.Slider(13, 80, value=22, step=1, label=lbl)

    # 3) UYKU / SÜRE saat cinsi
    if 'sleep' in low:
        return gr.Slider(0, 14, value=7, step=0.5, label=lbl)
    if 'screen_time' in low or 'hour' in low or ('time' in low and 'night' not in low):
        return gr.Slider(0, 14, value=3, step=0.5, label=lbl)

    # 4) Klinik ölçekler
    if 'gad' in low: return gr.Slider(0, 21, value=5, step=1, label=lbl)
    if 'phq' in low: return gr.Slider(0, 27, value=5, step=1, label=lbl)

    # 5) Sayılabilir öğeler
    if 'platform' in low or 'count' in low:
        return gr.Slider(1, 10, value=3, step=1, label=lbl)
    if 'check' in low:
        return gr.Slider(1, 80, value=15, step=1, label=lbl)

    # 6) VARSAYILAN: 1-5 Likert (gece_kullanim, fomo, vs hepsi buraya düşer)
    return gr.Slider(1, 5, value=3, step=1, label=lbl)

# ─────────────────────────────────────────
# Tahmin
# ─────────────────────────────────────────
LEVELS = {
    1: ("#059669", "✅ Sağlıklı",              "Sosyal medya kullanımın dengeli. Mevcut alışkanlıklarını korumaya devam et!"),
    2: ("#65a30d", "🟡 Dikkat",                 "Küçük risk işaretleri var. Ekran süreni takip etmeye başla."),
    3: ("#d97706", "🟠 Risk Altında",           "Belirgin bağımlılık örüntüleri var. Dijital detoks dene."),
    4: ("#dc2626", "🔴 Bağımlılık Başlıyor",    "Ciddi uyarı! Uzman desteği almayı değerlendir."),
    5: ("#991b1b", "🚨 Ciddi Bağımlılık",       "Profesyonel destek şiddetle tavsiye edilir."),
}
LEVEL_NAMES  = ['Sağlıklı', 'Dikkatli', 'Risk', 'Bağımlılık Başlıyor', 'Ciddi Bağımlılık']
LEVEL_COLORS = ['#059669', '#65a30d', '#d97706', '#dc2626', '#991b1b']

def predict(*args):
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

    bars = ""
    for i, (name, c) in enumerate(zip(LEVEL_NAMES, LEVEL_COLORS)):
        pct = probs[i] * 100
        active = (i + 1) == seviye
        weight = "700" if active else "500"
        text_color = "#0f172a" if active else "#475569"
        bars += f"""
        <div style="margin-bottom:12px;">
          <div style="display:flex;justify-content:space-between;margin-bottom:6px;
                      font-size:13px;font-weight:{weight};color:{text_color};">
            <span>{name}</span><span>{pct:.1f}%</span>
          </div>
          <div style="background:#f1f5f9;border-radius:999px;height:10px;overflow:hidden;">
            <div style="width:{pct:.1f}%;background:{c};height:100%;border-radius:999px;
                        transition:width 0.6s ease;"></div>
          </div>
        </div>"""

    return f"""
    <div style="background:white;border:1px solid #e2e8f0;border-radius:16px;padding:28px;
                font-family:system-ui,-apple-system,sans-serif;
                box-shadow:0 10px 40px -15px rgba(0,0,0,0.1);">
      
      <div style="text-align:center;margin-bottom:24px;">
        <div style="display:inline-block;padding:6px 14px;border-radius:999px;
                    background:{color}15;color:{color};font-size:11px;font-weight:700;
                    letter-spacing:1.5px;text-transform:uppercase;margin-bottom:14px;">
          Seviye {seviye} / 5
        </div>
        <div style="font-size:26px;font-weight:800;color:{color};">{label}</div>
      </div>
      
      <div style="background:{color}10;border-left:3px solid {color};
                  border-radius:8px;padding:14px 16px;margin-bottom:24px;
                  font-size:14px;line-height:1.6;color:#334155;">
        💡 {advice}
      </div>
      
      <div style="font-size:11px;font-weight:700;color:#64748b;
                  text-transform:uppercase;letter-spacing:1.5px;margin-bottom:14px;">
        Bağımlılık Olasılıkları
      </div>
      {bars}
    </div>
    """

# ─────────────────────────────────────────
# CSS — AÇIK TEMA, YÜKSEK KONTRAST
# ─────────────────────────────────────────
css = """
.gradio-container {
    background: #f8fafc !important;
    font-family: system-ui, -apple-system, 'Segoe UI', sans-serif !important;
    max-width: 1400px !important;
    margin: 0 auto !important;
}

/* Başlıklar */
.gradio-container h1, 
.gradio-container h2, 
.gradio-container h3,
.prose h1, .prose h2, .prose h3 {
    color: #0f172a !important;
    font-weight: 700 !important;
}
.prose, .prose p, .prose strong { color: #334155 !important; }

/* Form blokları — beyaz kart görünümü */
.gr-block, .gr-form, .gr-panel, .gr-box, .block {
    background: white !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 12px !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04) !important;
}

/* LABELS — KOYU VE NET */
label, 
.gr-form label,
span[data-testid="block-info"],
.gr-block label span {
    color: #1e293b !important;
    font-size: 14px !important;
    font-weight: 600 !important;
}

/* Inputs */
input, select, textarea,
.gr-input, .gr-dropdown {
    background: #f8fafc !important;
    color: #0f172a !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 8px !important;
    font-size: 14px !important;
}
input:focus, select:focus, textarea:focus {
    border-color: #8b5cf6 !important;
    outline: none !important;
    box-shadow: 0 0 0 3px rgba(139, 92, 246, 0.15) !important;
}

/* Slider */
input[type=range] { accent-color: #8b5cf6 !important; }
input[type=range]::-webkit-slider-thumb { background: #8b5cf6 !important; }
.gr-slider .min, .gr-slider .max { color: #64748b !important; }

/* Slider number kutusu */
input[type=number] {
    background: #f1f5f9 !important;
    color: #0f172a !important;
    font-weight: 600 !important;
}

/* Dropdown */
.gr-dropdown, 
select,
ul[role="listbox"] {
    background: white !important;
    color: #0f172a !important;
}
ul[role="listbox"] li { color: #0f172a !important; }
ul[role="listbox"] li:hover { background: #f1f5f9 !important; }

/* Ana buton */
button.primary, .gr-button-primary {
    background: linear-gradient(135deg, #8b5cf6, #6366f1) !important;
    color: white !important;
    border: none !important;
    font-weight: 600 !important;
    font-size: 15px !important;
    border-radius: 10px !important;
    padding: 12px 24px !important;
    box-shadow: 0 10px 25px -10px rgba(139, 92, 246, 0.5) !important;
    transition: all 0.2s ease !important;
}
button.primary:hover, .gr-button-primary:hover { 
    transform: translateY(-2px) !important;
    box-shadow: 0 15px 30px -10px rgba(139, 92, 246, 0.6) !important;
}

footer { display: none !important; }
"""

# ─────────────────────────────────────────
# Arayüz
# ─────────────────────────────────────────
with gr.Blocks(title="Sosyal Medya Bağımlılık Dedektörü", css=css,
               theme=gr.themes.Soft(primary_hue="violet", neutral_hue="slate")) as demo:

    gr.HTML("""
    <div style="text-align:center;padding:32px 0 16px;">
      <div style="font-size:12px;color:#8b5cf6;font-weight:700;letter-spacing:2px;
                  text-transform:uppercase;margin-bottom:10px;">
        Yapay Zeka Destekli Analiz
      </div>
      <div style="font-size:34px;font-weight:800;color:#0f172a;margin-bottom:10px;">
        🧠 Sosyal Medya Bağımlılık Dedektörü
      </div>
      <div style="font-size:15px;color:#64748b;max-width:600px;margin:0 auto;line-height:1.5;">
        Soruları dürüstçe yanıtla — sonuçların sana kalır, hiçbir veri kaydedilmez.
      </div>
    </div>
    """)

    with gr.Row():
        # ── Sol: Form ──
        with gr.Column(scale=3):
            gr.Markdown("### 📋 Anket")
            input_components = []
            for i in range(0, len(feature_cols), 2):
                with gr.Row():
                    for col in feature_cols[i:i+2]:
                        input_components.append(build_input_for(col))

            btn = gr.Button("🔍 Analiz Et", variant="primary", size="lg")

        # ── Sağ: Sonuç ──
        with gr.Column(scale=2):
            gr.Markdown("### 📊 Sonuç")
            result = gr.HTML(value="""
            <div style="background:white;border:2px dashed #e2e8f0;
                        border-radius:16px;padding:60px 30px;text-align:center;
                        font-family:system-ui,sans-serif;">
              <div style="font-size:56px;margin-bottom:18px;">🔮</div>
              <div style="color:#475569;font-size:15px;font-weight:500;line-height:1.6;">
                Soruları yanıtla ve<br>
                <strong style="color:#8b5cf6;font-size:17px;">Analiz Et</strong>
                butonuna bas
              </div>
            </div>
            """)

    btn.click(fn=predict, inputs=input_components, outputs=result)

if __name__ == '__main__':
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True, show_error=True)