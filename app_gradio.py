"""
Sosyal Medya Bağımlılık Dedektörü — Gradio Web Demo
"""
import numpy as np, json, pickle, os
import tensorflow as tf
import gradio as gr

# ─────────────────────────────────────────
# Artifacts Yükle
# ─────────────────────────────────────────
def find_artifacts_dir():
    for path in ['/kaggle/working', '.', './output', '/kaggle/input']:
        if os.path.exists(f'{path}/addiction_model.keras'):
            return path
    raise FileNotFoundError("❌ Model bulunamadı! Önce: python train_model.py")

BASE = find_artifacts_dir()
model          = tf.keras.models.load_model(f'{BASE}/addiction_model.keras')
with open(f'{BASE}/scaler.pkl', 'rb') as f:         scaler         = pickle.load(f)
with open(f'{BASE}/label_encoders.pkl', 'rb') as f: label_encoders = pickle.load(f)
with open(f'{BASE}/feature_cols.json') as f:        feature_cols   = json.load(f)

# ─────────────────────────────────────────
# Kolon Adı Türkçeleştirme
# ─────────────────────────────────────────
TR_LABELS = {
    'Age': 'Yaş', 'Gender': 'Cinsiyet',
    'Daily_Screen_Time_Hours': 'Günlük Ekran Süresi (saat)',
    'Late_Night_Usage': 'Gece Kullanım Sıklığı',
    'GAD_7_Score': 'Anksiyete Skoru (GAD-7)',
    'PHQ_9_Score': 'Depresyon Skoru (PHQ-9)',
    'Sleep_Duration_Hours': 'Günlük Uyku (saat)',
    'Sleep_Hours': 'Günlük Uyku (saat)',
    'Relationship_Status': 'İlişki Durumu', 'Occupation': 'Meslek',
    'Platforms_Used': 'Platform Sayısı',
    'User_Archetype': 'Kullanıcı Profili',
    'Primary_Platform': 'Ana Platform',
    'Dominant_Content_Type': 'Tüketilen İçerik Tipi',
    'Activity_Type': 'Aktivite Türü',
    'Social_Comparison_Trigger': 'Karşılaştırma Tetiği',
    'FOMO': 'FOMO Seviyesi', 'Self_Esteem': 'Özgüven',
    'Productivity': 'Verimlilik',
}
def tr(col): return TR_LABELS.get(col, col.replace('_', ' '))

# ─────────────────────────────────────────
# Dropdown Değerlerini Türkçeleştirme
# ─────────────────────────────────────────
VALUE_TR = {
    'Gender': {'Male': 'Erkek', 'Female': 'Kadın', 'Non-binary': 'Non-binary', 'Other': 'Diğer'},
    'User_Archetype': {
        'Average User': 'Ortalama Kullanıcı',
        'Digital Minimalist': 'Dijital Minimalist',
        'Hyper-Connected': 'Aşırı Bağımlı',
        'Passive Scroller': 'Pasif Kaydırıcı',
    },
    'Primary_Platform': {  # çoğu marka adı değişmez
        'Facebook':'Facebook','Instagram':'Instagram','TikTok':'TikTok',
        'Twitter':'Twitter / X','YouTube':'YouTube','Snapchat':'Snapchat',
        'LinkedIn':'LinkedIn','Reddit':'Reddit','WhatsApp':'WhatsApp',
    },
    'Dominant_Content_Type': {
        'Educational/Tech':'Eğitim / Teknoloji', 'News':'Haber',
        'Entertainment':'Eğlence', 'Sports':'Spor', 'Lifestyle':'Yaşam Tarzı',
        'Gaming':'Oyun', 'Politics':'Siyaset', 'Memes':'Mizah / Meme',
        'Fashion':'Moda', 'Food':'Yemek',
    },
    'Activity_Type': {
        'Active': 'Aktif (paylaşan)', 'Passive': 'Pasif (sadece tüketen)',
    },
    'Relationship_Status': {
        'Single':'Bekar','In Relationship':'İlişkide',
        'Married':'Evli','Divorced':'Boşanmış',
    },
    'Occupation': {
        'Student':'Öğrenci','Employee':'Çalışan','Freelancer':'Serbest Çalışan',
        'Unemployed':'İşsiz','Self-Employed':'Kendi İşinin Sahibi','Retired':'Emekli',
    },
    'Social_Comparison_Trigger': {
        'Low':'Düşük','Medium':'Orta','High':'Yüksek',
        'None':'Yok','Frequent':'Sık',
    },
}
def tr_val(col, eng):     return VALUE_TR.get(col, {}).get(eng, eng)
def rev_val(col, turkce):
    for eng, t in VALUE_TR.get(col, {}).items():
        if t == turkce: return eng
    return turkce

# ─────────────────────────────────────────
# Dinamik Form
# ─────────────────────────────────────────
def build_input_for(col):
    lbl = tr(col); low = col.lower()
    if col in label_encoders:
        eng = list(label_encoders[col].classes_)
        turkish = [tr_val(col, c) for c in eng]
        return gr.Dropdown(choices=turkish, value=turkish[0], label=lbl, interactive=True)
    if 'age' in low and 'usage' not in low:
        return gr.Slider(13, 80, value=22, step=1, label=lbl)
    if 'sleep' in low:                                  return gr.Slider(0, 14, value=7, step=0.5, label=lbl)
    if 'screen_time' in low or 'hour' in low:           return gr.Slider(0, 14, value=3, step=0.5, label=lbl)
    if 'gad' in low:                                     return gr.Slider(0, 21, value=5, step=1, label=lbl)
    if 'phq' in low:                                     return gr.Slider(0, 27, value=5, step=1, label=lbl)
    if 'platform' in low or 'count' in low:              return gr.Slider(1, 10, value=3, step=1, label=lbl)
    if 'check' in low:                                   return gr.Slider(1, 80, value=15, step=1, label=lbl)
    return gr.Slider(1, 5, value=3, step=1, label=lbl)

# ─────────────────────────────────────────
# Tahmin
# ─────────────────────────────────────────
LEVELS = {
    1: ("#10b981","✅ Sağlıklı",         "Kullanımın dengeli. Mevcut alışkanlıklarını koru!"),
    2: ("#84cc16","🟡 Dikkat",            "Küçük risk işaretleri var. Ekran süreni takip et."),
    3: ("#f59e0b","🟠 Risk Altında",      "Belirgin bağımlılık var. Dijital detoks dene."),
    4: ("#ef4444","🔴 Bağımlılık Başlıyor","Ciddi uyarı. Uzman desteği almayı değerlendir."),
    5: ("#991b1b","🚨 Ciddi Bağımlılık",  "Profesyonel destek şiddetle tavsiye edilir."),
}
LEVEL_NAMES  = ['Sağlıklı','Dikkatli','Risk','Bağımlılık Başlıyor','Ciddi Bağımlılık']
LEVEL_COLORS = ['#10b981','#84cc16','#f59e0b','#ef4444','#991b1b']

def predict(*args):
    kullanici = []
    for col, val in zip(feature_cols, args):
        if col in label_encoders:
            eng = rev_val(col, val)
            kullanici.append(float(label_encoders[col].transform([eng])[0]))
        else:
            kullanici.append(float(val))
    veri  = scaler.transform([kullanici])
    probs = model.predict(veri, verbose=0)[0]
    seviye = int(np.argmax(probs)) + 1
    color, label, advice = LEVELS[seviye]

    bars = ""
    for i,(name,c) in enumerate(zip(LEVEL_NAMES, LEVEL_COLORS)):
        pct = probs[i]*100
        active = (i+1)==seviye
        w = "700" if active else "500"
        tc = "#0f172a" if active else "#64748b"
        bars += f"""
        <div style="margin-bottom:12px;">
          <div style="display:flex;justify-content:space-between;margin-bottom:6px;
                      font-size:13px;font-weight:{w};color:{tc};">
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
                box-shadow:0 4px 20px -8px rgba(0,0,0,0.08);">
      <div style="text-align:center;margin-bottom:22px;">
        <div style="display:inline-block;padding:5px 13px;border-radius:999px;
                    background:{color}15;color:{color};font-size:11px;font-weight:700;
                    letter-spacing:1.5px;text-transform:uppercase;margin-bottom:12px;">
          Seviye {seviye} / 5
        </div>
        <div style="font-size:26px;font-weight:800;color:{color};">{label}</div>
      </div>
      <div style="background:{color}10;border-left:3px solid {color};border-radius:8px;
                  padding:14px 16px;margin-bottom:22px;font-size:14px;line-height:1.6;color:#334155;">
        💡 {advice}
      </div>
      <div style="font-size:11px;font-weight:700;color:#64748b;text-transform:uppercase;
                  letter-spacing:1.5px;margin-bottom:12px;">Olasılık Dağılımı</div>
      {bars}
    </div>"""

# ─────────────────────────────────────────
# CSS — Uyumlu renkler + dropdown aşağı açılsın
# ─────────────────────────────────────────
css = """
/* ─── Genel ─── */
.gradio-container {
    background: #f1f5f9 !important;
    font-family: system-ui, -apple-system, 'Segoe UI', sans-serif !important;
    max-width: 1400px !important; margin: 0 auto !important;
}
.gradio-container * { box-sizing: border-box; }

/* ─── Başlıklar / metin ─── */
.gradio-container h1, .gradio-container h2, .gradio-container h3,
.prose h1, .prose h2, .prose h3 { color: #0f172a !important; font-weight: 700 !important; }
.prose, .prose p { color: #475569 !important; }

/* ─── Kartlar ─── */
.gr-block, .gr-form, .gr-panel, .gr-box, .block {
    background: white !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 12px !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.03) !important;
}

/* ─── Label ─── */
label, label span, .gr-form label,
span[data-testid="block-info"], .gr-block label span {
    color: #1e293b !important;
    font-size: 14px !important;
    font-weight: 600 !important;
    background: transparent !important;   /* label arkaplanı temiz */
}

/* ─── Input / dropdown / slider sayı kutusu ─── */
input, select, textarea, .gr-input, .gr-dropdown,
input[type=number], input[type=text] {
    background: #f8fafc !important;
    color: #0f172a !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 8px !important;
    font-size: 14px !important;
}
input:focus, select:focus, textarea:focus {
    border-color: #8b5cf6 !important; outline: none !important;
    box-shadow: 0 0 0 3px rgba(139,92,246,0.15) !important;
}

/* ─── Slider ─── */
input[type=range] { accent-color: #8b5cf6 !important; }
.gr-slider .min, .gr-slider .max { color: #64748b !important; }

/* ─── DROPDOWN: popup aşağı açılsın ─── */
ul[role="listbox"],
.gradio-container ul[role="listbox"],
.wrap.svelte-1ixn6qd,
.options.svelte-yuohum {
    position: absolute !important;
    top: 100% !important;
    bottom: auto !important;
    left: 0 !important;
    transform: none !important;
    max-height: 220px !important;
    overflow-y: auto !important;
    background: white !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 8px !important;
    box-shadow: 0 10px 25px -8px rgba(0,0,0,0.15) !important;
    z-index: 9999 !important;
    margin-top: 4px !important;
}
ul[role="listbox"] li {
    color: #0f172a !important;
    background: white !important;
    padding: 8px 12px !important;
}
ul[role="listbox"] li:hover, ul[role="listbox"] li[aria-selected="true"] {
    background: #f1f5f9 !important;
}

/* Dropdown wrapper'ı relative yapıp popup'ın içinde konumlanmasını sağla */
.gr-dropdown, [data-testid="dropdown"] {
    position: relative !important;
    overflow: visible !important;
}

/* ─── Buton ─── */
button.primary, .gr-button-primary {
    background: linear-gradient(135deg, #8b5cf6, #6366f1) !important;
    color: white !important; border: none !important;
    font-weight: 600 !important; font-size: 15px !important;
    border-radius: 10px !important; padding: 12px 24px !important;
    box-shadow: 0 10px 20px -8px rgba(139,92,246,0.4) !important;
    transition: all 0.2s ease !important;
}
button.primary:hover { transform: translateY(-1px) !important; }

footer { display: none !important; }
"""

# ─────────────────────────────────────────
# Arayüz
# ─────────────────────────────────────────
with gr.Blocks(title="Sosyal Medya Bağımlılık Dedektörü", css=css,
               theme=gr.themes.Soft(primary_hue="violet", neutral_hue="slate")) as demo:

    gr.HTML("""
    <div style="text-align:center;padding:28px 0 12px;">
      <div style="font-size:12px;color:#8b5cf6;font-weight:700;letter-spacing:2px;
                  text-transform:uppercase;margin-bottom:10px;">Yapay Zeka Destekli Analiz</div>
      <div style="font-size:32px;font-weight:800;color:#0f172a;margin-bottom:8px;">
        🧠 Sosyal Medya Bağımlılık Dedektörü</div>
      <div style="font-size:15px;color:#64748b;max-width:600px;margin:0 auto;">
        Soruları dürüstçe yanıtla — sonuçların sana kalır, hiçbir veri kaydedilmez.</div>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=3):
            gr.Markdown("### 📋 Anket")
            inputs = []
            for i in range(0, len(feature_cols), 2):
                with gr.Row():
                    for col in feature_cols[i:i+2]:
                        inputs.append(build_input_for(col))
            btn = gr.Button("🔍 Analiz Et", variant="primary", size="lg")

        with gr.Column(scale=2):
            gr.Markdown("### 📊 Sonuç")
            result = gr.HTML(value="""
            <div style="background:white;border:2px dashed #e2e8f0;border-radius:16px;
                        padding:60px 30px;text-align:center;font-family:system-ui,sans-serif;">
              <div style="font-size:56px;margin-bottom:18px;">🔮</div>
              <div style="color:#64748b;font-size:15px;line-height:1.6;">
                Soruları yanıtla ve<br>
                <strong style="color:#8b5cf6;font-size:17px;">Analiz Et</strong> butonuna bas</div>
            </div>""")

    btn.click(fn=predict, inputs=inputs, outputs=result)

if __name__ == '__main__':
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True, show_error=True)