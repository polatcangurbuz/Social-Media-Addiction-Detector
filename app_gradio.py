"""
Sosyal Medya Bağımlılık Dedektörü — Gradio Web Demo
"""
import numpy as np, json, pickle, os
import tensorflow as tf
import gradio as gr

# ─────────────────────────────────────────
# Artifacts
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
# Türkçeleştirme
# ─────────────────────────────────────────
TR_LABELS = {
    'Age':'Yaş', 'Gender':'Cinsiyet',
    'Daily_Screen_Time_Hours':'Günlük Ekran Süresi (saat)',
    'Late_Night_Usage':'Gece Kullanım Sıklığı',
    'GAD_7_Score':'Anksiyete (GAD-7)',
    'PHQ_9_Score':'Depresyon (PHQ-9)',
    'Sleep_Duration_Hours':'Günlük Uyku (saat)',
    'Sleep_Hours':'Günlük Uyku (saat)',
    'Relationship_Status':'İlişki Durumu', 'Occupation':'Meslek',
    'Platforms_Used':'Platform Sayısı',
    'User_Archetype':'Kullanıcı Profili',
    'Primary_Platform':'Ana Platform',
    'Dominant_Content_Type':'Tüketilen İçerik',
    'Activity_Type':'Aktivite Türü',
    'Social_Comparison_Trigger':'Karşılaştırma Tetiği',
    'FOMO':'FOMO Seviyesi', 'Self_Esteem':'Özgüven',
    'Productivity':'Verimlilik',
}
def tr(col): return TR_LABELS.get(col, col.replace('_',' '))

VALUE_TR = {
    'Gender': {'Male':'Erkek','Female':'Kadın','Non-binary':'Non-binary','Other':'Diğer'},
    'User_Archetype': {
        'Average User':'Ortalama Kullanıcı','Digital Minimalist':'Dijital Minimalist',
        'Hyper-Connected':'Aşırı Bağımlı','Passive Scroller':'Pasif Kaydırıcı',
    },
    'Dominant_Content_Type': {
        'Educational/Tech':'Eğitim / Teknoloji','News':'Haber',
        'Entertainment':'Eğlence','Sports':'Spor','Lifestyle':'Yaşam Tarzı',
        'Gaming':'Oyun','Politics':'Siyaset','Memes':'Mizah',
        'Fashion':'Moda','Food':'Yemek',
    },
    'Activity_Type': {'Active':'Aktif (paylaşan)','Passive':'Pasif (tüketen)'},
    'Relationship_Status': {
        'Single':'Bekar','In Relationship':'İlişkide',
        'Married':'Evli','Divorced':'Boşanmış',
    },
    'Occupation': {
        'Student':'Öğrenci','Employee':'Çalışan','Freelancer':'Serbest',
        'Unemployed':'İşsiz','Self-Employed':'Kendi İşi','Retired':'Emekli',
    },
    'Social_Comparison_Trigger': {'Low':'Düşük','Medium':'Orta','High':'Yüksek'},
}
def tr_val(col, eng): return VALUE_TR.get(col, {}).get(eng, eng)
def rev_val(col, turkce):
    for eng, t in VALUE_TR.get(col, {}).items():
        if t == turkce: return eng
    return turkce

# ─────────────────────────────────────────
# Kolon kategorize
# ─────────────────────────────────────────
def categorize(col):
    low = col.lower()
    if any(x in low for x in ['age','gender','relationship','occupation','marital']):
        return 'demografik'
    if any(x in low for x in ['gad','phq','fomo','anxiety','depress','self_esteem',
                              'comparison','stress','mental','validation','restless']):
        return 'psikolojik'
    if any(x in low for x in ['sleep','productivity','purpose','harm','loss']):
        return 'yasam'
    return 'kullanim'

SECTIONS = {
    'demografik': ('👤 Demografik Bilgiler', '#3b82f6', 'Kişisel bilgiler'),
    'kullanim':   ('📱 Kullanım Alışkanlıkları', '#f97316', 'Sosyal medya davranışların'),
    'psikolojik': ('🧠 Psikolojik Göstergeler', '#a855f7', 'Duygusal durum'),
    'yasam':      ('💤 Yaşam Kalitesi', '#10b981', 'Günlük yaşama etkileri'),
}

# ─────────────────────────────────────────
# Input builder
# ─────────────────────────────────────────
def build_input_for(col):
    lbl = tr(col); low = col.lower()
    if col in label_encoders:
        eng = list(label_encoders[col].classes_)
        turkish = [tr_val(col, c) for c in eng]
        return gr.Dropdown(choices=turkish, value=turkish[0], label=lbl)
    if 'age' in low and 'usage' not in low:
        return gr.Slider(13, 80, value=22, step=1, label=lbl)
    if 'sleep' in low:                        return gr.Slider(0, 14, value=7, step=0.5, label=lbl)
    if 'screen_time' in low or 'hour' in low: return gr.Slider(0, 14, value=3, step=0.5, label=lbl)
    if 'gad' in low:                          return gr.Slider(0, 21, value=5, step=1, label=lbl)
    if 'phq' in low:                          return gr.Slider(0, 27, value=5, step=1, label=lbl)
    if 'platform' in low or 'count' in low:   return gr.Slider(1, 10, value=3, step=1, label=lbl)
    if 'check' in low:                        return gr.Slider(1, 80, value=15, step=1, label=lbl)
    return gr.Slider(1, 5, value=3, step=1, label=lbl)

# ─────────────────────────────────────────
# Tahmin
# ─────────────────────────────────────────
LEVELS = {
    1: ("#10b981","✅ Sağlıklı","Kullanımın dengeli. Mevcut alışkanlıklarını koru!"),
    2: ("#84cc16","🟡 Dikkat","Küçük risk işaretleri var. Ekran süreni takip et."),
    3: ("#f59e0b","🟠 Risk Altında","Belirgin bağımlılık var. Dijital detoks dene."),
    4: ("#ef4444","🔴 Bağımlılık Başlıyor","Ciddi uyarı. Uzman desteği değerlendir."),
    5: ("#991b1b","🚨 Ciddi Bağımlılık","Profesyonel destek tavsiye edilir."),
}
LEVEL_NAMES = ['Sağlıklı','Dikkatli','Risk','Bağımlılık Başlıyor','Ciddi Bağımlılık']
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
    logits = model.predict(veri, verbose=0)[0]
    temperature = 1.5   # >1 yumuşatır, <1 keskinleştirir
    softened = np.exp(np.log(logits + 1e-9) / temperature)
    probs = softened / softened.sum()
    seviye = int(np.argmax(probs)) + 1
    color, label, advice = LEVELS[seviye]

    bars = ""
    for i,(name,c) in enumerate(zip(LEVEL_NAMES, LEVEL_COLORS)):
        pct = probs[i]*100
        active = (i+1)==seviye
        w = "700" if active else "500"
        tc = "#0f172a" if active else "#64748b"
        bars += f"""
        <div style="margin-bottom:14px;">
          <div style="display:flex;justify-content:space-between;margin-bottom:6px;
                      font-size:13px;font-weight:{w};color:{tc};">
            <span>{name}</span><span>{pct:.1f}%</span>
          </div>
          <div style="background:#f1f5f9;border-radius:999px;height:10px;overflow:hidden;">
            <div style="width:{pct:.1f}%;background:{c};height:100%;border-radius:999px;"></div>
          </div>
        </div>"""

    return f"""
    <div style="background:white;border-radius:20px;padding:32px;
                font-family:system-ui,-apple-system,sans-serif;
                box-shadow:0 20px 50px -20px {color}44, 0 4px 20px -8px rgba(0,0,0,0.08);
                border-top:6px solid {color};">
      <div style="text-align:center;margin-bottom:24px;">
        <div style="display:inline-block;padding:5px 14px;border-radius:999px;
                    background:{color}18;color:{color};font-size:11px;font-weight:700;
                    letter-spacing:1.5px;text-transform:uppercase;margin-bottom:14px;">
          Seviye {seviye} / 5
        </div>
        <div style="font-size:28px;font-weight:800;color:{color};">{label}</div>
      </div>
      <div style="background:{color}0e;border-left:3px solid {color};border-radius:8px;
                  padding:15px 18px;margin-bottom:26px;font-size:14px;line-height:1.6;color:#334155;">
        💡 {advice}
      </div>
      <div style="font-size:11px;font-weight:700;color:#64748b;text-transform:uppercase;
                  letter-spacing:1.5px;margin-bottom:14px;">Olasılık Dağılımı</div>
      {bars}
    </div>"""

# ─────────────────────────────────────────
# CSS — SADECE ARKA PLAN VE BUTON. Form bileşenlerine dokunmuyor.
# ─────────────────────────────────────────
css = """
.gradio-container {
    background: linear-gradient(135deg, #eef2ff 0%, #fdf4ff 50%, #fff7ed 100%) !important;
    max-width: 1400px !important;
    margin: 0 auto !important;
}

button.primary {
    background: linear-gradient(135deg, #8b5cf6, #ec4899) !important;
    border: none !important;
    font-size: 16px !important;
    font-weight: 700 !important;
    padding: 14px 32px !important;
    box-shadow: 0 10px 30px -10px rgba(139,92,246,0.5) !important;
    transition: transform 0.2s !important;
}
button.primary:hover { transform: translateY(-2px) !important; }

footer { display: none !important; }
"""

# ─────────────────────────────────────────
# Arayüz
# ─────────────────────────────────────────
def section_header(title, color, desc):
    return f"""
    <div style="margin: 20px 0 10px 0; padding: 14px 20px;
                background: linear-gradient(90deg, {color}15, transparent);
                border-left: 4px solid {color}; border-radius: 8px;">
      <div style="font-size: 17px; font-weight: 700; color: #0f172a; margin-bottom: 2px;">
        {title}
      </div>
      <div style="font-size: 13px; color: #64748b;">{desc}</div>
    </div>
    """

with gr.Blocks(title="Sosyal Medya Bağımlılık Dedektörü", css=css,
               theme=gr.themes.Soft(primary_hue="violet", neutral_hue="slate")) as demo:

    gr.HTML("""
    <div style="text-align:center;padding:36px 0 20px;">
      <div style="display:inline-block;padding:6px 16px;
                  background:linear-gradient(135deg,#8b5cf6,#ec4899);
                  color:white;border-radius:999px;font-size:11px;font-weight:700;
                  letter-spacing:2px;text-transform:uppercase;margin-bottom:16px;
                  box-shadow:0 8px 20px -6px rgba(139,92,246,0.5);">
        ✨ Yapay Zeka Destekli Analiz
      </div>
      <div style="font-size:38px;font-weight:800;color:#0f172a;margin-bottom:10px;">
        🧠 Sosyal Medya Bağımlılık 
        <span style="background:linear-gradient(135deg,#8b5cf6,#ec4899);
                     -webkit-background-clip:text;background-clip:text;
                     -webkit-text-fill-color:transparent;">Dedektörü</span>
      </div>
      <div style="font-size:15px;color:#475569;max-width:620px;margin:0 auto;">
        Kısa bir anket doldur, anlık analiz al. Sonuçların sana kalır — hiçbir veri kaydedilmez.
      </div>
    </div>
    """)

    # Kolonları gruplara ayır
    grouped = {'demografik':[], 'kullanim':[], 'psikolojik':[], 'yasam':[]}
    for col in feature_cols:
        grouped[categorize(col)].append(col)

    with gr.Row():
        with gr.Column(scale=3):
            inputs_map = {}
            for section_key in ['demografik', 'kullanim', 'psikolojik', 'yasam']:
                cols = grouped[section_key]
                if not cols: continue
                title, color, desc = SECTIONS[section_key]
                gr.HTML(section_header(title, color, desc))
                for i in range(0, len(cols), 2):
                    with gr.Row():
                        for col in cols[i:i+2]:
                            inputs_map[col] = build_input_for(col)
            
            btn = gr.Button("🔍 Analiz Et", variant="primary", size="lg")

        with gr.Column(scale=2):
            gr.HTML("""
            <div style="margin: 20px 0 10px 0; padding: 14px 20px;
                        background: linear-gradient(90deg, #8b5cf615, transparent);
                        border-left: 4px solid #8b5cf6; border-radius: 8px;">
              <div style="font-size: 17px; font-weight: 700; color: #0f172a;">
                📊 Analiz Sonucu
              </div>
              <div style="font-size: 13px; color: #64748b;">Tahmin ve öneriler</div>
            </div>
            """)
            result = gr.HTML(value="""
            <div style="background:white;border:2px dashed #cbd5e1;border-radius:20px;
                        padding:70px 30px;text-align:center;font-family:system-ui,sans-serif;">
              <div style="font-size:72px;margin-bottom:20px;">🔮</div>
              <div style="color:#475569;font-size:15px;line-height:1.7;">
                Anketi doldurup<br>
                <strong style="color:#8b5cf6;font-size:18px;">🔍 Analiz Et</strong>
                butonuna bas
              </div>
            </div>""")

    # predict fonksiyonuna feature_cols sırasıyla input ver
    ordered_inputs = [inputs_map[col] for col in feature_cols]
    btn.click(fn=predict, inputs=ordered_inputs, outputs=result)

if __name__ == '__main__':
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True, show_error=True)