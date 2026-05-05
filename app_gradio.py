"""
Sosyal Medya Bağımlılık Dedektörü — Gradio Web Demo (v3)
=========================================================
v3 GÜNCELLEMELERİ:
  ✨ Tüm İngilizce dropdown değerleri Türkçeye çevrildi
  ✨ Content type ve platform artık dropdown (slider değil)
  ✨ v3 modelle uyumlu (label-encoded kategorikler)
"""
import numpy as np, json, pickle, os
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import gradio as gr

# ─────────────────────────────────────────
# Artifacts yükle
# ─────────────────────────────────────────
def find_artifacts_dir():
    for path in ['/kaggle/working', '.', './output', '/kaggle/input']:
        if os.path.exists(f'{path}/addiction_model.keras'):
            return path
    raise FileNotFoundError("❌ Model bulunamadı! Önce: python train_model.py")

BASE = find_artifacts_dir()
model = tf.keras.models.load_model(f'{BASE}/addiction_model.keras')

with open(f'{BASE}/scaler.pkl', 'rb') as f:         scaler         = pickle.load(f)
with open(f'{BASE}/label_encoders.pkl', 'rb') as f: label_encoders = pickle.load(f)
with open(f'{BASE}/tokenizer.pkl', 'rb') as f:      tokenizer      = pickle.load(f)
with open(f'{BASE}/feature_cols.json') as f:        feature_cols   = json.load(f)
with open(f'{BASE}/config.json') as f:              config         = json.load(f)

MAX_SEQ_LEN = config['max_seq_len']
print(f"✅ Artifacts yüklendi (v{config.get('version', '?')}, "
      f"max_seq_len={MAX_SEQ_LEN}, features={len(feature_cols)})")

# ─────────────────────────────────────────
# Türkçe etiketler
# ─────────────────────────────────────────
TR_LABELS = {
    'Age':'Yaş', 'Gender':'Cinsiyet',
    'Daily_Screen_Time_Hours':'Günlük Ekran Süresi (saat)',
    'Late_Night_Usage':'Gece Kullanım Sıklığı (1-5)',
    'GAD_7_Score':'Anksiyete Skoru (GAD-7, 0-21)',
    'PHQ_9_Score':'Depresyon Skoru (PHQ-9, 0-27)',
    'Sleep_Duration_Hours':'Günlük Uyku (saat)',
    'Sleep_Hours':'Günlük Uyku (saat)',
    'Relationship_Status':'İlişki Durumu', 'Occupation':'Meslek',
    'Platforms_Used':'Platform Sayısı',
    'User_Archetype':'Kullanıcı Profili',
    'Primary_Platform':'Ana Platform',
    'Dominant_Content_Type':'Tüketilen İçerik Türü',
    'Activity_Type':'Aktivite Türü',
    'Social_Comparison_Trigger':'Karşılaştırma Tetiği (0-1)',
    'FOMO':'FOMO Seviyesi', 'Self_Esteem':'Özgüven',
    'Productivity':'Verimlilik',
}
def tr(col): return TR_LABELS.get(col, col.replace('_', ' '))

# ─────────────────────────────────────────
# Türkçe dropdown değerleri (TÜM değerler tercüme edildi)
# ─────────────────────────────────────────
VALUE_TR = {
    'Gender': {
        'Male':'Erkek',
        'Female':'Kadın',
        'Non-binary':'Non-binary',
        'Other':'Diğer',
    },
    'User_Archetype': {
        'Average User':'Ortalama Kullanıcı',
        'Digital Minimalist':'Dijital Minimalist',
        'Hyper-Connected':'Aşırı Bağımlı',
        'Passive Scroller':'Pasif Kaydırıcı',
    },
    'Dominant_Content_Type': {
        'Educational/Tech':'Eğitim / Teknoloji',
        'News/Politics':'Haber / Siyaset',
        'News':'Haber',
        'Entertainment/Comedy':'Eğlence / Komedi',
        'Entertainment':'Eğlence',
        'Sports':'Spor',
        'Lifestyle/Fashion':'Yaşam Tarzı / Moda',
        'Lifestyle':'Yaşam Tarzı',
        'Gaming':'Oyun',
        'Self-Help/Motivation':'Kişisel Gelişim / Motivasyon',
        'Politics':'Siyaset',
        'Memes':'Mizah / Caps',
        'Fashion':'Moda',
        'Food':'Yemek',
    },
    'Activity_Type': {
        'Active':'Aktif (paylaşan)',
        'Passive':'Pasif (tüketen)',
    },
    'Primary_Platform': {
        'Instagram':'Instagram',
        'TikTok':'TikTok',
        'Twitter':'Twitter / X',
        'YouTube':'YouTube',
        'Facebook':'Facebook',
        'Snapchat':'Snapchat',
        'Reddit':'Reddit',
        'LinkedIn':'LinkedIn',
    },
    'Relationship_Status': {
        'Single':'Bekar',
        'In Relationship':'İlişkide',
        'Married':'Evli',
        'Divorced':'Boşanmış',
    },
    'Occupation': {
        'Student':'Öğrenci',
        'Employee':'Çalışan',
        'Freelancer':'Serbest Çalışan',
        'Unemployed':'İşsiz',
        'Self-Employed':'Kendi İşinde',
        'Retired':'Emekli',
    },
    'Social_Comparison_Trigger': {
        'Low':'Düşük',
        'Medium':'Orta',
        'High':'Yüksek',
    },
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
    if col == 'Age':
        return None
    lbl = tr(col); low = col.lower()

    # Tüm kategorik kolonlar artık dropdown
    if col in label_encoders:
        eng = list(label_encoders[col].classes_)
        turkish = [tr_val(col, c) for c in eng]
        return gr.Dropdown(choices=turkish, value=turkish[0], label=lbl)

    if 'comparison' in low or 'trigger' in low:
        return gr.Slider(0, 1, value=0.5, step=0.05, label=lbl)
    if 'sleep' in low:
        return gr.Slider(0, 14, value=7, step=0.5, label=lbl)
    if 'screen_time' in low or 'hour' in low:
        return gr.Slider(0, 14, value=3, step=0.5, label=lbl)
    if 'gad' in low:
        return gr.Slider(0, 21, value=5, step=1, label=lbl)
    if 'phq' in low:
        return gr.Slider(0, 27, value=5, step=1, label=lbl)
    if 'platform' in low or 'count' in low:
        return gr.Slider(1, 10, value=3, step=1, label=lbl)
    if 'check' in low:
        return gr.Slider(1, 80, value=15, step=1, label=lbl)
    if 'late_night' in low or 'night' in low:
        return gr.Slider(1, 5, value=3, step=1, label=lbl)
    return gr.Slider(1, 5, value=3, step=1, label=lbl)

# ─────────────────────────────────────────
# Klinik yorum yardımcıları
# ─────────────────────────────────────────
def gad7_severity(score):
    if score <= 4:  return "Minimal", "#10b981", "Anksiyete belirtin yok denecek kadar az."
    if score <= 9:  return "Hafif",   "#84cc16", "Hafif anksiyete belirtileri var."
    if score <= 14: return "Orta",    "#f59e0b", "Orta düzey anksiyete — takip önerilir."
    return "Şiddetli", "#ef4444", "Yüksek anksiyete — uzman desteği önerilir."

def phq9_severity(score):
    if score <= 4:  return "Minimal", "#10b981", "Depresif belirtilerin minimal düzeyde."
    if score <= 9:  return "Hafif",   "#84cc16", "Hafif depresif belirtiler var."
    if score <= 14: return "Orta",    "#f59e0b", "Orta düzey depresyon — takip önerilir."
    if score <= 19: return "Orta-Şiddetli", "#f97316", "Belirgin depresyon — destek almalısın."
    return "Şiddetli", "#ef4444", "Şiddetli depresyon — profesyonel destek gerekli."

def screen_status(hours):
    if hours <= 2: return "✅ Sağlıklı",  "#10b981"
    if hours <= 4: return "🟡 Ortalama",  "#84cc16"
    if hours <= 6: return "🟠 Yüksek",    "#f59e0b"
    return "🔴 Çok Yüksek", "#ef4444"

def sleep_status(hours):
    if hours < 6:  return "🔴 Yetersiz", "#ef4444", "Yetişkinler için 7-9 saat önerilir."
    if hours < 7:  return "🟠 Sınırda",  "#f59e0b", "Hafif uyku eksikliği var."
    if hours <= 9: return "✅ İdeal",    "#10b981", "Uyku süresi sağlıklı aralıkta."
    return "🟡 Fazla", "#84cc16", "9 saatten fazla uyku da yorgunluk yapabilir."

def get_personalized_recommendations(values, level):
    recs = []
    screen = float(values.get('Daily_Screen_Time_Hours', 0))
    night  = float(values.get('Late_Night_Usage', 0))
    sleep  = float(values.get('Sleep_Duration_Hours', 8))
    gad    = float(values.get('GAD_7_Score', 0))
    phq    = float(values.get('PHQ_9_Score', 0))
    comp   = float(values.get('Social_Comparison_Trigger', 0))

    if screen > 6:
        recs.append(("⏱️", "Ekran süresi limiti",
                    f"Şu an günlük {screen:.1f} saat. Hedef 3 saat. Telefonunda ekran süresi limit özelliğini aç."))
    elif screen > 4:
        recs.append(("⏱️", "Ekran süresi takibi",
                    f"{screen:.1f} saat ortalamanın üstünde. Haftalık raporları kontrol et."))

    if night >= 4:
        recs.append(("🌙", "Gece kullanımını kes",
                    "Yatmadan 1 saat önce telefonu başka odaya bırak. Mavi ışık uykuyu bozar."))
    elif night >= 3:
        recs.append(("🌙", "Yatak öncesi rutin",
                    "Uyumadan 30 dk önce ekrandan uzaklaş; kitap, müzik veya nefes egzersizi tercih et."))

    if sleep < 6:
        recs.append(("😴", "Uyku önceliği",
                    f"{sleep:.1f} saat çok az. Düzenli uyku saati belirle ve telefonu yatak odasından çıkar."))
    elif sleep < 7:
        recs.append(("😴", "Uyku düzeni",
                    "30 dakika daha erken yatmayı dene. Küçük değişiklikler büyük fark yaratır."))

    if gad > 10:
        recs.append(("🧘", "Anksiyete egzersizleri",
                    "4-7-8 nefes tekniği, 5-4-3-2-1 grounding egzersizi veya günlük 10 dk meditasyon dene."))
    elif gad > 5:
        recs.append(("🧘", "Stres yönetimi",
                    "Günde 5 dakika nefes egzersizi anksiyete seviyesini düzenler."))

    if phq > 10:
        recs.append(("🏃", "Fiziksel aktivite",
                    "Haftada 3 kez 30 dk yürüyüş, depresyon belirtilerini ilaca yakın oranda azaltır."))
    elif phq > 5:
        recs.append(("☀️", "Günlük rutin",
                    "Sabah 15 dk dışarıda zaman geçir — gün ışığı ruh halini düzenler."))

    if comp >= 0.6:
        recs.append(("🚫", "Karşılaştırma temizliği",
                    "Seni kötü hissettiren hesapları sustur veya takipten çık. Akış senin kontrolünde olmalı."))

    if level >= 3:
        recs.append(("📵", "Dijital detoks",
                    "Haftada 1 gün 'telefonsuz gün' belirle. Pazar günleri başlamak için iyi olabilir."))

    if level >= 4:
        recs.append(("💚", "Destek ağı",
                    "Güvendiğin biriyle hislerini paylaş. Yalnız savaşmak zorunda değilsin."))

    if not recs:
        recs.append(("✨", "Mevcut dengeyi koru",
                    "Alışkanlıkların sağlıklı görünüyor. Bu dengeyi sürdür ve kendine zaman ayırmaya devam et."))

    return recs

# ─────────────────────────────────────────
# Tahmin
# ─────────────────────────────────────────
LEVELS = {
    1: ("#10b981", "✅ Sağlıklı",            "Kullanımın dengeli. Mevcut alışkanlıklarını koru!"),
    2: ("#84cc16", "🟡 Dikkatli",            "Küçük risk işaretleri var. Ekran süreni takip et."),
    3: ("#f59e0b", "🟠 Risk Altında",        "Belirgin bağımlılık var. Dijital detoks dene."),
    4: ("#ef4444", "🔴 Bağımlılık Başlıyor", "Ciddi uyarı. Uzman desteği değerlendir."),
    5: ("#991b1b", "🚨 Ciddi Bağımlılık",    "Profesyonel destek tavsiye edilir."),
}
LEVEL_NAMES  = ['Sağlıklı', 'Dikkatli', 'Risk', 'Bağımlılık Başlıyor', 'Ciddi Bağımlılık']
LEVEL_COLORS = ['#10b981', '#84cc16', '#f59e0b', '#ef4444', '#991b1b']

def predict(user_text, *args):
    kullanici = []
    values_dict = {}
    for col, val in zip(feature_cols, args):
        if col in label_encoders:
            eng = rev_val(col, val)
            try:
                encoded = float(label_encoders[col].transform([eng])[0])
            except ValueError:
                encoded = 0.0
            kullanici.append(encoded)
            values_dict[col] = eng
        else:
            kullanici.append(float(val))
            values_dict[col] = float(val)

    veri_tab = scaler.transform([kullanici])

    # Metin tokenize
    if not user_text or not user_text.strip():
        veri_text = np.zeros((1, MAX_SEQ_LEN), dtype='int32')
        text_for_display = ""
    else:
        seq = tokenizer.texts_to_sequences([user_text])
        veri_text = pad_sequences(seq, maxlen=MAX_SEQ_LEN,
                                   padding='post', truncating='post')
        text_for_display = user_text.strip()

    logits = model.predict([veri_tab, veri_text], verbose=0)[0]
    temperature = 1.5
    softened = np.exp(np.log(logits + 1e-9) / temperature)
    probs = softened / softened.sum()
    seviye = int(np.argmax(probs)) + 1
    color, label, _ = LEVELS[seviye]

    return build_report(seviye, probs, values_dict, text_for_display, color, label)

# ─────────────────────────────────────────
# RUH SAĞLIĞI RAPORU
# ─────────────────────────────────────────
def build_report(seviye, probs, values, user_text, color, label):
    gad    = float(values.get('GAD_7_Score', 0))
    phq    = float(values.get('PHQ_9_Score', 0))
    screen = float(values.get('Daily_Screen_Time_Hours', 0))
    sleep  = float(values.get('Sleep_Duration_Hours', 8))
    night  = float(values.get('Late_Night_Usage', 0))

    gad_lbl, gad_color, gad_desc = gad7_severity(gad)
    phq_lbl, phq_color, phq_desc = phq9_severity(phq)
    screen_lbl, screen_color = screen_status(screen)
    sleep_lbl, sleep_color, _ = sleep_status(sleep)

    bars = ""
    for i, (name, c) in enumerate(zip(LEVEL_NAMES, LEVEL_COLORS)):
        pct = probs[i] * 100
        active = (i + 1) == seviye
        w = "700" if active else "500"
        tc = "#0f172a" if active else "#64748b"
        bars += f"""
        <div style="margin-bottom:10px;">
          <div style="display:flex;justify-content:space-between;margin-bottom:4px;
                      font-size:13px;font-weight:{w};color:{tc};">
            <span>{name}</span><span>{pct:.1f}%</span>
          </div>
          <div style="background:#f1f5f9;border-radius:999px;height:8px;overflow:hidden;">
            <div style="width:{pct:.1f}%;background:{c};height:100%;border-radius:999px;"></div>
          </div>
        </div>"""

    symptoms_html = f"""
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:16px;">
      <div style="background:white;border-left:3px solid {gad_color};padding:12px 14px;border-radius:8px;">
        <div style="font-size:10px;color:#64748b;font-weight:700;text-transform:uppercase;letter-spacing:0.5px;">Anksiyete (GAD-7)</div>
        <div style="font-size:22px;font-weight:800;color:{gad_color};margin:4px 0;">{int(gad)}<span style="font-size:13px;color:#94a3b8;">/21</span></div>
        <div style="font-size:13px;color:#0f172a;font-weight:700;">{gad_lbl}</div>
        <div style="font-size:11px;color:#64748b;margin-top:3px;line-height:1.4;">{gad_desc}</div>
      </div>
      <div style="background:white;border-left:3px solid {phq_color};padding:12px 14px;border-radius:8px;">
        <div style="font-size:10px;color:#64748b;font-weight:700;text-transform:uppercase;letter-spacing:0.5px;">Depresyon (PHQ-9)</div>
        <div style="font-size:22px;font-weight:800;color:{phq_color};margin:4px 0;">{int(phq)}<span style="font-size:13px;color:#94a3b8;">/27</span></div>
        <div style="font-size:13px;color:#0f172a;font-weight:700;">{phq_lbl}</div>
        <div style="font-size:11px;color:#64748b;margin-top:3px;line-height:1.4;">{phq_desc}</div>
      </div>
    </div>
    """

    behavior_html = f"""
    <div style="background:white;border-radius:10px;padding:14px 16px;margin-bottom:16px;">
      <div style="font-size:13px;color:#0f172a;margin-bottom:10px;font-weight:700;">📊 Davranış Örüntüsü</div>
      <div style="display:flex;justify-content:space-between;padding:7px 0;border-bottom:1px solid #f1f5f9;">
        <span style="color:#475569;font-size:13px;">📱 Günlük ekran</span>
        <span style="color:{screen_color};font-weight:700;font-size:13px;">{screen:.1f} saat — {screen_lbl}</span>
      </div>
      <div style="display:flex;justify-content:space-between;padding:7px 0;border-bottom:1px solid #f1f5f9;">
        <span style="color:#475569;font-size:13px;">😴 Uyku süresi</span>
        <span style="color:{sleep_color};font-weight:700;font-size:13px;">{sleep:.1f} saat — {sleep_lbl}</span>
      </div>
      <div style="display:flex;justify-content:space-between;padding:7px 0;">
        <span style="color:#475569;font-size:13px;">🌙 Gece kullanım sıklığı</span>
        <span style="color:#0f172a;font-weight:700;font-size:13px;">{int(night)}/5</span>
      </div>
    </div>
    """

    text_html = ""
    if user_text:
        text_html = f"""
        <div style="background:#faf5ff;border-radius:10px;padding:12px 14px;margin-bottom:16px;border-left:3px solid #8b5cf6;">
          <div style="font-size:10px;color:#6b21a8;font-weight:700;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:6px;">
            💬 Senin İfadelerin · LSTM ile analiz edildi
          </div>
          <div style="font-size:13px;color:#4c1d95;font-style:italic;line-height:1.5;">"{user_text}"</div>
        </div>
        """

    recs = get_personalized_recommendations(values, seviye)
    recs_html = '<div style="margin-bottom:16px;"><div style="font-size:13px;color:#0f172a;margin-bottom:10px;font-weight:700;">🎯 Kişiselleştirilmiş Öneriler</div>'
    for icon, title, desc in recs:
        recs_html += f"""
        <div style="background:white;border-radius:10px;padding:12px 14px;margin-bottom:8px;display:flex;gap:12px;
                    box-shadow:0 1px 3px rgba(0,0,0,0.04);">
          <div style="font-size:22px;flex-shrink:0;">{icon}</div>
          <div style="flex:1;">
            <div style="font-size:13px;font-weight:700;color:#0f172a;margin-bottom:3px;">{title}</div>
            <div style="font-size:12px;color:#475569;line-height:1.5;">{desc}</div>
          </div>
        </div>"""
    recs_html += '</div>'

    pro_html = ""
    if seviye >= 4 or gad >= 15 or phq >= 15:
        pro_html = """
        <div style="background:#fef2f2;border:2px solid #fecaca;border-radius:10px;padding:14px 16px;margin-bottom:14px;">
          <div style="font-size:13px;font-weight:700;color:#991b1b;margin-bottom:6px;">📞 Profesyonel Destek</div>
          <div style="font-size:12px;color:#7f1d1d;line-height:1.6;">
            Sonuçlar ciddi düzeyde belirtilere işaret ediyor. Bir psikolog ya da psikiyatristen destek alman önemli.<br>
            <strong>Türkiye'de:</strong> İntihar Önleme ve Psikolojik Destek Hattı <strong>182</strong> · 
            Sağlık Bakanlığı Ruh Sağlığı Birimleri · Üniversite hastaneleri PRM klinikleri.
          </div>
        </div>
        """

    return f"""
    <div style="background:linear-gradient(135deg,#fafbff 0%,#f1f5f9 100%);
                border-radius:20px;padding:22px;
                font-family:system-ui,-apple-system,sans-serif;
                box-shadow:0 20px 50px -20px {color}44, 0 4px 20px -8px rgba(0,0,0,0.08);
                border-top:6px solid {color};">

      <div style="text-align:center;margin-bottom:18px;">
        <div style="display:inline-block;padding:5px 14px;border-radius:999px;
                    background:{color}18;color:{color};font-size:10px;font-weight:700;
                    letter-spacing:1.5px;text-transform:uppercase;margin-bottom:10px;">
          🧠 Ruh Sağlığı Raporu · Seviye {seviye}/5
        </div>
        <div style="font-size:26px;font-weight:800;color:{color};">{label}</div>
      </div>

      {symptoms_html}
      {behavior_html}
      {text_html}
      {recs_html}

      <div style="margin-bottom:14px;">
        <div style="font-size:13px;color:#0f172a;margin-bottom:10px;font-weight:700;">📊 Olasılık Dağılımı</div>
        <div style="background:white;border-radius:10px;padding:14px;">
          {bars}
        </div>
      </div>

      {pro_html}

      <div style="text-align:center;font-size:11px;color:#94a3b8;margin-top:14px;line-height:1.5;
                  border-top:1px solid #e2e8f0;padding-top:12px;">
        ⚠️ Bu rapor bir AI modelinin tahminidir, tıbbi tanı niteliği taşımaz.<br>
        Gerçek değerlendirme için ruh sağlığı uzmanına başvurun.
      </div>
    </div>
    """

# ─────────────────────────────────────────
# CSS
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

# ─────────────────────────────────────────
# Arayüz
# ─────────────────────────────────────────
with gr.Blocks(title="Sosyal Medya Bağımlılık Dedektörü v3", css=css,
               theme=gr.themes.Soft(primary_hue="violet", neutral_hue="slate")) as demo:

    gr.HTML("""
    <div style="text-align:center;padding:36px 0 20px;">
      <div style="display:inline-block;padding:6px 16px;
                  background:linear-gradient(135deg,#8b5cf6,#ec4899);
                  color:white;border-radius:999px;font-size:11px;font-weight:700;
                  letter-spacing:2px;text-transform:uppercase;margin-bottom:16px;
                  box-shadow:0 8px 20px -6px rgba(139,92,246,0.5);">
        ✨ Multi-Modal AI · MLP + LSTM
      </div>
      <div style="font-size:38px;font-weight:800;color:#0f172a;margin-bottom:10px;">
        🧠 Sosyal Medya Bağımlılık 
        <span style="background:linear-gradient(135deg,#8b5cf6,#ec4899);
                     -webkit-background-clip:text;background-clip:text;
                     -webkit-text-fill-color:transparent;">Dedektörü</span>
      </div>
      <div style="font-size:15px;color:#475569;max-width:640px;margin:0 auto;">
        Anketi doldur ve duygularını birkaç cümlede anlat. Yapay zeka hem rakamları hem metnini analiz edip detaylı bir <strong>ruh sağlığı raporu</strong> hazırlasın.
      </div>
    </div>
    """)

    grouped = {'demografik': [], 'kullanim': [], 'psikolojik': [], 'yasam': []}
    for col in feature_cols:
        grouped[categorize(col)].append(col)

    with gr.Row():
        with gr.Column(scale=3):
            inputs_map = {}
            for section_key in ['demografik', 'kullanim', 'psikolojik', 'yasam']:
                cols = grouped[section_key]
                if not cols:
                    continue
                title, color, desc = SECTIONS[section_key]
                gr.HTML(section_header(title, color, desc))
                for i in range(0, len(cols), 2):
                    with gr.Row():
                        for col in cols[i:i+2]:
                            comp = build_input_for(col)
                            if comp is not None:
                                inputs_map[col] = comp

            gr.HTML(section_header(
                "💬 Bu Hafta Nasıl Hissettin?", "#8b5cf6",
                "Birkaç cümleyle yaz — yapay zeka duygularını LSTM ile analiz edecek."
            ))
            text_input = gr.Textbox(
                label="Haftalık hisler (opsiyonel ama önerilir)",
                placeholder="Örn: Bu hafta kendimi çok yorgun hissettim, geceleri telefondan kopamadım. Sürekli scroll ediyorum, dikkat dağınık...",
                lines=4,
                max_lines=6,
                value=""
            )

            btn = gr.Button("🔍 Ruh Sağlığı Raporumu Oluştur", variant="primary", size="lg")

        with gr.Column(scale=2):
            gr.HTML("""
            <div style="margin: 20px 0 10px 0; padding: 14px 20px;
                        background: linear-gradient(90deg, #8b5cf615, transparent);
                        border-left: 4px solid #8b5cf6; border-radius: 8px;">
              <div style="font-size: 17px; font-weight: 700; color: #0f172a;">
                📋 Ruh Sağlığı Raporu
              </div>
              <div style="font-size: 13px; color: #64748b;">Detaylı analiz ve kişiselleştirilmiş öneriler</div>
            </div>
            """)
            result = gr.HTML(value="""
            <div style="background:white;border:2px dashed #cbd5e1;border-radius:20px;
                        padding:60px 30px;text-align:center;font-family:system-ui,sans-serif;">
              <div style="font-size:64px;margin-bottom:18px;">🔮</div>
              <div style="color:#475569;font-size:15px;line-height:1.7;">
                Anketi doldur, duygularını yaz ve<br>
                <strong style="color:#8b5cf6;font-size:17px;">🔍 Raporumu Oluştur</strong><br>
                butonuna bas
              </div>
            </div>""")

    ordered_tab_inputs = [inputs_map[col] for col in feature_cols if col in inputs_map]
    btn.click(fn=predict, inputs=[text_input] + ordered_tab_inputs, outputs=result)

if __name__ == '__main__':
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True, show_error=True)
