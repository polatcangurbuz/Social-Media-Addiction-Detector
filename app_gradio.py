"""
Sosyal Medya Bağımlılık Dedektörü — Gradio Web Demo
=====================================================
Çalıştırma: python app_gradio.py
URL: http://localhost:7860
"""

import numpy as np
import json
import pickle
import os
import tensorflow as tf
import gradio as gr

# ─────────────────────────────────────────
# Model & Scaler yükle
# ─────────────────────────────────────────
def load_artifacts():
    if not os.path.exists('addiction_model.keras'):
        raise FileNotFoundError(
            "❌ Model bulunamadı!\n"
            "Önce şunu çalıştırın: python train_model.py"
        )
    model  = tf.keras.models.load_model('addiction_model.keras')
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('feature_cols.json') as f:
        feature_cols = json.load(f)
    return model, scaler, feature_cols

model, scaler, feature_cols = load_artifacts()

# ─────────────────────────────────────────
# Tahmin Fonksiyonu
# ─────────────────────────────────────────
GENDER_MAP     = {'Erkek': 0, 'Kadın': 1, 'Diğer': 2}
REL_MAP        = {'Bekar': 0, 'İlişkide': 1, 'Evli': 2}
OCC_MAP        = {'Öğrenci': 0, 'Çalışan': 1, 'Serbest': 2, 'İşsiz': 3}
LEVEL_CONFIGS  = {
    1: ("#22c55e", "✅ Sağlıklı Kullanıcı",       "Harika! Sosyal medya kullanımın dengeli ve kontrollü. Mevcut alışkanlıklarını korumaya devam et."),
    2: ("#84cc16", "🟡 Dikkat Gerektiriyor",       "Bazı küçük risk işaretleri var. Ekran süresini takip etmeye başlamanı öneririz."),
    3: ("#f97316", "🟠 Risk Altında",              "Belirgin bağımlılık örüntüleri gözlemleniyor. Haftada 1 gün dijital detoks uygula."),
    4: ("#ef4444", "🔴 Bağımlılık Başlıyor",       "Ciddi uyarı! Günlük ekran süresini kısıtla ve bir uzmanla görüşmeyi düşün."),
    5: ("#dc2626", "🚨 Ciddi Bağımlılık",          "Profesyonel destek almanı şiddetle tavsiye ederiz. Bir psikolog veya terapistle görüş."),
}

def predict(*args):
    (yaş, cinsiyet, ilişki, meslek,
     gunluk_saat, platform_sayisi, kontrol_sayisi, gece_kullanim,
     fomo, dikkat_dagilmasi, huzursuzluk, endise,
     depresyon, kendini_karsilastirma, onay_arama,
     uyku_sorunu, verimlilik_kaybi, iliski_zarari, amac_yoksunlugu) = args
    
    kullanici = [
        float(yaş),
        float(GENDER_MAP.get(cinsiyet, 0)),
        float(REL_MAP.get(ilişki, 0)),
        float(OCC_MAP.get(meslek, 0)),
        float(gunluk_saat),
        float(platform_sayisi),
        float(kontrol_sayisi),
        float(gece_kullanim),
        float(fomo),
        float(dikkat_dagilmasi),
        float(huzursuzluk),
        float(endise),
        float(depresyon),
        float(kendini_karsilastirma),
        float(onay_arama),
        float(uyku_sorunu),
        float(verimlilik_kaybi),
        float(iliski_zarari),
        float(amac_yoksunlugu),
    ]
    
    veri = scaler.transform([kullanici])
    probs = model.predict(veri, verbose=0)[0]
    seviye = int(np.argmax(probs)) + 1
    
    color, label, advice = LEVEL_CONFIGS[seviye]
    
    # HTML sonuç kartı
    prob_bars = ""
    level_names = ['Sağlıklı', 'Dikkatli', 'Risk', 'Bağımlılık Başlıyor', 'Ciddi Bağımlılık']
    level_colors = ['#22c55e', '#84cc16', '#f97316', '#ef4444', '#dc2626']
    
    for i, (name, c) in enumerate(zip(level_names, level_colors)):
        pct = probs[i] * 100
        bold = "font-weight:700;" if (i+1)==seviye else ""
        prob_bars += f"""
        <div style="margin-bottom:8px;">
          <div style="display:flex;justify-content:space-between;margin-bottom:3px;font-size:13px;{bold}color:#e2e8f0;">
            <span>{name}</span><span>{pct:.1f}%</span>
          </div>
          <div style="background:#1e293b;border-radius:6px;height:8px;overflow:hidden;">
            <div style="width:{pct:.1f}%;background:{c};height:100%;border-radius:6px;
                        transition:width 0.5s ease;"></div>
          </div>
        </div>"""
    
    result_html = f"""
    <div style="background:linear-gradient(135deg,#0f0f1a,#1a1a2e);
                border:2px solid {color};border-radius:16px;padding:28px;
                font-family:'Segoe UI',sans-serif;color:#e2e8f0;">
      
      <div style="text-align:center;margin-bottom:24px;">
        <div style="font-size:42px;font-weight:900;color:{color};
                    text-shadow:0 0 20px {color}66;">{label}</div>
        <div style="font-size:28px;font-weight:700;color:{color};margin-top:4px;">
          Seviye {seviye}/5
        </div>
      </div>
      
      <div style="background:{color}22;border-left:4px solid {color};
                  border-radius:8px;padding:14px;margin-bottom:24px;
                  font-size:15px;line-height:1.6;">
        💡 {advice}
      </div>
      
      <div style="font-size:14px;font-weight:700;color:#94a3b8;
                  text-transform:uppercase;letter-spacing:1px;margin-bottom:12px;">
        Bağımlılık Olasılıkları
      </div>
      {prob_bars}
    </div>
    """
    return result_html

# ─────────────────────────────────────────
# Gradio Arayüzü
# ─────────────────────────────────────────
css = """
body { background: #0a0a12 !important; }
.gradio-container { background: #0a0a12 !important; color: #e2e8f0 !important; }
.gr-panel { background: #111827 !important; border: 1px solid #1e293b !important; border-radius: 12px !important; }
label { color: #94a3b8 !important; font-size: 13px !important; }
.gr-slider { accent-color: #7c3aed !important; }
h1, h2, h3 { color: #f1f5f9 !important; }
footer { display: none !important; }
"""

with gr.Blocks(
    title="Sosyal Medya Bağımlılık Dedektörü",
    css=css,
    theme=gr.themes.Base(
        primary_hue="purple",
        secondary_hue="cyan",
        neutral_hue="slate"
    )
) as demo:
    
    gr.Markdown("""
    # 🧠 Sosyal Medya Bağımlılık Dedektörü
    **Yapay zeka destekli kişisel analiz** — Lütfen soruları dürüstçe yanıtlayın
    """)
    
    with gr.Row():
        # ── Sol: Anket Formu ──
        with gr.Column(scale=2):
            
            gr.Markdown("### 👤 Demografik Bilgiler")
            with gr.Row():
                yas   = gr.Slider(13, 65, value=22, step=1, label="Yaş")
                cins  = gr.Dropdown(['Erkek','Kadın','Diğer'], value='Erkek', label="Cinsiyet")
            with gr.Row():
                ilisk = gr.Dropdown(['Bekar','İlişkide','Evli'], value='Bekar', label="İlişki Durumu")
                mesl  = gr.Dropdown(['Öğrenci','Çalışan','Serbest','İşsiz'], value='Öğrenci', label="Meslek")
            
            gr.Markdown("### 📱 Kullanım Alışkanlıkları")
            gun_saat = gr.Slider(0, 12, value=3, step=0.5, label="Günlük sosyal medya süresi (saat)")
            plat_say = gr.Slider(1, 8,  value=3, step=1,   label="Kullandığın platform sayısı")
            kont_say = gr.Slider(1, 80, value=15,step=1,   label="Günde kaç kez kontrol ediyorsun?")
            gece_kul = gr.Slider(1, 5,  value=2, step=1,   label="Gece yatmadan önce kullanım (1=Hiç, 5=Her gece)")
            
            gr.Markdown("### 😰 Psikolojik Göstergeler")
            gr.Markdown("*1 = Hiç / 5 = Her zaman*")
            
            with gr.Row():
                fomo     = gr.Slider(1, 5, value=2, step=1, label="🎭 FOMO (kaçırma korkusu)")
                dikkat   = gr.Slider(1, 5, value=2, step=1, label="🌀 Dikkat dağınıklığı")
            with gr.Row():
                huzursuz = gr.Slider(1, 5, value=2, step=1, label="😤 Telefon yokken huzursuzluk")
                endise   = gr.Slider(1, 5, value=2, step=1, label="😟 Endişe")
            with gr.Row():
                depres   = gr.Slider(1, 5, value=2, step=1, label="😔 Depresyon belirtisi")
                karsilas = gr.Slider(1, 5, value=2, step=1, label="👁️ Kendinle başkalarını karşılaştırma")
            with gr.Row():
                onay     = gr.Slider(1, 5, value=2, step=1, label="❤️ Beğeni/onay arama")
                uyku     = gr.Slider(1, 5, value=2, step=1, label="😴 Uyku sorunları")
            with gr.Row():
                verim    = gr.Slider(1, 5, value=2, step=1, label="📉 Verimlilik kaybı")
                iliski   = gr.Slider(1, 5, value=2, step=1, label="💔 İlişkilere zarar")
            
            amac     = gr.Slider(1, 5, value=2, step=1, label="🎯 Amaçsız gezinme hissi")
            
            btn = gr.Button("🔍 Analiz Et", variant="primary", size="lg")
        
        # ── Sağ: Sonuç ──
        with gr.Column(scale=1):
            gr.Markdown("### 📊 Analiz Sonucu")
            result = gr.HTML(
                value="""
                <div style="background:#111827;border:1px dashed #334155;border-radius:16px;
                            padding:40px;text-align:center;color:#475569;font-family:sans-serif;">
                  <div style="font-size:48px;margin-bottom:16px;">🔍</div>
                  <div style="font-size:16px;">Anketi doldurup<br><strong style="color:#7c3aed">Analiz Et</strong> butonuna bas</div>
                </div>"""
            )
    
    # Örnek profiller
    gr.Markdown("---")
    gr.Markdown("### 🎯 Hızlı Test Profilleri")
    with gr.Row():
        def load_healthy():
            return (20,'Kadın','Bekar','Öğrenci', 1.5,2,5,1, 1,1,1,1, 1,1,1,1, 1,1,1)
        def load_risky():
            return (22,'Erkek','Bekar','Öğrenci', 5,4,30,4, 4,4,4,3, 3,4,4,3, 4,3,4)
        def load_addicted():
            return (24,'Kadın','Bekar','Çalışan', 8,6,60,5, 5,5,5,5, 4,5,5,5, 5,4,5)
        
        ex1 = gr.Button("✅ Sağlıklı Profil")
        ex2 = gr.Button("🟠 Risk Profili")
        ex3 = gr.Button("🚨 Bağımlılık Profili")
    
    all_inputs = [yas,cins,ilisk,mesl, gun_saat,plat_say,kont_say,gece_kul,
                  fomo,dikkat,huzursuz,endise, depres,karsilas,onay,uyku, verim,iliski,amac]
    
    btn.click(fn=predict, inputs=all_inputs, outputs=result)
    ex1.click(fn=load_healthy, outputs=all_inputs)
    ex2.click(fn=load_risky,   outputs=all_inputs)
    ex3.click(fn=load_addicted,outputs=all_inputs)

if __name__ == '__main__':
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        show_error=True
    )
