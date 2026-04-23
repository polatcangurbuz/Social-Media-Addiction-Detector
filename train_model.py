"""
Sosyal Medya Bağımlılık Dedektörü — Model Eğitim Pipeline'ı
============================================================
Kaggle veri seti: "Social Media & Mental Health"
Çalıştırma: python train_model.py
"""

import numpy as np
import pandas as pd
import os
import json
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras import layers, callbacks
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

OUTPUT_DIR = '/kaggle/working' if os.path.exists('/kaggle/working') else '.'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────
# Klinik Tabanlı Bağımlılık Skoru
# ─────────────────────────────────────────
def compute_addiction_score(df):
    """
    DSM-5 esinli + davranışsal faktörlerle zenginleştirilmiş skor.
    Tüm feature'lar ağırlıklı olarak katkı sağlar.
    """
    def norm(series, max_val, invert=False):
        v = np.clip(series / max_val, 0, 1)
        return 1 - v if invert else v

    # ── ANA SİNYALLER (klinik) ──
    screen = norm(df['Daily_Screen_Time_Hours'], 10)
    night  = norm(df['Late_Night_Usage'], 5)
    gad    = norm(df['GAD_7_Score'], 21)
    phq    = norm(df['PHQ_9_Score'], 27)

    # ── YAN SİNYALLER (davranışsal) ──
    sleep_risk = norm(df['Sleep_Duration_Hours'], 10, invert=True)   # az uyku → yüksek risk
    age_risk   = np.clip(1 - (df['Age'] - 13) / 25, 0, 1)            # gençlerde risk daha yüksek

    # ── KATEGORİK RISK MAPPİNGLER (veri setine göre) ──
    archetype_risk = df['User_Archetype'].map({
        'Hyper-Connected':    1.0,   # en riskli
        'Passive Scroller':   0.7,
        'Average User':       0.5,
        'Digital Minimalist': 0.1,   # en sağlıklı
    }).fillna(0.5)

    content_risk = df['Dominant_Content_Type'].map({
        'Entertainment/Comedy': 0.85,  # dopamin yoğun, bağımlılık yapıcı
        'Lifestyle/Fashion':    0.75,  # sosyal karşılaştırmayı tetikler
        'Gaming':               0.70,
        'News/Politics':        0.55,  # duygusal yüklü, ama bilgi odaklı
        'Self-Help/Motivation': 0.35,  # potansiyel pozitif
        'Educational/Tech':     0.20,  # en az riskli
    }).fillna(0.5)

    activity_risk = df['Activity_Type'].map({
        'Passive': 0.8,   # sadece scroll → daha bağımlılık yapıcı
        'Active':  0.3,   # paylaşan/üreten → daha az
    }).fillna(0.5)

    # Binary: 1 = karşılaştırma tetikleniyor, 0 = tetiklenmiyor
    comparison_risk = df['Social_Comparison_Trigger'].astype(float).clip(0, 1)

    # ── NON-LİNEER ETKİLEŞİMLER ──
    compound_risk = screen * (gad + phq) / 2             # ekran + ruh hali
    sleep_mental  = sleep_risk * (gad + phq) / 2         # uykusuzluk + ruh hali

    # ── AĞIRLIKLI TOPLAM (toplam = 1.0) ──
    raw = (
        screen          * 0.18 +
        night           * 0.10 +
        gad             * 0.14 +
        phq             * 0.10 +
        sleep_risk      * 0.08 +
        age_risk        * 0.05 +
        archetype_risk  * 0.08 +
        content_risk    * 0.05 +
        activity_risk   * 0.05 +
        comparison_risk * 0.07 +
        compound_risk   * 0.05 +
        sleep_mental    * 0.05
    )

    # Gerçekçi gürültü
    noise = np.random.RandomState(42).normal(0, 0.03, len(raw))
    raw = np.clip(raw + noise, 0, 1)
    return raw

# ─────────────────────────────────────────
# 1. SENTETİK VERİ ÜRET (Kaggle'da yoksa)
# ─────────────────────────────────────────
def generate_synthetic_data(n=600):
    """
    Gerçek Kaggle verisi yoksa sentetik veri üretir.
    Kaggle'dan indirilen CSV varsa bu fonksiyonu atlayın.
    """
    np.random.seed(42)
    
    records = []
    for _ in range(n):
        # Bağımlılık seviyesi 1-5 (dengeli dağılım)
        addiction = np.random.choice([1, 2, 3, 4, 5], p=[0.15, 0.20, 0.25, 0.22, 0.18])
        
        # Özellikleri bağımlılık seviyesine göre oluştur
        base = addiction / 5.0  # 0.2 → 1.0 arasında normalize baz
        
        record = {
            # Demografik
            'age':              int(np.clip(np.random.normal(22 + addiction, 5), 13, 60)),
            'gender':           np.random.choice(['Male', 'Female', 'Non-binary'], p=[0.45, 0.45, 0.10]),
            'relationship':     np.random.choice(['Single', 'In relationship', 'Married'], p=[0.5, 0.35, 0.15]),
            'occupation':       np.random.choice(['Student', 'Employee', 'Freelancer', 'Unemployed'], p=[0.45, 0.30, 0.15, 0.10]),
            
            # Kullanım alışkanlıkları
            'daily_hours':      float(np.clip(np.random.normal(1 + addiction * 1.2, 0.8), 0.5, 10)),
            'platforms_count':  int(np.clip(np.random.normal(1 + addiction, 1), 1, 8)),
            'checks_per_day':   int(np.clip(np.random.normal(5 + addiction * 8, 5), 1, 80)),
            'night_usage':      int(np.clip(np.random.normal(base * 4, 1), 1, 5)),
            
            # Psikolojik göstergeler
            'fomo_score':       int(np.clip(np.random.normal(base * 4, 1), 1, 5)),
            'distraction':      int(np.clip(np.random.normal(base * 4, 1), 1, 5)),
            'restlessness':     int(np.clip(np.random.normal(base * 4, 1), 1, 5)),
            'anxiety':          int(np.clip(np.random.normal(base * 4, 1), 1, 5)),
            'depression':       int(np.clip(np.random.normal(base * 3.5, 1), 1, 5)),
            'self_comparison':  int(np.clip(np.random.normal(base * 4, 1), 1, 5)),
            'validation_seek':  int(np.clip(np.random.normal(base * 4, 1), 1, 5)),
            
            # Uyku & günlük hayat
            'sleep_issues':     int(np.clip(np.random.normal(base * 4, 1), 1, 5)),
            'productivity_loss':int(np.clip(np.random.normal(base * 4, 1), 1, 5)),
            'relationship_harm':int(np.clip(np.random.normal(base * 3, 1), 1, 5)),
            'purpose_less':     int(np.clip(np.random.normal(base * 4, 1), 1, 5)),
            
            # Hedef
            'addiction_score': addiction
        }
        records.append(record)
    
    df = pd.DataFrame(records)
    df.to_csv('social_media_usage.csv', index=False)
    print(f"✅ {n} satırlık sentetik veri oluşturuldu → social_media_usage.csv")
    return df


# ─────────────────────────────────────────
# 2. VERİ YÜKLEME & ÖN İŞLEME
# ─────────────────────────────────────────
def load_and_preprocess(csv_path='/kaggle/input/datasets/bertnardomariouskono/social-media-and-mental-health/social_media_mental_health.csv'):
    if not os.path.exists(csv_path):
        print("📊 CSV bulunamadı, sentetik veri üretiliyor...")
        df = generate_synthetic_data()
    else:
        df = pd.read_csv(csv_path)
        print(f"✅ Veri yüklendi: {df.shape[0]} satır, {df.shape[1]} sütun")
    
    # ─────────────────────────────────────────
    # 🧠 1. ADDICTION SCORE OLUŞTUR
    # ─────────────────────────────────────────
    df['addiction_score_raw'] = compute_addiction_score(df)

    df['addiction_score'] = pd.qcut(
        df['addiction_score_raw'], 5, labels=[1,2,3,4,5]
    ).astype(int)

    print("\n📋 Addiction score dağılımı:")
    print(df['addiction_score'].value_counts().sort_index())

    # ─────────────────────────────────────────
    # 🧹 2. GEREKSİZ KOLONLARI SİL
    # ─────────────────────────────────────────
    if 'User_ID' in df.columns:
        df = df.drop(columns=['User_ID'])
    df = df.drop(columns=['addiction_score_raw'])

    # ─────────────────────────────────────────
    # 🔤 3. KATEGORİK ENCODE
    # ─────────────────────────────────────────
    label_encoders = {}
    cat_cols = df.select_dtypes(include='object').columns
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    # ─────────────────────────────────────────
    # 🎯 4. FEATURE / TARGET AYIR
    # ─────────────────────────────────────────
    target_col = 'addiction_score'
    feature_cols = [c for c in df.columns if c != target_col]

    X = df[feature_cols].values.astype(np.float32)
    y = (df[target_col].values - 1).astype(np.int32)

    # ─────────────────────────────────────────
    # ✂️ 5. TRAIN / TEST SPLIT (önce ham veri ile)
    # ─────────────────────────────────────────
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ─────────────────────────────────────────
    # ⚖️ 6. SCALE (sadece train'e fit — leakage önlenir)
    # ─────────────────────────────────────────
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test  = scaler.transform(X_test_raw)

    print(f"\n📐 Özellik sayısı: {X.shape[1]}")
    print(f"🎓 Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

    # ─────────────────────────────────────────
    # 💾 7. KAYDET
    # ─────────────────────────────────────────
    with open(f'{OUTPUT_DIR}/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    with open(f'{OUTPUT_DIR}/label_encoders.pkl', 'wb') as f:
        pickle.dump(label_encoders, f)
    with open(f'{OUTPUT_DIR}/feature_cols.json', 'w') as f:
        json.dump(feature_cols, f)

    return (X_train, X_test, y_train, y_test,
            X_test_raw, feature_cols, label_encoders, X.shape[1])


# ─────────────────────────────────────────
# 3. MODEL MİMARİSİ
# ─────────────────────────────────────────
def build_model(input_dim, num_classes=5):
    """
    Küçültülmüş + L2 regularize edilmiş model.
    Bu veri boyutu için daha uygun bir kapasite.
    """
    inputs = tf.keras.Input(shape=(input_dim,), name='input')
    
    x = layers.Dense(
        64,
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),
        name='dense_1'
    )(inputs)
    x = layers.BatchNormalization(name='bn_1')(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.4, name='drop_1')(x)   # 0.3 → 0.4
    
    x = layers.Dense(
        32,
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),
        name='dense_2'
    )(x)
    x = layers.BatchNormalization(name='bn_2')(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.3, name='drop_2')(x)
    
    outputs = layers.Dense(num_classes, activation='softmax', name='output')(x)
    
    model = tf.keras.Model(inputs, outputs, name='SocialMediaAddictionDetector')
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),  # 1e-3 → 5e-4
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


# ─────────────────────────────────────────
# 4. EĞİTİM
# ─────────────────────────────────────────
def train(model, X_train, y_train, epochs=80):
    # Class weight hesapla — dengesizlik varsa az olan sınıflara daha fazla önem ver
    class_weights = compute_class_weight(
        'balanced', classes=np.unique(y_train), y=y_train
    )
    class_weight_dict = dict(enumerate(class_weights))
    print(f"\n📊 Class weights: {class_weight_dict}")

    cb_list = [
        callbacks.EarlyStopping(
            monitor='val_loss', patience=12,
            restore_best_weights=True, verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5,
            patience=6, min_lr=1e-5, verbose=1
        ),
        callbacks.ModelCheckpoint(
            f'{OUTPUT_DIR}/best_model.keras', monitor='val_accuracy',
            save_best_only=True, verbose=0
        )
    ]
    
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=epochs,
        batch_size=32,
        callbacks=cb_list,
        class_weight=class_weight_dict,   # ← yeni
        verbose=1
    )
    return history


# ─────────────────────────────────────────
# 5. DEĞERLENDİRME & GRAFİKLER
# ─────────────────────────────────────────
def evaluate_and_plot(model, history, X_test, y_test):
    # Test doğruluğu
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n🎯 Test Accuracy: {acc:.4f} ({acc*100:.1f}%)")
    print(f"📉 Test Loss: {loss:.4f}")
    
    # Tahminler
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    
    labels = ['Sağlıklı', 'Dikkatli', 'Risk', 'Bağımlılık Başlıyor', 'Ciddi Bağımlılık']
    print("\n📊 Sınıflandırma Raporu:")
    print(classification_report(y_test, y_pred, target_names=labels))
    
    # Grafik
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.patch.set_facecolor('#0f0f1a')
    
    for ax in axes:
        ax.set_facecolor('#1a1a2e')
        ax.tick_params(colors='#aaaacc')
        ax.spines['bottom'].set_color('#333355')
        ax.spines['left'].set_color('#333355')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    # Accuracy
    axes[0].plot(history.history['accuracy'], color='#7c3aed', linewidth=2, label='Train')
    axes[0].plot(history.history['val_accuracy'], color='#06b6d4', linewidth=2, linestyle='--', label='Val')
    axes[0].set_title('Model Accuracy', color='white', fontsize=13, pad=10)
    axes[0].set_xlabel('Epoch', color='#aaaacc')
    axes[0].set_ylabel('Accuracy', color='#aaaacc')
    axes[0].legend(facecolor='#1a1a2e', labelcolor='white')
    axes[0].grid(alpha=0.15, color='#555577')
    
    # Loss
    axes[1].plot(history.history['loss'], color='#f59e0b', linewidth=2, label='Train')
    axes[1].plot(history.history['val_loss'], color='#ef4444', linewidth=2, linestyle='--', label='Val')
    axes[1].set_title('Model Loss', color='white', fontsize=13, pad=10)
    axes[1].set_xlabel('Epoch', color='#aaaacc')
    axes[1].set_ylabel('Loss', color='#aaaacc')
    axes[1].legend(facecolor='#1a1a2e', labelcolor='white')
    axes[1].grid(alpha=0.15, color='#555577')
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(
        cm, ax=axes[2],
        annot=True, fmt='d', cmap='RdPu',
        xticklabels=['S', 'D', 'R', 'B', 'CB'],
        yticklabels=['S', 'D', 'R', 'B', 'CB'],
        cbar=False
    )
    axes[2].set_title('Confusion Matrix', color='white', fontsize=13, pad=10)
    axes[2].set_xlabel('Tahmin', color='#aaaacc')
    axes[2].set_ylabel('Gerçek', color='#aaaacc')
    axes[2].tick_params(colors='#aaaacc')
    
    plt.suptitle('Sosyal Medya Bağımlılık Dedektörü — Eğitim Sonuçları',
                 color='white', fontsize=15, y=1.02)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/training_results.png', dpi=150, bbox_inches='tight',
                facecolor='#0f0f1a', edgecolor='none')
    print("\n📈 Grafik kaydedildi → training_results.png")
    
    return acc

# ─────────────────────────────────────────
# FEATURE IMPORTANCE (Permutation)
# ─────────────────────────────────────────
def feature_importance(model, X_test, y_test, feature_cols):
    """
    Her feature'ı rastgele karıştırıp doğruluğun ne kadar düştüğünü ölçer.
    Çok düşerse → feature önemli.  Değişmezse → feature işe yaramıyor.
    """
    baseline = model.evaluate(X_test, y_test, verbose=0)[1]
    importances = []
    
    np.random.seed(42)
    for i, col in enumerate(feature_cols):
        X_shuffled = X_test.copy()
        np.random.shuffle(X_shuffled[:, i])
        score = model.evaluate(X_shuffled, y_test, verbose=0)[1]
        importances.append((col, baseline - score))
    
    importances.sort(key=lambda x: x[1], reverse=True)
    max_imp = max(abs(x[1]) for x in importances) or 1
    
    print("\n" + "═" * 60)
    print("  🔍 FEATURE IMPORTANCE (Permutation Method)")
    print("═" * 60)
    print(f"  Baseline accuracy: {baseline:.4f}\n")
    print(f"  {'Özellik':<32}{'Önem':>10}   Bar")
    print("  " + "─" * 58)
    
    for col, imp in importances:
        bar_len = int((imp / max_imp) * 25) if imp > 0 else 0
        bar = '█' * bar_len
        flag = "  ← önemsiz" if imp < 0.005 else ""
        print(f"  {col:<32}{imp:>10.4f}   {bar}{flag}")
    print()
    return importances

# ─────────────────────────────────────────
# 6. DEMO FONKSİYONU
# ─────────────────────────────────────────
def bagimlilik_skoru(model, scaler, kullanici_verisi: list) -> dict:
    """
    Tek kullanıcı tahmini.
    kullanici_verisi: feature_cols sırasında sayısal değerler
    """
    veri = scaler.transform([kullanici_verisi])
    tahmin_probs = model.predict(veri, verbose=0)[0]
    seviye = int(np.argmax(tahmin_probs)) + 1
    
    etiketler = {
        1: ("✅ Sağlıklı",           "#22c55e", "Sosyal medya kullanımın dengeli. Böyle devam et!"),
        2: ("🟡 Dikkatli ol",        "#eab308", "Küçük riskler var. Ekran süresini takip etmeye başla."),
        3: ("🟠 Risk altında",       "#f97316", "Belirgin bağımlılık işaretleri var. Dijital detoks dene."),
        4: ("🔴 Bağımlılık başlıyor","#ef4444", "Ciddi uyarı! Uzman desteği faydalı olabilir."),
        5: ("🚨 Ciddi bağımlılık",   "#dc2626", "Profesyonel destek almanı şiddetle tavsiye ederiz.")
    }
    
    label, color, advice = etiketler[seviye]
    
    return {
        'level': seviye,
        'label': label,
        'color': color,
        'advice': advice,
        'probabilities': {
            'Sağlıklı':           float(tahmin_probs[0]),
            'Dikkatli':           float(tahmin_probs[1]),
            'Risk':               float(tahmin_probs[2]),
            'Bağımlılık Başlıyor':float(tahmin_probs[3]),
            'Ciddi Bağımlılık':   float(tahmin_probs[4])
        }
    }


# ─────────────────────────────────────────
# 7. ANA ÇALIŞTIRMA
# ─────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 55)
    print("  Sosyal Medya Bağımlılık Dedektörü — Model Eğitimi")
    print("=" * 55)
    
    # Veri
    (X_train, X_test, y_train, y_test,
 X_test_raw, feature_cols, label_encoders, input_dim) = load_and_preprocess()
    
    # Model
    model = build_model(input_dim)
    model.summary()
    
    # Eğitim
    print("\n🚀 Eğitim başlıyor...")
    history = train(model, X_train, y_train, epochs=80)
    
    # Değerlendirme
    acc = evaluate_and_plot(model, history, X_test, y_test)

    feature_importance(model, X_test, y_test, feature_cols)
    
    # Modeli kaydet
    model.save(f'{OUTPUT_DIR}/addiction_model.keras')
    print("💾 Model kaydedildi → addiction_model.keras")
    
    # Demo test
    # ───────────── Demo: rastgele bir test örneği seç ─────────────
    print("\n" + "═" * 60)
    print("  🎮 DEMO TEST")
    print("═" * 60)

    idx = np.random.randint(0, len(X_test))
    ornek_scaled   = X_test[idx:idx+1]
    ornek_original = X_test_raw[idx]
    gercek_seviye  = int(y_test[idx]) + 1

    # ───────────── Kullanıcı profilini tablo olarak göster ─────────────
    print("\n📋 TEST EDİLEN KULLANICI PROFİLİ\n")

    print(f"  {'#':<4}{'Özellik':<32}{'Değer':<20}")
    print("  " + "─" * 54)

    for i, (col, val) in enumerate(zip(feature_cols, ornek_original), start=1):
        # Kategorik kolonsa encoded değeri orijinal etikete çevir
        if col in label_encoders:
            val_str = label_encoders[col].inverse_transform([int(val)])[0]
        elif float(val).is_integer():
            val_str = str(int(val))
        else:
            val_str = f"{val:.2f}"
        print(f"  {i:<4}{col:<32}{val_str:<20}")

    print("  " + "─" * 54)

    # ───────────── Tahmin yap ve sonucu göster ─────────────
    tahmin_probs = model.predict(ornek_scaled, verbose=0)[0]
    seviye = int(np.argmax(tahmin_probs)) + 1

    etiketler = {
        1: ("✅ Sağlıklı",            "Sosyal medya kullanımın dengeli. Böyle devam et!"),
        2: ("🟡 Dikkatli ol",         "Küçük riskler var. Ekran süresini takip etmeye başla."),
        3: ("🟠 Risk altında",        "Belirgin bağımlılık işaretleri var. Dijital detoks dene."),
        4: ("🔴 Bağımlılık başlıyor", "Ciddi uyarı! Uzman desteği faydalı olabilir."),
        5: ("🚨 Ciddi bağımlılık",    "Profesyonel destek almanı şiddetle tavsiye ederiz.")
    }
    label, advice = etiketler[seviye]
    dogru_mu = "✅ DOĞRU" if seviye == gercek_seviye else "❌ YANLIŞ"

    print(f"\n🎯 Gerçek Seviye : {gercek_seviye} — {etiketler[gercek_seviye][0]}")
    print(f"🤖 Model Tahmini : {seviye} — {label}")
    print(f"📌 Sonuç         : {dogru_mu}")
    print(f"💡 Tavsiye       : {advice}")

    print("\n📊 Olasılık Dağılımı:")
    siniflar = ['Sağlıklı', 'Dikkatli', 'Risk', 'Bağımlılık Başlıyor', 'Ciddi Bağımlılık']
    for k, v in zip(siniflar, tahmin_probs):
        bar = '█' * int(v * 30)
        print(f"  {k:22s} {bar:<30} {v:>6.1%}")

    print(f"\n✅ Tüm işlem tamamlandı! Test accuracy: {acc*100:.1f}%")
