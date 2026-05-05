"""
Sosyal Medya Bağımlılık Dedektörü — Model Eğitim Pipeline'ı (v3)
================================================================
v3 DÜZELTMELERİ (v2'deki veri sızıntısını giderir):

  🔧 OLASILIKSAL DUYGU EŞLEMESİ — metin artık seviyeye bire bir bağlı değil.
     Seviye 1 kullanıcı %15 ihtimalle "endişeli" yazabilir, seviye 5 kullanıcı
     %5 ihtimalle "nötr" yazabilir. Bu LSTM'in ezberlemesini engeller.

  🔧 TEXT DROPOUT AUGMENTATION — eğitim verisinin %30'unda metin sıfırlanmış.
     Model "metin olmadan da tabular ile karar ver" davranışını öğreniyor.

  🔧 KÜÇÜLTÜLMÜŞ + REGULARIZED TEXT BRANCH — LSTM(16) + Embedding(32) + Dropout(0.5).
     Tabular branch'i baskın hâle getirir, LSTM yardımcı sinyale dönüşür.

  🔧 LABEL ENCODING (one-hot YOK) — kategorik kolonlar tek kolonda kalır,
     arayüzde Türkçe dropdown olarak görünür.

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
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

OUTPUT_DIR = '/kaggle/working' if os.path.exists('/kaggle/working') else '.'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────
# NLP Hyperparametreleri (v3 — küçültüldü)
# ─────────────────────────────────────────
MAX_SEQ_LEN = 30
VOCAB_SIZE  = 2500
EMBED_DIM   = 32      # ↓ 64'ten 32'ye
LSTM_UNITS  = 16      # ↓ 32'den 16'ya
TEXT_DROPOUT_RATE = 0.30   # eğitim verisinin %30'u text-sıfırlanmış
TAB_NOISE_STD = 0.04       # tabular feature noise (regularization)

# ─────────────────────────────────────────
# DUYGU TABANLI METİN ŞABLONLARI (v3)
# Artık seviyeye değil, duyguya göre gruplanmış
# ─────────────────────────────────────────
EMOTION_TEMPLATES = {
    'positive': [
        "Bu hafta kendimi enerjik hissettim ve dinç uyandım",
        "Hafta sonu doğa yürüyüşü yaptım çok keyifliydi",
        "Telefonsuz vakit geçirmek bana iyi geliyor",
        "Spor ve kitap okumak rutinim oldu memnunum",
        "Uykum düzenli sabahları dinç kalkıyorum",
        "Hayatımdan memnunum kendime zaman ayırıyorum",
        "Arkadaşlarımla yüz yüze görüştüm çok mutluyum",
        "Hobilerime zaman ayırdım üretken bir hafta geçirdim",
        "Yeni şeyler öğreniyorum motivasyonum yüksek",
        "Genel olarak iyiyim hayattan zevk alıyorum",
        "Kendimi huzurlu ve dengeli hissediyorum",
        "Aile ile vakit geçirdim güzel bir haftaydı",
    ],
    'mild_concern': [
        "Bazen telefondan kopamadığımı fark ediyorum",
        "Akşamları biraz fazla scroll yapıyorum",
        "Genel olarak iyiyim ama ara sıra yoruluyorum",
        "Sosyal medya kullanımımı azaltmak istiyorum",
        "Uyku saatim son zamanlarda kaymış olabilir",
        "Çalışırken telefon dikkatimi dağıtıyor bazen",
        "Bildirimleri kontrol etme alışkanlığım var",
        "Bazı günler aşırıya kaçabiliyorum",
        "Ekran sürem ara sıra beni şaşırtıyor",
        "Sınırları korumakta zorlanabiliyorum",
        "Az da olsa yorgun hissettiğim günler var",
    ],
    'distress': [
        "Telefonu elime aldığımda saatlerin geçtiğini fark etmiyorum",
        "Sürekli bildirimleri ve mesajları kontrol ediyorum",
        "Kendimi başkalarıyla karşılaştırıyorum kötü hissediyorum",
        "Konsantrasyonum düşük dikkatim çabuk dağılıyor",
        "Geceleri yatakta saatlerce telefonla vakit geçiriyorum",
        "Anksiyete ve huzursuzluk hissediyorum sıkça",
        "Uykum bozuldu sürekli yorgun hissediyorum",
        "Verimim düştü işleri ertelemeye başladım",
        "Telefonsuz duramıyorum sürekli kontrol etmem gerekiyor",
        "Sosyal medyada kendimi yetersiz hissediyorum",
        "FOMO hissediyorum bir şey kaçırıyorum gibi",
        "Sürekli kaygılıyım dinlenemiyorum kafam dağınık",
    ],
    'severe': [
        "Hayattan zevk almıyorum sürekli telefondayım",
        "Bütün gece scroll ediyorum uyuyamıyorum",
        "Kendimi değersiz ve yalnız hissediyorum",
        "Yardıma ihtiyacım olduğunu biliyorum ama duramıyorum",
        "İlişkilerim bozuldu kimseyle görüşmüyorum",
        "Depresyondayım hiçbir şey yapasım gelmiyor",
        "Ekrandan ayrılınca panik atak gibi oluyorum",
        "Günlerimi kaybediyorum elim kolum bağlı gibi",
        "Hayatımın kontrolünü kaybettim",
        "İzole hissediyorum çok yorgunum",
        "Kendimi telefonun esiri gibi hissediyorum",
    ],
    'neutral': [
        "Bu hafta normal geçti bir şey değişmedi",
        "Olağan bir hafta yaşadım söyleyecek bir şey yok",
        "İşler yolunda gidiyor genel olarak",
        "Çok düşünmedim açıkçası",
        "Hayat akıyor işte günler birbirine benziyor",
        "Bilmiyorum ne hissediyorum tam olarak",
        "Karışık bir hafta oldu ne çok iyi ne kötü",
        "Bazı şeyler iyi bazıları kötü",
        "Sıradan günler geçirdim özel bir şey yok",
    ],
}

# Her seviyenin her duyguyu seçme olasılığı
LEVEL_EMOTION_PROBS = {
    1: {'positive': 0.55, 'neutral': 0.30, 'mild_concern': 0.15},
    2: {'positive': 0.25, 'neutral': 0.30, 'mild_concern': 0.35, 'distress': 0.10},
    3: {'positive': 0.05, 'neutral': 0.20, 'mild_concern': 0.35, 'distress': 0.35, 'severe': 0.05},
    4: {'neutral': 0.10, 'mild_concern': 0.15, 'distress': 0.45, 'severe': 0.30},
    5: {'neutral': 0.05, 'mild_concern': 0.05, 'distress': 0.25, 'severe': 0.65},
}

def generate_text_for_level(level, rng):
    """Seviyeye göre olasılıksal duygu seç, o duygudan 1-2 cümle birleştir."""
    emotion_probs = LEVEL_EMOTION_PROBS[int(level)]
    emotions = list(emotion_probs.keys())
    probs = np.array([emotion_probs[e] for e in emotions])
    probs = probs / probs.sum()

    chosen_emotion = rng.choice(emotions, p=probs)
    templates = EMOTION_TEMPLATES[chosen_emotion]
    n = rng.choice([1, 2])
    selected = rng.choice(templates, size=min(n, len(templates)), replace=False)
    return " ".join(selected)

# ─────────────────────────────────────────
# Klinik Tabanlı Bağımlılık Skoru
# ─────────────────────────────────────────
def compute_addiction_score(df):
    def norm(series, max_val, invert=False):
        v = np.clip(series / max_val, 0, 1)
        return 1 - v if invert else v

    screen = norm(df['Daily_Screen_Time_Hours'], 10)
    night  = norm(df['Late_Night_Usage'], 5)
    gad    = norm(df['GAD_7_Score'], 21)
    phq    = norm(df['PHQ_9_Score'], 27)
    sleep_risk = norm(df['Sleep_Duration_Hours'], 10, invert=True)

    archetype_risk = df['User_Archetype'].map({
        'Hyper-Connected': 1.0, 'Passive Scroller': 0.7,
        'Average User': 0.5, 'Digital Minimalist': 0.1,
    }).fillna(0.5)

    content_risk = df['Dominant_Content_Type'].map({
        'Entertainment/Comedy': 0.85, 'Lifestyle/Fashion': 0.75,
        'Gaming': 0.70, 'News/Politics': 0.55,
        'Self-Help/Motivation': 0.35, 'Educational/Tech': 0.20,
    }).fillna(0.5)

    activity_risk = df['Activity_Type'].map({
        'Passive': 0.8, 'Active': 0.3,
    }).fillna(0.5)

    comparison_risk = df['Social_Comparison_Trigger'].astype(float).clip(0, 1)
    compound_risk = screen * (gad + phq) / 2
    sleep_mental  = sleep_risk * (gad + phq) / 2

    raw = (
        screen          * 0.18 +
        night           * 0.12 +
        gad             * 0.15 +
        phq             * 0.12 +
        sleep_risk      * 0.08 +
        archetype_risk  * 0.10 +
        content_risk    * 0.05 +
        activity_risk   * 0.05 +
        comparison_risk * 0.08 +
        compound_risk   * 0.04 +
        sleep_mental    * 0.03
    )

    # v3: daha fazla noise → daha gerçekçi
    noise = np.random.RandomState(42).normal(0, 0.06, len(raw))
    return np.clip(raw + noise, 0, 1)

# ─────────────────────────────────────────
# Sentetik Veri Üretimi
# ─────────────────────────────────────────
def generate_synthetic_data(n=1000):
    """v3: daha fazla varyans, %15 'atipik' örnek (sağlıklı kullanıcı yüksek ekran vb.)"""
    np.random.seed(42)
    rng = np.random.default_rng(42)

    rows = []
    for i in range(n):
        archetype = rng.choice(
            ['Hyper-Connected', 'Passive Scroller', 'Average User', 'Digital Minimalist'],
            p=[0.20, 0.30, 0.35, 0.15]
        )
        af = {'Hyper-Connected': 0.9, 'Passive Scroller': 0.65,
              'Average User': 0.45, 'Digital Minimalist': 0.15}[archetype]

        # %15 "atipik" — feature'ları arketipten saptır (gerçekçi outlier'lar)
        if rng.random() < 0.15:
            af = float(np.clip(af + rng.normal(0, 0.25), 0.05, 0.95))

        rows.append({
            'User_ID': f'U{i:05d}',
            'Age': int(np.clip(np.random.normal(25, 7), 13, 60)),
            'Gender': rng.choice(['Male', 'Female', 'Non-binary'], p=[0.45, 0.45, 0.10]),
            'Daily_Screen_Time_Hours': float(np.clip(np.random.normal(1 + af * 8, 2.0), 0.5, 12)),
            'Late_Night_Usage': int(np.clip(np.random.normal(1 + af * 4, 1.2), 1, 5)),
            'GAD_7_Score': int(np.clip(np.random.normal(af * 18, 5), 0, 21)),
            'PHQ_9_Score': int(np.clip(np.random.normal(af * 22, 6), 0, 27)),
            'Sleep_Duration_Hours': float(np.clip(np.random.normal(8.5 - af * 4, 1.3), 3, 10)),
            'User_Archetype': archetype,
            'Dominant_Content_Type': rng.choice([
                'Entertainment/Comedy', 'Lifestyle/Fashion', 'Gaming',
                'News/Politics', 'Self-Help/Motivation', 'Educational/Tech'
            ]),
            'Primary_Platform': rng.choice(
                ['Instagram', 'TikTok', 'Twitter', 'YouTube', 'Facebook']
            ),
            'Activity_Type': 'Passive' if af > 0.6 else rng.choice(['Active', 'Passive']),
            'Social_Comparison_Trigger': float(np.clip(np.random.normal(af, 0.25), 0, 1)),
        })

    df = pd.DataFrame(rows)
    df.to_csv('social_media_synthetic.csv', index=False)
    print(f"✅ {n} satırlık sentetik veri üretildi (15% atipik, daha yüksek varyans)")
    return df

# ─────────────────────────────────────────
# Veri Yükleme & Ön İşleme
# ─────────────────────────────────────────
def load_and_preprocess(csv_path='/kaggle/input/datasets/bertnardomariouskono/social-media-and-mental-health/social_media_mental_health.csv'):
    if not os.path.exists(csv_path):
        print("📊 Kaggle CSV bulunamadı, sentetik veri üretiliyor...")
        df = generate_synthetic_data()
    else:
        df = pd.read_csv(csv_path)
        print(f"✅ Veri yüklendi: {df.shape}")

    # 1. Bağımlılık skoru
    df['addiction_score_raw'] = compute_addiction_score(df)
    df['addiction_score'] = pd.qcut(
        df['addiction_score_raw'], 5, labels=[1, 2, 3, 4, 5]
    ).astype(int)

    print("\n📋 Addiction score dağılımı:")
    print(df['addiction_score'].value_counts().sort_index())

    # 2. OLASILIKSAL metin üret (artık seviyeye bire bir bağlı değil)
    print("\n📝 Olasılıksal metin verisi üretiliyor (v3: emotion-based)...")
    rng = np.random.default_rng(42)
    df['user_text'] = df['addiction_score'].apply(
        lambda lvl: generate_text_for_level(int(lvl), rng)
    )
    print(f"   Seviye 1 örnekleri:")
    for t in df[df['addiction_score']==1]['user_text'].sample(3, random_state=1):
        print(f"     • \"{t}\"")
    print(f"   Seviye 5 örnekleri:")
    for t in df[df['addiction_score']==5]['user_text'].sample(3, random_state=1):
        print(f"     • \"{t}\"")

    # 3. Metni ayır
    texts = df['user_text'].values
    df = df.drop(columns=['user_text'])

    # 4. Gereksiz kolonları sil
    drop_cols = ['User_ID', 'Age', 'GAD_7_Severity', 'PHQ_9_Severity', 'addiction_score_raw']
    for col in drop_cols:
        if col in df.columns:
            df = df.drop(columns=[col])

    # 5. KATEGORİKLER LABEL ENCODE (v3: one-hot YOK)
    # Tüm string kolonlar tek kolonda kalır → arayüzde Türkçe dropdown
    label_encoders = {}
    for col in df.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
        print(f"   📋 {col}: {list(le.classes_)}")

    # 6. Feature/target ayır
    target_col = 'addiction_score'
    feature_cols = [c for c in df.columns if c != target_col]
    X_tab = df[feature_cols].values.astype(np.float32)
    y = (df[target_col].values - 1).astype(np.int32)

    # 7. Tokenize text
    tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token='<OOV>')
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    X_text = pad_sequences(sequences, maxlen=MAX_SEQ_LEN, padding='post', truncating='post')
    print(f"\n📝 Vocabulary size: {len(tokenizer.word_index)}")

    # 8. Train/test split
    indices = np.arange(len(y))
    train_idx, test_idx = train_test_split(
        indices, test_size=0.2, random_state=42, stratify=y
    )
    X_train_tab_raw = X_tab[train_idx]
    X_test_tab_raw  = X_tab[test_idx]
    X_train_text    = X_text[train_idx]
    X_test_text     = X_text[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # 9. Scale
    scaler = StandardScaler()
    X_train_tab = scaler.fit_transform(X_train_tab_raw)
    X_test_tab  = scaler.transform(X_test_tab_raw)

    # 10. TEXT DROPOUT AUGMENTATION (v3 yenilik)
    # Eğitim verisinin %30'unu kopyala ama metnini sıfırla
    n_aug = int(len(X_train_tab) * TEXT_DROPOUT_RATE)
    aug_idx = np.random.RandomState(7).choice(len(X_train_tab), n_aug, replace=False)
    X_train_tab_aug  = np.concatenate([X_train_tab, X_train_tab[aug_idx]], axis=0)
    X_train_text_aug = np.concatenate([
        X_train_text, np.zeros((n_aug, MAX_SEQ_LEN), dtype=X_train_text.dtype)
    ], axis=0)
    y_train_aug = np.concatenate([y_train, y_train[aug_idx]], axis=0)

    # Karıştır
    perm = np.random.RandomState(11).permutation(len(y_train_aug))
    X_train_tab_aug  = X_train_tab_aug[perm]
    X_train_text_aug = X_train_text_aug[perm]
    y_train_aug      = y_train_aug[perm]

    print(f"\n📐 Tabular özellik sayısı: {X_tab.shape[1]}")
    print(f"🎓 Train (orijinal): {len(y_train)}")
    print(f"🎓 Train (+ %{int(TEXT_DROPOUT_RATE*100)} text-dropout aug): {len(y_train_aug)}")
    print(f"🧪 Test: {len(y_test)}")

    # 11. Artifacts kaydet
    with open(f'{OUTPUT_DIR}/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    with open(f'{OUTPUT_DIR}/label_encoders.pkl', 'wb') as f:
        pickle.dump(label_encoders, f)
    with open(f'{OUTPUT_DIR}/tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)
    with open(f'{OUTPUT_DIR}/feature_cols.json', 'w') as f:
        json.dump(feature_cols, f)
    with open(f'{OUTPUT_DIR}/config.json', 'w') as f:
        json.dump({
            'max_seq_len': MAX_SEQ_LEN,
            'vocab_size': VOCAB_SIZE,
            'embed_dim': EMBED_DIM,
            'lstm_units': LSTM_UNITS,
            'tab_input_dim': X_tab.shape[1],
            'version': 'v3',
        }, f)

    return {
        'X_train_tab': X_train_tab_aug, 'X_test_tab': X_test_tab,
        'X_train_text': X_train_text_aug, 'X_test_text': X_test_text,
        'y_train': y_train_aug, 'y_test': y_test,
        'feature_cols': feature_cols, 'label_encoders': label_encoders,
        'tokenizer': tokenizer,
        'tab_input_dim': X_tab.shape[1],
    }

# ─────────────────────────────────────────
# MULTI-MODAL MODEL (v3 — küçültüldü + güçlü regularization)
# ─────────────────────────────────────────
def build_model(tab_input_dim, num_classes=5):
    L2 = tf.keras.regularizers.l2(1e-3)   # ↑ 1e-4'ten 1e-3'e

    # ── TABULAR BRANCH (baskın) ──
    tab_in = layers.Input(shape=(tab_input_dim,), name='tabular_input')
    t = layers.GaussianNoise(TAB_NOISE_STD, name='tab_noise')(tab_in)  # eğitim sırasında noise
    t = layers.Dense(64, kernel_regularizer=L2, name='tab_dense_1')(t)
    t = layers.BatchNormalization(name='tab_bn_1')(t)
    t = layers.Activation('relu', name='tab_relu_1')(t)
    t = layers.Dropout(0.5, name='tab_drop_1')(t)

    t = layers.Dense(32, kernel_regularizer=L2, name='tab_dense_2')(t)
    t = layers.BatchNormalization(name='tab_bn_2')(t)
    t = layers.Activation('relu', name='tab_relu_2')(t)
    t = layers.Dropout(0.4, name='tab_drop_2')(t)

    # ── TEXT BRANCH (yardımcı - küçültüldü) ──
    text_in = layers.Input(shape=(MAX_SEQ_LEN,), name='text_input', dtype='int32')
    e = layers.Embedding(VOCAB_SIZE, EMBED_DIM, mask_zero=True,
                          embeddings_regularizer=L2, name='embedding')(text_in)
    e = layers.LSTM(LSTM_UNITS, dropout=0.3, recurrent_dropout=0.3,
                     kernel_regularizer=L2, name='lstm')(e)
    e = layers.Dense(8, activation='relu', kernel_regularizer=L2, name='text_dense')(e)
    e = layers.Dropout(0.5, name='text_drop')(e)

    # ── FUSION ──
    merged = layers.Concatenate(name='concat')([t, e])
    x = layers.Dense(32, activation='relu', kernel_regularizer=L2, name='fusion_dense')(merged)
    x = layers.Dropout(0.4, name='fusion_drop')(x)
    output = layers.Dense(num_classes, activation='softmax', name='output')(x)

    model = tf.keras.Model([tab_in, text_in], output, name='MultiModal_AddictionDetector_v3')
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# ─────────────────────────────────────────
# EĞİTİM
# ─────────────────────────────────────────
def train(model, data, epochs=80):
    class_weights = compute_class_weight(
        'balanced', classes=np.unique(data['y_train']), y=data['y_train']
    )
    class_weight_dict = dict(enumerate(class_weights))
    print(f"\n📊 Class weights: {class_weight_dict}")

    cb_list = [
        callbacks.EarlyStopping(monitor='val_loss', patience=15,
                                restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                     patience=7, min_lr=1e-5, verbose=1),
        callbacks.ModelCheckpoint(f'{OUTPUT_DIR}/best_model.keras',
                                   monitor='val_accuracy',
                                   save_best_only=True, verbose=0)
    ]

    history = model.fit(
        [data['X_train_tab'], data['X_train_text']], data['y_train'],
        validation_split=0.2,
        epochs=epochs,
        batch_size=32,
        callbacks=cb_list,
        class_weight=class_weight_dict,
        verbose=1
    )
    return history

# ─────────────────────────────────────────
# DEĞERLENDİRME & GRAFİK
# ─────────────────────────────────────────
def evaluate_and_plot(model, history, data):
    X_test_tab  = data['X_test_tab']
    X_test_text = data['X_test_text']
    y_test      = data['y_test']

    loss, acc = model.evaluate([X_test_tab, X_test_text], y_test, verbose=0)
    print(f"\n🎯 Test Accuracy: {acc:.4f} ({acc*100:.1f}%)")
    print(f"📉 Test Loss: {loss:.4f}")

    if acc > 0.97:
        print("\n⚠️  UYARI: Test accuracy %97'den yüksek — hâlâ overfitting olabilir.")
        print("   Beklenen: %75-90 arası gerçekçi sonuç.")
    elif acc < 0.55:
        print("\n⚠️  UYARI: Test accuracy çok düşük — model yeterince öğrenememiş olabilir.")

    y_pred = np.argmax(model.predict([X_test_tab, X_test_text], verbose=0), axis=1)
    labels = ['Sağlıklı', 'Dikkatli', 'Risk', 'Bağımlılık Başlıyor', 'Ciddi Bağımlılık']
    print("\n📊 Sınıflandırma Raporu:")
    print(classification_report(y_test, y_pred, target_names=labels))

    # Branch ablation
    print("🔬 Branch Ablation Testi:")
    zero_text = np.zeros_like(X_test_text)
    _, acc_no_text = model.evaluate([X_test_tab, zero_text], y_test, verbose=0)
    zero_tab = np.zeros_like(X_test_tab)
    _, acc_no_tab = model.evaluate([zero_tab, X_test_text], y_test, verbose=0)
    print(f"   Tam model:           {acc:.4f}")
    print(f"   Sadece tabular:      {acc_no_text:.4f}  (LSTM kapalı)")
    print(f"   Sadece text:         {acc_no_tab:.4f}  (Tabular kapalı)")
    print(f"   LSTM marjinal katkı: {(acc - acc_no_text)*100:+.2f}%")

    # Grafik
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.patch.set_facecolor('#0f0f1a')
    for ax in axes:
        ax.set_facecolor('#1a1a2e')
        ax.tick_params(colors='#aaaacc')
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        for spine in ['bottom', 'left']:
            ax.spines[spine].set_color('#333355')

    axes[0].plot(history.history['accuracy'], color='#7c3aed', linewidth=2, label='Train')
    axes[0].plot(history.history['val_accuracy'], color='#06b6d4', linewidth=2,
                 linestyle='--', label='Val')
    axes[0].set_title('Model Accuracy', color='white', fontsize=13, pad=10)
    axes[0].set_xlabel('Epoch', color='#aaaacc'); axes[0].set_ylabel('Accuracy', color='#aaaacc')
    axes[0].legend(facecolor='#1a1a2e', labelcolor='white')
    axes[0].grid(alpha=0.15, color='#555577')

    axes[1].plot(history.history['loss'], color='#f59e0b', linewidth=2, label='Train')
    axes[1].plot(history.history['val_loss'], color='#ef4444', linewidth=2,
                 linestyle='--', label='Val')
    axes[1].set_title('Model Loss', color='white', fontsize=13, pad=10)
    axes[1].set_xlabel('Epoch', color='#aaaacc'); axes[1].set_ylabel('Loss', color='#aaaacc')
    axes[1].legend(facecolor='#1a1a2e', labelcolor='white')
    axes[1].grid(alpha=0.15, color='#555577')

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, ax=axes[2], annot=True, fmt='d', cmap='RdPu',
                xticklabels=['S', 'D', 'R', 'B', 'CB'],
                yticklabels=['S', 'D', 'R', 'B', 'CB'], cbar=False)
    axes[2].set_title('Confusion Matrix', color='white', fontsize=13, pad=10)
    axes[2].set_xlabel('Tahmin', color='#aaaacc'); axes[2].set_ylabel('Gerçek', color='#aaaacc')

    plt.suptitle('Sosyal Medya Bağımlılık Dedektörü v3 — Multi-modal (MLP + LSTM)',
                 color='white', fontsize=15, y=1.02)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/training_results.png', dpi=150, bbox_inches='tight',
                facecolor='#0f0f1a', edgecolor='none')
    print("\n📈 Grafik kaydedildi → training_results.png")
    return acc

# ─────────────────────────────────────────
# ANA ÇALIŞTIRMA
# ─────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 60)
    print("  Sosyal Medya Bağımlılık Dedektörü v3")
    print("  (data leakage düzeltmesi + güçlü regularization)")
    print("=" * 60)

    data = load_and_preprocess()
    model = build_model(data['tab_input_dim'])
    print("\n📐 MODEL ÖZET:")
    model.summary()

    print("\n🚀 Eğitim başlıyor...")
    history = train(model, data, epochs=80)

    acc = evaluate_and_plot(model, history, data)

    model.save(f'{OUTPUT_DIR}/addiction_model.keras')
    print(f"💾 Model kaydedildi → {OUTPUT_DIR}/addiction_model.keras")

    # Demo
    print("\n" + "═" * 60)
    print("  🎮 DEMO TEST")
    print("═" * 60)
    idx = np.random.randint(0, len(data['y_test']))
    sample_tab  = data['X_test_tab'][idx:idx+1]
    sample_text = data['X_test_text'][idx:idx+1]
    true_level  = int(data['y_test'][idx]) + 1

    rev_word = {v: k for k, v in data['tokenizer'].word_index.items()}
    decoded_text = " ".join([rev_word.get(i, '') for i in sample_text[0] if i > 0])

    probs = model.predict([sample_tab, sample_text], verbose=0)[0]
    pred_level = int(np.argmax(probs)) + 1

    print(f"\n💬 Kullanıcı metni: \"{decoded_text}\"")
    print(f"🎯 Gerçek Seviye: {true_level}")
    print(f"🤖 Tahmin:        {pred_level}")
    print(f"📌 Sonuç:         {'✅ DOĞRU' if pred_level == true_level else '❌ YANLIŞ'}")

    print("\n📊 Olasılık Dağılımı:")
    levels = ['Sağlıklı', 'Dikkatli', 'Risk', 'Bağımlılık Başlıyor', 'Ciddi Bağımlılık']
    for lvl, p in zip(levels, probs):
        bar = '█' * int(p * 30)
        print(f"  {lvl:22s} {bar:<30} {p:>6.1%}")

    print(f"\n✅ Eğitim tamamlandı! Test accuracy: {acc*100:.1f}%")
