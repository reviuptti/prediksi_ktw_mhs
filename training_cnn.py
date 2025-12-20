import os
import sys
import gc
import getpass
from urllib.parse import quote_plus
import numpy as np
import pandas as pd
import matplotlib
# Gunakan 'Agg' agar bisa jalan di terminal tanpa error display driver
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import seaborn as sns 
import joblib
import optuna 

# Import Visualisasi Optuna
from optuna.visualization.matplotlib import plot_optimization_history
from optuna.visualization.matplotlib import plot_param_importances
from optuna.visualization.matplotlib import plot_slice

# Scikit-learn
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve # <--- Added Metrics
from sklearn.utils import class_weight 

# TensorFlow & Keras
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, Concatenate, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical 
from tensorflow.keras.backend import clear_session

# Database
from sqlalchemy import create_engine

# ==========================================
# 1. KONFIGURASI KOLOM
# ==========================================
COL_CONFIG = {
    'sks_total_smt_cols': [f'sks_total_smt_{i}' for i in range(1, 9)],   
    'ipk_cols': [f'ipk_{i}' for i in range(1, 9)],   
    'ips_cols': [f'ips_{i}' for i in range(1, 9)],   
    'sks_smt_cols': [f'sks_smt_{i}' for i in range(1, 9)], 
    'bayar_ukt_cols': [f'bayar_ukt_{i}' for i in range(1, 9)], 
    'status_beasiswa_cols': [f'status_beasiswa_{i}' for i in range(1, 9)], 
    'status_akademik_cols': [f'kdstatusakademik_{i}' for i in range(1, 9)],
    'cat_features': ['kdstratapendidikan', 'kdfakultas', 'kdjeniskelamin', 'kdeseleksi'],
    'num_features': ['umur_saat_masuk', 'rata_rata_sks', 'rata_rata_ips', 'rata_rata_ipk', 'tren_ipk', 'stddev_ips', 'jumlah_smt_gagal'], 
    'target': 'status_lulus_tepat_waktu'
}

# ==========================================
# 2. SETUP GPU
# ==========================================
def setup_desktop_gpu():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"[INFO] ✅ GPU Siap: {len(gpus)} unit (GTX 1660 SUPER).")
        else:
            print("[WARN] ⚠️ GPU tidak terdeteksi.")
    except Exception as e:
        print(f"[ERROR] Setup GPU: {e}")

# ==========================================
# 3. PREPROCESSING
# ==========================================
def prepare_hybrid_data(df, config):
    print("[INFO] Memproses Data Hybrid...")
    
    status_map = {'A': 0, 'C': 1, 'N': 2, 'K': 3,'0': 4} 
    NUM_CLASSES_STATUS = 5 
    
    status_matrix = []
    for col in config['status_akademik_cols']:
        if col not in df.columns: df[col] = '0'
        clean_col = df[col].astype(str).str.upper().str.strip().fillna('0')
        mapped_col = clean_col.map(status_map).fillna(3).astype(int).values
        status_matrix.append(mapped_col)
    
    status_matrix = np.stack(status_matrix, axis=1) 
    X_status_seq = to_categorical(status_matrix, num_classes=NUM_CLASSES_STATUS)

    seq_cols = config['ipk_cols'] + config['ips_cols']+ config['sks_smt_cols'] + config['sks_total_smt_cols']+ config['bayar_ukt_cols'] + config['status_beasiswa_cols']
    for col in seq_cols:
        if col not in df.columns: df[col] = 0.0
        df[col] = df[col].fillna(0.0)

    data_arrays = []
    scalers_dict = {} 
    
    group_mapping = {
        'ipk': config['ipk_cols'],
        'ips': config['ips_cols'],
        'sks_smt': config['sks_smt_cols'],
        'sks_total': config['sks_total_smt_cols'],
        'bayar_ukt': config['bayar_ukt_cols'],
        'beasiswa': config['status_beasiswa_cols']
    }

    for key, cols in group_mapping.items():
        scaler = StandardScaler()
        scaled_values = scaler.fit_transform(df[cols].values)
        data_arrays.append(scaled_values)
        scalers_dict[key] = scaler

    joblib.dump(scalers_dict, 'scaler_seq.pkl')
    print("[INFO] Scaler sequential berhasil disimpan ke 'scaler_seq.pkl'")
    
    X_numeric_seq = np.dstack(data_arrays).astype('float32')
    X_seq_final = np.concatenate([X_numeric_seq, X_status_seq], axis=2).astype('float32')

    valid_cat = [c for c in config['cat_features'] if c in df.columns]
    valid_num = [c for c in config['num_features'] if c in df.columns]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), valid_num),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), valid_cat)
        ], remainder='drop'
    )
    
    X_static = preprocessor.fit_transform(df).astype('float32')
    joblib.dump(preprocessor, 'preprocessor_static.pkl')
    
    return X_seq_final, X_static, df[config['target']].values.astype('float32')

# ==========================================
# HELPER: ADVANCED PLOTTING (BARU)
# ==========================================
def save_advanced_plots(history, y_true, y_pred_proba, suffix="_cnn"):
    print(f"[INFO] Membuat Plot Lanjutan ({suffix})...")
    
    # 1. Loss Curve
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title(f'Model Loss Convergence {suffix}')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f'grafik_loss{suffix}.png')
    plt.close()

    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve {suffix}')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.savefig(f'grafik_roc{suffix}.png')
    plt.close()

    # 3. Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='green', lw=2, label='Precision-Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve {suffix}')
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.savefig(f'grafik_pr_curve{suffix}.png')
    plt.close()

    # 4. Probability Histogram
    plt.figure(figsize=(8, 6))
    plt.hist(y_pred_proba[y_true==0], bins=30, alpha=0.5, color='red', label='Aktual: Terlambat')
    plt.hist(y_pred_proba[y_true==1], bins=30, alpha=0.5, color='blue', label='Aktual: Tepat Waktu')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Count')
    plt.title(f'Distribution of Predictions {suffix}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f'grafik_prob_dist{suffix}.png')
    plt.close()

# ==========================================
# 4. BUILD MODEL
# ==========================================
def build_hybrid_model(trial, seq_shape, static_shape):
    filters_1 = trial.suggest_categorical('filters_1', [32, 64, 128])
    filters_2 = trial.suggest_categorical('filters_2', [64, 128, 256])
    kernel_size = trial.suggest_int('kernel_size', 2, 4)
    dense_static = trial.suggest_int('dense_static', 32, 128)
    dense_fusion_1 = trial.suggest_int('dense_fusion_1', 64, 256)
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)

    input_seq = Input(shape=seq_shape, name="Input_Seq")
    x1 = Conv1D(filters=filters_1, kernel_size=kernel_size, padding='same', activation='relu')(input_seq)
    x1 = BatchNormalization()(x1)
    x1 = MaxPooling1D(pool_size=2)(x1)
    x1 = Conv1D(filters=filters_2, kernel_size=kernel_size, padding='same', activation='relu')(x1)
    x1 = BatchNormalization()(x1)
    x1 = Flatten()(x1)
    
    input_static = Input(shape=(static_shape,), name="Input_Static")
    x2 = Dense(dense_static, activation='relu')(input_static)
    x2 = BatchNormalization()(x2)
    x2 = Dropout(dropout_rate)(x2)
    
    combined = Concatenate()([x1, x2])
    z = Dense(dense_fusion_1, activation='relu')(combined)
    z = Dropout(dropout_rate)(z)
    z = Dense(64, activation='relu')(z) 
    
    output = Dense(1, activation='sigmoid')(z)
    
    model = Model(inputs=[input_seq, input_static], outputs=output)
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# ==========================================
# 5. MAIN EXECUTION
# ==========================================
def main():
    setup_desktop_gpu()

    print("\n[1/7] Loading Data...")
    db_host = ''
    db_port = ''
    db_name = ''
    db_user = ''
    encoded_pass = quote_plus('')
    
    try:
        engine = create_engine(f'postgresql+psycopg2://{db_user}:{encoded_pass}@{db_host}:{db_port}/{db_name}')
        df = pd.read_sql("SELECT DISTINCT mm.umur_saat_masuk, mm.kdseleksi::text AS kdseleksi, mm.kdstratapendidikan, mm.kdjeniskelamin, mm.kdfakultas, mm.sks_total_smt_1, mm.sks_total_smt_2, mm.sks_total_smt_3, mm.sks_total_smt_4, mm.sks_total_smt_5, mm.sks_total_smt_6, mm.sks_total_smt_7, mm.sks_total_smt_8, mm.ipk_1, mm.ipk_2, mm.ipk_3, mm.ipk_4, mm.ipk_5, mm.ipk_6, mm.ipk_7, mm.ipk_8, mm.ips_1, mm.ips_2, mm.ips_3, mm.ips_4, mm.ips_5, mm.ips_6, mm.ips_7, mm.ips_8, mm.sks_smt_1, mm.sks_smt_2, mm.sks_smt_3, mm.sks_smt_4, mm.sks_smt_5, mm.sks_smt_6, mm.sks_smt_7, mm.sks_smt_8, mm.kdstatusakademik_1, mm.kdstatusakademik_2, mm.kdstatusakademik_3, mm.kdstatusakademik_4, mm.kdstatusakademik_5, mm.kdstatusakademik_6, mm.kdstatusakademik_7, mm.kdstatusakademik_8, mm.bayar_ukt_1, mm.bayar_ukt_2, mm.bayar_ukt_3, mm.bayar_ukt_4, mm.bayar_ukt_5, mm.bayar_ukt_6, mm.bayar_ukt_7, mm.bayar_ukt_8, mm.status_beasiswa_1, mm.status_beasiswa_2, mm.status_beasiswa_3, mm.status_beasiswa_4, mm.status_beasiswa_5, mm.status_beasiswa_6, mm.status_beasiswa_7, mm.status_beasiswa_8, mm.status_lulus_tepat_waktu, mm.rata_rata_sks, mm.rata_rata_ips, mm.rata_rata_ipk, mm.tren_ipk, mm.stddev_ips, mm.jumlah_smt_gagal FROM dl.master_mhs mm WHERE mm.status_lulus_tepat_waktu IS NOT NULL and CASE WHEN mm.kdstratapendidikan IN ( '10' , '30' ) THEN lama_studi_semester <= 14 WHEN mm.kdstratapendidikan IN ( '03' , '04' ) THEN mm.lama_studi_semester <= 10 WHEN mm.kdstratapendidikan IN ( '20' ) THEN mm.lama_studi_semester <= 8 ELSE lama_studi_semester <= 5 END", engine)
        
        if COL_CONFIG['target'] not in df.columns:
            def temp_target(r):
                limit = 6 if 'diploma' in str(r.get('kdstratapendidikan','')).lower() else 8
                return 1 if r.get('lama_studi_semester', 10) <= limit else 0
            df[COL_CONFIG['target']] = df.apply(temp_target, axis=1)
        df = df.dropna(subset=[COL_CONFIG['target']])
        
    except Exception as e:
        print(f"[ERROR] {e}"); sys.exit(1)

    print("\n[2/7] Preprocessing...")
    X_seq, X_static, y = prepare_hybrid_data(df, COL_CONFIG)
    SEQ_SHAPE = (X_seq.shape[1], X_seq.shape[2])
    STATIC_SHAPE = X_static.shape[1]
    del df; gc.collect()

    # --- HITUNG CLASS WEIGHTS ---
    print("\n[3/7] Menghitung Class Weights Global...")
    class_weights_vals = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y),
        y=y
    )
    class_weights_dict = dict(enumerate(class_weights_vals))
    print(f"   -> Class Weights: {class_weights_dict}")

    # --- C. OPTUNA OPTIMIZATION (MODIFIED) ---
    print("\n[4/7] Memulai Optuna Tuning (5-Fold CV)...")
    BATCH_SIZE = 128
    
    def objective(trial):
        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = []
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X_seq, y)):
            model = build_hybrid_model(trial, SEQ_SHAPE, STATIC_SHAPE)
            history = model.fit(
                [X_seq[train_idx], X_static[train_idx]], y[train_idx],
                validation_data=([X_seq[val_idx], X_static[val_idx]], y[val_idx]),
                epochs=5, 
                batch_size=BATCH_SIZE, 
                class_weight=class_weights_dict, 
                verbose=0
            )
            scores.append(max(history.history['val_accuracy']))
            del model; clear_session(); gc.collect()
        return np.mean(scores)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=10) 

    print("\n[INFO] Menyimpan Plot Visualisasi Optuna...")
    try:
        plot_optimization_history(study).figure.savefig('optuna_1_history_cnn.png')
        plot_param_importances(study).figure.savefig('optuna_2_importance_cnn.png')
        plot_slice(study).figure.savefig('optuna_3_slice_cnn.png')
    except Exception as e: print(f"   -> Gagal plot Optuna: {e}")

    # --- D. FINAL TRAINING ---
    print("\n[5/7] Training Model Final...")
    indices = np.arange(len(y))
    idx_train, idx_test = train_test_split(indices, test_size=0.2, stratify=y, random_state=42)
    
    class FixedTrial:
        def suggest_categorical(self, name, choices): return study.best_params[name]
        def suggest_int(self, name, low, high): return study.best_params[name]
        def suggest_float(self, name, low, high, log=False): return study.best_params[name]
    
    final_model = build_hybrid_model(FixedTrial(), SEQ_SHAPE, STATIC_SHAPE)
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
    ]
    
    history = final_model.fit(
        [X_seq[idx_train], X_static[idx_train]], y[idx_train],
        validation_data=([X_seq[idx_test], X_static[idx_test]], y[idx_test]),
        epochs=50, 
        batch_size=BATCH_SIZE, 
        callbacks=callbacks, 
        class_weight=class_weights_dict, 
        verbose=1
    )

    # --- E. EVALUASI & VISUALISASI LENGKAP ---
    print("\n[6/7] Evaluasi & Menyimpan Hasil...")
    final_model.save('model_cnn.h5')
    
    y_pred_proba = final_model.predict([X_seq[idx_test], X_static[idx_test]], batch_size=BATCH_SIZE)
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    print("\n--- CLASSIFICATION REPORT ---")
    print(classification_report(y[idx_test], y_pred, target_names=['Terlambat', 'Tepat Waktu']))
    
    # 1. Confusion Matrix
    cm = confusion_matrix(y[idx_test], y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Terlambat', 'Tepat Waktu'],
                yticklabels=['Terlambat', 'Tepat Waktu'])
    plt.title('Confusion Matrix - CNN Hybrid')
    plt.xlabel('Predicted Label')
    plt.ylabel('Actual Label')
    plt.tight_layout()
    plt.savefig('grafik_confusion_matrix_cnn.png')
    plt.close()

    # 2. Accuracy Plot
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title(f'Final Training Accuracy (Best: {study.best_value:.3f})')
    plt.legend()
    plt.savefig('grafik_training_accuracy_cnn.png')
    plt.close()

    # 3. ADVANCED PLOTS (Loss, ROC, PR, Hist)
    save_advanced_plots(history, y[idx_test], y_pred_proba.flatten(), suffix="_cnn")

    print("[SELESAI] Semua proses dan visualisasi selesai.")

if __name__ == "__main__":
    main()
