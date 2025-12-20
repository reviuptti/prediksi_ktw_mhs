import os
import sys
import gc
import getpass
import numpy as np
import pandas as pd
import matplotlib
# Gunakan 'Agg' agar bisa jalan di background/terminal tanpa monitor
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import seaborn as sns 
import joblib
import optuna
from functools import partial 

# Import visualisasi Optuna
from optuna.visualization.matplotlib import plot_optimization_history
from optuna.visualization.matplotlib import plot_param_importances
from optuna.visualization.matplotlib import plot_slice

from sqlalchemy import create_engine
from urllib.parse import quote_plus
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.utils import class_weight 
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve # <--- Added Metrics

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ==========================================
# 1. KONFIGURASI GPU
# ==========================================
def setup_desktop_gpu():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"[INFO] ✅ GPU Siap: {len(gpus)} unit.")
        else:
            print("[WARN] ⚠️ GPU tidak terdeteksi, menggunakan CPU.")
    except Exception as e:
        print(f"[ERROR] Gagal konfigurasi GPU: {e}")

# ==========================================
# 2. HELPER: PREPROCESSING CERDAS
# ==========================================
def sanitize_status(status_raw):
    if pd.isna(status_raw): return "AKTIF"
    status = str(status_raw).upper().strip()
    if status in ['LULUS', 'YUDISIUM', 'WISUDA', 'L']: return 'AKTIF'
    elif status in ['CUTI', 'C']: return 'CUTI'
    elif status in ['NON-AKTIF', 'N', 'DO', 'D', 'KELUAR', 'K']: return 'NON-AKTIF'
    return 'AKTIF'

def smart_imputation(df):
    df_clean = df.copy()
    seq_prefixes = ['ipk', 'ips', 'sks_smt', 'bayar_ukt', 'status_beasiswa']
    for prefix in seq_prefixes:
        cols = [c for c in df_clean.columns if c.startswith(f"{prefix}_")]
        try: cols.sort(key=lambda x: int(x.split('_')[-1]))
        except: continue
        if not cols: continue

        if prefix in ['ipk', 'ips', 'sks_smt']:
            for col in cols:
                df_clean[col] = df_clean[col].replace(0.0, np.nan).replace(0, np.nan)
        
        df_clean[cols] = df_clean[cols].ffill(axis=1)
        df_clean[cols] = df_clean[cols].fillna(0)

    if 'umur_saat_masuk' in df_clean.columns:
        df_clean['umur_saat_masuk'] = df_clean['umur_saat_masuk'].fillna(20)

    return df_clean

def run_eda_and_importance(X_raw, y_raw, categorical_cols, numeric_cols):
    plt.figure(figsize=(6, 4))
    sns.countplot(x=y_raw)
    plt.title("Distribusi Target (Lulus Tepat Waktu)")
    plt.savefig('analisis_1_target_dist_mlp.png') # Renamed output
    plt.close()
    
    if len(numeric_cols) > 1:
        df_num = X_raw[numeric_cols].copy()
        df_num['target'] = y_raw
        corr = df_num.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr[['target']].sort_values(by='target', ascending=False), 
                    annot=True, cmap='coolwarm')
        plt.title("Korelasi Fitur vs Target")
        plt.tight_layout()
        plt.savefig('analisis_2_correlation_mlp.png') # Renamed output
        plt.close()

# ==========================================
# HELPER: ADVANCED PLOTTING (BARU)
# ==========================================
def save_advanced_plots(history, y_true, y_pred_proba, suffix="_mlp"):
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
# 3. MODEL & OPTUNA
# ==========================================
def create_model(trial, input_dim):
    model = Sequential()
    model.add(Input(shape=(input_dim,)))
    n_layers = trial.suggest_int('n_layers', 1, 3)
    
    for i in range(n_layers):
        units = trial.suggest_int(f'units_{i}', 32, 128, step=32)
        model.add(Dense(units, activation='relu'))
        model.add(BatchNormalization())
        dropout = trial.suggest_float(f'dropout_{i}', 0.1, 0.4)
        model.add(Dropout(dropout))
    
    model.add(Dense(1, activation='sigmoid'))
    lr = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    model.compile(optimizer=Adam(learning_rate=lr), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# MODIFIED OBJECTIVE FOR 5-FOLD
def objective(trial, X, y, class_weights):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        tf.keras.backend.clear_session()
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]

        model = create_model(trial, X.shape[1])
        
        history = model.fit(
            X_train_fold, y_train_fold,
            validation_data=(X_val_fold, y_val_fold),
            epochs=5, 
            batch_size=trial.suggest_categorical('batch_size', [32, 64]),
            class_weight=class_weights,
            verbose=0
        )
        scores.append(max(history.history['val_accuracy']))
        del model; gc.collect()
    
    return np.mean(scores)

# ==========================================
# 4. MAIN PROGRAM
# ==========================================
def main():
    setup_desktop_gpu()

    print("\n[1/7] Loading Data...")
    db_host, db_port = '', ''
    db_name, db_user = '', ''
    encoded_pass = quote_plus('')

    try:
        engine = create_engine(f'postgresql+psycopg2://{db_user}:{encoded_pass}@{db_host}:{db_port}/{db_name}')
        df = pd.read_sql("SELECT DISTINCT mm.umur_saat_masuk, mm.kdseleksi::text AS kdseleksi, mm.kdstratapendidikan, mm.kdjeniskelamin, mm.kdfakultas, mm.sks_total_smt_1, mm.sks_total_smt_2, mm.sks_total_smt_3, mm.sks_total_smt_4, mm.sks_total_smt_5, mm.sks_total_smt_6, mm.sks_total_smt_7, mm.sks_total_smt_8, mm.ipk_1, mm.ipk_2, mm.ipk_3, mm.ipk_4, mm.ipk_5, mm.ipk_6, mm.ipk_7, mm.ipk_8, mm.ips_1, mm.ips_2, mm.ips_3, mm.ips_4, mm.ips_5, mm.ips_6, mm.ips_7, mm.ips_8, mm.sks_smt_1, mm.sks_smt_2, mm.sks_smt_3, mm.sks_smt_4, mm.sks_smt_5, mm.sks_smt_6, mm.sks_smt_7, mm.sks_smt_8, mm.kdstatusakademik_1, mm.kdstatusakademik_2, mm.kdstatusakademik_3, mm.kdstatusakademik_4, mm.kdstatusakademik_5, mm.kdstatusakademik_6, mm.kdstatusakademik_7, mm.kdstatusakademik_8, mm.bayar_ukt_1, mm.bayar_ukt_2, mm.bayar_ukt_3, mm.bayar_ukt_4, mm.bayar_ukt_5, mm.bayar_ukt_6, mm.bayar_ukt_7, mm.bayar_ukt_8, mm.status_beasiswa_1, mm.status_beasiswa_2, mm.status_beasiswa_3, mm.status_beasiswa_4, mm.status_beasiswa_5, mm.status_beasiswa_6, mm.status_beasiswa_7, mm.status_beasiswa_8, mm.status_lulus_tepat_waktu, mm.rata_rata_sks, mm.rata_rata_ips, mm.rata_rata_ipk, mm.tren_ipk, mm.stddev_ips, mm.jumlah_smt_gagal FROM dl.master_mhs mm WHERE mm.status_lulus_tepat_waktu IS NOT NULL and CASE WHEN mm.kdstratapendidikan IN ( '10' , '30' ) THEN lama_studi_semester <= 14 WHEN mm.kdstratapendidikan IN ( '03' , '04' ) THEN mm.lama_studi_semester <= 10 WHEN mm.kdstratapendidikan IN ( '20' ) THEN mm.lama_studi_semester <= 8 ELSE lama_studi_semester <= 5 END", engine)
        print(f"   -> Data dimuat: {df.shape}")
    except Exception as e:
        print(f"[ERROR] Database: {e}")
        return

    print("\n[2/7] Cleaning & Prep...")
    # status_cols = [c for c in df.columns if 'kdstatusakademik' in c]
    # for col in status_cols: df[col] = df[col].apply(sanitize_status)
    # df = smart_imputation(df)
    
    # cols_to_drop = ['status_lulus_tepat_waktu', 'lama_studi_semester'] + [c for c in df.columns if 'sks_total_smt' in c] 
    # X_raw = df.drop(cols_to_drop, errors='ignore')
    y_raw = df['status_lulus_tepat_waktu']
    X_raw=df.drop(['status_lulus_tepat_waktu'], axis=1)

    
    categorical_features = [col for col in X_raw.columns if X_raw[col].dtype == 'object']
    numeric_features = [col for col in X_raw.columns if col not in categorical_features]
    run_eda_and_importance(X_raw, y_raw, categorical_features, numeric_features)

    print("\n[3/7] Transformasi Data...")
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ]
    )
    X_processed = preprocessor.fit_transform(X_raw).astype('float32')
    joblib.dump(preprocessor, 'preprocessor_pipeline.pkl')
    y_values = y_raw.values.astype('float32')

    # Hitung Class Weights Global
    print("   -> Menghitung Class Weights...")
    class_weights_vals = class_weight.compute_class_weight(
        class_weight='balanced', classes=np.unique(y_values), y=y_values
    )
    class_weights_dict = dict(enumerate(class_weights_vals))

    # --- F. OPTUNA TUNING (5-Fold) ---
    print("\n[5/7] Memulai Optuna Tuning (5-Fold CV)...")
    
    objective_with_args = partial(objective, X=X_processed, y=y_values, class_weights=class_weights_dict)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective_with_args, n_trials=10) 

    print("\n[6/7] Training Model Terbaik...")
    best_params = study.best_params
    
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y_values, test_size=0.2, random_state=42, stratify=y_values)

    final_model = Sequential()
    final_model.add(Input(shape=(X_train.shape[1],)))
    for i in range(best_params['n_layers']):
        final_model.add(Dense(best_params[f'units_{i}'], activation='relu'))
        final_model.add(BatchNormalization())
        final_model.add(Dropout(best_params[f'dropout_{i}']))
    final_model.add(Dense(1, activation='sigmoid'))
    final_model.compile(optimizer=Adam(learning_rate=best_params['learning_rate']),
                        loss='binary_crossentropy', metrics=['accuracy'])
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
    
    history = final_model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=50,
        batch_size=best_params['batch_size'],
        callbacks=[early_stopping, reduce_lr],
        class_weight=class_weights_dict,
        verbose=1
    )
    
    final_model.save('model_mlp.h5')

    # --- G. EVALUASI & VISUALISASI LENGKAP ---
    print("\n[7/7] Evaluasi Akhir & Plotting...")
    y_pred_prob = final_model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int)
    
    print(classification_report(y_test, y_pred, target_names=['Terlambat', 'Tepat Waktu']))
    
    # 1. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Terlambat', 'Tepat Waktu'],
                yticklabels=['Terlambat', 'Tepat Waktu'])
    plt.title('Confusion Matrix - MLP')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('grafik_confusion_matrix_mlp.png')
    plt.close()
    
    # 2. Accuracy Plot
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title('Training & Validation Accuracy - MLP')
    plt.legend()
    plt.tight_layout()
    plt.savefig('grafik_training_accuracy_mlp.png')
    plt.close()

    # 3. ADVANCED PLOTS (Loss, ROC, PR, Hist)
    save_advanced_plots(history, y_test, y_pred_prob.flatten(), suffix="_mlp")

    # 4. Optuna Plots
    try:
        plot_optimization_history(study).figure.savefig('optuna_history_mlp.png')
        plot_param_importances(study).figure.savefig('optuna_importance_mlp.png')
    except: pass
    
    print("[SELESAI] Semua proses dan visualisasi selesai.")

if __name__ == "__main__":
    main()
