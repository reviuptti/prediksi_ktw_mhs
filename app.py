import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
from sqlalchemy import create_engine
from urllib.parse import quote_plus
from tensorflow.keras.utils import to_categorical

# ==========================================
# 1. KONFIGURASI HALAMAN & CSS
# ==========================================
st.set_page_config(
    page_title="Sistem Prediksi Kelulusan",
    page_icon="🎓",
    layout="wide"
)

# Custom CSS untuk mempercantik tampilan
st.markdown("""
    <style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #007bff;
    }
    .stAlert {
        padding: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. LOAD ASSETS & CONFIG (CACHED)
# ==========================================

# Konfigurasi Kolom
CNN_CONFIG = {
    'sks_total_smt_cols': [f'sks_total_smt_{i}' for i in range(1, 9)],
    'ipk_cols': [f'ipk_{i}' for i in range(1, 9)],
    'ips_cols': [f'ips_{i}' for i in range(1, 9)],
    'sks_smt_cols': [f'sks_smt_{i}' for i in range(1, 9)],
    'bayar_ukt_cols': [f'bayar_ukt_{i}' for i in range(1, 9)],
    'status_beasiswa_cols': [f'status_beasiswa_{i}' for i in range(1, 9)],
    'status_akademik_cols': [f'kdstatusakademik_{i}' for i in range(1, 9)],
    'cat_features': ['kdstratapendidikan', 'kdfakultas', 'kdjeniskelamin', 'kdeseleksi'],
    'num_features': ['umur_saat_masuk', 'rata_rata_sks', 'rata_rata_ips', 'rata_rata_ipk', 'tren_ipk', 'stddev_ips', 'jumlah_smt_gagal']
}

@st.cache_resource
def load_models_and_scalers():
    """Memuat model dan scaler sekali saja di awal untuk performa."""
    try:
        m_cnn = tf.keras.models.load_model('model_cnn.h5')
        m_mlp = tf.keras.models.load_model('model_mlp.h5')
        s_seq = joblib.load('scaler_seq.pkl')
        p_static = joblib.load('preprocessor_static.pkl')
        p_pipe = joblib.load('preprocessor_pipeline.pkl')
        return m_cnn, m_mlp, s_seq, p_static, p_pipe
    except Exception as e:
        st.error(f"Gagal memuat model/scaler: {e}")
        return None, None, None, None, None

model_cnn, model_mlp, scaler_seq, prep_static, prep_pipeline = load_models_and_scalers()

@st.cache_resource
def get_db_engine():
    """Koneksi database menggunakan st.secrets"""
    try:
        # Mengambil config dari .streamlit/secrets.toml
        db = st.secrets["postgres"]
        encoded_pass = quote_plus(db['password'])
        uri = f"postgresql+psycopg2://{db['user']}:{encoded_pass}@{db['host']}:{db['port']}/{db['dbname']}"
        return create_engine(uri)
    except Exception as e:
        st.error(f"Konfigurasi Database Error: {e}")
        return None

# ==========================================
# 3. HELPER FUNCTIONS (PREPROCESSING)
# ==========================================

def prepare_cnn_input(df_row):
    df = df_row.copy()
    status_map = {'A': 0, 'C': 1, 'N': 2, 'K': 3,'0': 4}
    NUM_CLASSES_STATUS = 5
    
    status_matrix = []
    for col in CNN_CONFIG['status_akademik_cols']:
        if col not in df.columns: df[col] = '0'
        val = str(df[col].iloc[0]).upper().strip()
        val = val if val != 'NONE' and val != 'NAN' else '0'
        mapped_val = status_map.get(val, 3)
        status_matrix.append(mapped_val)
    
    status_matrix = np.array([status_matrix]) 
    X_status_seq = to_categorical(status_matrix, num_classes=NUM_CLASSES_STATUS)

    group_mapping = {
        'ipk': CNN_CONFIG['ipk_cols'],
        'ips': CNN_CONFIG['ips_cols'],
        'sks_smt': CNN_CONFIG['sks_smt_cols'],
        'sks_total': CNN_CONFIG['sks_total_smt_cols'],
        'bayar_ukt': CNN_CONFIG['bayar_ukt_cols'],
        'beasiswa': CNN_CONFIG['status_beasiswa_cols']
    }
    
    data_arrays = []
    for key, cols in group_mapping.items():
        vals = df[cols].fillna(0.0).values
        scaled_vals = scaler_seq[key].transform(vals)
        data_arrays.append(scaled_vals)
    
    X_numeric_seq = np.dstack(data_arrays).astype('float32')
    X_seq_final = np.concatenate([X_numeric_seq, X_status_seq], axis=2)

    for c in CNN_CONFIG['num_features']:
        if c not in df.columns: df[c] = 0
        df[c] = df[c].fillna(0)
        
    X_static = prep_static.transform(df).astype('float32')
    return [X_seq_final, X_static]

def prepare_mlp_input(df_row):
    try:
        return prep_pipeline.transform(df_row)
    except Exception as e:
        st.warning(f"MLP Preprocessing Error: {e}")
        return None

# ==========================================
# 4. UI UTAMA
# ==========================================

st.title("🎓 Dashboard Prediksi Kelulusan (AI)")
st.markdown("Sistem prediksi kelulusan mahasiswa tepat waktu menggunakan *Hybrid Deep Learning (CNN + MLP)*.")

# --- Bagian Pencarian ---
with st.container():
    col1, col2 = st.columns([3, 1])
    with col1:
        nim_input = st.text_input("Masukkan NIM Mahasiswa:", placeholder="Contoh: 18241010xxxx")
    with col2:
        st.write("") # Spacer
        st.write("") 
        search_btn = st.button("🔍 Analisa & Prediksi", type="primary", use_container_width=True)

if search_btn and nim_input:
    engine = get_db_engine()
    if engine:
        try:
            with st.spinner('Mengambil data dan melakukan prediksi...'):
                # Query Database
                query = f"SELECT * FROM dl.master_mhs WHERE reverse(nim) = '{nim_input}'"
                df = pd.read_sql(query, engine)
                
                if df.empty:
                    st.error(f"❌ Data NIM **{nim_input}** tidak ditemukan di database.")
                else:
                    # --- 1. Tampilkan Data Statis ---
                    st.divider()
                    st.subheader("📋 Profil Mahasiswa")
                    
                    # Mengambil data baris pertama
                    row = df.iloc[0]
                    
                    c1, c2, c3, c4 = st.columns(4)
                    c1.markdown(f"**Fakultas**\n{row.get('kdfakultas', '-')}")
                    c2.markdown(f"**Prodi**\n{row.get('kdstratapendidikan', '-')}")
                    c3.markdown(f"**Jenis Kelamin**\n{row.get('kdjeniskelamin', '-')}")
                    c4.markdown(f"**Angkatan/Umur**\n{row.get('umur_saat_masuk', '-')}")
                    
                    st.write("")
                    c1, c2, c3, c4 = st.columns(4)
                    c1.markdown(f"**IPK Akhir**\n{row.get('ipk_saat_ini', 0):.2f}")
                    c2.markdown(f"**Total SKS**\n{row.get('total_sks_saat_ini', 0)}")
                    c3.markdown(f"**Lama Studi**\n{row.get('lama_studi_semester', '-')}")
                    c4.markdown(f"**Status Terakhir**\n{row.get('real_status_akademik_terakhir', '-')}")

                    # --- 2. Tabel Sekuensial ---
                    st.subheader("📊 Riwayat Akademik (Semester 1-8)")
                    
                    history_data = []
                    for i in range(1, 9):
                        ukt_val = row.get(f'nominal_ukt_{i}', 0)
                        bea_val = row.get(f'status_beasiswa_{i}', 0)
                        
                        history_data.append({
                            "Semester": f"Smt {i}",
                            "SKS Smt": row.get(f'sks_smt_{i}', 0),
                            "Total SKS": row.get(f'sks_total_smt_{i}', 0),
                            "IPS": f"{row.get(f'ips_{i}', 0.0):.2f}",
                            "IPK": f"{row.get(f'ipk_{i}', 0.0):.2f}",
                            "UKT (Rp)": f"{int(ukt_val):,}" if pd.notnull(ukt_val) else "0",
                            "Beasiswa": "Ya" if bea_val == 1 else "Tidak",
                            "Status": row.get(f'kdstatusakademik_{i}', '-')
                        })
                    
                    df_history = pd.DataFrame(history_data)
                    st.dataframe(df_history, use_container_width=True, hide_index=True)

                    # --- 3. Proses Prediksi AI ---
                    st.divider()
                    st.subheader("🤖 Hasil Prediksi AI")

                    # Prediksi CNN
                    in_cnn = prepare_cnn_input(df)
                    prob_cnn = model_cnn.predict(in_cnn)[0][0]
                    
                    # Prediksi MLP
                    in_mlp = prepare_mlp_input(df)
                    prob_mlp = model_mlp.predict(in_mlp)[0][0] if in_mlp is not None else 0.0
                    
                    # Rata-rata (Ensemble)
                    avg_prob = (prob_cnn + prob_mlp) / 2
                    
                    # Visualisasi Hasil
                    col_cnn, col_mlp, col_final = st.columns(3)
                    
                    def get_status_style(prob):
                        label = "TEPAT WAKTU" if prob > 0.5 else "TERLAMBAT"
                        delta_color = "normal" if prob > 0.5 else "inverse"
                        return label, delta_color

                    with col_cnn:
                        lbl, color = get_status_style(prob_cnn)
                        st.metric(label="Model Sequential (CNN)", value=f"{prob_cnn*100:.1f}%", delta=lbl, delta_color=color)
                    
                    with col_mlp:
                        lbl, color = get_status_style(prob_mlp)
                        st.metric(label="Model Tabular (MLP)", value=f"{prob_mlp*100:.1f}%", delta=lbl, delta_color=color)
                    
                    with col_final:
                        # Final Verdict Box
                        verdict = "LULUS TEPAT WAKTU" if avg_prob > 0.5 else "TERLAMBAT LULUS"
                        bg_color = "#d4edda" if avg_prob > 0.5 else "#f8d7da"
                        text_color = "#155724" if avg_prob > 0.5 else "#721c24"
                        
                        st.markdown(f"""
                        <div style="background-color: {bg_color}; color: {text_color}; padding: 10px; border-radius: 5px; text-align: center;">
                            <h4 style="margin:0;">KESIMPULAN</h4>
                            <h3 style="margin:0;">{verdict}</h3>
                            <p style="margin:0;">Confidence: {avg_prob*100:.1f}%</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
        except Exception as e:
            st.error(f"Terjadi Kesalahan Aplikasi: {str(e)}")
