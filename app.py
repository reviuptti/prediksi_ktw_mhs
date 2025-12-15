import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import requests
import io
from tensorflow.keras.utils import to_categorical

# ==========================================
# 1. KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(
    page_title="Sistem Prediksi Kelulusan",
    page_icon="🎓",
    layout="wide"
)

st.markdown("""
    <style>
    .info-box {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 5px;
        padding: 15px;
        margin-bottom: 15px;
    }
    .info-label {
        font-weight: bold;
        color: #495057;
        font-size: 0.9rem;
    }
    .info-value {
        color: #212529;
        font-size: 1rem;
        margin-bottom: 8px;
    }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. LOAD ASSETS
# ==========================================

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

@st.cache_data(ttl=3600)
def load_data_from_github():
    try:
        url = st.secrets["github"]["csv_url"]
        token = st.secrets["github"]["token"]
        headers = {"Authorization": f"token {token}"}
        
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            # Baca CSV dengan tipe data object dulu agar tidak error saat parsing awal
            df = pd.read_csv(io.StringIO(response.content.decode('utf-8')), low_memory=False, dtype=str)
            return df
        else:
            st.error(f"Gagal ambil data. Status: {response.status_code}")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Error Koneksi Data: {e}")
        return pd.DataFrame()

# ==========================================
# 3. HELPER FUNCTIONS (FIXED)
# ==========================================

def clean_and_convert_numeric(df, cols):
    """
    Fungsi pembantu untuk memaksa kolom menjadi angka float.
    Nilai error/string akan diubah jadi 0.0
    """
    subset = df[cols].copy()
    # 1. Ganti koma dengan titik (jika format Indonesia)
    subset = subset.apply(lambda x: x.str.replace(',', '.', regex=False) if x.dtype == "object" else x)
    # 2. Paksa ke numeric, error jadi NaN
    subset = subset.apply(pd.to_numeric, errors='coerce')
    # 3. Isi NaN dengan 0.0
    return subset.fillna(0.0)

def prepare_cnn_input(df_row):
    df = df_row.copy()
    
    # --- 1. STATUS AKADEMIK (Tetap String -> Kategori) ---
    status_map = {'A': 0, 'C': 1, 'N': 2, 'K': 3,'0': 4}
    NUM_CLASSES_STATUS = 5
    
    status_matrix = []
    for col in CNN_CONFIG['status_akademik_cols']:
        if col not in df.columns: df[col] = '0'
        # Pastikan string bersih
        val = str(df[col].iloc[0]).upper().strip()
        val = val if val not in ['NONE', 'NAN', ''] else '0'
        mapped_val = status_map.get(val, 3)
        status_matrix.append(mapped_val)
    
    status_matrix = np.array([status_matrix]) 
    X_status_seq = to_categorical(status_matrix, num_classes=NUM_CLASSES_STATUS)

    # --- 2. DATA NUMERIK (IPK, IPS, UKT, DLL) ---
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
        # [SOLUSI UTAMA] Bersihkan data sebelum masuk Scaler
        cleaned_data = clean_and_convert_numeric(df, cols)
        vals = cleaned_data.values
        
        # Transform
        scaled_vals = scaler_seq[key].transform(vals)
        data_arrays.append(scaled_vals)
    
    X_numeric_seq = np.dstack(data_arrays).astype('float32')
    X_seq_final = np.concatenate([X_numeric_seq, X_status_seq], axis=2)

    # --- 3. DATA STATIS (UMUR, DLL) ---
    # Bersihkan fitur statis numerik
    for c in CNN_CONFIG['num_features']:
        if c not in df.columns: df[c] = 0
    
    # Bersihkan kolom numerik statis di DF utama sebelum transform
    df_cleaned = df.copy()
    df_cleaned[CNN_CONFIG['num_features']] = clean_and_convert_numeric(df, CNN_CONFIG['num_features'])
        
    X_static = prep_static.transform(df_cleaned).astype('float32')
    return [X_seq_final, X_static]

def prepare_mlp_input(df_row):
    try:
        # Bersihkan data numerik sebelum masuk pipeline MLP juga
        df_clean = df_row.copy()
        
        # List semua kolom numerik yang mungkin dipakai MLP
        all_numeric_cols = CNN_CONFIG['num_features'] + \
                           CNN_CONFIG['ipk_cols'] + CNN_CONFIG['ips_cols'] + \
                           CNN_CONFIG['sks_smt_cols'] + CNN_CONFIG['sks_total_smt_cols']
        
        # Bersihkan
        for col in all_numeric_cols:
            if col in df_clean.columns:
                 # Logic pembersihan manual per kolom
                 val = str(df_clean[col].iloc[0])
                 try:
                     val = float(val)
                 except:
                     val = 0.0
                 df_clean[col] = val
                 
        return prep_pipeline.transform(df_clean)
    except Exception as e:
        # Silent fail agar aplikasi tidak crash total, kembalikan None
        print(f"MLP Error: {e}")
        return None

# ==========================================
# 4. UI UTAMA
# ==========================================

st.title("🎓 Sistem Prediksi Kelulusan Mahasiswa")
st.markdown("Dashboard Prediksi Kelulusan (Deep Learning - Hybrid Model)")

# --- A. LOAD DATA ---
with st.spinner("Sedang memuat data dari Repository..."):
    df_all = load_data_from_github()

if df_all.empty:
    st.stop()

# --- B. INPUT FRAME ---
with st.container():
    c1, c2 = st.columns([3, 1])
    with c1:
        if 'nim' in df_all.columns:
            # Pastikan NIM string bersih
            nims = df_all['nim'].astype(str).str.split('.').str[0].unique().tolist()
            selected_nim = st.selectbox("Cari NIM:", options=nims, index=None, placeholder="Ketik atau pilih NIM...")
        else:
            st.error("Kolom 'nim' tidak ditemukan di CSV.")
            st.stop()
            
    with c2:
        st.write("") 
        st.write("")
        btn_predict = st.button("🔍 ANALISA & PREDIKSI", type="primary", use_container_width=True)

# --- C. PROSES & HASIL ---
if btn_predict and selected_nim:
    # Filter Data
    # Normalisasi kolom NIM di dataframe agar cocok dengan input
    df_all['nim_clean'] = df_all['nim'].astype(str).str.split('.').str[0]
    df = df_all[df_all['nim_clean'] == selected_nim]
    
    if df.empty:
        st.error("Data NIM tidak ditemukan.")
    else:
        row = df.iloc[0] 
        
        # --- D. DETAIL DATA FRAME ---
        st.divider()
        st.subheader("📋 Parameter Fitur Mahasiswa")
        
        def get_val(col): return row.get(col, '-')
        
        with st.container(border=True):
            # Layout Grid
            c1, c2, c3 = st.columns(3)
            c1.markdown(f"<div class='info-label'>Fakultas</div><div class='info-value'>{get_val('kdfakultas')}</div>", unsafe_allow_html=True)
            c2.markdown(f"<div class='info-label'>Prodi/Strata</div><div class='info-value'>{get_val('kdstratapendidikan')}</div>", unsafe_allow_html=True)
            c3.markdown(f"<div class='info-label'>Gender</div><div class='info-value'>{get_val('kdjeniskelamin')}</div>", unsafe_allow_html=True)
            
            c1, c2, c3 = st.columns(3)
            c1.markdown(f"<div class='info-label'>Umur Masuk</div><div class='info-value'>{get_val('umur_saat_masuk')}</div>", unsafe_allow_html=True)
            c2.markdown(f"<div class='info-label'>SKS Tempuh</div><div class='info-value'>{get_val('total_sks_saat_ini')}</div>", unsafe_allow_html=True)
            c3.markdown(f"<div class='info-label'>IPK Akhir</div><div class='info-value'>{get_val('ipk_saat_ini')}</div>", unsafe_allow_html=True)
            
            c1, c2, c3 = st.columns(3)
            c1.markdown(f"<div class='info-label'>Status Saat Ini</div><div class='info-value'>{get_val('real_status_akademik_terakhir')}</div>", unsafe_allow_html=True)
            c2.markdown(f"<div class='info-label'>Lama Studi (Smt)</div><div class='info-value'>{get_val('lama_studi_semester')}</div>", unsafe_allow_html=True)
            c3.markdown(f"<div class='info-label'>Rata-rata SKS</div><div class='info-value'>{get_val('rata_rata_sks')}</div>", unsafe_allow_html=True)
            
            c1, c2, c3 = st.columns(3)
            c1.markdown(f"<div class='info-label'>Rata-rata IPS</div><div class='info-value'>{get_val('rata_rata_ips')}</div>", unsafe_allow_html=True)
            c2.markdown(f"<div class='info-label'>Rata-rata IPK</div><div class='info-value'>{get_val('rata_rata_ipk')}</div>", unsafe_allow_html=True)
            c3.markdown(f"<div class='info-label'>Tren IPK</div><div class='info-value'>{get_val('tren_ipk')}</div>", unsafe_allow_html=True)
            
            c1, c2, c3 = st.columns(3)
            c1.markdown(f"<div class='info-label'>StdDev IPS</div><div class='info-value'>{get_val('stddev_ips')}</div>", unsafe_allow_html=True)
            c2.markdown(f"<div class='info-label'>Jumlah Smt Gagal</div><div class='info-value'>{get_val('jumlah_smt_gagal')}</div>", unsafe_allow_html=True)
            c3.write("")

        # --- E. TABEL SEKUENSIAL ---
        st.subheader("📊 Riwayat Akademik (Semester 1-8)")
        
        history_data = []
        for i in range(1, 9):
            # Ambil data mentah
            ukt_val = row.get(f'nominal_ukt_{i}', 0)
            bea_val = row.get(f'status_beasiswa_{i}', 0)
            raw_status = row.get(f'kdstatusakademik_{i}', '-')
            
            # Format Angka untuk Tampilan
            try:
                ukt_fmt = f"{float(str(ukt_val).replace(',','.')):,.0f}"
            except:
                ukt_fmt = "0"
                
            try:
                bea_fmt = "Ya" if float(str(bea_val)) == 1 else "Tidak"
            except:
                bea_fmt = "Tidak"

            history_data.append({
                "Smt": f"Smt {i}",
                "SKS Smt": row.get(f'sks_smt_{i}', 0),
                "Tot SKS": row.get(f'sks_total_smt_{i}', 0),
                "IPS": row.get(f'ips_{i}', 0),
                "IPK": row.get(f'ipk_{i}', 0),
                "Bayar UKT": ukt_fmt,
                "Beasiswa": bea_fmt,
                "Status Akademik": str(raw_status) 
            })
        
        st.dataframe(pd.DataFrame(history_data), use_container_width=True, hide_index=True)

        # --- F. HASIL PREDIKSI ---
        st.divider()
        st.subheader("🤖 Hasil Prediksi AI")
        
        try:
            # 1. CNN
            in_cnn = prepare_cnn_input(df)
            prob_cnn = model_cnn.predict(in_cnn)[0][0]
            
            # 2. MLP
            in_mlp = prepare_mlp_input(df)
            if in_mlp is not None:
                prob_mlp = model_mlp.predict(in_mlp)[0][0]
            else:
                prob_mlp = prob_cnn # Fallback jika MLP error
            
            # 3. Average
            avg_prob = (prob_cnn + prob_mlp) / 2
            
            # Visualisasi
            c_cnn, c_mlp, c_final = st.columns(3)
            
            def get_style(prob):
                txt = "TEPAT WAKTU" if prob > 0.5 else "TERLAMBAT"
                col = "normal" if prob > 0.5 else "inverse"
                return txt, col
            
            with c_cnn:
                lbl, col = get_style(prob_cnn)
                st.metric("Model Hybrid (CNN)", f"{prob_cnn*100:.1f}%", lbl, delta_color=col)
                
            with c_mlp:
                lbl, col = get_style(prob_mlp)
                st.metric("Model Baseline (MLP)", f"{prob_mlp*100:.1f}%", lbl, delta_color=col)
            
            with c_final:
                verdict = "LULUS TEPAT WAKTU" if avg_prob > 0.5 else "TERLAMBAT LULUS"
                bg = "#d4edda" if avg_prob > 0.5 else "#f8d7da"
                fg = "#155724" if avg_prob > 0.5 else "#721c24"
                
                st.markdown(f"""
                <div style="background-color: {bg}; color: {fg}; padding: 15px; border-radius: 8px; text-align: center; border: 1px solid {fg};">
                    <h5 style="margin:0;">KESIMPULAN SISTEM</h5>
                    <h3 style="margin:5px 0;">{verdict}</h3>
                    <p style="margin:0; font-weight:bold;">Confidence: {avg_prob*100:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
                
        except Exception as e:
            st.error(f"Terjadi kesalahan saat prediksi: {e}")
