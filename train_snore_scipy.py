#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Entrenamiento TinyML para detección de ronquidos
USA SCIPY para MFCCs - Compatible con Raspberry Pi Zero (sin librosa)

El código de MFCC es IDENTICO al que usa snore_detectorv2.py
"""

import os, glob, wave
import numpy as np
import tensorflow as tf
from scipy.signal import stft, get_window
from scipy.fftpack import dct
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# ============================================================
# CONFIGURACION - IDENTICA A snore_detectorv2.py
# ============================================================
SR = 16000
WIN_SEC = 1.5
N_MFCC = 20
HOP = 160
N_FFT = 512
N_MELS = 40
FMIN, FMAX = 80.0, 6000.0
LABELS = ["background", "snore"]
T_FRAMES = int(WIN_SEC * SR / HOP)  # 150
SEED = 42
BATCH_SIZE = 32
EPOCHS = 100

np.random.seed(SEED)
tf.random.set_seed(SEED)

# ============================================================
# MFCC CON SCIPY - IDENTICO A snore_detectorv2.py
# ============================================================
def hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + f / 700.0)

def mel_to_hz(m):
    return 700.0 * (10.0**(m / 2595.0) - 1.0)

def mel_filterbank(sr, n_fft, n_mels, fmin=80.0, fmax=6000.0):
    n_freqs = n_fft // 2 + 1
    m_min, m_max = hz_to_mel(fmin), hz_to_mel(min(fmax, sr/2.0))
    m_pts = np.linspace(m_min, m_max, n_mels + 2)
    f_pts = mel_to_hz(m_pts)
    bins = np.floor((n_fft + 1) * f_pts / sr).astype(int)
    fb = np.zeros((n_mels, n_freqs), dtype=np.float32)
    for m in range(1, n_mels + 1):
        f0 = int(np.clip(bins[m - 1], 0, n_freqs - 1))
        fm = int(np.clip(bins[m], 0, n_freqs - 1))
        f1 = int(np.clip(bins[m + 1] if m + 1 < len(bins) else fm + 1, 0, n_freqs - 1))
        if fm == f0: fm = min(fm + 1, n_freqs - 1)
        if f1 == fm: f1 = min(f1 + 1, n_freqs - 1)
        if f0 < fm:
            fb[m-1, f0:fm] = (np.arange(f0, fm) - f0) / max(1, fm - f0)
        if fm < f1:
            fb[m-1, fm:f1] = (f1 - np.arange(fm, f1)) / max(1, f1 - fm)
    return fb

def compute_mfcc_scipy(x_f32, sr=SR, n_fft=N_FFT, hop=HOP, n_mels=N_MELS, 
                       n_mfcc=N_MFCC, fmin=FMIN, fmax=FMAX, target_frames=T_FRAMES):
    """
    Calcula MFCCs usando scipy - IDENTICO a snore_detectorv2.py
    """
    # Normalizar
    x_f32 = x_f32 / (np.max(np.abs(x_f32)) + 1e-8)
    
    win = get_window("hann", n_fft, fftbins=True)
    _, _, Z = stft(x_f32, fs=sr, window=win, nperseg=n_fft, noverlap=n_fft - hop,
                   nfft=n_fft, boundary="zeros", padded=True)
    S_pow = (np.abs(Z) ** 2).astype(np.float32)
    fb = mel_filterbank(sr, n_fft, n_mels, fmin=fmin, fmax=fmax)
    S_mel = np.dot(fb, S_pow)
    S_log = np.log(np.maximum(S_mel, 1e-10))
    M = dct(S_log, type=2, axis=0, norm='ortho')
    MFCC = M[:n_mfcc, :].T

    T = MFCC.shape[0]
    if T < target_frames:
        pad = np.zeros((target_frames - T, n_mfcc), dtype=np.float32)
        MFCC = np.vstack([MFCC, pad])
    elif T > target_frames:
        MFCC = MFCC[:target_frames, :]

    # Normalizar por canal
    mean = MFCC.mean(axis=0, keepdims=True)
    std = MFCC.std(axis=0, keepdims=True)
    MFCC = (MFCC - mean) / (std + 1e-8)
    return MFCC.astype(np.float32)

# ============================================================
# AUMENTACION DE DATOS
# ============================================================
def random_gain(y, min_gain=0.7, max_gain=1.3):
    return y * np.random.uniform(min_gain, max_gain)

def add_noise(y, noise_factor=0.002):
    noise = np.random.normal(0, noise_factor, len(y))
    return y + noise

def time_shift(y, sr=SR, max_shift_seconds=0.3):
    shift = int(np.random.uniform(-max_shift_seconds * sr, max_shift_seconds * sr))
    return np.roll(y, shift)

def augment_audio(y):
    y = random_gain(y)
    if np.random.random() > 0.5:
        y = add_noise(y)
    if np.random.random() > 0.5:
        y = time_shift(y)
    return y

# ============================================================
# CARGA DE DATOS
# ============================================================
def load_wav_scipy(path):
    """Carga WAV usando solo scipy/wave (sin librosa)"""
    with wave.open(path, 'rb') as wf:
        sr = wf.getframerate()
        n_frames = wf.getnframes()
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        
        raw = wf.readframes(n_frames)
        
        if sampwidth == 2:
            data = np.frombuffer(raw, dtype=np.int16)
        elif sampwidth == 1:
            data = np.frombuffer(raw, dtype=np.uint8).astype(np.int16) - 128
        elif sampwidth == 4:
            data = np.frombuffer(raw, dtype=np.int32).astype(np.float32)
        else:
            raise ValueError(f"Unsupported sample width: {sampwidth}")
        
        # Convertir a mono si es estéreo
        if n_channels == 2:
            data = data.reshape(-1, 2).mean(axis=1)
        
        # Convertir a float32 normalizado
        data = data.astype(np.float32)
        if sampwidth <= 2:
            data = data / 32768.0
        
        # Resamplear si es necesario
        if sr != SR:
            # Resampleo simple por interpolación
            duration = len(data) / sr
            new_len = int(duration * SR)
            indices = np.linspace(0, len(data) - 1, new_len)
            data = np.interp(indices, np.arange(len(data)), data)
        
        return data.astype(np.float32)

def ensure_len(y, sr=SR, sec=WIN_SEC):
    need = int(sr * sec)
    if len(y) > need:
        start = np.random.randint(0, len(y) - need)
        return y[start:start + need]
    return np.pad(y, (0, max(0, need - len(y))))[:need]

def load_dataset(base, augment=False):
    X, y = [], []
    for li, lbl in enumerate(LABELS):
        paths = glob.glob(os.path.join(base, lbl, "*.wav"))
        print(f"Cargando {lbl}: {len(paths)} archivos")
        
        for p in paths:
            try:
                sig = load_wav_scipy(p)
                sig = ensure_len(sig)
                
                # Muestra original
                mfcc = compute_mfcc_scipy(sig)
                X.append(mfcc[..., None])
                y.append(li)
                
                # Muestras aumentadas (solo para ronquidos)
                if augment and lbl == "snore":
                    for _ in range(2):
                        aug_sig = augment_audio(sig.copy())
                        mfcc_aug = compute_mfcc_scipy(aug_sig)
                        X.append(mfcc_aug[..., None])
                        y.append(li)
                        
            except Exception as e:
                print(f"  [WARN] Error cargando {p}: {e}")
                
    return np.stack(X), np.array(y)

# ============================================================
# MODELO
# ============================================================
def build_model():
    # Tiny DS-CNN optimizado
    inp = tf.keras.Input(shape=(T_FRAMES, N_MFCC, 1))
    
    # First conv block
    x = tf.keras.layers.Conv2D(16, 3, padding='same', use_bias=False)(inp)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    
    # DS Conv blocks
    for filters in [32, 48]:
        x = tf.keras.layers.DepthwiseConv2D(3, padding='same', use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Conv2D(filters, 1, use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
    
    # Global pooling and output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    out = tf.keras.layers.Dense(len(LABELS), activation='softmax')(x)
    
    model = tf.keras.Model(inp, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# ============================================================
# EXPORTAR TFLITE
# ============================================================
def export_tflite(model, X_test, output_dir):
    def representative_dataset():
        for i in np.random.choice(len(X_test), min(100, len(X_test)), False):
            yield [X_test[i:i+1].astype(np.float32)]
    
    # FP32 version
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_fp32 = converter.convert()
    fp32_path = os.path.join(output_dir, "snore_model_fp32.tflite")
    with open(fp32_path, 'wb') as f:
        f.write(tflite_fp32)
    print(f"✓ Modelo FP32: {fp32_path} ({len(tflite_fp32)/1024:.1f} KB)")
    
    # INT8 version
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    converter.representative_dataset = representative_dataset
    
    tflite_int8 = converter.convert()
    int8_path = os.path.join(output_dir, "snore_model_int8.tflite")
    with open(int8_path, 'wb') as f:
        f.write(tflite_int8)
    print(f"✓ Modelo INT8: {int8_path} ({len(tflite_int8)/1024:.1f} KB)")
    
    return fp32_path, int8_path

# ============================================================
# MAIN
# ============================================================
def main():
    base = "data"
    artifacts = "artifacts"
    os.makedirs(artifacts, exist_ok=True)
    
    print("="*60)
    print("ENTRENAMIENTO CON SCIPY (Compatible con Raspberry Pi Zero)")
    print("="*60)
    
    # Cargar dataset
    print("\n1. Cargando y aumentando dataset...")
    X, y = load_dataset(base, augment=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )
    print(f"   Train: {X_train.shape} Test: {X_test.shape}")
    
    # Entrenar modelo
    print("\n2. Entrenando modelo...")
    model = build_model()
    model.summary()
    
    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True
        ),
    ]
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluar
    print("\n3. Evaluando modelo...")
    y_pred = model.predict(X_test).argmax(axis=1)
    report = classification_report(y_test, y_pred, target_names=LABELS, digits=4)
    cm = confusion_matrix(y_test, y_pred)
    print(report)
    print("\nMatriz de confusión:")
    print(cm)
    
    # Guardar reporte
    with open(os.path.join(artifacts, "report.txt"), "w") as f:
        f.write("MODELO ENTRENADO CON SCIPY (Compatible con RPi Zero)\n")
        f.write("="*50 + "\n\n")
        f.write(report + "\n\n")
        f.write("Matriz de confusión:\n")
        f.write(str(cm))
    
    # Guardar modelo y labels
    print("\n4. Guardando artefactos...")
    model.save(os.path.join(artifacts, "snore_model.keras"))
    with open(os.path.join(artifacts, "labels.txt"), "w") as f:
        f.write("\n".join(LABELS))
    
    # Exportar TFLite
    print("\n5. Exportando versiones TFLite...")
    fp32_path, int8_path = export_tflite(model, X_test, artifacts)
    
    print("\n" + "="*60)
    print("¡LISTO! Modelo entrenado con scipy.")
    print("Ahora snore_detectorv2.py usará las mismas features.")
    print("="*60)

if __name__ == "__main__":
    main()
