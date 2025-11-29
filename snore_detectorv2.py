#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Snore Detector v2 - Streaming desde Arduino + ML Inference
Optimizado para Raspberry Pi Zero (usa scipy, NO librosa)
"""

import argparse, os, wave, datetime as dt, time, json, tempfile
import numpy as np
from collections import deque
from scipy.signal import stft, get_window
from scipy.fftpack import dct

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter

import serial

HDR0, HDR1 = 0xAA, 0x55

# -------------------- Serial helpers --------------------
def find_sync(port):
    """Busca el header de sincronización 0xAA 0x55"""
    timeout_count = 0
    while timeout_count < 100:
        b = port.read(1)
        if not b:
            timeout_count += 1
            continue
        timeout_count = 0
        if b[0] == HDR0:
            b2 = port.read(1)
            if b2 and b2[0] == HDR1:
                return True
    return False

def read_packet(port):
    """Lee un paquete de audio del Arduino"""
    if not find_sync(port):
        return None
    ln = port.read(2)
    if len(ln) < 2:
        return None
    n = ln[0] | (ln[1] << 8)
    if n == 0 or n > 1024:
        return None
    payload = port.read(n * 2)
    if len(payload) < n * 2:
        return None
    return np.frombuffer(payload, dtype='<i2')

# -------------------- Model helpers --------------------
def load_tflite(path):
    it = Interpreter(model_path=path)
    it.allocate_tensors()
    return it, it.get_input_details()[0], it.get_output_details()[0]

def get_hw(det):
    s = det['shape']
    if len(s) == 4:
        return (int(s[1]), int(s[2])) if s[1] > 1 and s[2] > 1 else (int(s[2]), int(s[3]))
    if len(s) == 2:
        return (int(s[1]), 1)
    return (150, 20)

def q_in(x, det):
    dt_type = det['dtype']
    if dt_type == np.float32:
        return x.astype(np.float32)
    scale, zp = det.get('quantization', (1.0, 0))
    if scale == 0:
        scale = 1.0
    if dt_type == np.int8:
        return np.clip((x / scale + zp), -128, 127).astype(np.int8)
    if dt_type == np.uint8:
        return np.clip((x / scale + zp), 0, 255).astype(np.uint8)
    return x.astype(np.float32)

def dq_out(y, det):
    dt_type = det['dtype']
    if dt_type == np.float32:
        return y.astype(np.float32)
    scale, zp = det.get('quantization', (1.0, 0))
    if scale == 0:
        scale = 1.0
    return scale * (y.astype(np.float32) - zp)

def softmax(v):
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    e = np.exp(v - np.max(v))
    s = e.sum()
    return e / s if s > 0 else np.zeros_like(v)

# -------------------- MFCC CON SCIPY (Igual que train_snore_scipy.py) --------------------
SR = 16000
WIN_SEC = 1.5
N_MFCC = 20
HOP = 160
N_FFT = 512
N_MELS = 40
FMIN, FMAX = 80.0, 6000.0
T_FRAMES = int(WIN_SEC * SR / HOP)  # 150

def hz_to_mel(f):  return 2595.0 * np.log10(1.0 + f / 700.0)
def mel_to_hz(m): return 700.0 * (10.0**(m / 2595.0) - 1.0)

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

def compute_mfcc_window(x_f32, sr, n_fft=512, hop=160, n_mels=40, n_mfcc=20,
                        fmin=80.0, fmax=6000.0, target_frames=150):
    """Calcula MFCCs idéntico al entrenamiento"""
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

    mean = MFCC.mean(axis=0, keepdims=True)
    std = MFCC.std(axis=0, keepdims=True)
    MFCC = (MFCC - mean) / (std + 1e-8)
    return MFCC.astype(np.float32)

# -------------------- Acoustic gates --------------------
def band_energy_ratio(x, sr, f_lo=70.0, f_hi=300.0):
    N = len(x)
    if N == 0:
        return 0.0
    X = np.fft.rfft(x * np.hanning(N))
    ps = (np.abs(X) ** 2)
    freqs = np.fft.rfftfreq(N, 1.0/sr)
    total = ps.sum() + 1e-12
    mask = (freqs >= f_lo) & (freqs <= f_hi)
    band = ps[mask].sum()
    return float(band / total)

def band_energy(x, sr, f_lo, f_hi):
    N = len(x)
    if N == 0:
        return 1e-12
    X = np.fft.rfft(x * np.hanning(N))
    ps = (np.abs(X) ** 2)
    freqs = np.fft.rfftfreq(N, 1.0/sr)
    m = (freqs >= f_lo) & (freqs <= f_hi)
    return float(ps[m].sum() + 1e-12)

# -------------------- Main --------------------
def main():
    ap = argparse.ArgumentParser(description="Detector de ronquidos con streaming de audio")
    ap.add_argument('--model', required=True, help="Ruta al modelo TFLite")
    ap.add_argument('--serial', default='/dev/ttyACM0')
    ap.add_argument('--sr', type=int, default=16000)
    ap.add_argument('--win', type=float, default=1.5)
    ap.add_argument('--hop', type=float, default=0.5)
    
    # AJUSTES PRINCIPALES - con librosa ahora debería funcionar mejor
    ap.add_argument('--threshold', type=float, default=0.75, help='Umbral de detección (subido, librosa deberia dar mejores scores)')
    ap.add_argument('--hyst', type=float, default=0.15)
    ap.add_argument('--avg-k', type=int, default=3, help='Ventanas para promediar')
    ap.add_argument('--cooldown', type=float, default=3.0)
    ap.add_argument('--beep-ms', type=int, default=800)
    ap.add_argument('--pos-index', type=int, default=1)
    
    # Gates para filtrar ruidos que no son ronquidos
    ap.add_argument('--gate-rms', type=float, default=0.01, help='RMS mínimo')
    ap.add_argument('--gate-band', type=float, default=0.10, help='Energía en banda de ronquido 70-300Hz')
    ap.add_argument('--gate-lowmid', type=float, default=1.0, help='Ratio baja/media frecuencia (ronquidos tienen más bajos)')
    ap.add_argument('--min-consec', type=int, default=2, help='Ventanas consecutivas necesarias')
    
    ap.add_argument('--outdir', default='data')
    ap.add_argument('--log-raw', action='store_true')
    ap.add_argument('--debug', action='store_true')
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    raw_dir = os.path.join(args.outdir, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    events_csv = os.path.join(args.outdir, "events.csv")
    if not os.path.exists(events_csv):
        with open(events_csv, 'w', encoding='utf-8') as f:
            f.write("timestamp,score,avg,rms,band,lowmid,threshold\n")

    status_path = os.path.join(args.outdir, "status.json")
    cmd_path = os.path.join(args.outdir, "cmd.json")

    print(f"[INFO] Cargando modelo: {args.model}")
    it, in_det, out_det = load_tflite(args.model)
    inp_idx = it.get_input_details()[0]['index']
    out_idx = it.get_output_details()[0]['index']
    H, W = get_hw(in_det)
    print(f"[INFO] Modelo input: {H}x{W}")
    print(f"[INFO] MFCC: scipy (compatible con RPi Zero)")
    print(f"[INFO] Threshold={args.threshold} | Gates: RMS>{args.gate_rms}, Band>{args.gate_band}, L/M>{args.gate_lowmid}")

    print(f"[INFO] Conectando a {args.serial}...")
    ser = serial.Serial(args.serial, 115200, timeout=1)
    time.sleep(2)
    print("[INFO] Conectado!")

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    wav_path = os.path.join(raw_dir, f"mic_{ts}.wav")
    wf = wave.open(wav_path, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(args.sr)

    win_n = int(args.win * args.sr)
    hop_n = int(args.hop * args.sr)
    circ = deque(maxlen=win_n)
    for _ in range(win_n):
        circ.append(0)
    acc = 0

    scores = deque(maxlen=max(1, args.avg_k))
    state_on = False
    last_event_time = 0.0
    consec = 0

    try:
        print("[INFO] Ejecutando... Ctrl+C para salir")
        print("="*60)
        
        while True:
            s = read_packet(ser)
            if s is None:
                continue

            wf.writeframes(s.tobytes())
            for v in s:
                circ.append(int(v))
            acc += len(s)

            # Comando web
            try:
                if os.path.exists(cmd_path):
                    with open(cmd_path, "r", encoding="utf-8") as f:
                        cmd = json.load(f)
                    os.remove(cmd_path)
                    ms = max(1, min(5000, int(cmd.get("beep_ms", args.beep_ms))))
                    ser.write(b'B')
                    ser.write(bytes([ms & 0xFF, (ms >> 8) & 0xFF]))
            except:
                pass

            while acc >= hop_n:
                acc -= hop_n

                x = np.array(circ, dtype=np.int16).astype(np.float32)
                mx = float(np.max(np.abs(x)))
                if mx > 0:
                    x = x / mx

                rms = float(np.sqrt(np.mean(x**2)))
                band = band_energy_ratio(x, args.sr, 70.0, 300.0)
                e_low = band_energy(x, args.sr, 80.0, 300.0)
                e_mid = band_energy(x, args.sr, 300.0, 2000.0)
                lowmid = float(e_low / e_mid) if e_mid > 0 else 0.0

                # Calcular MFCC con scipy (igual que entrenamiento)
                MFCC = compute_mfcc_window(x, args.sr, n_fft=N_FFT, hop=HOP, n_mels=N_MELS, n_mfcc=N_MFCC,
                                           fmin=FMIN, fmax=FMAX, target_frames=T_FRAMES)
                tin = MFCC[None, :, :, None]
                it.set_tensor(inp_idx, q_in(tin, in_det))
                it.invoke()
                y = dq_out(it.get_tensor(out_idx), out_det)
                
                v = np.asarray(y).squeeze().reshape(-1)
                if len(v) >= 2:
                    if (v.min() < 0.0) or (v.max() > 1.0) or not (0.98 <= float(v.sum()) <= 1.02):
                        v = softmax(v)
                    score = float(v[args.pos_index])
                else:
                    score = float(v[0]) if len(v) > 0 else 0.0

                scores.append(score)
                avg = float(np.mean(scores))

                if args.debug:
                    print(f"[DBG] v={v} score={score:.3f}")

                ok_rms = rms >= args.gate_rms
                ok_band = band >= args.gate_band
                ok_lowmid = lowmid >= args.gate_lowmid
                ok_gates = ok_rms and ok_band and ok_lowmid

                icon = "🔴" if state_on else ("🟡" if avg >= args.threshold * 0.8 else "⚪")
                gates = f"[{'✓' if ok_rms else '✗'}{'✓' if ok_band else '✗'}{'✓' if ok_lowmid else '✗'}]"
                print(f"{icon} score={score:.2f} avg={avg:.2f} rms={rms:.3f} band={band:.2f} L/M={lowmid:.1f} {gates}")

                try:
                    status = {
                        "ts": dt.datetime.now().isoformat(timespec="seconds"),
                        "score": round(score, 4),
                        "avg": round(avg, 4),
                        "rms": round(rms, 4),
                        "band": round(band, 4),
                        "lowmid": round(lowmid, 4),
                        "threshold": args.threshold,
                        "state_on": state_on
                    }
                    with tempfile.NamedTemporaryFile("w", delete=False, dir=args.outdir) as tf:
                        json.dump(status, tf)
                        tmpname = tf.name
                    os.replace(tmpname, status_path)
                except:
                    pass

                if (avg >= args.threshold) and ok_gates:
                    consec += 1
                else:
                    consec = 0

                now = time.time()
                if not state_on:
                    if (consec >= args.min_consec) and (now - last_event_time) >= args.cooldown:
                        state_on = True
                        last_event_time = now
                        print(f"🔔 ¡RONQUIDO! score={score:.2f} avg={avg:.2f}")
                        with open(events_csv, 'a', encoding='utf-8') as f:
                            f.write(f"{dt.datetime.now().isoformat(timespec='seconds')},{score:.4f},{avg:.4f},{rms:.4f},{band:.3f},{lowmid:.2f},{args.threshold}\n")
                        try:
                            ms = max(1, min(5000, args.beep_ms))
                            ser.write(b'B')
                            ser.write(bytes([ms & 0xFF, (ms >> 8) & 0xFF]))
                        except Exception as e:
                            print(f"[WARN] buzzer: {e}")
                else:
                    if avg < (args.threshold - args.hyst):
                        state_on = False
                        consec = 0

    except KeyboardInterrupt:
        print("\n[SALIR]")
    finally:
        try: wf.close()
        except: pass
        try: ser.close()
        except: pass
        print(f"[INFO] Audio: {wav_path}")
        print(f"[INFO] Eventos: {events_csv}")

if __name__ == "__main__":
    main()
