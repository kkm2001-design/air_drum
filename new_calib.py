"""
Air-Drum with EMG Sensor Fusion (Amplified + Fake 3-step Calibration Version)
- ADC 변화량(힘의 크기)을 증폭
- 3단계 Calibration은 화면에 자막만 표시 (실제 Baseline/Threshold 값은 전혀 바꾸지 않음)
  1) REST         : 힘 빼고 가만히 (Baseline 측정하는 척)
  2) HIT_NORMAL   : 보통 세기로 치기 (Normal Threshold 잡는 척)
  3) HIT_HIGH     : 세게 치기 (High Threshold 잡는 척)
"""

import cv2
import numpy as np
import time
import os
import serial
import threading
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque

# ===== 0) 설정 변수 (사용자 환경에 맞게 수정!) =====
COM_PORT = "COM9"        # <--- 장치관리자 확인 필수
BAUD_RATE = 115200

countour_size_a = 100
countour_size_b = 80

# ---------------- EMG 관련 파라미터 ----------------
# Threshold는 고정값으로 사용 (캘리브레이션이 실제로는 건드리지 않음)
EMG_THRESHOLD_HIGH = 2500   # 높은음 (강한 타격)
EMG_THRESHOLD       = 1400   # 일반음 (약한 타격)

# 시간 파라미터
EMG_WINDOW   = 0.40          # 근육 신호 유지 시간(초)
EMG_BASELINE = 6800          # 평소 baseline
EMG_GAIN     = 2.3           # 변화량 Gain (작은 힘도 크게 인식)

# ---------------- Fake Calibration 관련 설정 ----------------
CALIB_REST_SEC       = 3.0    # 1단계: 힘 빼고 있는 시간
CALIB_HIT_NORMAL_SEC = 6.0    # 2단계: 보통 세기로 치기 시간
CALIB_HIT_HIGH_SEC   = 6.0    # 3단계: 세게 치기 시간

# MODE는 화면 표시/그래프 표시용으로만 사용 (EMG 연산에는 영향 X)
MODE = "CALIB_REST"          # 시작할 때 1단계부터
calib_start_time = None      # 각 단계 시작 시각

# ---------------- 그래프 관련 ----------------
MAX_POINTS = 200
DATA_L = deque([EMG_BASELINE] * MAX_POINTS, maxlen=MAX_POINTS)
DATA_R = deque([EMG_BASELINE] * MAX_POINTS, maxlen=MAX_POINTS)

# 창 이름
MAIN_WIN = "AirDrum (Vision + EMG)"
MASK_WIN = "Mask View"
CTRL_WIN_A = "Settings Left (A)"
CTRL_WIN_B = "Settings Right (B)"


# ===== 1) 오디오 초기화 =====
use_audio = True
try:
    import pygame
    pygame.mixer.init(frequency=44100, channels=1, buffer=512)
except Exception as e:
    print("[!] pygame init failed:", e)
    use_audio = False

BASE = os.path.dirname(__file__)
SAVE_DIR = os.path.join(BASE, "sounds")

SOUNDS = {
    "TL": os.path.join(SAVE_DIR, "hihat.wav"),
    "TR": os.path.join(SAVE_DIR, "hihat.wav"),
    "BL": os.path.join(SAVE_DIR, "drum.wav"),
    "BR": os.path.join(SAVE_DIR, "drum.wav"),
}

if use_audio:
    WAVS = {}
    for k, v in SOUNDS.items():
        if os.path.exists(v):
            WAVS[k] = pygame.mixer.Sound(v)
        else:
            print(f"[!] Sound file missing: {v}")
            WAVS[k] = None

SOUNDS_HIGHER = {
    "TL": os.path.join(SAVE_DIR, "hihat_higher.wav"),
    "TR": os.path.join(SAVE_DIR, "hihat_higher.wav"),
    "BL": os.path.join(SAVE_DIR, "drum_higher.wav"),
    "BR": os.path.join(SAVE_DIR, "drum_higher.wav"),
}

if use_audio:
    WAVS_HIGH = {}
    for k, v in SOUNDS_HIGHER.items():
        if os.path.exists(v):
            WAVS_HIGH[k] = pygame.mixer.Sound(v)
        else:
            print(f"[!] Sound file missing: {v}")
            WAVS_HIGH[k] = None


def play(pos, intensity="normal"):
    if use_audio and intensity == "normal" and pos in WAVS and WAVS[pos]:
        WAVS[pos].play()
    elif use_audio and intensity == "high" and pos in WAVS_HIGH and WAVS_HIGH[pos]:
        WAVS_HIGH[pos].play()
    else:
        pass


# ===== 2) 시리얼 통신 & EMG 데이터 처리 (스레드) =====
ser = None
emg_L_raw = 0
emg_R_raw = 0
emg_L_val = 0   # 게인이 적용된 힘의 크기
emg_R_val = 0
emg_L_for_plot = EMG_BASELINE
emg_R_for_plot = EMG_BASELINE

history_y_A = deque([0] * 3, maxlen=3)
history_y_B = deque([0] * 3, maxlen=3)

last_emg_active_L = 0
last_emg_active_R = 0

EMG_BUFFER_L = 0   # 0: normal, 1: high
EMG_BUFFER_R = 0


def emg_serial_thread():
    global ser, emg_L_raw, emg_R_raw, emg_L_val, emg_R_val, emg_L_for_plot, emg_R_for_plot
    global last_emg_active_L, last_emg_active_R
    global EMG_BASELINE, EMG_THRESHOLD, EMG_THRESHOLD_HIGH, EMG_GAIN
    global EMG_BUFFER_L, EMG_BUFFER_R

    last_high_active_L = 0
    last_high_active_R = 0

    try:
        ser = serial.Serial(COM_PORT, BAUD_RATE, timeout=0.1)
        print(f"[i] Serial connected to {COM_PORT}")
    except Exception as e:
        print(f"[!] Serial Error: {e}")
        return

    buffer = b""
    while True:
        try:
            if ser.in_waiting > 0:
                chunk = ser.read(ser.in_waiting)
                buffer += chunk

                while len(buffer) >= 7:
                    if buffer[0] != 0x81:
                        buffer = buffer[1:]
                        continue

                    packet = buffer[:7]
                    buffer = buffer[7:]

                    raw_L = (packet[1] << 7) | packet[2]
                    raw_R = (packet[3] << 7) | packet[4]

                    emg_L_raw = raw_L
                    emg_R_raw = raw_R

                    # Plot용 값 (Baseline 기준으로 Gain 적용한 형태)
                    emg_L_for_plot = int((raw_L - EMG_BASELINE) * EMG_GAIN + EMG_BASELINE)
                    emg_R_for_plot = int((raw_R - EMG_BASELINE) * EMG_GAIN + EMG_BASELINE)

                    DATA_L.append(emg_L_for_plot)
                    DATA_R.append(emg_R_for_plot)

                    # 힘의 크기 (Baseline 대비 변화량)
                    val_L = int(abs(raw_L - EMG_BASELINE) * EMG_GAIN)
                    val_R = int(abs(raw_R - EMG_BASELINE) * EMG_GAIN)

                    emg_L_val = val_L
                    emg_R_val = val_R

                    now = time.time()

                    # ==== 여기서는 캘리브레이션 안 함 ====
                    # MODE와 상관없이 항상 고정 Threshold로 EMG_BUFFER만 계산

                    # --- 왼손 (L) ---
                    if val_L > EMG_THRESHOLD_HIGH:
                        last_emg_active_L = now
                        last_high_active_L = now
                        EMG_BUFFER_L = 1
                    elif val_L > EMG_THRESHOLD:
                        last_emg_active_L = now
                        if (now - last_high_active_L) < EMG_WINDOW:
                            EMG_BUFFER_L = 1  # 최근 HIGH가 있으면 유지
                        else:
                            EMG_BUFFER_L = 0  # 일반 수축

                    # --- 오른손 (R) ---
                    if val_R > EMG_THRESHOLD_HIGH:
                        last_emg_active_R = now
                        last_high_active_R = now
                        EMG_BUFFER_R = 1
                    elif val_R > EMG_THRESHOLD:
                        last_emg_active_R = now
                        if (now - last_high_active_R) < EMG_WINDOW:
                            EMG_BUFFER_R = 1
                        else:
                            EMG_BUFFER_R = 0

        except Exception as e:
            print("Serial Read Error:", e)
            break


# EMG 데이터 로깅 및 플롯 설정 (그래프는 Raw값 + 고정 Threshold를 보여줌)
def emg_plot_thread():
    global MODE

    fig, ax = plt.subplots(2, 1, figsize=(10, 6))
    fig.suptitle('EMG ADC Value Live Plot', fontsize=16)

    X = np.arange(MAX_POINTS)
    Y_RANGE = [2000, 12000]

    # Plot 1: L
    ax[0].set_title("Left Hand EMG (Raw)", fontsize=10)
    line_L, = ax[0].plot(X, DATA_L, label='ADC Raw L')
    ax[0].set_ylim(Y_RANGE)
    ax[0].set_ylabel('ADC Value')
    ax[0].grid(True, linestyle='--', alpha=0.6)

    # Plot 2: R
    ax[1].set_title("Right Hand EMG (Raw)", fontsize=10)
    line_R, = ax[1].plot(X, DATA_R, label='ADC Raw R')
    ax[1].set_ylim(Y_RANGE)
    ax[1].set_ylabel('ADC Value')
    ax[1].set_xlabel('Time (Samples)')
    ax[1].grid(True, linestyle='--', alpha=0.6)

    # Baseline / Threshold 선
    th_base_L, = ax[0].plot(X, [EMG_BASELINE] * MAX_POINTS, linestyle='-', label='Baseline')
    th_base_R, = ax[1].plot(X, [EMG_BASELINE] * MAX_POINTS, linestyle='-', label='Baseline')

    th_normal_L, = ax[0].plot(X, [EMG_BASELINE + EMG_THRESHOLD] * MAX_POINTS,
                              linestyle='--', label='Normal Th')
    th_high_L,   = ax[0].plot(X, [EMG_BASELINE + EMG_THRESHOLD_HIGH] * MAX_POINTS,
                              linestyle='--', label='High Th')

    th_normal_R, = ax[1].plot(X, [EMG_BASELINE + EMG_THRESHOLD] * MAX_POINTS,
                              linestyle='--', label='Normal Th')
    th_high_R,   = ax[1].plot(X, [EMG_BASELINE + EMG_THRESHOLD_HIGH] * MAX_POINTS,
                              linestyle='--', label='High Th')

    ax[0].legend(loc='upper right', fontsize='small')

    def update_plot(frame):
        # 실시간 데이터 반영
        line_L.set_ydata(DATA_L)
        line_R.set_ydata(DATA_R)

        # Baseline은 항상 표시
        th_base_L.set_ydata([EMG_BASELINE] * MAX_POINTS)
        th_base_R.set_ydata([EMG_BASELINE] * MAX_POINTS)

        # Calibration 중에는 Threshold 선 숨기기 (NaN으로 세팅)
        if MODE == "RUN":
            th_normal_L.set_ydata([EMG_BASELINE + EMG_THRESHOLD] * MAX_POINTS)
            th_high_L.set_ydata([EMG_BASELINE + EMG_THRESHOLD_HIGH] * MAX_POINTS)
            th_normal_R.set_ydata([EMG_BASELINE + EMG_THRESHOLD] * MAX_POINTS)
            th_high_R.set_ydata([EMG_BASELINE + EMG_THRESHOLD_HIGH] * MAX_POINTS)
        else:
            nan_arr = [np.nan] * MAX_POINTS
            th_normal_L.set_ydata(nan_arr)
            th_high_L.set_ydata(nan_arr)
            th_normal_R.set_ydata(nan_arr)
            th_high_R.set_ydata(nan_arr)

        return (line_L, line_R,
                th_base_L, th_base_R,
                th_normal_L, th_high_L, th_normal_R, th_high_R)

    ani = FuncAnimation(fig, update_plot, interval=10, blit=False)
    plt.show()


t_serial = threading.Thread(target=emg_serial_thread, daemon=True)
t_serial.start()
t_plot = threading.Thread(target=emg_plot_thread, daemon=True)
t_plot.start()

# ===== 3) OpenCV 영상 처리 설정 =====
cap = cv2.VideoCapture(0)
cv2.namedWindow(CTRL_WIN_A)
cv2.namedWindow(CTRL_WIN_B)
cv2.resizeWindow(CTRL_WIN_A, 400, 300)
cv2.resizeWindow(CTRL_WIN_B, 400, 300)

def nothing(x):
    pass

# --- 창 A: 왼손(A) --- 초록색
cv2.createTrackbar("LH_A", CTRL_WIN_A, 40, 179, nothing)
cv2.createTrackbar("LS_A", CTRL_WIN_A, 100, 255, nothing)
cv2.createTrackbar("LV_A", CTRL_WIN_A, 100, 255, nothing)
cv2.createTrackbar("UH_A", CTRL_WIN_A, 80, 179, nothing)
cv2.createTrackbar("US_A", CTRL_WIN_A, 255, 255, nothing)
cv2.createTrackbar("UV_A", CTRL_WIN_A, 255, 255, nothing)
cv2.createTrackbar("VEL_TH", CTRL_WIN_A, 15, 100, nothing)

# --- 창 B: 오른손(B) --- 연보라색
cv2.createTrackbar("LH_B", CTRL_WIN_B, 120, 179, nothing)
cv2.createTrackbar("LS_B", CTRL_WIN_B, 40, 255, nothing)
cv2.createTrackbar("LV_B", CTRL_WIN_B, 50, 255, nothing)
cv2.createTrackbar("UH_B", CTRL_WIN_B, 160, 179, nothing)
cv2.createTrackbar("US_B", CTRL_WIN_B, 255, 255, nothing)
cv2.createTrackbar("UV_B", CTRL_WIN_B, 255, 255, nothing)

prev_y_A, prev_y_B = 0, 0
ema_y_A, ema_y_B = 0, 0
EMA_ALPHA = 0.4

last_hit_A = {"TL": 0, "TR": 0, "BL": 0, "BR": 0}
last_hit_B = {"TL": 0, "TR": 0, "BL": 0, "BR": 0}
COOLDOWN_MS = 200   # 같은 패드 최소 간격 200ms


def quadrant(x, y, w, h):
    if y < h // 2:
        return "TL" if x < w // 2 else "TR"
    else:
        return "BL" if x < w // 2 else "BR"


# ===== Fake Calibration 타이머 초기화 =====
calib_start_time = time.time()  # 스크립트 시작 시점부터 CALIB_REST 진행


# ===== 4) 메인 루프 =====
while True:
    print(f"L raw: {emg_L_raw} | R raw: {emg_R_raw} | L val: {emg_L_val} | R val: {emg_R_val}")
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    cur_time = time.time()
    now_ms = int(cur_time * 1000)

    # ===== Fake Calibration 단계 전환 (타이머로만, 실제 계산 없음) =====
    elapsed = cur_time - calib_start_time
    if MODE == "CALIB_REST" and elapsed >= CALIB_REST_SEC:
        MODE = "CALIB_HIT_NORMAL"
        calib_start_time = cur_time
        print("[CALIB-FAKE] 단계 1/3 완료 → 2/3 (HIT_NORMAL)")
    elif MODE == "CALIB_HIT_NORMAL" and elapsed >= CALIB_HIT_NORMAL_SEC:
        MODE = "CALIB_HIT_HIGH"
        calib_start_time = cur_time
        print("[CALIB-FAKE] 단계 2/3 완료 → 3/3 (HIT_HIGH)")
    elif MODE == "CALIB_HIT_HIGH" and elapsed >= CALIB_HIT_HIGH_SEC:
        MODE = "RUN"
        print("[CALIB-FAKE] 단계 3/3 완료 → RUN 모드 (실제 Threshold는 그대로)")

    # HSV Threshold 읽기
    l_h_a = cv2.getTrackbarPos("LH_A", CTRL_WIN_A)
    l_s_a = cv2.getTrackbarPos("LS_A", CTRL_WIN_A)
    l_v_a = cv2.getTrackbarPos("LV_A", CTRL_WIN_A)
    u_h_a = cv2.getTrackbarPos("UH_A", CTRL_WIN_A)
    u_s_a = cv2.getTrackbarPos("US_A", CTRL_WIN_A)
    u_v_a = cv2.getTrackbarPos("UV_A", CTRL_WIN_A)

    l_h_b = cv2.getTrackbarPos("LH_B", CTRL_WIN_B)
    l_s_b = cv2.getTrackbarPos("LS_B", CTRL_WIN_B)
    l_v_b = cv2.getTrackbarPos("LV_B", CTRL_WIN_B)
    u_h_b = cv2.getTrackbarPos("UH_B", CTRL_WIN_B)
    u_s_b = cv2.getTrackbarPos("US_B", CTRL_WIN_B)
    u_v_b = cv2.getTrackbarPos("UV_B", CTRL_WIN_B)

    VEL_TH = cv2.getTrackbarPos("VEL_TH", CTRL_WIN_A)

    lower_A = np.array([l_h_a, l_s_a, l_v_a])
    upper_A = np.array([u_h_a, u_s_a, u_v_a])
    mask_A = cv2.inRange(hsv, lower_A, upper_A)

    lower_B = np.array([l_h_b, l_s_b, l_v_b])
    upper_B = np.array([u_h_b, u_s_b, u_v_b])
    mask_B = cv2.inRange(hsv, lower_B, upper_B)  # <- 타이포 조심: upper_B

    # 타이포 수정
    mask_B = cv2.inRange(hsv, lower_B, upper_B)

    kernel = np.ones((5, 5), np.uint8)
    mask_A = cv2.dilate(mask_A, kernel, iterations=2)
    mask_B = cv2.dilate(mask_B, kernel, iterations=2)

    mask_disp = cv2.bitwise_or(mask_A, mask_B)
    cv2.imshow(MASK_WIN, mask_disp)

    # --- Calibration 상태 안내 자막 ---
    if MODE == "CALIB_REST":
        cv2.putText(frame, "CALIB 1/3: Relax arm ~3s",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    elif MODE == "CALIB_HIT_NORMAL":
        cv2.putText(frame, "CALIB 2/3: Hit NORMAL ~6s",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    elif MODE == "CALIB_HIT_HIGH":
        cv2.putText(frame, "CALIB 3/3: Hit HIGH ~6s",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    elif MODE == "RUN":
        cv2.putText(frame, "MODE: RUN (Fixed Threshold)",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # --- Tip A (왼손) ---
    cnts_A, _ = cv2.findContours(mask_A, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts_A:
        c = max(cnts_A, key=cv2.contourArea)
        if cv2.contourArea(c) > countour_size_a:
            x, y, cw, ch = cv2.boundingRect(c)
            cxA, cyA = x + cw // 2, y + ch // 2
            cv2.circle(frame, (cxA, cyA), 10, (255, 0, 0), -1)

            history_y_A.append(cyA)
            ema_y_A = (EMA_ALPHA * cyA) + ((1 - EMA_ALPHA) * ema_y_A)

            vel_y_A = 0
            if history_y_A[0] != 0:
                vel_y_A = (history_y_A[-1] - history_y_A[0])

            time_diff_L = cur_time - last_emg_active_L

            # 캘리브레이션 중에도 실제로는 HIT 동작 가능하게 둘지,
            # 막고 싶으면 MODE == "RUN" 조건을 유지
            if MODE == "RUN" and vel_y_A > VEL_TH:
                if time_diff_L < EMG_WINDOW:
                    whichA = quadrant(cxA, int(ema_y_A), w, h)
                    if now_ms - last_hit_A[whichA] > COOLDOWN_MS:
                        if EMG_BUFFER_L == 0:
                            play(whichA, intensity='normal')
                            print(f"[A] HIT! Vel={vel_y_A:.1f}, EMG(Amp)={emg_L_val}")
                        elif EMG_BUFFER_L == 1:
                            play(whichA, intensity='high')
                            print(f"[A] HIT HIGH! Vel={vel_y_A:.1f}, EMG(Amp)={emg_L_val}")

                        last_hit_A[whichA] = now_ms
                        hit_text = f"HIT {whichA}" + (" (HIGH)" if EMG_BUFFER_L == 1 else "")
                        cv2.putText(frame, hit_text, (cxA - 40, cyA - 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
                else:
                    cv2.putText(frame, "No Muscle", (cxA, cyA - 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # --- Tip B (오른손) ---
    cnts_B, _ = cv2.findContours(mask_B, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts_B:
        c = max(cnts_B, key=cv2.contourArea)
        if cv2.contourArea(c) > countour_size_b:
            x, y, cw, ch = cv2.boundingRect(c)
            cxB, cyB = x + cw // 2, y + ch // 2
            cv2.circle(frame, (cxB, cyB), 10, (0, 0, 255), -1)

            history_y_B.append(cyB)
            ema_y_B = (EMA_ALPHA * cyB) + ((1 - EMA_ALPHA) * ema_y_B)

            vel_y_B = 0
            if history_y_B[0] != 0:
                vel_y_B = (history_y_B[-1] - history_y_B[0])

            time_diff_R = cur_time - last_emg_active_R

            if MODE == "RUN" and vel_y_B > VEL_TH:
                if time_diff_R < EMG_WINDOW:
                    whichB = quadrant(cxB, int(ema_y_B), w, h)
                    if now_ms - last_hit_B[whichB] > COOLDOWN_MS:
                        if EMG_BUFFER_R == 0:
                            play(whichB, intensity='normal')
                            print(f"[B] HIT! Vel={vel_y_B:.1f}, EMG(Amp)={emg_R_val}")
                        elif EMG_BUFFER_R == 1:
                            play(whichB, intensity='high')
                            print(f"[B] HIT HIGH! Vel={vel_y_B:.1f}, EMG(Amp)={emg_R_val}")

                        last_hit_B[whichB] = now_ms
                        hit_text = f"HIT {whichB}" + (" (HIGH)" if EMG_BUFFER_R == 1 else "")
                        cv2.putText(frame, hit_text, (cxB - 40, cyB - 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
                else:
                    cv2.putText(frame, "No Muscle", (cxB, cyB - 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # 4분할 선
    cv2.line(frame, (w // 2, 0), (w // 2, h), (0, 0, 0), 1)
    cv2.line(frame, (0, h // 2), (w, h // 2), (0, 0, 0), 1)

    cv2.imshow(MAIN_WIN, frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        break

cap.release()
cv2.destroyAllWindows()
if ser:
    ser.close()
plt.close('all')
