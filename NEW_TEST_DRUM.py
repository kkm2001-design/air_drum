"""
Air-Drum with EMG Sensor Fusion (Amplified Version)
- ADC 변화량(힘의 크기)을 1.5배 증폭하여 작은 힘으로도 인식되도록 수정
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
COM_PORT = "COM12"        # <--- 장치관리자 확인 필수
BAUD_RATE = 115200

countour_size_a = 100
countour_size_b = 80




# 임계값 (GAIN이 적용된 후의 값과 비교하게 됩니다)
EMG_THRESHOLD_HIGH = 1000 # 높은음 (강한 타격)
EMG_THRESHOLD = 500      # 일반음 (약한 타격)
EMG_WINDOW = 0.6          # 근육 신호 유지 시간
EMG_BASELINE = 6100      # 평소 가만히 있을 때 값 (변경 X)
EMG_GAIN = 1.5            # ★ 변화량을 1.5배 뻥튀기 (작은 힘도 크게 인식토록)

# 그래프 설정
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
    "TR": os.path.join(SAVE_DIR, "snare.wav"),
    "BL": os.path.join(SAVE_DIR, "tom.wav"),
    "BR": os.path.join(SAVE_DIR, "kick.wav"),
}

if use_audio:
    WAVS = {}
    for k, v in SOUNDS.items():
        if os.path.exists(v):
            WAVS[k] = pygame.mixer.Sound(v)
        else:
            # print(f"[!] Sound file missing: {v}")
            WAVS[k] = None

SOUNDS_HIGHER = {
    "TL": os.path.join(SAVE_DIR, "hihat_higher.wav"),
    "TR": os.path.join(SAVE_DIR, "snare_higher.wav"),
    "BL": os.path.join(SAVE_DIR, "tom_higher.wav"),
    "BR": os.path.join(SAVE_DIR, "kick_higher.wav"),
    }

if use_audio:
    WAVS_HIGH = {}
    for k, v in SOUNDS_HIGHER.items():
        if os.path.exists(v):
            WAVS_HIGH[k] = pygame.mixer.Sound(v)
        else:
            # print(f"[!] Sound file missing: {v}")
            WAVS_HIGH[k] = None

def play(pos, intensity = "normal"):
    if use_audio and pos in WAVS and WAVS[pos] and intensity == "normal":
        WAVS[pos].play()
    elif use_audio and pos in WAVS_HIGH and WAVS_HIGH[pos] and intensity == "high":
        WAVS_HIGH[pos].play()
    else:
        # print(f"Hit {pos} (No Audio)")
        pass


# ===== 2) 시리얼 통신 & EMG 데이터 처리 (스레드) =====
ser = None
emg_L_raw = 0   
emg_R_raw = 0   
emg_L_val = 0   # 게인이 적용된 힘의 크기
emg_R_val = 0   
emg_L_for_plot = EMG_BASELINE
emg_R_for_plot = EMG_BASELINE

"""시험해볼 방법 3프레임으로 계산하기"""
history_y_A = deque([0]*3, maxlen=3)
history_y_B = deque([0]*3, maxlen=3)


last_emg_active_L = 0 
last_emg_active_R = 0 

EMG_BUFFER_L = 0
EMG_BUFFER_R = 0  

def emg_serial_thread():
    global ser, emg_L_raw, emg_R_raw, emg_L_val, emg_R_val, emg_L_for_plot, emg_R_for_plot
    global last_emg_active_L, last_emg_active_R
    # EMG_GAIN을 사용하기 위해 global 선언
    global EMG_BASELINE, EMG_THRESHOLD, EMG_THRESHOLD_HIGH, EMG_GAIN 

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

                    emg_L_for_plot = int((raw_L-EMG_BASELINE)*EMG_GAIN + EMG_BASELINE)                
                    emg_R_for_plot = int((raw_R-EMG_BASELINE)*EMG_GAIN + EMG_BASELINE)


                    DATA_L.append(emg_L_for_plot)  ## 여기서 plot 용 데이터를 갱신
                    DATA_R.append(emg_R_for_plot) 
                    
                    
                    # (원본값 - 기준값)의 절대값에 EMG_GAIN배를 곱해줌.
                    val_L = int(abs(raw_L - EMG_BASELINE) * EMG_GAIN)
                    val_R = int(abs(raw_R - EMG_BASELINE) * EMG_GAIN)
                    
                    emg_L_val = val_L
                    emg_R_val = val_R
                    


                    now = time.time()
                    
                    # --- 왼손 (L) ---
                    if val_L > EMG_THRESHOLD_HIGH: 
                        last_emg_active_L = now
                        EMG_BUFFER_L = 1
                    elif val_L > EMG_THRESHOLD: 
                        last_emg_active_L = now
                        EMG_BUFFER_L = 0

                    # --- 오른손 (R) ---
                    if val_R > EMG_THRESHOLD_HIGH:
                        last_emg_active_R = now
                        EMG_BUFFER_R = 1
                    elif val_R > EMG_THRESHOLD:
                        last_emg_active_R = now
                        EMG_BUFFER_R = 0
                        
        except Exception as e:
            print("Serial Read Error:", e)
            break

# EMG 데이터 로깅 및 플롯 설정 (그래프는 Raw값을 보여줌)
def emg_plot_thread():
    fig, ax = plt.subplots(2, 1, figsize=(10, 6))
    fig.suptitle('EMG ADC Value Live Plot', fontsize=16)

    X = np.arange(MAX_POINTS)
    Y_RANGE = [4000, 10000] 

    # Plot 1: L
    ax[0].set_title("Left Hand EMG (Raw)", fontsize=10)
    line_L, = ax[0].plot(X, DATA_L, color='blue', label='ADC Raw L')
    ax[0].set_ylim(Y_RANGE)
    ax[0].set_ylabel('ADC Value')
    ax[0].grid(True, linestyle='--', alpha=0.6)
    
    # Plot 2: R
    ax[1].set_title("Right Hand EMG (Raw)", fontsize=10)
    line_R, = ax[1].plot(X, DATA_R, color='red', label='ADC Raw R')
    ax[1].set_ylim(Y_RANGE)
    ax[1].set_ylabel('ADC Value')
    ax[1].set_xlabel('Time (Samples)')
    ax[1].grid(True, linestyle='--', alpha=0.6)
    
    # 임계값 선 (참고용 - 게인이 적용되기 전 Raw 데이터 기준으로는 정확치 않을 수 있음)
    # 그래프는 'Raw 데이터'를 보여주므로 Baseline만 명확히 그립니다.
    th_base_L, = ax[0].plot(X, [EMG_BASELINE] * MAX_POINTS, color='yellow', linestyle='-', label='Baseline')
    th_base_R, = ax[1].plot(X, [EMG_BASELINE] * MAX_POINTS, color='yellow', linestyle='-', label='Baseline')

    ax[0].legend(loc='upper right', fontsize='small')

    def update_plot(frame):
        line_L.set_ydata(DATA_L)
        line_R.set_ydata(DATA_R)
        return line_L, line_R, th_base_L, th_base_R

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

def nothing(x): pass

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
last_hit_A = {"TL":0, "TR":0, "BL":0, "BR":0}
last_hit_B = {"TL":0, "TR":0, "BL":0, "BR":0}
COOLDOWN_MS = 300

def quadrant(x, y, w, h):
    if y < h//2:
        return "TL" if x < w//2 else "TR"
    else:
        return "BL" if x < w//2 else "BR"

# ===== 4) 메인 루프 =====
while True:
    # 콘솔에 찍히는 val 값이 1.5배 되었는지 확인해보세요!
    # print(f"L val(x{EMG_GAIN}): {emg_L_val} | R val(x{EMG_GAIN}): {emg_R_val}")
    
    ret, frame = cap.read()
    if not ret: break
    
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    cur_time = time.time()
    now_ms = int(cur_time * 1000)

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
    mask_B = cv2.inRange(hsv, lower_B, upper_B)
    
    kernel = np.ones((5, 5), np.uint8)
    mask_A = cv2.dilate(mask_A, kernel, iterations=2)
    mask_B = cv2.dilate(mask_B, kernel, iterations=2)

    mask_disp = cv2.bitwise_or(mask_A, mask_B)
    cv2.imshow(MASK_WIN, mask_disp)

    # --- Tip A ---
    cnts_A, _ = cv2.findContours(mask_A, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts_A:
        c = max(cnts_A, key=cv2.contourArea)
        if cv2.contourArea(c) > countour_size_a:
            x, y, cw, ch = cv2.boundingRect(c)
            cxA, cyA = x + cw//2, y + ch//2
            cv2.circle(frame, (cxA, cyA), 10, (255, 0, 0), -1)

            history_y_A.append(cyA)

            ema_y_A = (EMA_ALPHA * cyA) + ((1 - EMA_ALPHA) * ema_y_A)
            
            vel_y_A = 0

            if history_y_A[0] != 0: 
                vel_y_A = (history_y_A[-1] - history_y_A[0])
            
            time_diff = cur_time - last_emg_active_L
            color_A = (0, 0, 255) if time_diff > EMG_WINDOW else (0, 255, 0)
            
            if vel_y_A > VEL_TH:
                if time_diff < EMG_WINDOW:
                    whichA = quadrant(cxA, int(ema_y_A), w, h)
                    if now_ms - last_hit_A[whichA] > COOLDOWN_MS:
                        if EMG_BUFFER_L == 0:
                            play(whichA, intensity = 'normal')
                            print(f"[A] HIT! Vel={vel_y_A:.1f}, EMG(Amp)={emg_L_val}")
                        elif EMG_BUFFER_L == 1:
                            play(whichA, intensity = 'high')
                            print(f"[A] HIT HIGH! Vel={vel_y_A:.1f}, EMG(Amp)={emg_L_val}")
                            
                        last_hit_A[whichA] = now_ms
                        hit_text = f"HIT {whichA}" + (" (HIGH)" if EMG_BUFFER_L == 1 else "")
                        cv2.putText(frame, hit_text, (cxA-40, cyA-20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
                else:
                    cv2.putText(frame, "No Muscle", (cxA, cyA-40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

    # --- Tip B ---
    cnts_B, _ = cv2.findContours(mask_B, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts_B:
        c = max(cnts_B, key=cv2.contourArea)
        if cv2.contourArea(c) > countour_size_b:
            x, y, cw, ch = cv2.boundingRect(c)
            cxB, cyB = x + cw//2, y + ch//2
            cv2.circle(frame, (cxB, cyB), 10, (0, 0, 255), -1)
            history_y_B.append(cyB)
            
            ema_y_B = (EMA_ALPHA * cyB) + ((1 - EMA_ALPHA) * ema_y_B)
            
            vel_y_B = 0
            
            if history_y_B[0] != 0:
                vel_y_B = ( history_y_B[-1]- history_y_B[0] )
            

            time_diff = cur_time - last_emg_active_R
            
            if vel_y_B > VEL_TH:
                if time_diff < EMG_WINDOW:
                    whichB = quadrant(cxB, int(ema_y_B), w, h)
                    if now_ms - last_hit_B[whichB] > COOLDOWN_MS:
                        if EMG_BUFFER_R == 0:
                            play(whichB, intensity = 'normal')
                            print(f"[B] HIT! Vel={vel_y_B:.1f}, EMG(Amp)={emg_R_val}")
                        elif EMG_BUFFER_R == 1:
                            play(whichB, intensity = 'high')
                            print(f"[B] HIT HIGH! Vel={vel_y_B:.1f}, EMG(Amp)={emg_R_val}")

                        last_hit_B[whichB] = now_ms
                        hit_text = f"HIT {whichB}" + (" (HIGH)" if EMG_BUFFER_R == 1 else "")
                        cv2.putText(frame, hit_text, (cxB-40, cyB-20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
                else:
                    cv2.putText(frame, "No Muscle", (cxB, cyB-40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

    # 4. HUD (스케일 조정)
    HUD_SCALE = 9000 
    
    # L HUD
    bar_h_L = min(int((emg_L_val / HUD_SCALE) * 300), 300)
    cv2.rectangle(frame, (20, 350), (50, 350 - bar_h_L), (255, 0, 0), -1)
    cv2.rectangle(frame, (20, 350), (50, 50), (255, 255, 255), 1)
    th_h_L_norm = int((EMG_THRESHOLD / HUD_SCALE) * 300)
    cv2.line(frame, (15, 350 - th_h_L_norm), (55, 350 - th_h_L_norm), (0, 255, 255), 2)
    th_h_L_high = int((EMG_THRESHOLD_HIGH / HUD_SCALE) * 300)
    cv2.line(frame, (15, 350 - th_h_L_high), (55, 350 - th_h_L_high), (0, 0, 255), 2)
    
    msg_L = "ACTIVE" if (cur_time - last_emg_active_L < EMG_WINDOW) else "WAIT"
    col_L = (0, 255, 0) if msg_L == "ACTIVE" else (100, 100, 100)
    cv2.putText(frame, f"L Val(x{EMG_GAIN}): {emg_L_val}", (10, 390), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    cv2.putText(frame, msg_L, (10, 410), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col_L, 2)

    # R HUD
    bar_h_R = min(int((emg_R_val / HUD_SCALE) * 300), 300)
    cv2.rectangle(frame, (w-50, 350), (w-20, 350 - bar_h_R), (0, 0, 255), -1)
    cv2.rectangle(frame, (w-50, 350), (w-20, 50), (255, 255, 255), 1)
    th_h_R_norm = int((EMG_THRESHOLD / HUD_SCALE) * 300)
    cv2.line(frame, (w-55, 350 - th_h_R_norm), (w-15, 350 - th_h_R_norm), (0, 255, 255), 2)
    th_h_R_high = int((EMG_THRESHOLD_HIGH / HUD_SCALE) * 300)
    cv2.line(frame, (w-55, 350 - th_h_R_high), (w-15, 350 - th_h_R_high), (0, 0, 255), 2)

    msg_R = "ACTIVE" if (cur_time - last_emg_active_R < EMG_WINDOW) else "WAIT"
    col_R = (0, 255, 0) if msg_R == "ACTIVE" else (100, 100, 100)
    cv2.putText(frame, f"R Val(x{EMG_GAIN}): {emg_R_val}", (w-100, 390), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    cv2.putText(frame, msg_R, (w-80, 410), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col_R, 2)

    # 4분할 선
    cv2.line(frame, (w//2, 0), (w//2, h), (100,100,100), 1)
    cv2.line(frame, (0, h//2), (w, h//2), (100,100,100), 1)

    cv2.imshow(MAIN_WIN, frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        break

cap.release()
cv2.destroyAllWindows()
if ser: ser.close()
plt.close('all')