"""
Air-Drum with EMG Sensor Fusion
- 웹캠: 동작 속도(Velocity) 감지
- MSP430: 근전도(EMG) 신호 감지 (Serial)
- 로직: (동작 빠름) AND (최근 0.3초 내 근육 힘) ==> 소리 재생
"""

import cv2
import numpy as np
import time
import os
import serial
import threading

# ===== 0) 설정 변수 (사용자 환경에 맞게 수정!) =====
COM_PORT = "COM3"       # <--- 장치관리자에서 확인 후 수정 필수!
BAUD_RATE = 115200      # MSP430 설정과 동일해야 함

# EMG 임계값 설정 (화면 보면서 튜닝 필요)
EMG_BASELINE = 7000      # 평소 가만히 있을 때 나오는 값 (오프셋 보드면 2000~2500, 아니면 0)
EMG_THRESHOLD = 500     # 평소 값보다 이만큼 더 커지면 힘준 걸로 인정
EMG_WINDOW = 0.4        # 근육 신호 후 0.4초까지는 타격 인정 (Sticky Flag 시간)

# 창 이름
MAIN_WIN = "AirDrum (Vision + EMG)"
MASK_WIN = "Mask View"
CTRL_WIN_A = "Settings Left (A)"   # 왼손 설정 창 이름
CTRL_WIN_B = "Settings Right (B)"  # 오른손 설정 창 이름

# ===== 1) 오디오 초기화 =====
use_audio = True
try:
    import pygame
    pygame.mixer.init(frequency=44100, channels=1, buffer=512)
except Exception as e:
    print("[!] pygame init failed:", e)
    use_audio = False

BASE = os.path.dirname(__file__)
# 같은 폴더에 wav 파일들이 있어야 함
SOUNDS = {
    "TL": os.path.join(BASE, "hihat.wav"),
    "TR": os.path.join(BASE, "snare.wav"),
    "BL": os.path.join(BASE, "tom.wav"),
    "BR": os.path.join(BASE, "kick.wav"),
}

if use_audio:
    WAVS = {}
    for k, v in SOUNDS.items():
        if os.path.exists(v):
            WAVS[k] = pygame.mixer.Sound(v)
        else:
            print(f"[!] Sound file missing: {v}")
            WAVS[k] = None

def play(pos):
    if use_audio and pos in WAVS and WAVS[pos]:
        WAVS[pos].play()
    else:
        print(f"Hit {pos} (No Audio)")

# ===== 2) 시리얼 통신 & EMG 데이터 처리 (스레드) =====
ser = None
emg_L_raw = 0   # 왼팔 원본값
emg_R_raw = 0   # 오른팔 원본값
emg_L_val = 0   # 기준값 뺀 절대값 (힘의 크기)
emg_R_val = 0   # 기준값 뺀 절대값

last_emg_active_L = 0  # 마지막으로 왼팔 힘준 시간
last_emg_active_R = 0  # 마지막으로 오른팔 힘준 시간

def emg_serial_thread():
    global ser, emg_L_raw, emg_R_raw, emg_L_val, emg_R_val
    global last_emg_active_L, last_emg_active_R

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
                
                # 패킷 길이(7바이트) 이상 쌓이면 처리
                while len(buffer) >= 7:
                    # 헤더(0x81) 찾기
                    if buffer[0] != 0x81:
                        buffer = buffer[1:] # 헤더 나올 때까지 버림
                        continue
                    
                    # 패킷 추출 [Header, L_H, L_L, R_H, R_L, 0, 0]
                    packet = buffer[:7]
                    buffer = buffer[7:]
                    
                    # 데이터 조립 (7비트 쉬프트)
                    raw_L = (packet[1] << 7) | packet[2]
                    raw_R = (packet[3] << 7) | packet[4]
                    
                    # 전역 변수 업데이트 (화면 표시용)
                    emg_L_raw = raw_L #adc1 = (int)((long)ADC12MEM0 * 9000 / 4096) - 4500 + 7000;이런식으로 받아옴
                    emg_R_raw = raw_R
                    
                    # 힘의 크기 계산 (절대값)
                    val_L = abs(raw_L - EMG_BASELINE)
                    val_R = abs(raw_R - EMG_BASELINE)
                    
                    emg_L_val = val_L
                    emg_R_val = val_R
                    
                    # 임계값 체크 -> Sticky Flag 타임스탬프 갱신
                    now = time.time()
                    if val_L > EMG_THRESHOLD:
                        last_emg_active_L = now
                    if val_R > EMG_THRESHOLD:
                        last_emg_active_R = now
                        
        except Exception as e:
            print("Serial Read Error:", e)
            break

# 스레드 시작
t = threading.Thread(target=emg_serial_thread, daemon=True)
t.start()

# ===== 3) OpenCV 영상 처리 설정 =====
cap = cv2.VideoCapture(0)
# [수정] 설정 창 2개 생성 및 크기 조절
cv2.namedWindow(CTRL_WIN_A) # 창생성
cv2.namedWindow(CTRL_WIN_B)
cv2.resizeWindow(CTRL_WIN_A, 400, 300) # 너비 400, 높이 300
cv2.resizeWindow(CTRL_WIN_B, 400, 300)

def nothing(x): pass

# --- 창 A: 왼손(A) + 공통 설정(속도감도) ---
cv2.createTrackbar("LH_A", CTRL_WIN_A, 100, 179, nothing)
cv2.createTrackbar("LS_A", CTRL_WIN_A, 150, 255, nothing)
cv2.createTrackbar("LV_A", CTRL_WIN_A, 150, 255, nothing)
cv2.createTrackbar("UH_A", CTRL_WIN_A, 120, 179, nothing)
cv2.createTrackbar("US_A", CTRL_WIN_A, 255, 255, nothing)
cv2.createTrackbar("UV_A", CTRL_WIN_A, 255, 255, nothing)
cv2.createTrackbar("VEL_TH", CTRL_WIN_A, 15, 100, nothing) # 감도는 A창에

# --- 창 B: 오른손(B) 설정 ---
cv2.createTrackbar("LH_B", CTRL_WIN_B, 10, 179, nothing)
cv2.createTrackbar("LS_B", CTRL_WIN_B, 150, 255, nothing)
cv2.createTrackbar("LV_B", CTRL_WIN_B, 150, 255, nothing)
cv2.createTrackbar("UH_B", CTRL_WIN_B, 30, 179, nothing)
cv2.createTrackbar("US_B", CTRL_WIN_B, 255, 255, nothing)
cv2.createTrackbar("UV_B", CTRL_WIN_B, 255, 255, nothing)

prev_y_A, prev_y_B = 0, 0
ema_y_A, ema_y_B = 0, 0
EMA_ALPHA = 0.4 # EMA 계수 (0~1, 클수록 최근값 비중 큼)
last_hit_A = {"TL":0, "TR":0, "BL":0, "BR":0}
last_hit_B = {"TL":0, "TR":0, "BL":0, "BR":0}
COOLDOWN_MS = 150 # 타격 후 쿨타임 (밀리초)

# 4분할 판정 함수
def quadrant(x, y, w, h):
    if y < h//2:
        return "TL" if x < w//2 else "TR"
    else:
        return "BL" if x < w//2 else "BR"

# ===== 4) 메인 루프 =====
while True:
    ret, frame = cap.read()
    if not ret: break
    
    frame = cv2.flip(frame, 1) # 좌우 반전
    h, w, _ = frame.shape
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    cur_time = time.time()
    now_ms = int(cur_time * 1000)

    # 1. 트랙바 값 읽기
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

    VEL_TH = cv2.getTrackbarPos("VEL_TH", CTRL_WIN_A) #여기서 값 읽어서 변수에 저장

    # 2. 마스크 생성
    lower_A = np.array([l_h_a, l_s_a, l_v_a])
    upper_A = np.array([u_h_a, u_s_a, u_v_a])
    mask_A = cv2.inRange(hsv, lower_A, upper_A) #조건 만족하는값만 1(흰색) 나머진 0(검정색)

    lower_B = np.array([l_h_b, l_s_b, l_v_b])
    upper_B = np.array([u_h_b, u_s_b, u_v_b])
    mask_B = cv2.inRange(hsv, lower_B, upper_B)
    
    # 합쳐서 보여주기용 마스크
    mask_disp = cv2.bitwise_or(mask_A, mask_B)
    cv2.imshow(MASK_WIN, mask_disp)

    # 3. 컨투어 추적 & 타격 판정
    # --- Tip A (왼손 가정) ---
    cnts_A, _ = cv2.findContours(mask_A, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # mask_A에서 외곽선 좌표롤검출
    if cnts_A:
        c = max(cnts_A, key=cv2.contourArea) # 가장 큰 외곽선 선택(면적 기준 key = countourArea)
        if cv2.contourArea(c) > 500: # 500 픽셀 이상만
            x, y, cw, ch = cv2.boundingRect(c)
            cxA, cyA = x + cw//2, y + ch//2
            cv2.circle(frame, (cxA, cyA), 10, (255, 0, 0), -1) # 파란 점

            # 속도 계산 (EMA)
            if prev_y_A == 0: prev_y_A = cyA
            ema_y_A = (EMA_ALPHA * cyA) + ((1 - EMA_ALPHA) * ema_y_A)
            vel_y_A = (cyA - prev_y_A) # 아래로 내려갈때가 양수(+)
            prev_y_A = cyA
            
            # [판정 로직] 속도 > TH 그리고 최근(EMG_WINDOW이내)에 근육 힘(L) 줬음
            time_diff = cur_time - last_emg_active_L
            
            # EMG 활성화 여부 확인 (HUD 표시용 텍스트 색상 변경)
            color_A = (0, 0, 255) if time_diff > EMG_WINDOW else (0, 255, 0)
            
            if vel_y_A > VEL_TH:
                # 여기에 AND 조건 추가!
                if time_diff < EMG_WINDOW:
                    whichA = quadrant(cxA, int(ema_y_A), w, h)
                    if now_ms - last_hit_A[whichA] > COOLDOWN_MS:
                        play(whichA)
                        last_hit_A[whichA] = now_ms
                        cv2.putText(frame, f"HIT {whichA}", (cxA-20, cyA-20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
                        print(f"[A] HIT! Vel={vel_y_A:.1f}, EMG={emg_L_val} (diff {time_diff:.2f}s)")
                else:
                    # 속도는 빠른데 근육 힘이 없어서 무시됨
                    cv2.putText(frame, "No Muscle", (cxA, cyA-40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

    # --- Tip B (오른손 가정) ---
    cnts_B, _ = cv2.findContours(mask_B, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts_B:
        c = max(cnts_B, key=cv2.contourArea)
        if cv2.contourArea(c) > 500:
            x, y, cw, ch = cv2.boundingRect(c)
            cxB, cyB = x + cw//2, y + ch//2
            cv2.circle(frame, (cxB, cyB), 10, (0, 0, 255), -1) # 빨간 점

            if prev_y_B == 0: prev_y_B = cyB
            ema_y_B = (EMA_ALPHA * cyB) + ((1 - EMA_ALPHA) * ema_y_B)
            vel_y_B = (cyB - prev_y_B)
            prev_y_B = cyB

            # [판정 로직]
            time_diff = cur_time - last_emg_active_R
            
            if vel_y_B > VEL_TH:
                if time_diff < EMG_WINDOW:
                    whichB = quadrant(cxB, int(ema_y_B), w, h)
                    if now_ms - last_hit_B[whichB] > COOLDOWN_MS:
                        play(whichB)
                        last_hit_B[whichB] = now_ms
                        cv2.putText(frame, f"HIT {whichB}", (cxB-20, cyB-20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
                        print(f"[B] HIT! Vel={vel_y_B:.1f}, EMG={emg_R_val} (diff {time_diff:.2f}s)")
                else:
                    cv2.putText(frame, "No Muscle", (cxB, cyB-40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

    # 4. HUD 그리기 (EMG 상태바)
    # 왼쪽 (Tip A, Left Hand)
    bar_h_L = min(int((emg_L_val / 4500) * 300), 300) # 높이 스케일링
    cv2.rectangle(frame, (20, 350), (50, 350 - bar_h_L), (255, 0, 0), -1) # 파란바
    cv2.rectangle(frame, (20, 350), (50, 50), (255, 255, 255), 1) # 테두리
    # 임계값 선
    th_h_L = int((EMG_THRESHOLD / 4500) * 300)
    cv2.line(frame, (15, 350 - th_h_L), (55, 350 - th_h_L), (0, 255, 255), 2)
    # 텍스트
    msg_L = "ACTIVE" if (cur_time - last_emg_active_L < EMG_WINDOW) else "WAIT"
    col_L = (0, 255, 0) if msg_L == "ACTIVE" else (100, 100, 100)
    cv2.putText(frame, f"L: {emg_L_raw}", (10, 370), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    cv2.putText(frame, msg_L, (10, 390), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col_L, 2)

    # 오른쪽 (Tip B, Right Hand)
    bar_h_R = min(int((emg_R_val / 4500) * 300), 300)
    cv2.rectangle(frame, (w-50, 350), (w-20, 350 - bar_h_R), (0, 0, 255), -1) # 빨간바
    cv2.rectangle(frame, (w-50, 350), (w-20, 50), (255, 255, 255), 1)
    # 임계값 선
    th_h_R = int((EMG_THRESHOLD / 4500) * 300)
    cv2.line(frame, (w-55, 350 - th_h_R), (w-15, 350 - th_h_R), (0, 255, 255), 2)
    # 텍스트
    msg_R = "ACTIVE" if (cur_time - last_emg_active_R < EMG_WINDOW) else "WAIT"
    col_R = (0, 255, 0) if msg_R == "ACTIVE" else (100, 100, 100)
    cv2.putText(frame, f"R: {emg_R_raw}", (w-60, 370), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    cv2.putText(frame, msg_R, (w-60, 390), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col_R, 2)

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