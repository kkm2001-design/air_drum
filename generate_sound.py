import numpy as np
from scipy.io.wavfile import write

# 설정
RATE = 44100

def save_wav(filename, data, volume=1.0):
    """
    volume: 1.0 = 최대 크기, 0.1 = 1/10 크기
    """
    # 1. 데이터의 모양(음색)은 유지하되, -1~1 사이로 꽉 채우기 (정규화)
    max_val = np.max(np.abs(data))
    if max_val > 0:
        data = data / max_val
    
    # 2. 볼륨 적용
    data = data * volume
    
    # 3. 16비트 변환
    normalized = np.int16(data * 32767)
    write(filename, RATE, normalized)
    print(f"Generated: {filename} (Volume: {volume})")

# --- 사운드 생성 함수들 (이전과 동일하지만 하나로 통일됨) ---

def generate_kick(duration=0.3):
    t = np.linspace(0, duration, int(RATE * duration))
    # 묵직한 킥 사운드
    freq_sweep = np.linspace(150, 50, len(t))
    phase = 2 * np.pi * np.cumsum(freq_sweep) / RATE
    sine = np.sin(phase)
    click = np.random.uniform(-1, 1, len(t)) * np.exp(-60 * t) # 타격감
    envelope = np.exp(-8 * t)
    return (sine * envelope) + (click * 0.3)

def generate_snare(duration=0.2):
    t = np.linspace(0, duration, int(RATE * duration))
    # 찰진 스네어 사운드
    noise = np.random.uniform(-1, 1, len(t))
    noise = np.diff(noise, prepend=0) # 고음 강조
    tone = np.sin(2 * np.pi * 200 * t) * np.exp(-12 * t)
    envelope = np.exp(-15 * t)
    return (noise * envelope * 0.8) + (tone * 0.2)

def generate_hihat(duration=0.1):
    t = np.linspace(0, duration, int(RATE * duration))
    # 짦고 날카로운 하이햇
    noise = np.random.uniform(-1, 1, len(t))
    noise = np.diff(noise, prepend=0) # 고음 강조
    envelope = np.exp(-40 * t)
    return noise * envelope

def generate_tom(duration=0.3):
    t = np.linspace(0, duration, int(RATE * duration))
    # 둥~ 하는 톰 사운드
    freq_sweep = np.linspace(200, 100, len(t))
    phase = 2 * np.pi * np.cumsum(freq_sweep) / RATE
    envelope = np.exp(-5 * t)
    return np.sin(phase) * envelope

# ===== 메인 실행 =====
if __name__ == "__main__":
    print("--- 볼륨 차이 10배 드럼 생성 시작 ---")

    # [핵심 변경 사항]
    # 1. generate 함수에 들어가는 파라미터는 완전히 동일하게 설정 (소리 모양 통일)
    # 2. save_wav의 volume 파라미터만 0.1 vs 1.0 으로 설정 (크기 10배 차이)

    # 1. Kick
    kick_sound = generate_kick() # 소리 생성은 한 번만 정의해도 됨
    save_wav("kick.wav", kick_sound, volume=0.1)        # 작게
    save_wav("kick_higher.wav", kick_sound, volume=1.0) # 10배 크게

    # 2. Snare
    snare_sound = generate_snare()
    save_wav("snare.wav", snare_sound, volume=0.1)
    save_wav("snare_higher.wav", snare_sound, volume=1.0)

    # 3. Hi-hat
    hihat_sound = generate_hihat()
    save_wav("hihat.wav", hihat_sound, volume=0.1)
    save_wav("hihat_higher.wav", hihat_sound, volume=1.0)

    # 4. Tom
    tom_sound = generate_tom()
    save_wav("tom.wav", tom_sound, volume=0.1)
    save_wav("tom_higher.wav", tom_sound, volume=1.0)

    print("="*30)
    print("완료! 'higher' 파일은 일반 파일보다 정확히 10배 큰 파형(Amplitude)을 가집니다.")