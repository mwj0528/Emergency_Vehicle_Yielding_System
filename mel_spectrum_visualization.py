import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# 상수 정의
SAMPLE_RATE = 22050
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512

def visualize_mel_spectrum(audio_path):
    """
    오디오 파일의 mel-spectrum을 시각화합니다.
    
    Args:
        audio_path (str): 오디오 파일 경로
    """
    # 오디오 파일 로드
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
    
    # 길이 표준화 (3초)
    if len(y) < SAMPLE_RATE * 3:
        y = np.pad(y, (0, SAMPLE_RATE * 3 - len(y)), 'constant')
    else:
        y = y[:SAMPLE_RATE * 3]  # 3초만 사용
    
    # Mel 스펙트로그램 계산
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # 정규화
    mel_spec_db = (mel_spec_db - mel_spec_db.mean()) / (mel_spec_db.std() + 1e-8)
    
    # 시각화
    plt.figure(figsize=(12, 4))
    librosa.display.specshow(mel_spec_db, sr=sr, hop_length=HOP_LENGTH, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-Spectrogram')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    audio_path = "D:\project\siren\siren.wav"
    visualize_mel_spectrum(audio_path)