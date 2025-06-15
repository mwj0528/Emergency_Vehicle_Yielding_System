import os
import librosa
import librosa.display
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import torchaudio.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau

# 데이터 경로 설정
DATASET_PATH = r"D:\project\siren\urbansound8k"
META_PATH = os.path.join(DATASET_PATH, "UrbanSound8K.csv")

# 하이퍼파라미터
SAMPLE_RATE = 22050
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RANDOM_SEED = 42

# 재현성을 위한 시드 설정
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

# TensorBoard 설정
writer = SummaryWriter("runs/siren_detection")

# UrbanSound8K 데이터셋 로드
metadata = pd.read_csv(META_PATH)

# 이진 분류를 위한 라벨 설정 (사이렌 vs 비사이렌)
metadata['is_siren'] = (metadata['class'] == 'siren').astype(int)

# 데이터셋 균형 확인
siren_count = metadata['is_siren'].sum()
non_siren_count = len(metadata) - siren_count
print(f"Siren samples: {siren_count}, Non-siren samples: {non_siren_count}")

# 데이터 증강 함수
def time_shift(wav, sr, shift_limit=0.1):
    """시간 이동 증강"""
    shift = np.random.uniform(-shift_limit, shift_limit) * len(wav)
    return np.roll(wav, int(shift))

def pitch_shift(wav, sr, pitch_limit=4):
    """피치 이동 증강"""
    pitch_factor = np.random.uniform(-pitch_limit, pitch_limit)
    return librosa.effects.pitch_shift(wav, sr=sr, n_steps=pitch_factor)

def add_noise(wav, noise_factor=0.005):
    """노이즈 추가 증강"""
    noise = np.random.randn(len(wav))
    return wav + noise_factor * noise

# Mel 스펙트로그램 추출 함수
def get_mel_spectrogram(wav_path, augment=False):
    y, sr = librosa.load(wav_path, sr=SAMPLE_RATE)
    
    # 데이터 증강 적용 (훈련시에만)
    if augment and np.random.random() < 0.5:  # 50% 확률로 증강 적용
        aug_type = np.random.choice(['time', 'pitch', 'noise'])
        if aug_type == 'time':
            y = time_shift(y, sr)
        elif aug_type == 'pitch':
            y = pitch_shift(y, sr)
        elif aug_type == 'noise':
            y = add_noise(y)
    
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
    
    return mel_spec_db

# 커스텀 데이터셋 클래스
class UrbanSoundDataset(Dataset):
    def __init__(self, metadata, root_dir, is_train=False):
        self.metadata = metadata
        self.root_dir = root_dir
        self.is_train = is_train
        self.filepaths = metadata.apply(lambda row: os.path.join(root_dir, f"fold{row['fold']}/{row['slice_file_name']}"), axis=1)
        self.labels = torch.tensor(metadata['is_siren'].values, dtype=torch.float32)
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        wav_path = self.filepaths.iloc[idx]
        mel_spec = get_mel_spectrogram(wav_path, augment=self.is_train)
        mel_spec = torch.tensor(mel_spec, dtype=torch.float32).unsqueeze(0)  # 채널 추가
        label = self.labels[idx]
        return mel_spec, label

# 데이터셋 분할 (K-fold 대신 stratified split 사용)
from sklearn.model_selection import train_test_split

train_meta, test_meta = train_test_split(
    metadata, 
    test_size=0.2, 
    random_state=RANDOM_SEED,
    stratify=metadata['is_siren']  # 클래스 비율 유지
)

# 검증 세트 추가 분리
train_meta, val_meta = train_test_split(
    train_meta, 
    test_size=0.2, 
    random_state=RANDOM_SEED,
    stratify=train_meta['is_siren']  # 클래스 비율 유지
)

print(f"Train: {len(train_meta)}, Validation: {len(val_meta)}, Test: {len(test_meta)}")

# 데이터 로더 준비
train_dataset = UrbanSoundDataset(train_meta, DATASET_PATH, is_train=True)
val_dataset = UrbanSoundDataset(val_meta, DATASET_PATH, is_train=False)
test_dataset = UrbanSoundDataset(test_meta, DATASET_PATH, is_train=False)

# 가중치 샘플링으로 클래스 불균형 처리
train_class_weights = 1.0 / np.bincount(train_meta['is_siren'].values)
train_sample_weights = train_class_weights[train_meta['is_siren'].values]
train_sampler = torch.utils.data.WeightedRandomSampler(
    weights=train_sample_weights,
    num_samples=len(train_sample_weights),
    replacement=True
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 개선된 CNN 모델 정의
class ImprovedCNNClassifier(nn.Module):
    def __init__(self, num_classes=1):
        super(ImprovedCNNClassifier, self).__init__()
        # 컨볼루션 레이어
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.2),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.2),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.3),
        )
        
        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # 전결합 레이어
        self.fc_layers = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

best_model_state = r'D:\project\siren\2D\best_siren_classifier.pth'
# # 모델 학습 및 평가 함수
def train_and_evaluate_model(model, train_loader, val_loader, test_loader, epochs, lr):
    model.to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    
    # 학습률 스케줄러 추가
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    
    best_val_loss = float('inf')
    best_model_state = None
    
    for epoch in range(epochs):
        # 훈련 단계
        model.train()
        train_loss, train_correct = 0, 0
        train_preds, train_targets = [], []
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for mel_specs, labels in loop:
            mel_specs, labels = mel_specs.to(DEVICE), labels.to(DEVICE).unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(mel_specs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * mel_specs.size(0)
            
            preds = (outputs > 0.5).float()
            train_correct += (preds == labels).sum().item()
            
            train_preds.extend(preds.cpu().detach().numpy())
            train_targets.extend(labels.cpu().detach().numpy())
            
            loop.set_postfix(loss=loss.item())
        
        train_loss = train_loss / len(train_loader.dataset)
        train_acc = train_correct / len(train_loader.dataset)
        train_f1 = f1_score(train_targets, train_preds)
        
        # 검증 단계
        model.eval()
        val_loss, val_correct = 0, 0
        val_preds, val_targets = [], []
        
        with torch.no_grad():
            for mel_specs, labels in val_loader:
                mel_specs, labels = mel_specs.to(DEVICE), labels.to(DEVICE).unsqueeze(1)
                outputs = model(mel_specs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * mel_specs.size(0)
                
                preds = (outputs > 0.5).float()
                val_correct += (preds == labels).sum().item()
                
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(labels.cpu().numpy())
        
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_correct / len(val_loader.dataset)
        val_f1 = f1_score(val_targets, val_preds)
        
        # 학습률 스케줄러 갱신
        scheduler.step(val_loss)
        
        # TensorBoard 기록
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Validation', val_loss, epoch)
        writer.add_scalar('Accuracy/Train', train_acc, epoch)
        writer.add_scalar('Accuracy/Validation', val_acc, epoch)
        writer.add_scalar('F1-Score/Train', train_f1, epoch)
        writer.add_scalar('F1-Score/Validation', val_f1, epoch)
        
        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Train F1: {train_f1:.4f}, Val F1: {val_f1:.4f}")
        
        # 최고 성능 모델 저장
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            torch.save(model.state_dict(), "best_siren_classifier.pth")
            print(f"Model saved at epoch {epoch+1}")
    
    # 최고 성능 모델로 복원
    model.load_state_dict(best_model_state).to(DEVICE)
    
    # 테스트 세트 평가
    model.eval()
    test_loss, test_correct = 0, 0
    test_preds, test_targets = [], []
    all_probs = []
    
    with torch.no_grad():
        for mel_specs, labels in test_loader:
            mel_specs, labels = mel_specs.to(DEVICE), labels.to(DEVICE).unsqueeze(1)
            outputs = model(mel_specs)
            loss = criterion(outputs, labels)
            
            test_loss += loss.item() * mel_specs.size(0)
            
            probs = outputs.cpu().numpy()
            preds = (outputs > 0.5).float()
            test_correct += (preds == labels).sum().item()
            
            test_preds.extend(preds.cpu().numpy())
            test_targets.extend(labels.cpu().numpy())
            all_probs.extend(probs)
    
    test_loss = test_loss / len(test_loader.dataset)
    test_acc = test_correct / len(test_loader.dataset)
    test_f1 = f1_score(test_targets, test_preds)
    
    try:
        test_auc = roc_auc_score(test_targets, all_probs)
    except:
        test_auc = 0.0
    
    print("\nTest Results:")
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}, Test F1: {test_f1:.4f}, Test AUC: {test_auc:.4f}")
    
    # 혼동 행렬
    cm = confusion_matrix(test_targets, test_preds)
    print("\nConfusion Matrix:")
    print(cm)
    
    # 혼동 행렬 시각화 및 저장
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    classes = ['Non-Siren', 'Siren']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    
    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    
    return model

# 실행
print(f"Using device: {DEVICE}")
model = ImprovedCNNClassifier()
trained_model = train_and_evaluate_model(model, train_loader, val_loader, test_loader, EPOCHS, LEARNING_RATE)

# 모델 저장
torch.save(trained_model.state_dict(), "final_siren_classifier.pth")

# TensorBoard 종료
writer.close()

# 추론 함수 정의
def predict_audio(model, audio_path):
    model.eval()
    mel_spec = get_mel_spectrogram(audio_path)
    mel_spec = torch.tensor(mel_spec, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # 배치, 채널 추가
    
    with torch.no_grad():
        mel_spec = mel_spec.to(DEVICE)
        output = model(mel_spec)
        prob = output.item()
        pred = 1 if prob > 0.5 else 0
        
    return pred, prob

# 모델 사용 예시
print("\nModel Usage Example:")
print("To predict a new audio file:")
print("model = ImprovedCNNClassifier()")
print("model.load_state_dict(torch.load('best_siren_classifier.pth'))")
print("is_siren, probability = predict_audio(model, 'path/to/your/audio.wav')")
print("print(f'Is siren: {is_siren}, Probability: {probability:.4f}')")

def visualize_mel_spectrogram(wav_path, save_path=None):
    """Mel 스펙트로그램을 시각화하는 함수"""
    # 오디오 로드
    y, sr = librosa.load(wav_path, sr=SAMPLE_RATE)
    
    # Mel 스펙트로그램 계산
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # 시각화
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mel_spec_db, sr=sr, hop_length=HOP_LENGTH, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel Spectrogram')
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

# 예제 사용법
if __name__ == "__main__":
    # 예제 오디오 파일 경로 (첫 번째 훈련 샘플 사용)
    example_audio = os.path.join(DATASET_PATH, f"fold{train_meta.iloc[0]['fold']}", train_meta.iloc[0]['slice_file_name'])
    
    # Mel 스펙트로그램 시각화
    visualize_mel_spectrogram(example_audio, save_path='example_mel_spectrogram.png')