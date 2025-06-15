import carla
import random
import time
import torch
import torchaudio
import threading
import pygame
import torch.nn as nn
import keyboard

# ======================== 설정 상수 ========================
SIREN_AUDIO_PATH = "siren/siren.wav"
MODEL_PATH = r"D:\project\siren\best_siren_classifier.pth"
NUM_NORMAL_VEHICLES = 5
SIREN_DETECTION_RADIUS = 30.0

# ======================== 0. 환경 초기화 ========================
def clean_actors(world):
    for actor in world.get_actors():
        if actor.type_id.startswith('vehicle.') or actor.type_id.startswith('sensor.') or actor.type_id.startswith('static.'):
            try:
                actor.destroy()
            except Exception as e:
                print(f"❌ actor 제거 실패: {actor.id}, 이유: {e}")
    print("🧹 기존 actor 제거 완료")

client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
client.load_world('Town02')
world = client.get_world()
blueprint_library = world.get_blueprint_library()
spawn_points = world.get_map().get_spawn_points()
clean_actors(world)

# ======================== 1. 사이렌 감지 모델 정의 ========================
class ImprovedCNNClassifier(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.2),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.2),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.3),
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)
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
        return self.fc_layers(x)

model = ImprovedCNNClassifier()
model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
model.eval()

import librosa
import numpy as np

SAMPLE_RATE = 22050
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512

def get_mel_tensor(path):
    y, sr = librosa.load(path, sr=SAMPLE_RATE)
    # 3초 길이로 맞추기
    if len(y) < SAMPLE_RATE * 3:
        y = np.pad(y, (0, SAMPLE_RATE * 3 - len(y)), 'constant')
    else:
        y = y[:SAMPLE_RATE * 3]
    # Mel-spectrogram 계산
    mel = librosa.feature.melspectrogram(y=y, sr=SAMPLE_RATE, n_fft=N_FFT,
                                         hop_length=HOP_LENGTH, n_mels=N_MELS)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    # 정규화
    mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-8)
    mel_tensor = torch.tensor(mel_db).unsqueeze(0).unsqueeze(0).float()  # [B, C, H, W]
    return mel_tensor

def detect_siren(path=SIREN_AUDIO_PATH):
    input_tensor = get_mel_tensor(path)
    with torch.no_grad():
        output = model(input_tensor)
        return output.item() > 0.5

# ======================== 2. 차량 생성 ========================
def spawn_vehicles():
    ambulance_bp = blueprint_library.find('vehicle.ford.ambulance')
    ambulance = world.try_spawn_actor(ambulance_bp, spawn_points[0])
    if ambulance: ambulance.set_autopilot(True)

    normal_vehicles = []
    for i in range(1, NUM_NORMAL_VEHICLES + 1):
        if i >= len(spawn_points): break
        car_bp = random.choice(blueprint_library.filter('vehicle.*.*'))
        while car_bp.id == 'vehicle.ford.ambulance':
            car_bp = random.choice(blueprint_library.filter('vehicle.*.*'))
        vehicle = world.try_spawn_actor(car_bp, spawn_points[i])
        if vehicle:
            vehicle.set_autopilot(True)
            normal_vehicles.append(vehicle)
    
    return ambulance, normal_vehicles

# ======================== 3. 사이렌 감지 및 정지 ========================
siren_on = False

def is_siren_nearby(vehicle, ambulance):
    return vehicle.get_location().distance(ambulance.get_location()) <= SIREN_DETECTION_RADIUS

def stop_vehicle(vehicle):
    control = vehicle.get_control()
    control.throttle = 0.0
    control.brake = 1.0
    vehicle.apply_control(control)
    print(f"🛑 차량 {vehicle.id} 정지됨")

def monitor_vehicle(vehicle, ambulance):
    is_stopped = False  # 현재 차량이 수동 제어 중인지 여부
    while True:
        if not siren_on:
            time.sleep(1)
            continue

        # 사이렌 감지 & 근처에 있음
        if detect_siren() and is_siren_nearby(vehicle, ambulance):
            if not is_stopped:
                vehicle.set_autopilot(False)
                control = carla.VehicleControl()
                control.throttle = 0.0
                control.brake = 1.0
                vehicle.apply_control(control)
                print(f"🛑 차량 {vehicle.id} 정지됨")
                is_stopped = True

        # 더 이상 감지되지 않으면 Autopilot 다시 켜기
        elif is_stopped:
            vehicle.set_autopilot(True)
            print(f"✅ 차량 {vehicle.id} Autopilot 다시 ON")
            is_stopped = False

        time.sleep(1)

# ======================== 4. 사이렌 토글 및 pygame ========================
def toggle_siren():
    global siren_on
    siren_on = not siren_on
    if siren_on:
        print("🚨 사이렌 ON")
        ambulance.set_light_state(carla.VehicleLightState(carla.VehicleLightState.Special1))
        siren_sound.play(-1)
    else:
        print("🔇 사이렌 OFF")
        ambulance.set_light_state(carla.VehicleLightState.NONE)
        siren_sound.stop()

def key_listener():
    while running:
        if keyboard.is_pressed('s'):
            toggle_siren()
            time.sleep(0.5)
        time.sleep(0.1)

# ======================== 5. 메인 실행 ========================
running = True
ambulance, normal_vehicles = spawn_vehicles()

pygame.mixer.init()
siren_sound = pygame.mixer.Sound(SIREN_AUDIO_PATH)

try:
    print("🚘 차량 자율 주행 시작 (s 키로 사이렌 ON/OFF)")
    threading.Thread(target=key_listener, daemon=True).start()

    for vehicle in normal_vehicles:
        threading.Thread(target=monitor_vehicle, args=(vehicle, ambulance), daemon=True).start()

    while running:
        time.sleep(1)

except KeyboardInterrupt:
    running = False

finally:
    if ambulance: ambulance.destroy()
    for vehicle in normal_vehicles:
        vehicle.destroy()
    pygame.quit()
    print("🧹 종료 완료 및 자원 정리")
