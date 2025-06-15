import carla
import random
import time
import torch
import torchaudio
import threading
import pygame
import torch.nn as nn
import keyboard
import numpy as np
import cv2
from ultralytics import YOLO
import librosa
import queue

# ======================== 설정 상수 ========================
YOLO_MODEL_PATH = "D:/project/yolo/0.9.15/yolov8s_finetuned/weights/best.pt"
SIREN_AUDIO_PATH = "D:/project/siren/siren.wav"
SIREN_MODEL_PATH = "D:/project/siren/final_siren_classifier.pth"
AMBULANCE_CLASS_NAME = 'ambulance'
CONFIDENCE_THRESHOLD = 0.6
SIREN_DETECTION_RADIUS = 30.0
NUM_NORMAL_VEHICLES = 5

# 오디오 처리 설정
SAMPLE_RATE = 22050
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512

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

# 모델 로드
yolo_model = YOLO(YOLO_MODEL_PATH)
siren_model = ImprovedCNNClassifier()
siren_model.load_state_dict(torch.load(SIREN_MODEL_PATH, map_location='cpu'))
siren_model.eval()

def get_mel_tensor(path):
    y, sr = librosa.load(path, sr=SAMPLE_RATE)
    if len(y) < SAMPLE_RATE * 3:
        y = np.pad(y, (0, SAMPLE_RATE * 3 - len(y)), 'constant')
    else:
        y = y[:SAMPLE_RATE * 3]
    mel = librosa.feature.melspectrogram(y=y, sr=SAMPLE_RATE, n_fft=N_FFT,
                                       hop_length=HOP_LENGTH, n_mels=N_MELS)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-8)
    mel_tensor = torch.tensor(mel_db).unsqueeze(0).unsqueeze(0).float()
    return mel_tensor

def detect_siren(path=SIREN_AUDIO_PATH):
    input_tensor = get_mel_tensor(path)
    with torch.no_grad():
        output = siren_model(input_tensor)
        return output.item() > 0.5

# ======================== 2. 차량 생성 ========================
def clean_actors(world):
    for actor in world.get_actors():
        if actor.type_id.startswith('vehicle.') or actor.type_id.startswith('sensor.') or actor.type_id.startswith('static.'):
            try:
                actor.destroy()
            except Exception as e:
                print(f"❌ actor 제거 실패: {actor.id}, 이유: {e}")
    print("🧹 기존 actor 제거 완료")

def spawn_vehicles(world, blueprint_library, spawn_points):
    ambulance_bp = blueprint_library.find('vehicle.ford.ambulance')
    ambulance = world.try_spawn_actor(ambulance_bp, spawn_points[0])
    if ambulance: 
        ambulance.set_autopilot(True)
        print("🚑 구급차 생성 완료")

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
            print(f"🚗 일반 차량 {vehicle.id} 생성 완료")
    
    return ambulance, normal_vehicles

# ======================== 3. 카메라 설정 ========================
def attach_rear_camera(vehicle, world):
    blueprint_library = world.get_blueprint_library()
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', '640')
    camera_bp.set_attribute('image_size_y', '480')
    camera_bp.set_attribute('fov', '90')
    
    camera_transform = carla.Transform(carla.Location(x=-5.0, z=2.0), carla.Rotation(yaw=180))
    camera = world.try_spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
    
    if camera:
        print(f"📸 차량 {vehicle.id}에 후방 카메라 부착 완료")
        return camera
    return None

# ======================== 4. 구급차 탐지 및 정지 ========================
siren_on = False

def is_ambulance_nearby(vehicle, ambulance):
    return vehicle.get_location().distance(ambulance.get_location()) <= SIREN_DETECTION_RADIUS

def stop_vehicle(vehicle):
    control = vehicle.get_control()
    control.throttle = 0.0
    control.brake = 1.0
    vehicle.apply_control(control)
    print(f"🛑 차량 {vehicle.id} 정지됨")

def process_image(image, vehicle, ambulance):
    if not siren_on:
        return

    # 이미지 처리
    img = np.frombuffer(image.raw_data, dtype=np.uint8)
    img = img.reshape((image.height, image.width, 4))[:, :, :3]
    
    # YOLO 탐지
    results = yolo_model(img)
    ambulance_detected = False
    
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            if yolo_model.names[cls_id] == AMBULANCE_CLASS_NAME and conf >= CONFIDENCE_THRESHOLD:
                ambulance_detected = True
                break
        if ambulance_detected:
            break
    
    # YOLO로 구급차가 탐지되고, 사이렌이 감지되며, 근처에 있는 경우
    if ambulance_detected and detect_siren() and is_ambulance_nearby(vehicle, ambulance):
        vehicle.set_autopilot(False)
        stop_vehicle(vehicle)
    else:
        vehicle.set_autopilot(True)

def monitor_vehicle(vehicle, ambulance, world):
    camera = attach_rear_camera(vehicle, world)
    if camera:
        camera.listen(lambda image: process_image(image, vehicle, ambulance))

# ======================== 5. 사이렌 토글 및 pygame ========================
def toggle_siren(ambulance):
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

def key_listener(ambulance):
    while running:
        if keyboard.is_pressed('s'):
            toggle_siren(ambulance)
            time.sleep(0.5)
        time.sleep(0.1)

# ======================== 6. 메인 실행 ========================
def main():
    global running, siren_sound
    running = True
    
    # CARLA 클라이언트 설정
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    client.load_world('Town02')
    world = client.get_world()
    blueprint_library = world.get_blueprint_library()
    spawn_points = world.get_map().get_spawn_points()
    
    # 기존 액터 제거
    clean_actors(world)
    
    # 차량 생성
    ambulance, normal_vehicles = spawn_vehicles(world, blueprint_library, spawn_points)
    
    # Pygame 초기화
    pygame.mixer.init()
    siren_sound = pygame.mixer.Sound(SIREN_AUDIO_PATH)
    
    try:
        print("🚘 차량 자율 주행 시작 (s 키로 사이렌 ON/OFF)")
        threading.Thread(target=key_listener, args=(ambulance,), daemon=True).start()
        
        for vehicle in normal_vehicles:
            threading.Thread(target=monitor_vehicle, args=(vehicle, ambulance, world), daemon=True).start()
        
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

if __name__ == "__main__":
    main() 