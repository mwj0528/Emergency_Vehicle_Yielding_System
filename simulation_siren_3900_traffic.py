import carla
import random
import math
import time
import numpy as np
import cv2 # OpenCV 필요
from ultralytics import YOLO # ultralytics 패키지 필요
import queue # 센서 데이터 처리를 위한 큐
import argparse # 커맨드 라인 인자 처리를 위한 모듈
import json
from datetime import datetime
import os
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel
from PyQt5.QtCore import Qt, QTimer
import sys
import torch
import torch.nn as nn
import torchaudio
import pygame
import librosa
import threading
import keyboard

# --- 기본 설정 ---
YOLO_MODEL_PATH = "D:/project/yolo/0.9.15/yolov8s_finetuned/weights/last.pt"
SIREN_AUDIO_PATH = "D:/project/siren/siren.wav"
SIREN_MODEL_PATH = "D:/project/siren/final_siren_classifier.pth"
AMBULANCE_CLASS_NAME = 'ambulance'
CONFIDENCE_THRESHOLD = 0.6

# 오디오 처리 설정
SAMPLE_RATE = 22050
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512

# 구급차 속도 설정
AMBULANCE_SPEED_PERCENTAGE = -150  # 제한 속도 대비 속도 차이 (음수: 더 빠름, 양수: 더 느림) - -100에서 -150으로 증가

# 구급차 우선권 설정
AMBULANCE_IGNORE_LIGHTS = 100  # 신호등 무시 비율 (%)
AMBULANCE_IGNORE_SIGNS = 100   # 표지판 무시 비율 (%)
AMBULANCE_IGNORE_VEHICLES = 0  # 다른 차량 무시 비율 (%) - 안전을 위해 0%로 설정
AMBULANCE_SAFE_DISTANCE = 20.0  # 구급차와 다른 차량 사이의 안전 거리 (미터)

# 일반 차량 설정
REGULAR_CAR_IGNORE_LIGHTS = 0  # 일반 차량의 신호등 무시 비율 (%)
REGULAR_CAR_IGNORE_SIGNS = 0   # 일반 차량의 표지판 무시 비율 (%)
REGULAR_CAR_IGNORE_VEHICLES = 0  # 일반 차량의 다른 차량 무시 비율 (%)
REGULAR_CAR_SAFE_DISTANCE = 10.0  # 일반 차량 간 안전 거리 (미터)

# 양보 동작 설정
YIELD_DISTANCE = 50.0  # 구급차 감지 시 양보를 시작할 거리 (미터)
YIELD_SPEED_REDUCTION = 70  # 양보 시 속도 감소 비율 (%)
YIELD_DURATION = 7.0  # 양보 상태 유지 시간 (초)
DEFAULT_CAR_SPEED = -30  # 일반 차량의 기본 속도 (제한 속도 대비) - -10에서 -30으로 증가
INTERSECTION_RADIUS = 50.0  # 교차로 영향 반경 (미터) - 50.0에서 60.0으로 증가
INTERSECTION_APPROACH_DISTANCE = 30.0  # 교차로 접근 거리 (미터) - 30.0에서 40.0으로 증가
INTERSECTION_EXIT_DISTANCE = 20.0  # 교차로 이탈 거리 (미터) - 20.0에서 30.0으로 증가
INTERSECTION_SPEED_REDUCTION = 50  # 교차로 통과 시 속도 감소 비율 (%) - 30에서 50으로 증가
INTERSECTION_STOP_DISTANCE = 15.0  # 교차로 진입 전 정지 거리 (미터) - 15.0에서 20.0으로 증가
STOPPED_VEHICLE_SAFE_DISTANCE = 12.0  # 정지된 차량과의 안전 거리 (미터) - 12.0에서 15.0으로 증가

# HUD 설정
HUD_FONT = cv2.FONT_HERSHEY_SIMPLEX
HUD_FONT_SCALE = 1.0
HUD_FONT_THICKNESS = 2
HUD_FONT_COLOR = (255, 255, 255)  # 흰색
HUD_BG_COLOR = (0, 0, 0)  # 검은색
HUD_PADDING = 10

# NUM_REGULAR_CARS = 10       # 스폰할 일반 차량 수
DETECTION_DISTANCE_THRESHOLD = 120.0 # 구급차 후방 탐지 활성화 거리 (미터) - 100.0에서 120.0으로 증가
LANE_CHECK_DISTANCE = 50.0   # 차선 비교를 시작할 거리 (DETECTION_DISTANCE_THRESHOLD보다 약간 작게) - 40.0에서 50.0으로 증가
LATERAL_THRESHOLD = 3.5      # 같은 차선으로 간주할 측면 거리 임계값 (미터) - 3.0에서 3.5로 증가

REAR_CAMERA_TRANSFORM = carla.Transform(carla.Location(x=-5.5, z=2.2), carla.Rotation(yaw=0)) # 후방 카메라 위치/방향 (차량 모델에 맞게 조절)
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
FIXED_DELTA_SECONDS = 0.05 # 시뮬레이션 스텝 시간

# 구급차 목적지 (예시: Town04의 특정 지점, 실제 맵과 경로에 맞게 수정 필요)
# 이 예시에서는 스폰 포인트 중 하나를 랜덤하게 목적지로 설정합니다.
# 더 정확한 실험을 위해서는 고정된 목적지 또는 경로를 설정해야 합니다.
AMBULANCE_DESTINATION_SPAWN_INDEX = -1 # 마지막 스폰 포인트를 목적지로 (예시)
DESTINATION_REACH_THRESHOLD = 10.0 # 목적지 도착으로 간주할 거리 (미터)
TOTAL_DISTANCE_THRESHOLD = 3900.0 # 구급차가 이동해야 할 총 거리 (미터)

# --- 고정된 경로 설정 (Town04 기준) ---
AMBULANCE_ROUTE = [
    carla.Location(x=100, y=100, z=0),
    carla.Location(x=200, y=100, z=0),
    carla.Location(x=300, y=100, z=0)
]

REGULAR_CAR_POSITIONS = [
    # 구급차 경로 주변에 일반 차량 배치
    carla.Location(x=90, y=90, z=0),
    carla.Location(x=110, y=90, z=0),
    carla.Location(x=190, y=90, z=0),
    carla.Location(x=210, y=90, z=0),
    carla.Location(x=290, y=90, z=0),
    carla.Location(x=310, y=90, z=0),
    # 추가 위치...
]

# --- 실험 결과 저장 설정 ---
RESULTS_DIR = "simulation_results"
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

def save_experiment_results(results, args):
    """실험 결과를 JSON 파일로 저장합니다."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{RESULTS_DIR}/experiment_{timestamp}.json"
    
    # 실험 설정과 결과를 함께 저장
    data = {
        "timestamp": timestamp,
        "configuration": {
            "map": args.map,
            "use_yolo_yielding": args.use_yolo_yielding,
            "seed": args.seed,
            "duration": args.duration,
            "num_regular_cars": args.num_regular_cars
        },
        "results": results
    }
    
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Results saved to {filename}")

# --- 도우미 함수 (제공된 코드와 유사하게 사용) ---
def get_forward_vector(rotation):
    rad_yaw = math.radians(rotation.yaw)
    return carla.Vector3D(x=math.cos(rad_yaw), y=math.sin(rad_yaw), z=0)

def is_ambulance_behind_in_same_general_direction(ego_vehicle, ambulance_vehicle, world_map, max_dist, lateral_thresh):
    """
    구급차가 일반 차량 뒤쪽의 동일/인접 차선에 있는지 확인합니다.
    차량의 방향성도 간략히 고려합니다.
    """
    if not ego_vehicle or not ambulance_vehicle:
        return False

    ego_tf = ego_vehicle.get_transform()
    ambulance_tf = ambulance_vehicle.get_transform()
    ego_loc = ego_tf.location
    ambulance_loc = ambulance_tf.location

    distance = ego_loc.distance(ambulance_loc)
    if distance > max_dist:
        return False

    # 1. 구급차가 일반 차량 뒤에 있는지 확인 (벡터 내적)
    vec_to_ambulance = ambulance_loc - ego_loc
    ego_fwd_vec = ego_tf.get_forward_vector() # x, y, z 벡터
    dot_product_position = vec_to_ambulance.x * ego_fwd_vec.x + vec_to_ambulance.y * ego_fwd_vec.y

    if dot_product_position > 0: # 구급차가 일반 차량보다 앞에 있으면 False
        return False

    # 2. 대략적인 방향 일치 확인 (두 차량이 서로 마주보고 있지 않은지)
    ambulance_fwd_vec = ambulance_tf.get_forward_vector()
    dot_product_direction = ego_fwd_vec.x * ambulance_fwd_vec.x + ego_fwd_vec.y * ambulance_fwd_vec.y
    if dot_product_direction < 0: # 두 차량이 서로 반대 방향을 향하고 있다면 (마주보고 있다면) False
        return False

    # 3. Waypoint API를 사용한 차선 기반 측면 거리 확인
    ego_wp = world_map.get_waypoint(ego_loc, project_to_road=True, lane_type=carla.LaneType.Driving)
    ambulance_wp = world_map.get_waypoint(ambulance_loc, project_to_road=True, lane_type=carla.LaneType.Driving)

    if ego_wp is None or ambulance_wp is None:
        return False

    # 같은 도로, 같은 차선 또는 바로 옆 차선에 있는지 확인
    if ego_wp.road_id == ambulance_wp.road_id:
        # 차선 ID는 방향에 따라 부호가 다를 수 있음.
        # 같은 방향으로 주행 중일 때 차선 ID 차이가 작아야 함.
        if ego_wp.lane_id * ambulance_wp.lane_id >= 0: # 같은 방향의 차선들인지 (부호가 같거나 둘 중 하나가 0)
            lane_diff = abs(ego_wp.lane_id - ambulance_wp.lane_id)
            if lane_diff <= 1: # 같은 차선(0) 또는 바로 옆 차선(1)
                return True
    return False

def draw_simulation_time(image, current_time):
    """시뮬레이션 시간을 이미지에 그립니다."""
    # 이미지 복사본 생성
    img_copy = image.copy()
    
    time_text = f"Simulation Time: {current_time:.2f}s"
    
    # 텍스트 크기 계산
    (text_width, text_height), _ = cv2.getTextSize(time_text, HUD_FONT, HUD_FONT_SCALE, HUD_FONT_THICKNESS)
    
    # 배경 사각형 그리기
    cv2.rectangle(img_copy, 
                 (img_copy.shape[1] - text_width - HUD_PADDING*2, HUD_PADDING),
                 (img_copy.shape[1] - HUD_PADDING, text_height + HUD_PADDING*2),
                 HUD_BG_COLOR, -1)
    
    # 텍스트 그리기
    cv2.putText(img_copy, time_text,
                (img_copy.shape[1] - text_width - HUD_PADDING, text_height + HUD_PADDING),
                HUD_FONT, HUD_FONT_SCALE, HUD_FONT_COLOR, HUD_FONT_THICKNESS)
    
    return img_copy

class SimulationTimerWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Simulation Timer")
        self.setGeometry(100, 100, 300, 100)  # x, y, width, height
        
        # 중앙 정렬된 라벨 생성
        self.time_label = QLabel("Simulation Time: 0.00s", self)
        self.time_label.setAlignment(Qt.AlignCenter)
        self.time_label.setStyleSheet("""
            QLabel {
                color: white;
                background-color: black;
                font-size: 24px;
                padding: 10px;
                border-radius: 5px;
            }
        """)
        self.setCentralWidget(self.time_label)
        
        # 창을 항상 위에 표시
        self.setWindowFlags(Qt.WindowStaysOnTopHint)
        
    def update_time(self, current_time, start_time):
        elapsed_time = current_time - start_time
        self.time_label.setText(f"Simulation Time: {elapsed_time:.2f}s")

def is_in_same_lane(ego_vehicle, ambulance_vehicle, world_map):
    """
    두 차량이 같은 차선에 있는지 확인합니다.
    """
    if not ego_vehicle or not ambulance_vehicle:
        return False

    ego_loc = ego_vehicle.get_location()
    ambulance_loc = ambulance_vehicle.get_location()

    ego_wp = world_map.get_waypoint(ego_loc, project_to_road=True, lane_type=carla.LaneType.Driving)
    ambulance_wp = world_map.get_waypoint(ambulance_loc, project_to_road=True, lane_type=carla.LaneType.Driving)

    if ego_wp is None or ambulance_wp is None:
        return False

    # 같은 도로인지 확인
    if ego_wp.road_id != ambulance_wp.road_id:
        return False

    # 차선 ID의 절대값이 같으면 같은 방향의 차선으로 간주
    if abs(ego_wp.lane_id) == abs(ambulance_wp.lane_id):
        print(f"\n=== Same Lane Detected ===")
        print(f"Car ID: {ego_vehicle.id}, Lane: {ego_wp.lane_id}")
        print(f"Distance to ambulance: {ego_loc.distance(ambulance_loc):.2f}m")
        print(f"Car location: {ego_loc}")
        print(f"Ambulance location: {ambulance_loc}")
        print("===========================\n")
        return True

    return False

def is_near_intersection(vehicle, world_map):
    """
    차량이 교차로 근처에 있는지 확인합니다.
    """
    vehicle_location = vehicle.get_location()
    vehicle_waypoint = world_map.get_waypoint(vehicle_location, project_to_road=True, lane_type=carla.LaneType.Driving)
    
    if vehicle_waypoint is None:
        return False
    
    # 현재 위치에서 INTERSECTION_RADIUS 반경 내의 모든 waypoint 확인
    nearby_waypoints = vehicle_waypoint.next(INTERSECTION_RADIUS)
    for wp in nearby_waypoints:
        if wp.is_intersection:
            # 교차로까지의 거리 계산
            intersection_distance = vehicle_location.distance(wp.transform.location)
            print(f"Vehicle is {intersection_distance:.2f}m away from intersection")
            # 교차로가 가까울수록 더 높은 확률로 신호 무시
            if intersection_distance < INTERSECTION_RADIUS:
                return True
    
    return False

def is_vehicle_changing_lane(vehicle, world_map):
    """
    차량이 차선 변경 중인지 확인합니다.
    """
    if not vehicle:
        return False
    
    vehicle_loc = vehicle.get_location()
    vehicle_vel = vehicle.get_velocity()
    
    # 차량이 거의 정지해 있는지 확인 (속도가 매우 낮은 경우)
    if vehicle_vel.length() < 1.0:  # 1.0 m/s 미만이면 거의 정지 상태로 간주
        return True
    
    # 차량의 현재 waypoint와 다음 waypoint 확인
    current_wp = world_map.get_waypoint(vehicle_loc, project_to_road=True, lane_type=carla.LaneType.Driving)
    if current_wp is None:
        return False
    
    # 차량의 진행 방향으로 약간 앞쪽의 waypoint 확인
    next_wp = current_wp.next(5.0)[0] if current_wp.next(5.0) else None
    if next_wp is None:
        return False
    
    # 현재 차선과 다음 차선이 다른 경우 차선 변경 중으로 간주
    return current_wp.lane_id != next_wp.lane_id

def has_completed_lane_change(vehicle, world_map, last_lane_id):
    """
    차량이 차선 변경을 완료했는지 확인합니다.
    """
    if not vehicle:
        return False
    
    vehicle_loc = vehicle.get_location()
    current_wp = world_map.get_waypoint(vehicle_loc, project_to_road=True, lane_type=carla.LaneType.Driving)
    
    if current_wp is None:
        return False
    
    # 현재 차선이 이전 차선과 다르고, 속도가 정상인 경우 차선 변경 완료로 간주
    return current_wp.lane_id != last_lane_id and vehicle.get_velocity().length() > 1.0

def is_same_direction(ego_vehicle, ambulance_vehicle):
    """
    두 차량이 같은 방향으로 진행 중인지 확인합니다.
    """
    if not ego_vehicle or not ambulance_vehicle:
        return False

    ego_vel = ego_vehicle.get_velocity()
    ambulance_vel = ambulance_vehicle.get_velocity()

    # 속도 벡터의 내적을 계산하여 방향 일치 여부 확인
    dot_product = ego_vel.x * ambulance_vel.x + ego_vel.y * ambulance_vel.y
    return dot_product > 0  # 내적이 양수면 같은 방향

def is_in_ambulance_path(ego_vehicle, ambulance_vehicle, world_map):
    """
    차량이 구급차의 진행 경로에 있는지 확인합니다.
    """
    if not ego_vehicle or not ambulance_vehicle:
        return False
    
    ego_loc = ego_vehicle.get_location()
    ambulance_loc = ambulance_vehicle.get_location()
    ambulance_vel = ambulance_vehicle.get_velocity()
    
    # 구급차의 속도가 너무 낮은 경우 처리
    if ambulance_vel.length() < 0.1:  # 0.1 m/s 미만이면 구급차의 방향을 waypoint로 판단
        ambulance_wp = world_map.get_waypoint(ambulance_loc, project_to_road=True, lane_type=carla.LaneType.Driving)
        if ambulance_wp is None:
            return False
        
        # 다음 waypoint를 통해 방향 계산
        next_wp = ambulance_wp.next(5.0)[0] if ambulance_wp.next(5.0) else None
        if next_wp is None:
            return False
        
        # waypoint 방향으로 벡터 생성
        ambulance_direction = carla.Vector3D(
            x=next_wp.transform.location.x - ambulance_loc.x,
            y=next_wp.transform.location.y - ambulance_loc.y,
            z=0
        )
    else:
        # 구급차의 진행 방향 벡터
        ambulance_direction = carla.Vector3D(
            x=ambulance_vel.x,
            y=ambulance_vel.y,
            z=0
        )
    
    # 벡터 길이가 너무 작은 경우 처리
    if ambulance_direction.length() < 0.1:
        return False
    
    # 벡터 정규화
    ambulance_direction = ambulance_direction.make_unit_vector()
    
    # 차량에서 구급차까지의 벡터
    to_ambulance = carla.Vector3D(
        x=ambulance_loc.x - ego_loc.x,
        y=ambulance_loc.y - ego_loc.y,
        z=0
    )
    
    # 두 벡터의 내적을 계산하여 구급차 진행 방향에 있는지 확인
    dot_product = to_ambulance.x * ambulance_direction.x + to_ambulance.y * ambulance_direction.y
    
    # 구급차로부터의 거리
    distance = ego_loc.distance(ambulance_loc)
    
    # 구급차 진행 방향에 있고, 일정 거리 이내에 있는 경우
    return dot_product > 0 and distance < 50.0  # 50m 이내

def is_vehicle_stopped(vehicle):
    """
    차량이 정지 상태인지 확인합니다.
    """
    if not vehicle:
        return False
    velocity = vehicle.get_velocity()
    return velocity.length() < 0.1  # 0.1 m/s 미만이면 정지 상태로 간주

def get_safe_distance(vehicle, world_map):
    """
    차량의 현재 상황에 따른 안전 거리를 계산합니다.
    """
    if not vehicle:
        return REGULAR_CAR_SAFE_DISTANCE
    
    # 정지 상태 확인
    if is_vehicle_stopped(vehicle):
        return STOPPED_VEHICLE_SAFE_DISTANCE
    
    # 교차로 근처 확인
    if is_near_intersection(vehicle, world_map):
        return REGULAR_CAR_SAFE_DISTANCE * 1.5
    
    return REGULAR_CAR_SAFE_DISTANCE

def is_approaching_intersection(vehicle, world_map):
    """
    차량이 교차로에 접근 중인지 확인합니다.
    """
    vehicle_location = vehicle.get_location()
    vehicle_waypoint = world_map.get_waypoint(vehicle_location, project_to_road=True, lane_type=carla.LaneType.Driving)
    
    if vehicle_waypoint is None:
        return False
    
    # 현재 위치에서 INTERSECTION_APPROACH_DISTANCE 반경 내의 모든 waypoint 확인
    nearby_waypoints = vehicle_waypoint.next(INTERSECTION_APPROACH_DISTANCE)
    for wp in nearby_waypoints:
        if wp.is_intersection:
            return True
    
    return False

def is_exiting_intersection(vehicle, world_map):
    """
    차량이 교차로를 이탈 중인지 확인합니다.
    """
    vehicle_location = vehicle.get_location()
    vehicle_waypoint = world_map.get_waypoint(vehicle_location, project_to_road=True, lane_type=carla.LaneType.Driving)
    
    if vehicle_waypoint is None:
        return False
    
    # 현재 위치에서 INTERSECTION_EXIT_DISTANCE 반경 내의 모든 waypoint 확인
    nearby_waypoints = vehicle_waypoint.next(INTERSECTION_EXIT_DISTANCE)
    for wp in nearby_waypoints:
        if not wp.is_intersection:
            return True
    
    return False

def is_opposite_direction(ego_vehicle, ambulance_vehicle):
    """
    두 차량이 반대 방향으로 진행 중인지 확인합니다.
    """
    if not ego_vehicle or not ambulance_vehicle:
        return False
    
    ego_vel = ego_vehicle.get_velocity()
    ambulance_vel = ambulance_vehicle.get_velocity()
    
    # 속도 벡터의 내적을 계산하여 방향 일치 여부 확인
    dot_product = ego_vel.x * ambulance_vel.x + ego_vel.y * ambulance_vel.y
    return dot_product < 0  # 내적이 음수면 반대 방향

def is_near_intersection_approach(vehicle, world_map):
    """
    차량이 교차로 진입 전 정지 구역에 있는지 확인합니다.
    """
    vehicle_location = vehicle.get_location()
    vehicle_waypoint = world_map.get_waypoint(vehicle_location, project_to_road=True, lane_type=carla.LaneType.Driving)
    
    if vehicle_waypoint is None:
        return False
    
    # 현재 위치에서 INTERSECTION_STOP_DISTANCE 반경 내의 모든 waypoint 확인
    nearby_waypoints = vehicle_waypoint.next(INTERSECTION_STOP_DISTANCE)
    for wp in nearby_waypoints:
        if wp.is_intersection:
            # 교차로까지의 거리 계산
            intersection_distance = vehicle_location.distance(wp.transform.location)
            # 교차로 진입 전 정지 구역에 있는지 확인
            if intersection_distance < INTERSECTION_STOP_DISTANCE and intersection_distance > 5.0:
                return True
    
    return False

def is_ambulance_in_intersection(ambulance, world_map):
    """
    구급차가 교차로 반경 내에 있는지 확인
    """
    if not ambulance:
        return False
    ambulance_loc = ambulance.get_location()
    ambulance_wp = world_map.get_waypoint(ambulance_loc, project_to_road=True, lane_type=carla.LaneType.Driving)
    if ambulance_wp is None:
        return False
    return ambulance_wp.is_intersection

def is_recently_exited_intersection(vehicle, world_map, exit_distance=15.0):
    """
    차량이 최근에 교차로를 벗어났는지(지정 거리 이내) 확인합니다.
    """
    vehicle_location = vehicle.get_location()
    vehicle_waypoint = world_map.get_waypoint(vehicle_location, project_to_road=True, lane_type=carla.LaneType.Driving)
    if vehicle_waypoint is None or vehicle_waypoint.is_intersection:
        return False
    # 뒤쪽으로 일정 거리 내에 교차로가 있었는지 확인
    prev_waypoints = vehicle_waypoint.previous(exit_distance)
    for wp in prev_waypoints:
        if wp.is_intersection:
            return True
    return False

def should_stop_for_ambulance(car, ambulance, world_map):
    """
    차량이 구급차를 위해 정지해야 하는지 확인합니다.
    """
    if not car or not ambulance:
        return False

    car_loc = car.get_location()
    ambulance_loc = ambulance.get_location()
    distance = car_loc.distance(ambulance_loc)

    # 구급차가 교차로 반경 내에 있거나 접근 중일 때
    if is_ambulance_in_intersection(ambulance, world_map) or is_approaching_intersection(ambulance, world_map):
        # 구급차와 같은 방향이면 정지하지 않음
        if is_same_direction(car, ambulance):
            return False

        # 이미 교차로 내부에 진입한 차량은 정지하지 않음
        car_wp = world_map.get_waypoint(car_loc, project_to_road=True, lane_type=carla.LaneType.Driving)
        if car_wp and car_wp.is_intersection:
            return False

        # 교차로를 막 지난 차량(15m 이내)은 정지하지 않음
        if is_recently_exited_intersection(car, world_map, exit_distance=15.0):
            return False

        # 구급차의 진행 경로에 있는 차량은 정지하지 않음
        if is_in_ambulance_path(car, ambulance, world_map):
            return False

        # 구급차와의 거리가 너무 멀면 정지하지 않음 (거리 증가)
        if distance > 50.0:  # 30m에서 50m로 증가
            return False

        # 구급차의 진행 방향과 차량의 진행 방향이 수직인 경우에만 정지
        ambulance_vel = ambulance.get_velocity()
        car_vel = car.get_velocity()
        
        # 속도 벡터의 내적을 계산하여 방향 관계 확인
        dot_product = ambulance_vel.x * car_vel.x + ambulance_vel.y * car_vel.y
        if abs(dot_product) > 0.5:  # 내적이 0.5보다 크면 같은 방향 또는 반대 방향으로 간주
            return False

        # 차량이 교차로 진입 전 정지 구역에 있는지 확인
        return is_near_intersection_approach(car, world_map)

    return False

def is_ambulance_passed(car, ambulance, world_map):
    """
    구급차가 차량을 지나갔는지 확인합니다.
    """
    if not car or not ambulance:
        return False
    
    car_loc = car.get_location()
    ambulance_loc = ambulance.get_location()
    
    # 구급차의 진행 방향 벡터
    ambulance_vel = ambulance.get_velocity()
    if ambulance_vel.length() < 0.1:  # 구급차가 거의 정지 상태
        return False
    
    # 구급차의 진행 방향으로의 벡터
    ambulance_direction = ambulance_vel.make_unit_vector()
    
    # 차량에서 구급차까지의 벡터
    to_ambulance = carla.Vector3D(
        x=ambulance_loc.x - car_loc.x,
        y=ambulance_loc.y - car_loc.y,
        z=0
    )
    
    # 두 벡터의 내적을 계산
    dot_product = to_ambulance.x * ambulance_direction.x + to_ambulance.y * ambulance_direction.y
    
    # 구급차가 차량을 지나갔는지 확인 (내적이 음수면 구급차가 차량을 지나감)
    return dot_product < 0

def is_safe_to_change_lane(car, world_map, target_lane_wp, world):
    """
    차선 변경이 안전한지 확인합니다.
    """
    if not car or not target_lane_wp:
        return False

    car_loc = car.get_location()
    car_vel = car.get_velocity()
    car_speed = car_vel.length()

    # 현재 차량의 waypoint
    current_wp = world_map.get_waypoint(car_loc, project_to_road=True, lane_type=carla.LaneType.Driving)
    if not current_wp:
        return False

    # 목표 차선의 다음 waypoint
    next_wp = target_lane_wp.next(5.0)[0] if target_lane_wp.next(5.0) else None
    if not next_wp:
        return False

    # 주변 차량 확인
    nearby_vehicles = world.get_actors().filter('vehicle.*')
    for vehicle in nearby_vehicles:
        if vehicle.id == car.id:
            continue

        vehicle_loc = vehicle.get_location()
        vehicle_vel = vehicle.get_velocity()
        vehicle_speed = vehicle_vel.length()

        # 목표 차선에 있는 차량과의 거리 확인
        if world_map.get_waypoint(vehicle_loc, project_to_road=True, lane_type=carla.LaneType.Driving).lane_id == target_lane_wp.lane_id:
            distance = car_loc.distance(vehicle_loc)
            if distance < 20.0:  # 20m 이내에 차량이 있으면 안전하지 않음
                return False

    return True

def optimize_intersection_behavior(car, traffic_manager, world_map, ambulance=None, current_time=0.0, world=None):
    """
    교차로 통과 시 차량의 동작을 최적화합니다.
    """
    if ambulance:
        car_loc = car.get_location()
        ambulance_loc = ambulance.get_location()
        distance = car_loc.distance(ambulance_loc)

        # 구급차가 교차로에 접근 중이거나 교차로 내부에 있을 때
        if is_ambulance_in_intersection(ambulance, world_map) or is_approaching_intersection(ambulance, world_map):
            if should_stop_for_ambulance(car, ambulance, world_map):
                # 구급차와 다른 방향이고 교차로 진입 전 정지 구역에 있으면 정지
                traffic_manager.vehicle_percentage_speed_difference(car, 100)  # 완전 정지
                traffic_manager.distance_to_leading_vehicle(car, STOPPED_VEHICLE_SAFE_DISTANCE)
                print(f"[{current_time:.2f}s] Car {car.id} stopping before intersection for ambulance")
            elif is_approaching_intersection(car, world_map):
                # 구급차가 교차로에 접근 중일 때 차선 변경 시도
                car_wp = world_map.get_waypoint(car_loc, project_to_road=True, lane_type=carla.LaneType.Driving)
                if car_wp:
                    # 오른쪽 차선 우선 확인
                    right_lane_wp = car_wp.get_right_lane()
                    left_lane_wp = car_wp.get_left_lane()
                    
                    # 양쪽 차선 확인
                    can_change_right = right_lane_wp and right_lane_wp.lane_type == carla.LaneType.Driving
                    can_change_left = left_lane_wp and left_lane_wp.lane_type == carla.LaneType.Driving
                    
                    # 차선 변경 가능 여부 확인
                    if can_change_right or can_change_left:
                        # 오른쪽 차선 우선
                        if can_change_right and is_safe_to_change_lane(car, world_map, right_lane_wp, world):
                            traffic_manager.force_lane_change(car, True)  # 오른쪽으로 차선 변경
                        else:
                            traffic_manager.force_lane_change(car, False)  # 왼쪽으로 차선 변경
                    else:
                        # 차선 변경이 불가능한 경우 속도를 높여 추월
                        print(f"[{current_time:.2f}s] Car {car.id} cannot change lanes - increasing speed to overtake")
                        # 속도 증가 (제한 속도 대비 80% 증가)
                        traffic_manager.vehicle_percentage_speed_difference(car, -80)  # -50에서 -80으로 증가
                        # 추월 상태로 설정
                        car_data['state'] = 'overtaking'
                        car_data['speed_increase_time'] = current_time
                        car_data['target_lane'] = 'right' if can_change_right else 'left'
            elif is_ambulance_passed(car, ambulance, world_map):
                # 구급차가 지나갔으면 정상 주행으로 복귀
                traffic_manager.vehicle_percentage_speed_difference(car, DEFAULT_CAR_SPEED)
                traffic_manager.distance_to_leading_vehicle(car, REGULAR_CAR_SAFE_DISTANCE)
                print(f"[{current_time:.2f}s] Car {car.id} resuming normal driving after ambulance passed")
    elif is_approaching_intersection(car, world_map):
        # 일반적인 교차로 접근 시 속도 감소
        traffic_manager.vehicle_percentage_speed_difference(car, INTERSECTION_SPEED_REDUCTION)
        traffic_manager.distance_to_leading_vehicle(car, REGULAR_CAR_SAFE_DISTANCE * 1.5)
    elif is_exiting_intersection(car, world_map):
        # 교차로 이탈 시 정상 속도로 복귀
        traffic_manager.vehicle_percentage_speed_difference(car, DEFAULT_CAR_SPEED)
        traffic_manager.distance_to_leading_vehicle(car, REGULAR_CAR_SAFE_DISTANCE)

def remove_stopped_vehicle(regular_cars, world, current_time):
    """
    가장 오래 정지한 차량을 찾아서 제거합니다.
    """
    longest_stopped_car = None
    longest_stop_time = 0.0
    longest_stopped_car_id = None

    for car_id, car_data in regular_cars.items():
        car = car_data['actor']
        if not car or not car.is_alive:
            continue

        # 차량이 정지 상태인지 확인
        if is_vehicle_stopped(car):
            # 정지 시작 시간 기록
            if 'stop_start_time' not in car_data:
                car_data['stop_start_time'] = current_time
            # 정지 시간 계산
            stop_time = current_time - car_data['stop_start_time']
            # 10초 이상 정지 상태이고, 가장 오래 정지한 차량인 경우
            if stop_time > 10.0 and stop_time > longest_stop_time:
                longest_stopped_car = car
                longest_stop_time = stop_time
                longest_stopped_car_id = car_id
        else:
            # 움직이면 정지 시작 시간 초기화
            car_data.pop('stop_start_time', None)
    
    # 가장 오래 정지한 차량이 있으면 제거
    if longest_stopped_car and longest_stopped_car_id:
        print(f"[{current_time:.2f}s] Removing longest stopped car {longest_stopped_car_id} (stopped for {longest_stop_time:.2f}s)")
        if longest_stopped_car.is_alive:
            longest_stopped_car.destroy()
        if longest_stopped_car_id in regular_cars:
            del regular_cars[longest_stopped_car_id]
        return True
    
    return False  # 제거된 차량이 없음

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

# --- 메인 로직 ---
def game_loop(args):
    # PyQt 애플리케이션 생성
    app = QApplication(sys.argv)
    timer_window = SimulationTimerWindow()
    timer_window.show()

    client = carla.Client(args.host, args.port)
    client.set_timeout(20.0)
    world = None
    original_settings = None
    traffic_manager = None
    actor_list = []
    regular_cars = {} # {actor_id: {'actor': vehicle_actor, 'state': 'driving'/'yielding', 'camera': None, 'queue': None, 'last_yield_time': 0.0}}
    ambulance = None
    yolo_model = None
    simulation_start_time = 0.0
    ambulance_start_time = 0.0
    ambulance_arrival_time = 0.0
    ambulance_destination_transform = None
    spectator = None  # 관전자 변수 추가
    wall_clock_start_time = time.time()  # wall clock 시작 시간 추가
    travel_time = 0.0  # 구급차 이동 시간 초기화
    total_simulation_ran_time = 0.0  # 시뮬레이션 실행 시간 초기화
    last_removal_time = 0.0  # 마지막 차량 제거 시간

    # 사이렌 모델 로드
    siren_model = ImprovedCNNClassifier()
    siren_model.load_state_dict(torch.load(SIREN_MODEL_PATH, map_location='cpu'))
    siren_model.eval()

    # Pygame 초기화
    pygame.mixer.init()
    siren_sound = pygame.mixer.Sound(SIREN_AUDIO_PATH)
    siren_on = False

    def toggle_siren():
        nonlocal siren_on
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

    # 키 리스너 스레드 시작
    threading.Thread(target=key_listener, daemon=True).start()

    try:
        world = client.load_world(args.map) # 지정된 맵 로드
        original_settings = world.get_settings()
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = FIXED_DELTA_SECONDS
        world.apply_settings(settings)

        # spectator 초기화
        spectator = world.get_spectator()

        traffic_manager = client.get_trafficmanager(args.tm_port)
        traffic_manager.set_synchronous_mode(True)
        if args.seed is not None:
            traffic_manager.set_random_device_seed(args.seed) # TM 시드 고정
        traffic_manager.set_global_distance_to_leading_vehicle(7.0)  # 차량 간 거리를 7m로 증가 (5m에서 7m로)
        # 모든 차량이 신호를 지키도록 설정
        traffic_manager.global_percentage_speed_difference(0)  # 기본 속도 유지
        print("All vehicles will obey traffic lights and signs")

        blueprint_library = world.get_blueprint_library()
        world_map = world.get_map()

        # 랜덤 시드 고정 (Python, NumPy)
        if args.seed is not None:
            random.seed(args.seed)
            np.random.seed(args.seed)
            print(f"Random seeds (Python, NumPy, TM) set to: {args.seed}")

        spawn_points = world_map.get_spawn_points()
        if not spawn_points:
            print("Error: No spawn points found in the map!")
            return

        # --- 1. 차량 스폰 ---
        # 구급차 스폰
        ambulance_bp = blueprint_library.find('vehicle.ford.ambulance') # 구급차 블루프린트

        # 구급차 출발지 설정 - 랜덤 spawn point 사용
        ambulance_spawn_point = random.choice(spawn_points)
        print(f"Using random spawn point for ambulance: {ambulance_spawn_point.location}")

        ambulance = world.try_spawn_actor(ambulance_bp, ambulance_spawn_point)

        if ambulance:
            actor_list.append(ambulance)
            ambulance.set_autopilot(True, traffic_manager.get_port())
            # 구급차 설정
            traffic_manager.ignore_lights_percentage(ambulance, 100)  # 구급차 신호등 무시
            traffic_manager.ignore_signs_percentage(ambulance, 100)   # 구급차 표지판 무시
            traffic_manager.ignore_vehicles_percentage(ambulance, 0)  # 다른 차량 무시하지 않음
            traffic_manager.distance_to_leading_vehicle(ambulance, AMBULANCE_SAFE_DISTANCE)  # 안전 거리 설정
            traffic_manager.vehicle_percentage_speed_difference(ambulance, AMBULANCE_SPEED_PERCENTAGE)  # 구급차 속도 설정
            traffic_manager.auto_lane_change(ambulance, True)  # 자동 차선 변경 활성화
            print(f"Spawned Ambulance (ID: {ambulance.id}) at {ambulance_spawn_point.location}")
            print(f"Ambulance settings: Ignoring traffic lights and signs, maintaining safe distance")
            spawn_points.remove(ambulance_spawn_point)
        else:
            print("Error: Failed to spawn ambulance.")
            return

        # 목적지 없이 총 이동거리로 종료
        ambulance_destination_transform = None
        print("No destination set. Simulation will end when ambulance travels 3900m.")

        # 일반 차량 스폰
        car_bp_choices = [
            blueprint_library.find('vehicle.tesla.model3'),
            blueprint_library.find('vehicle.lincoln.mkz_2017')
        ]

        for i in range(args.num_regular_cars):
            if not spawn_points: break
            spawn_point = random.choice(spawn_points)
            spawn_points.remove(spawn_point)
            
            car_bp = random.choice(car_bp_choices)
            car = world.try_spawn_actor(car_bp, spawn_point)
            if car:
                actor_list.append(car)
                car.set_autopilot(True, traffic_manager.get_port())
                # 일반 차량 설정 - YOLO 양보 시스템 사용 시에만 신호 무시 가능
                if args.use_yolo_yielding:
                    traffic_manager.ignore_lights_percentage(car, 0)  # 기본적으로 신호등 준수
                    traffic_manager.ignore_signs_percentage(car, 0)   # 기본적으로 표지판 준수
                else:
                    traffic_manager.ignore_lights_percentage(car, 0)  # 항상 신호등 준수
                    traffic_manager.ignore_signs_percentage(car, 0)   # 항상 표지판 준수
                traffic_manager.ignore_vehicles_percentage(car, 0)  # 다른 차량 무시하지 않음
                traffic_manager.distance_to_leading_vehicle(car, REGULAR_CAR_SAFE_DISTANCE)
                traffic_manager.auto_lane_change(car, True)  # 평소 자동 차선 변경 허용
                traffic_manager.vehicle_percentage_speed_difference(car, DEFAULT_CAR_SPEED)  # 기본 속도 설정
                regular_cars[car.id] = {
                    'actor': car, 
                    'state': 'driving', 
                    'camera': None, 
                    'queue': None, 
                    'last_yield_time': 0.0,
                    'original_speed': None
                }
                print(f"Spawned regular car (ID: {car.id}, Type: {car.type_id}) at {spawn_point.location}")
            else:
                print(f"Failed to spawn regular car {i+1}.")

        # --- 2. YOLO 모델 로드 (use_yolo_yielding 플래그가 True일 때만) ---
        if args.use_yolo_yielding:
            print(f"Loading YOLO model from: {YOLO_MODEL_PATH}")
            try:
                yolo_model = YOLO(YOLO_MODEL_PATH)
                # 모델 클래스 이름 확인 (디버깅용)
                print("YOLO Model classes:", yolo_model.names)
                # AMBULANCE_CLASS_NAME이 모델에 있는지 확인
                class_id_found = False
                for class_id, name in yolo_model.names.items():
                    if name == AMBULANCE_CLASS_NAME:
                        class_id_found = True
                        break
                if not class_id_found:
                    print(f"Warning: Ambulance class name '{AMBULANCE_CLASS_NAME}' not found in the YOLO model's class list!")
                    print("Please check YOLO_MODEL_PATH and AMBULANCE_CLASS_NAME in your script.")
                    # return # 또는 그냥 진행하고 탐지 안되도록
            except Exception as e:
simulation_3900_traffic