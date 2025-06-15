import carla
import random
import os
import numpy as np
import cv2
import time

client = None  # 전역으로 client 선언

def clear_world(world):
    """ 맵에 존재하는 모든 차량과 보행자 제거 """
    actors = world.get_actors()
    for actor in actors:
        if 'vehicle' in actor.type_id or 'walker' in actor.type_id:
            actor.destroy()

def find_safe_spawn_point(world, ego_vehicle, min_distance=10, emergency_vehicle=None):
    """ Ego 차량 후방에서 안전한 스폰 지점 찾기 (구급차와의 최소 거리 유지) """
    spawn_points = world.get_map().get_spawn_points()
    ego_location = ego_vehicle.get_location() if ego_vehicle else carla.Location(0, 0, 0)
    safe_points = []

    # 이미 차량이 존재하는 위치를 제외하고 안전한 지점 찾기
    actors = world.get_actors()
    occupied_locations = set()  # 차량이 있는 위치를 기록
    
    for actor in actors:
        if 'vehicle' in actor.type_id:
            occupied_locations.add((actor.get_location().x, actor.get_location().y))
    
    # 안전한 스폰 지점 필터링
    for sp in spawn_points:
        if (sp.location.x, sp.location.y) not in occupied_locations and sp.location.distance(ego_location) > min_distance:
            if emergency_vehicle:
                # 기존 구급차와 Ego 차량 간의 최소 거리도 고려하여 안전한 지점 찾기
                emergency_location = emergency_vehicle.get_location()
                if sp.location.distance(emergency_location) > min_distance:
                    safe_points.append(sp)
            else:
                safe_points.append(sp)
    
    return random.choice(safe_points) if safe_points else spawn_points[0]

def setup_carla_world():
    """ Carla 서버 연결 및 월드 설정 """
    global client
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    return world

def spawn_vehicle(world, model_filter='vehicle.*', ego_vehicle=None):
    """ 차량 스폰 """
    blueprint_library = world.get_blueprint_library()

    if model_filter == 'vehicle.tesla.model3':
        vehicle_bps = [blueprint_library.find('vehicle.tesla.model3')]
    else:
        vehicle_bps = blueprint_library.filter(model_filter)
    
    if not vehicle_bps:  # 필터 결과가 비어 있으면 모든 차량 블루프린트 사용
        vehicle_bps = blueprint_library.filter('vehicle.*')
        print(f"'{model_filter}' 블루프린트 없음. 'vehicle.*'에서 랜덤 선택")
    
    vehicle_bp = random.choice(vehicle_bps)
    spawn_point = find_safe_spawn_point(world, ego_vehicle)
    
    try:
        vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        return vehicle
    except Exception as e:
        print(f"차량 스폰 실패: {e}")
        return None

def get_forward_vector(rotation):
    """ 차량의 회전 정보를 바탕으로 전방 벡터 계산 """
    rad = np.radians(rotation.yaw)
    x = np.cos(rad)
    y = np.sin(rad)
    return carla.Vector3D(x, y, 0)

def capture_images(output_dir, num_images=500, images_per_batch=100, run_id=1):
    """ 이미지 캡쳐 메인 함수 """
    world = setup_carla_world()
    camera = None
    ego_vehicle = None
    emergency_vehicle = None
    total_images_collected = 0
    image_index = 1  # 이미지 인덱스를 1부터 시작
    
    try:
        clear_world(world)  # 기존 차량 및 보행자 제거
        
        # Ego 차량의 위치를 랜덤으로 설정
        ego_vehicle = spawn_vehicle(world)
        if not ego_vehicle:
            print("Ego 차량 스폰 실패")
            return
        
        camera = attach_camera(world, ego_vehicle)
        
        def image_callback(image):
            nonlocal image_index
            save_image(image, image_index, output_dir, run_id)  # output_dir, run_id 인자 추가
            print(f"이미지 {image_index} 저장 완료")
            image_index += 1
        
        camera.listen(image_callback)
        
        while total_images_collected < num_images:
            # 100개씩 이미지를 수집 후 배경 변경
            captured_images_in_batch = 0
            while captured_images_in_batch < images_per_batch:
                # 응급차량 위치를 Ego 차량 뒤쪽에서 랜덤으로 설정
                ego_location = ego_vehicle.get_location()
                ego_rotation = ego_vehicle.get_transform().rotation

                # Ego 차량의 방향을 향하는 벡터 계산
                forward_vector = get_forward_vector(ego_rotation)
                
                # 응급차량을 Ego 차량의 뒤에 스폰 (뒤쪽으로 설정)
                distance = random.uniform(10, 30)  # 10m ~ 30m 사이의 거리
                emergency_spawn_location = ego_location - forward_vector * distance
                
                # 응급차가 좌우 차선으로 이동하도록 설정
                lane_offset = random.uniform(-5, 5)  # 좌우 차선 이동 (Ego 차량의 위치에서 좌우로 이동)
                emergency_spawn_location.x += lane_offset
                
                # 응급차 스폰 (단, 구급차가 한 번만 스폰됨)
                if not emergency_vehicle:
                    emergency_vehicle = spawn_vehicle(world, 'vehicle.tesla.model3', ego_vehicle)
                    if not emergency_vehicle:
                        print("응급차 스폰 실패")
                        continue
                    emergency_vehicle.set_transform(carla.Transform(emergency_spawn_location, carla.Rotation(yaw=ego_rotation.yaw)))
                    world.tick()

                # 구급차 위치를 좌우 차선으로 조금씩 변경
                offset_x = random.uniform(-2, 2)  # 작은 범위에서 좌우로 위치 이동
                new_emergency_location = emergency_vehicle.get_location()
                new_emergency_location.x += offset_x
                emergency_vehicle.set_location(new_emergency_location)

                # 구급차가 사이렌을 켤 수 있도록 설정
                control_emergency_vehicle_with_siren(emergency_vehicle, ego_vehicle)

                # 구급차 위치가 카메라 시점 내에 있어야 이미지를 캡쳐
                print(f"응급차 위치: {emergency_vehicle.get_location()}")
                
                # 카메라를 통해 이미지를 캡처하려면 world.tick()을 호출하여 씬을 렌더링
                world.tick()

                captured_images_in_batch += 1
                total_images_collected += 1
            
            # 배경을 변경하는 함수 호출 (예: 맵을 바꿔서 다른 환경에서 이미지 수집)
            print(f"{images_per_batch}개의 이미지를 수집했습니다. 배경을 변경합니다.")
            change_background(world)  # 배경을 바꿔주는 함수 추가
        
    except Exception as e:
        print(f"에러 발생: {e}")
    
    finally:
        if camera:
            camera.stop()
            camera.destroy()
        if emergency_vehicle:
            emergency_vehicle.destroy()
        if ego_vehicle:
            ego_vehicle.destroy()

def change_background(world):
    """ 배경을 변경하는 함수 (같은 맵 내에서 스폰 지점만 변경) """
    clear_world(world)  # 기존 차량 제거
    
    ego_vehicle = spawn_vehicle(world)
    if not ego_vehicle:
        print("새로운 Ego 차량 스폰 실패")
        return
    
    emergency_vehicle = spawn_vehicle(world, 'vehicle.tesla.model3', ego_vehicle)
    if not emergency_vehicle:
        print("새로운 응급차 스폰 실패")
        return
    
    # 새로 스폰된 차량을 기존처럼 연결하고, 위치 설정
    camera = attach_camera(world, ego_vehicle)
    
    # 구급차 위치를 새롭게 설정
    emergency_spawn_location = ego_vehicle.get_location() - get_forward_vector(ego_vehicle.get_transform().rotation) * random.uniform(10, 30)
    emergency_vehicle.set_transform(carla.Transform(emergency_spawn_location, carla.Rotation(yaw=ego_vehicle.get_transform().rotation.yaw)))
    
    control_emergency_vehicle_with_siren(emergency_vehicle, ego_vehicle)
    
    world.tick()  # 씬 업데이트
    return ego_vehicle, emergency_vehicle, camera

def attach_camera(world, ego_vehicle):
    """ 후방 카메라 부착 """
    blueprint_library = world.get_blueprint_library()
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    
    camera_bp.set_attribute('image_size_x', '1920')
    camera_bp.set_attribute('image_size_y', '1080')
    camera_bp.set_attribute('fov', '110')

    # 카메라 위치와 회전 조정 (Ego 차량의 뒤쪽으로 카메라 배치)
    camera_transform = carla.Transform(carla.Location(x=-5, z=2), carla.Rotation(pitch=0, yaw=180))  # 후방을 바라보도록 설정
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=ego_vehicle)
    return camera

def save_image(image, image_index, output_dir, run_id):
    """ 이미지 저장 (이미지 인덱스를 사용하여 이름 지정) """
    image_data = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    image_data = np.reshape(image_data, (image.height, image.width, 4))
    image_data = image_data[:, :, :3]
    
    # output_dir 경로에 이미지 저장
    os.makedirs(output_dir, exist_ok=True)
    
    # 실행 번호(run_id)를 파일명에 추가하여 고유하게 만듬
    filename = f'emergency_vehicle_{run_id}_{image_index:04d}.png'
    filepath = os.path.join(output_dir, filename)
    
    # 이미지 저장
    cv2.imwrite(filepath, image_data)

def control_emergency_vehicle_with_siren(emergency_vehicle, ego_vehicle, min_distance=3, lane_offset_range=(-3, 3), fixed_distance=13):
    """ 구급차량이 Ego 차량을 추적하도록 제어하며, 사이렌을 켬. 
        최소 3m 거리 유지 및 차선 하나 옆으로 이동, Ego 차량의 뒤에서 10m로 고정 """
    
    control = carla.VehicleControl()
    
    # Ego 차량의 위치와 방향을 가져옴
    ego_location = ego_vehicle.get_location()
    ego_rotation = ego_vehicle.get_transform().rotation  # Ego 차량의 회전 정보를 가져옴
    
    # 구급차의 위치를 계속 추적하면서 최소 거리 3m 이상 유지하도록 제어
    emergency_location = emergency_vehicle.get_location()
    direction_vector = ego_location - emergency_location
    distance = np.linalg.norm([direction_vector.x, direction_vector.y])  # 거리 계산

    # 최소 거리 이상이 되도록 위치를 조정
    if distance < min_distance:
        # 구급차가 너무 가까워지면 속도를 줄여서 거리를 유지
        control.throttle = 0.2
    else:
        control.throttle = 0.5  # 정상 속도

    control.steer = 0.0  # 항상 직진하도록 조정
    
    # Ego 차량의 뒤쪽으로 10m 위치 계산
    forward_vector = get_forward_vector(ego_rotation)
    fixed_location = ego_location - forward_vector * fixed_distance  # Ego 차량의 뒤쪽으로 10m 떨어진 위치

    # 차선 하나 정도 좌우로 이동하도록 설정 (랜덤으로 차선 내에서 이동)
    lane_offset = random.uniform(*lane_offset_range)  # 차선 이동 범위 (-5m ~ 5m)
    fixed_location.x += lane_offset  # 좌우 차선 이동
    
    # 구급차의 위치를 계산된 위치로 설정
    emergency_vehicle.set_location(fixed_location)
    
    # 구급차의 회전값을 Ego 차량의 회전값과 동일하게 맞추기
    emergency_vehicle.set_transform(carla.Transform(fixed_location, carla.Rotation(yaw=ego_rotation.yaw)))
    
    # 사이렌 켜기 (Position | Special1 -> 사이렌 켜는 설정)
    light_state = carla.VehicleLightState.Position | carla.VehicleLightState.Special1
    emergency_vehicle.set_light_state(carla.VehicleLightState(light_state))
    
    emergency_vehicle.apply_control(control)



# 실행
if __name__ == '__main__':
    for i in range(40):
        run_id = i + 1
        print(f"generate_data.py 실행 {run_id}/5")
        capture_images(num_images=100, images_per_batch=100, output_dir='./project/data/tesla_vehicle_dataset', run_id=run_id)
        print(f"{run_id}번째 실행 완료")

# capture_images(num_images=100, images_per_batch=100, output_dir='./emergency_vehicle_dataset', run_id=2)

# python project/generate_data.py


# 7~106까지만 살리고 나머지 지우기