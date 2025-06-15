from ultralytics import YOLO

# 사전 학습된 YOLOv8s 모델 로드
model = YOLO('best.pt')

# 데이터셋 설정
data_yaml = """
path: D:/kr_ambulance  # 데이터셋 루트 디렉토리
train: train/images/M14  # 학습 이미지 경로
val: val/images/M14  # 검증 이미지 경로

# 클래스 설정
names:
  0: ambulance
"""

# YAML 파일 저장
with open('ambulance_data.yaml', 'w') as f:
    f.write(data_yaml)

# 모델 학습
results = model.train(
    data='ambulance_data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    patience=50,
    save=True,
    device='0'  # GPU 사용 (CPU 사용시 'cpu'로 변경)
) 

# python train_ambulance.py