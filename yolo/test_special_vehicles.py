from ultralytics import YOLO
import os
from pathlib import Path
import cv2
import numpy as np
import json
from collections import defaultdict
from datetime import datetime

def calculate_iou(box1, box2):
    # box1, box2: [x1, y1, x2, y2]
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - intersection
    
    return intersection / union if union > 0 else 0

def convert_yolo_to_xyxy(box, img_width, img_height):
    # YOLO format: [x_center, y_center, width, height] (normalized)
    x_center, y_center, width, height = box
    x1 = (x_center - width/2) * img_width
    y1 = (y_center - height/2) * img_height
    x2 = (x_center + width/2) * img_width
    y2 = (y_center + height/2) * img_height
    return [x1, y1, x2, y2]

def convert_gt_to_xyxy(gt_box, scale_x, scale_y):
    try:
        if 'coords' in gt_box:
            coords = gt_box['coords']
            x1 = coords['tl']['x'] * scale_x
            y1 = coords['tl']['y'] * scale_y
            x2 = coords['br']['x'] * scale_x
            y2 = coords['br']['y'] * scale_y
            return [x1, y1, x2, y2]
        elif all(k in gt_box for k in ['left', 'top', 'width', 'height']):
            x1 = gt_box['left'] * scale_x
            y1 = gt_box['top'] * scale_y
            x2 = (gt_box['left'] + gt_box['width']) * scale_x
            y2 = (gt_box['top'] + gt_box['height']) * scale_y
            return [x1, y1, x2, y2]
        else:
            print(f"Warning: Unknown bbox format in {gt_box}")
            return None
    except Exception as e:
        print(f"Error converting bbox: {e}")
        print(f"Problematic bbox: {gt_box}")
        return None

# YOLOv8 모델 로드
model = YOLO(r'D:\last.pt')

# 모델의 클래스 목록 출력
print('\n=== YOLOv8 모델 클래스 목록 ===')
for class_id, class_name in model.names.items():
    print(f'클래스 ID {class_id}: {class_name}')
print('==============================\n')

# 테스트할 이미지와 라벨 폴더 경로
image_dir = r'D:\kr_ambulance\val\images\M14'
label_dir = r'D:\kr_ambulance\val\labels\M14'

# 결과를 저장할 폴더 생성
results_dir = 'test_results'
os.makedirs(results_dir, exist_ok=True)

# 성능 측정을 위한 변수들
iou_threshold = 0.5
conf_threshold = 0.5
total_images = 0
total_correct = 0
total_predictions = 0
total_gt = 0
ambulance_metrics = {'tp': 0, 'fp': 0, 'fn': 0}

# 감지된 이미지 파일명을 저장할 리스트
detected_images = []

# 모든 jpg 파일 찾기
image_files = []
for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.lower().endswith('.jpg'):
            image_files.append(os.path.join(root, file))

print(f'총 {len(image_files)}개의 이미지 파일을 찾았습니다.')

# 각 이미지에 대해 예측 수행
for img_path in image_files:
    # 이미지 파일명 가져오기
    img_name = os.path.splitext(os.path.basename(img_path))[0]
    label_path = os.path.join(label_dir, f'{img_name}.txt')
    
    # 원본 이미지 로드 (BGR)
    img = cv2.imread(img_path)
    orig_height, orig_width = img.shape[:2]

    # RGB 변환 및 640x640 리사이즈
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (640, 640))
    scale_x = 640 / orig_width
    scale_y = 640 / orig_height
    
    # 라벨 파일 읽기
    gt_boxes = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                class_id, x_center, y_center, width, height = map(float, line.strip().split())
                # YOLO 형식을 xyxy 형식으로 변환 (원본 이미지 크기 기준)
                x1 = (x_center - width/2) * orig_width
                y1 = (y_center - height/2) * orig_height
                x2 = (x_center + width/2) * orig_width
                y2 = (y_center + height/2) * orig_height
                gt_boxes.append([x1, y1, x2, y2])
                print(f"라벨 좌표 변환: x1={x1:.1f}, y1={y1:.1f}, x2={x2:.1f}, y2={y2:.1f}")
    else:
        print(f'라벨 파일을 찾을 수 없습니다: {label_path}')
    
    # YOLOv8 예측 (RGB 640x640)
    results = model(img_resized, classes=[0])  # ambulance 클래스만 탐지
    
    # 결과 시각화
    for r in results:
        im_array = r.plot()  # 예측 결과가 표시된 이미지 배열
        im = cv2.cvtColor(im_array, cv2.COLOR_RGB2BGR)
        
        # 결과 이미지 저장
        save_path = os.path.join(results_dir, f'result_{os.path.basename(img_path)}')
        cv2.imwrite(save_path, im)
        
        # 성능 측정
        total_images += 1
        total_gt += len(gt_boxes)
        
        # 예측된 박스들을 신뢰도 기준으로 정렬
        pred_boxes = []
        for box in r.boxes:
            if float(box.conf[0]) >= conf_threshold:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                # 640x640 크기의 좌표를 원본 이미지 크기로 변환
                x_center, y_center, width, height = box.xywh[0].tolist()
                # 640x640 크기에서의 좌표를 원본 크기로 변환
                x1 = (x_center - width/2) / scale_x
                y1 = (y_center - height/2) / scale_y
                x2 = (x_center + width/2) / scale_x
                y2 = (y_center + height/2) / scale_y
                pred_boxes.append((cls, conf, [x1, y1, x2, y2]))
                print(f"예측 좌표 변환: x1={x1:.1f}, y1={y1:.1f}, x2={x2:.1f}, y2={y2:.1f}")
                print(f"원본 좌표: x_center={x_center:.3f}, y_center={y_center:.3f}, width={width:.3f}, height={height:.3f}")
                print(f"스케일: scale_x={scale_x:.3f}, scale_y={scale_y:.3f}")
        
        pred_boxes.sort(key=lambda x: x[1], reverse=True)
        total_predictions += len(pred_boxes)
        
        # 디버깅 정보 출력
        print(f"\n이미지: {img_name}")
        print(f"예측된 박스 수: {len(pred_boxes)}")
        print(f"실제 라벨 박스 수: {len(gt_boxes)}")
        
        # 매칭 수행
        matched_gt = set()
        for pred_cls, pred_conf, pred_box in pred_boxes:
            best_iou = 0
            best_gt_idx = -1
            
            for gt_idx, gt_box in enumerate(gt_boxes):
                if gt_idx in matched_gt:
                    continue
                
                iou = calculate_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            if best_iou >= iou_threshold:
                matched_gt.add(best_gt_idx)
                total_correct += 1
                ambulance_metrics['tp'] += 1
                print(f"매칭 성공 - IoU: {best_iou:.3f}, 신뢰도: {pred_conf:.3f}")
            else:
                ambulance_metrics['fp'] += 1
                print(f"매칭 실패 - 최고 IoU: {best_iou:.3f}, 신뢰도: {pred_conf:.3f}")
        
        # False Negative 계산
        for gt_idx in range(len(gt_boxes)):
            if gt_idx not in matched_gt:
                ambulance_metrics['fn'] += 1
                print(f"False Negative - 라벨 {gt_idx}")
        
        # 감지된 이미지 파일명 저장
        if len(pred_boxes) > 0:
            detected_images.append(os.path.basename(img_path))

# 성능 지표 계산 및 출력
print('\n=== Ambulance 클래스 성능 평가 결과 ===')
print(f'총 이미지 수: {total_images}')
print(f'총 Ground Truth 수: {total_gt}')
print(f'총 예측 수: {total_predictions}')
print(f'정확한 예측 수: {total_correct}')

# Ambulance 클래스 성능 지표
tp = ambulance_metrics['tp']
fp = ambulance_metrics['fp']
fn = ambulance_metrics['fn']

precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print(f'\nAmbulance 클래스:')
print(f'Precision: {precision:.3f}')
print(f'Recall: {recall:.3f}')
print(f'F1-Score: {f1_score:.3f}')

# 결과를 txt 파일로 저장
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
result_txt_path = os.path.join(results_dir, f'test_results_{timestamp}.txt')

with open(result_txt_path, 'w', encoding='utf-8') as f:
    f.write('=== Ambulance 클래스 성능 평가 결과 ===\n')
    f.write(f'테스트 시간: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n\n')
    f.write(f'총 이미지 수: {total_images}\n')
    f.write(f'총 Ground Truth 수: {total_gt}\n')
    f.write(f'총 예측 수: {total_predictions}\n')
    f.write(f'정확한 예측 수: {total_correct}\n\n')
    f.write('Ambulance 클래스:\n')
    f.write(f'Precision: {precision:.3f}\n')
    f.write(f'Recall: {recall:.3f}\n')
    f.write(f'F1-Score: {f1_score:.3f}\n\n')
    f.write(f'True Positives: {tp}\n')
    f.write(f'False Positives: {fp}\n')
    f.write(f'False Negatives: {fn}\n\n')
    f.write(f'테스트 설정:\n')
    f.write(f'IoU 임계값: {iou_threshold}\n')
    f.write(f'신뢰도 임계값: {conf_threshold}\n')
    f.write(f'이미지 크기: 640x640\n\n')
    
    # 감지된 이미지 파일명 출력
    f.write('=== 감지된 이미지 목록 ===\n')
    for img_name in detected_images:
        f.write(f'{img_name}\n')

print(f'\n테스트가 완료되었습니다.')
print(f'결과 이미지는 test_results 폴더에서 확인할 수 있습니다.')
print(f'상세 결과는 {result_txt_path}에서 확인할 수 있습니다.')

# python test_special_vehicles.py