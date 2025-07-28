import os
import cv2
import numpy as np
import time
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from PIL import Image
import uuid

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['PROCESSED_FOLDER'] = 'processed'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# 허용된 파일 확장자
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

# 업로드 폴더 생성
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def perspective_crop(image_path, corners):
    """
    포토부스 사진을 perspective crop하는 함수
    corners: 4개의 코너점 좌표 (정규화된 값 0~1)
    """
    # 이미지 로드
    image = cv2.imread(image_path)
    if image is None:
        return None
    
    h, w = image.shape[:2]
    
    # 정규화된 좌표를 실제 픽셀 좌표로 변환
    pixel_corners = []
    for corner in corners:
        x = corner[0] * w
        y = corner[1] * h
        pixel_corners.append([x, y])
    
    # 코너점을 numpy 배열로 변환
    src_points = np.array(pixel_corners, dtype=np.float32)
    
    # 코너점 순서 정렬 (시계방향: 좌상단, 우상단, 우하단, 좌하단)
    # 각 코너점의 위치를 기준으로 정렬
    center = np.mean(src_points, axis=0)
    
    def get_angle(point):
        return np.arctan2(point[1] - center[1], point[0] - center[0])
    
    # 각도에 따라 정렬 (시계방향)
    angles = [get_angle(point) for point in src_points]
    sorted_indices = np.argsort(angles)
    src_points = src_points[sorted_indices]
    
    # 목표 크기 계산 (사각형의 최대 너비와 높이)
    # 대각선 길이를 계산하여 적절한 크기 결정
    width1 = np.linalg.norm(src_points[1] - src_points[0])
    width2 = np.linalg.norm(src_points[2] - src_points[3])
    height1 = np.linalg.norm(src_points[3] - src_points[0])
    height2 = np.linalg.norm(src_points[2] - src_points[1])
    
    # 평균값 사용으로 더 안정적인 결과
    target_width = int((width1 + width2) / 2)
    target_height = int((height1 + height2) / 2)
    
    # 최소 크기 보장
    target_width = max(target_width, 100)
    target_height = max(target_height, 100)
    
    # 목표 좌표 설정 (정사각형 또는 원본 비율 유지)
    dst_points = np.array([
        [0, 0],
        [target_width, 0],
        [target_width, target_height],
        [0, target_height]
    ], dtype=np.float32)
    
    # perspective transform matrix 계산
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    
    # perspective transform 적용
    result = cv2.warpPerspective(image, matrix, (target_width, target_height))
    
    return result

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': '파일이 선택되지 않았습니다.'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '파일이 선택되지 않았습니다.'}), 400
    
    if file and allowed_file(file.filename):
        # 고유한 파일명 생성
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        
        # 파일 저장
        file.save(filepath)
        
        return jsonify({
            'success': True,
            'filename': unique_filename,
            'message': '파일이 성공적으로 업로드되었습니다.'
        })
    
    return jsonify({'error': '허용되지 않는 파일 형식입니다.'}), 400

@app.route('/crop', methods=['POST'])
def crop_photo():
    data = request.get_json()
    
    if not data or 'filename' not in data or 'corners' not in data:
        return jsonify({'error': '필수 데이터가 누락되었습니다.'}), 400
    
    filename = data['filename']
    corners = data['corners']
    
    # 파일 경로 확인
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(filepath):
        return jsonify({'error': '파일을 찾을 수 없습니다.'}), 404
    
    try:
        # perspective crop 수행
        cropped_image = perspective_crop(filepath, corners)
        
        if cropped_image is None:
            return jsonify({'error': '이미지 처리 중 오류가 발생했습니다.'}), 500
        
        # 결과 이미지 저장 (매번 새로운 파일명 생성)
        timestamp = str(int(time.time()))
        output_filename = f"cropped_{timestamp}_{filename}"
        output_path = os.path.join(app.config['PROCESSED_FOLDER'], output_filename)
        cv2.imwrite(output_path, cropped_image)
        
        return jsonify({
            'success': True,
            'cropped_filename': output_filename,
            'message': '이미지가 성공적으로 크롭되었습니다.'
        })
        
    except Exception as e:
        return jsonify({'error': f'이미지 처리 중 오류가 발생했습니다: {str(e)}'}), 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/processed/<filename>')
def processed_file(filename):
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename)

@app.route('/image-info/<filename>')
def get_image_info(filename):
    """이미지 정보를 반환하는 API"""
    filepath = os.path.join(app.config['PROCESSED_FOLDER'], filename)
    
    if not os.path.exists(filepath):
        return jsonify({'error': '파일을 찾을 수 없습니다.'}), 404
    
    try:
        # 이미지 크기 정보 가져오기
        image = cv2.imread(filepath)
        if image is None:
            return jsonify({'error': '이미지를 읽을 수 없습니다.'}), 500
        
        height, width = image.shape[:2]
        file_size = os.path.getsize(filepath)
        
        return jsonify({
            'width': width,
            'height': height,
            'file_size': file_size,
            'aspect_ratio': f"{width}:{height}",
            'filename': filename
        })
        
    except Exception as e:
        return jsonify({'error': f'이미지 정보를 가져오는 중 오류가 발생했습니다: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5002) 