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

def detect_edges_and_corners(image_path):
    """
    엣지 디텍팅을 이용해 이미지의 4개 코너점을 자동으로 검출하는 함수 (기본 버전)
    """
    # 이미지 로드
    image = cv2.imread(image_path)
    if image is None:
        return None
    
    # 이미지 크기
    h, w = image.shape[:2]
    
    # 그레이스케일 변환
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 기본 전처리 방법들 시도
    preprocess_methods = [
        # 1. 가우시안 블러 + Canny
        lambda img: cv2.Canny(cv2.GaussianBlur(img, (5, 5), 0), 50, 150),
        
        # 2. 양방향 필터 + Canny
        lambda img: cv2.Canny(cv2.bilateralFilter(img, 9, 75, 75), 30, 100),
        
        # 3. 적응형 히스토그램 평활화 + Canny
        lambda img: cv2.Canny(cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(img), 40, 120),
        
        # 4. 노이즈 제거 + Canny
        lambda img: cv2.Canny(cv2.fastNlMeansDenoising(img), 60, 180),
    ]
    
    for method in preprocess_methods:
        try:
            edges = method(gray)
            
            # 모폴로지 연산으로 엣지 강화
            kernel = np.ones((3, 3), np.uint8)
            enhanced_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
            
            # 윤곽선 찾기
            contours, _ = cv2.findContours(enhanced_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                continue
            
            # 면적 기준으로 윤곽선 필터링
            min_area = (w * h) * 0.005  # 이미지의 0.5% 이상
            max_area = (w * h) * 0.98   # 이미지의 98% 이하
            
            valid_contours = [cnt for cnt in contours if min_area < cv2.contourArea(cnt) < max_area]
            
            if not valid_contours:
                continue
            
            # 가장 큰 윤곽선 선택
            largest_contour = max(valid_contours, key=cv2.contourArea)
            
            # 윤곽선 단순화 (다양한 epsilon 값 시도)
            epsilon_values = [0.02, 0.05, 0.1]
            
            for epsilon_val in epsilon_values:
                epsilon = epsilon_val * cv2.arcLength(largest_contour, True)
                approx = cv2.approxPolyDP(largest_contour, epsilon, True)
                
                corners = process_approximated_contour(approx, largest_contour, w, h)
                if corners is not None:
                    return corners
            
            # 모든 epsilon 값이 실패하면 외접 사각형 사용
            rect = cv2.minAreaRect(largest_contour)
            box = cv2.boxPoints(rect)
            corners = np.int0(box)
            
            # 정규화된 좌표로 변환
            normalized_corners = []
            for corner in corners:
                x = max(0, min(1, corner[0] / w))
                y = max(0, min(1, corner[1] / h))
                normalized_corners.append([x, y])
            
            # 코너점 순서 정렬 (좌상단부터 시계방향)
            normalized_corners = sort_corners_clockwise(normalized_corners)
            
            # 코너점이 유효한지 확인
            if validate_corner_points(normalized_corners):
                return normalized_corners
                
        except Exception as e:
            continue
    
    # 모든 방법이 실패하면 기본 사각형 반환 (좌상단부터 시계방향)
    return [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]]

def detect_photo_booth_corners(image_path):
    """
    포토부스 사진의 테두리를 더 정확하게 검출하는 고급 함수
    """
    # 이미지 로드
    image = cv2.imread(image_path)
    if image is None:
        return None
    
    # 이미지 크기
    h, w = image.shape[:2]
    
    # 그레이스케일 변환
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 여러 전처리 방법 시도
    preprocessed_images = []
    
    # 1. 기본 전처리
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    blurred = cv2.bilateralFilter(enhanced, 9, 75, 75)
    preprocessed_images.append(blurred)
    
    # 2. 가우시안 블러 + 적응형 임계값
    gaussian_blur = cv2.GaussianBlur(gray, (5, 5), 0)
    adaptive_thresh = cv2.adaptiveThreshold(gaussian_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    preprocessed_images.append(adaptive_thresh)
    
    # 3. 노이즈 제거 + 샤프닝
    denoised = cv2.fastNlMeansDenoising(gray)
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(denoised, -1, kernel)
    preprocessed_images.append(sharpened)
    
    # 4. 히스토그램 평활화
    hist_eq = cv2.equalizeHist(gray)
    preprocessed_images.append(hist_eq)
    
    # 각 전처리 방법으로 코너점 검출 시도
    for i, processed_img in enumerate(preprocessed_images):
        corners = try_detect_corners(processed_img, w, h, method=f"preprocess_{i}")
        if corners is not None:
            return corners
    
    # 모든 전처리 방법이 실패하면 원본 이미지로 시도
    return try_detect_corners(gray, w, h, method="original")

def try_detect_corners(gray_image, width, height, method="default"):
    """
    다양한 엣지 디텍션 방법으로 코너점 검출을 시도하는 함수
    """
    # 여러 엣지 디텍션 방법 시도
    edge_methods = [
        # Canny 엣지 디텍션 (다양한 임계값)
        lambda img: cv2.Canny(img, 30, 200),
        lambda img: cv2.Canny(img, 50, 150),
        lambda img: cv2.Canny(img, 20, 100),
        lambda img: cv2.Canny(img, 100, 300),
        
        # Sobel 엣지 디텍션
        lambda img: np.uint8(np.absolute(cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)) + np.absolute(cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3))),
        
        # Laplacian 엣지 디텍션
        lambda img: cv2.Laplacian(img, cv2.CV_8U, ksize=3),
    ]
    
    for edge_method in edge_methods:
        try:
            edges = edge_method(gray_image)
            
            # 모폴로지 연산으로 엣지 강화
            kernel_sizes = [3, 5, 7]
            for kernel_size in kernel_sizes:
                kernel = np.ones((kernel_size, kernel_size), np.uint8)
                
                # 다양한 모폴로지 연산 시도
                morph_operations = [
                    lambda e, k: cv2.morphologyEx(e, cv2.MORPH_CLOSE, k),
                    lambda e, k: cv2.dilate(e, k, iterations=1),
                    lambda e, k: cv2.morphologyEx(e, cv2.MORPH_CLOSE, k) + cv2.dilate(e, k, iterations=1),
                    lambda e, k: cv2.morphologyEx(e, cv2.MORPH_GRADIENT, k),
                ]
                
                for morph_op in morph_operations:
                    enhanced_edges = morph_op(edges, kernel)
                    
                    # 윤곽선 찾기
                    contours, _ = cv2.findContours(enhanced_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    if not contours:
                        continue
                    
                    # 면적 기준으로 윤곽선 필터링
                    min_area = (width * height) * 0.01  # 이미지의 1% 이상
                    max_area = (width * height) * 0.95  # 이미지의 95% 이하
                    
                    valid_contours = [cnt for cnt in contours if min_area < cv2.contourArea(cnt) < max_area]
                    
                    if not valid_contours:
                        continue
                    
                    # 가장 큰 윤곽선 선택
                    largest_contour = max(valid_contours, key=cv2.contourArea)
                    
                    # 윤곽선 단순화 (다양한 epsilon 값 시도)
                    epsilon_values = [0.01, 0.02, 0.05, 0.1]
                    
                    for epsilon_val in epsilon_values:
                        epsilon = epsilon_val * cv2.arcLength(largest_contour, True)
                        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
                        
                        corners = process_approximated_contour(approx, largest_contour, width, height)
                        if corners is not None:
                            return corners
                    
                    # 윤곽선이 너무 복잡한 경우, 외접 사각형 사용
                    rect = cv2.minAreaRect(largest_contour)
                    box = cv2.boxPoints(rect)
                    corners = np.int0(box)
                    
                    # 정규화된 좌표로 변환
                    normalized_corners = []
                    for corner in corners:
                        x = max(0, min(1, corner[0] / width))
                        y = max(0, min(1, corner[1] / height))
                        normalized_corners.append([x, y])
                    
                    # 코너점이 유효한지 확인
                    if validate_corner_points(normalized_corners):
                        return normalized_corners
                        
        except Exception as e:
            continue
    
    return None

def process_approximated_contour(approx, original_contour, width, height):
    """
    근사화된 윤곽선을 처리하여 코너점을 추출하는 함수 (4개 제약 제거)
    """
    if len(approx) >= 3:  # 최소 3개 이상의 꼭지점이 있으면 처리
        corners = approx.reshape(-1, 2)
        
        # 너무 많은 점이면 중요한 점들만 선택
        if len(corners) > 8:
            # 중심점으로부터 가장 먼 점들을 선택 (최대 8개)
            center = np.mean(corners, axis=0)
            distances = [np.linalg.norm(corner - center) for corner in corners]
            indices = np.argsort(distances)[-8:]
            corners = corners[indices]
            
    elif len(approx) < 3:
        # 3개 미만이면 외접 사각형 사용
        rect = cv2.minAreaRect(original_contour)
        box = cv2.boxPoints(rect)
        corners = np.int0(box)
    else:
        return None
    
    # 정규화된 좌표로 변환
    normalized_corners = []
    for corner in corners:
        x = max(0, min(1, corner[0] / width))
        y = max(0, min(1, corner[1] / height))
        normalized_corners.append([x, y])
    
    # 코너점 순서 정렬 (좌상단부터 시계방향)
    normalized_corners = sort_corners_clockwise(normalized_corners)
    
    # 코너점이 유효한지 확인 (4개 제약 제거)
    if validate_corner_points_flexible(normalized_corners):
        return normalized_corners
    
    return None

def sort_corners_clockwise(corners):
    """
    코너점들을 좌상단부터 시계방향으로 정렬하는 함수
    """
    if len(corners) < 3:
        return corners
    
    # 중심점 계산
    center_x = sum(corner[0] for corner in corners) / len(corners)
    center_y = sum(corner[1] for corner in corners) / len(corners)
    
    # 각 코너점의 각도 계산 (좌상단이 0도, 시계방향)
    def get_angle(corner):
        # 좌상단을 기준으로 각도 계산
        dx = corner[0] - center_x
        dy = corner[1] - center_y
        angle = np.arctan2(dy, dx)
        
        # 좌상단(-135도)을 0도로 변환
        # 좌상단은 dx < 0, dy < 0 이므로 -135도
        angle = angle + 3*np.pi/4  # -135도를 0도로 변환
        
        # 음수 각도를 양수로 변환
        if angle < 0:
            angle += 2 * np.pi
            
        return angle
    
    # 각도에 따라 정렬
    sorted_corners = sorted(corners, key=get_angle)
    
    return sorted_corners

def validate_corner_points(corners):
    """
    검출된 코너점들이 유효한지 검증하는 함수 (4개 제약)
    """
    if len(corners) != 4:
        return False
    
    # 모든 좌표가 0~1 범위 내에 있는지 확인
    for corner in corners:
        if not (0 <= corner[0] <= 1 and 0 <= corner[1] <= 1):
            return False
    
    # 사각형의 최소 크기 확인 (너무 작으면 무시)
    corners_array = np.array(corners)
    
    # 대각선 길이 계산
    diagonal1 = np.linalg.norm(corners_array[0] - corners_array[2])
    diagonal2 = np.linalg.norm(corners_array[1] - corners_array[3])
    
    # 최소 대각선 길이 (이미지 대각선의 20% 이상)
    min_diagonal = 0.2 * np.sqrt(2)
    
    if diagonal1 < min_diagonal or diagonal2 < min_diagonal:
        return False
    
    # 코너점들이 서로 너무 가깝지 않은지 확인
    for i in range(4):
        for j in range(i + 1, 4):
            distance = np.linalg.norm(corners_array[i] - corners_array[j])
            if distance < 0.05:  # 5% 이내면 너무 가까움
                return False
    
    return True

def validate_corner_points_flexible(corners):
    """
    검출된 코너점들이 유효한지 검증하는 함수 (유연한 개수)
    """
    if len(corners) < 3:  # 최소 3개 이상
        return False
    
    # 모든 좌표가 0~1 범위 내에 있는지 확인
    for corner in corners:
        if not (0 <= corner[0] <= 1 and 0 <= corner[1] <= 1):
            return False
    
    # 코너점들이 서로 너무 가깝지 않은지 확인
    corners_array = np.array(corners)
    for i in range(len(corners)):
        for j in range(i + 1, len(corners)):
            distance = np.linalg.norm(corners_array[i] - corners_array[j])
            if distance < 0.03:  # 3% 이내면 너무 가까움
                return False
    
    # 영역의 최소 크기 확인
    if len(corners) >= 3:
        # 볼록 껍질(Convex Hull) 계산
        try:
            from scipy.spatial import ConvexHull
            hull = ConvexHull(corners_array)
            hull_area = hull.volume  # 2D에서는 volume이 면적
            min_area = 0.01  # 최소 1% 면적
            if hull_area < min_area:
                return False
        except ImportError:
            # scipy가 없으면 간단한 면적 계산
            if len(corners) == 3:
                # 삼각형 면적
                area = abs((corners[1][0] - corners[0][0]) * (corners[2][1] - corners[0][1]) - 
                          (corners[2][0] - corners[0][0]) * (corners[1][1] - corners[0][1])) / 2
            else:
                # 다각형 면적 (간단한 근사)
                area = 0.1  # 기본값
            if area < 0.005:
                return False
    
    return True

def perspective_crop(image_path, corners):
    """
    포토부스 사진을 perspective crop하는 함수 (유연한 개수 지원)
    corners: 코너점 좌표 리스트 (정규화된 값 0~1)
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
    
    # 코너점 개수에 따른 처리
    if len(corners) == 4:
        # 4개 점인 경우 기존 방식 사용
        result = crop_with_4_points(src_points, image)
    elif len(corners) == 3:
        # 3개 점인 경우 삼각형 영역 크롭
        result = crop_with_3_points(src_points, image)
    else:
        # 5개 이상인 경우 볼록 껍질 사용
        result = crop_with_multiple_points(src_points, image)
    
    return result

def crop_with_4_points(src_points, image):
    """4개 점으로 perspective crop"""
    # 코너점 순서 정렬 (시계방향)
    center = np.mean(src_points, axis=0)
    
    def get_angle(point):
        return np.arctan2(point[1] - center[1], point[0] - center[0])
    
    angles = [get_angle(point) for point in src_points]
    sorted_indices = np.argsort(angles)
    src_points = src_points[sorted_indices]
    
    # 목표 크기 계산
    width1 = np.linalg.norm(src_points[1] - src_points[0])
    width2 = np.linalg.norm(src_points[2] - src_points[3])
    height1 = np.linalg.norm(src_points[3] - src_points[0])
    height2 = np.linalg.norm(src_points[2] - src_points[1])
    
    target_width = int((width1 + width2) / 2)
    target_height = int((height1 + height2) / 2)
    
    # 최소 크기 보장
    target_width = max(target_width, 100)
    target_height = max(target_height, 100)
    
    # 목표 좌표 설정
    dst_points = np.array([
        [0, 0],
        [target_width, 0],
        [target_width, target_height],
        [0, target_height]
    ], dtype=np.float32)
    
    # perspective transform
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    result = cv2.warpPerspective(image, matrix, (target_width, target_height))
    
    return result

def crop_with_3_points(src_points, image):
    """3개 점으로 삼각형 영역 크롭"""
    # 삼각형의 경계 상자 계산
    min_x = int(min(src_points[:, 0]))
    max_x = int(max(src_points[:, 0]))
    min_y = int(min(src_points[:, 1]))
    max_y = int(max(src_points[:, 1]))
    
    # 경계 상자로 크롭
    cropped = image[min_y:max_y, min_x:max_x]
    
    # 삼각형 마스크 생성
    mask = np.zeros((max_y - min_y, max_x - min_x), dtype=np.uint8)
    triangle_points = src_points - np.array([min_x, min_y])
    cv2.fillPoly(mask, [triangle_points.astype(np.int32)], 255)
    
    # 마스크 적용
    result = cv2.bitwise_and(cropped, cropped, mask=mask)
    
    return result

def crop_with_multiple_points(src_points, image):
    """여러 점으로 다각형 영역 크롭"""
    try:
        from scipy.spatial import ConvexHull
        # 볼록 껍질 계산
        hull = ConvexHull(src_points)
        hull_points = src_points[hull.vertices]
        
        # 경계 상자 계산
        min_x = int(min(hull_points[:, 0]))
        max_x = int(max(hull_points[:, 0]))
        min_y = int(min(hull_points[:, 1]))
        max_y = int(max(hull_points[:, 1]))
        
        # 경계 상자로 크롭
        cropped = image[min_y:max_y, min_x:max_x]
        
        # 다각형 마스크 생성
        mask = np.zeros((max_y - min_y, max_x - min_x), dtype=np.uint8)
        polygon_points = hull_points - np.array([min_x, min_y])
        cv2.fillPoly(mask, [polygon_points.astype(np.int32)], 255)
        
        # 마스크 적용
        result = cv2.bitwise_and(cropped, cropped, mask=mask)
        
        return result
        
    except ImportError:
        # scipy가 없으면 경계 상자만 사용
        min_x = int(min(src_points[:, 0]))
        max_x = int(max(src_points[:, 0]))
        min_y = int(min(src_points[:, 1]))
        max_y = int(max(src_points[:, 1]))
        
        return image[min_y:max_y, min_x:max_x]

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

@app.route('/detect-corners', methods=['POST'])
def detect_corners():
    """엣지 디텍팅을 이용해 자동으로 코너점을 검출하는 API"""
    data = request.get_json()
    
    if not data or 'filename' not in data:
        return jsonify({'error': '파일명이 누락되었습니다.'}), 400
    
    filename = data['filename']
    detection_method = data.get('method', 'advanced')  # 'basic' or 'advanced'
    
    # 파일 경로 확인
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(filepath):
        return jsonify({'error': '파일을 찾을 수 없습니다.'}), 404
    
    try:
        # 선택된 방법에 따라 코너점 검출
        if detection_method == 'basic':
            corners = detect_edges_and_corners(filepath)
        else:
            corners = detect_photo_booth_corners(filepath)
        
        if corners is None:
            return jsonify({'error': '코너점을 검출할 수 없습니다. 다른 이미지를 시도해보세요.'}), 400
        
        return jsonify({
            'success': True,
            'corners': corners,
            'message': f'{detection_method} 방법으로 코너점이 검출되었습니다.'
        })
        
    except Exception as e:
        return jsonify({'error': f'코너점 검출 중 오류가 발생했습니다: {str(e)}'}), 500

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