# 📸 포토부스 사진 크롭 도구

Python, OpenCV, Flask를 활용한 웹 애플리케이션으로, 갤러리에서 사진을 업로드하여 포토부스 사진을 perspective crop할 수 있는 도구입니다.

## 🚀 주요 기능

- **갤러리 사진 업로드**: 드래그 앤 드롭 또는 파일 선택으로 이미지 업로드
- **Interactive Corner Selection**: 4개의 모서리 점을 드래그하여 포토부스 영역 선택
- **Perspective Crop**: OpenCV를 사용한 정확한 perspective transform
- **실시간 미리보기**: 크롭 결과를 즉시 확인
- **이미지 다운로드**: 처리된 이미지를 로컬에 저장

## 📋 요구사항

- Python 3.7+
- OpenCV
- Flask
- NumPy
- Pillow

## 🛠️ 설치 및 실행

### 1. 의존성 설치

```bash
pip install -r requirements.txt
```

### 2. 애플리케이션 실행

```bash
python app.py
```

### 3. 웹 브라우저에서 접속

```
http://localhost:5001
```

## 📖 사용 방법

1. **이미지 업로드**
   - "갤러리에서 사진 선택하기" 버튼 클릭
   - 또는 이미지를 드래그하여 업로드 영역에 드롭

2. **포토부스 영역 선택**
   - 업로드된 이미지에서 4개의 빨간 점이 나타남
   - 각 점을 드래그하여 포토부스의 모서리와 정확히 일치시킴

3. **크롭 실행**
   - "크롭하기" 버튼 클릭
   - 처리된 이미지가 하단에 표시됨

4. **결과 다운로드**
   - "이미지 다운로드" 버튼으로 최종 결과 저장

## 🏗️ 프로젝트 구조

```
python/
├── app.py                 # Flask 메인 애플리케이션
├── requirements.txt       # Python 의존성
├── templates/
│   └── index.html        # 웹 인터페이스
├── uploads/              # 업로드된 이미지 저장소
├── processed/            # 처리된 이미지 저장소
└── README.md            # 프로젝트 문서
```

## 🔧 기술 스택

- **Backend**: Python, Flask
- **Image Processing**: OpenCV, NumPy
- **Frontend**: HTML5, CSS3, JavaScript (Vanilla)
- **File Handling**: Pillow, Werkzeug

## 🎯 주요 알고리즘

### Perspective Transform
```python
def perspective_crop(image_path, corners):
    # 1. 이미지 로드
    image = cv2.imread(image_path)
    
    # 2. 소스 포인트를 numpy 배열로 변환
    src_points = np.array(corners, dtype=np.float32)
    
    # 3. 목표 크기 계산
    width = max(
        np.linalg.norm(src_points[1] - src_points[0]),
        np.linalg.norm(src_points[2] - src_points[3])
    )
    height = max(
        np.linalg.norm(src_points[3] - src_points[0]),
        np.linalg.norm(src_points[2] - src_points[1])
    )
    
    # 4. 목표 좌표 설정
    dst_points = np.array([
        [0, 0], [width, 0], [width, height], [0, height]
    ], dtype=np.float32)
    
    # 5. Perspective transform matrix 계산 및 적용
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    result = cv2.warpPerspective(image, matrix, (width, height))
    
    return result
```

## 🌟 특징

- **반응형 디자인**: 모바일과 데스크톱에서 모두 사용 가능
- **직관적인 UI**: 드래그 앤 드롭 인터페이스
- **실시간 피드백**: 사용자 액션에 대한 즉시적인 응답
- **에러 처리**: 다양한 예외 상황에 대한 적절한 처리
- **보안**: 파일 업로드 보안 검증

## 🔒 보안 고려사항

- 파일 확장자 검증
- 파일 크기 제한 (16MB)
- 고유한 파일명 생성으로 충돌 방지
- 업로드된 파일의 안전한 저장

## 📝 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

## 🤝 기여하기

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📞 문의

프로젝트에 대한 문의사항이 있으시면 이슈를 생성해 주세요. 