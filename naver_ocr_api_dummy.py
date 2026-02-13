import requests
import uuid
import time
import json
import cv2
import numpy as np

# 사용자 설정 (API URL 및 Secret Key 입력 필요)
# 환경 변수나 별도의 설정 파일에서 불러오는 방식을 권장합니다.
NAVER_OCR_API_URL = '' 
NAVER_OCR_SECRET_KEY = '' 

def call_naver_ocr(image_data, api_url=None, secret_key=None):
    """
    네이버 OCR API를 호출하여 텍스트 분석 결과를 반환합니다.
    
    Args:
        image_data: 파일 경로(str) 또는 이미지 binary data(bytes).
                    cv2 이미지(numpy array)인 경우 인코딩 과정을 거쳐야 함.
        api_url: 네이버 OCR API URL
        secret_key: 네이버 OCR Secret Key
    
    Returns:
        json_response: API 응답 (dict)
    """
    if api_url is None:
        api_url = NAVER_OCR_API_URL
    if secret_key is None:
        secret_key = NAVER_OCR_SECRET_KEY

    if not api_url or not secret_key:
        raise ValueError("Naver OCR API URL and Secret Key must be provided.")

    request_json = {
        'images': [
            {
                'format': 'jpg',
                'name': 'demo'
            }
        ],
        'requestId': str(uuid.uuid4()),
        'version': 'V2',
        'timestamp': int(round(time.time() * 1000))
    }

    payload = {'message': json.dumps(request_json).encode('UTF-8')}
    
    files = []
    if isinstance(image_data, str):
        # 파일 경로인 경우
        files = [('file', open(image_data,'rb'))]
    elif isinstance(image_data, bytes):
        # 바이너리 데이터인 경우
        files = [('file', image_data)]
    else:
        raise ValueError("image_data must be file path or bytes")

    headers = {
        'X-OCR-SECRET': secret_key
    }

    response = requests.request("POST", api_url, headers=headers, data=payload, files=files)
    
    # 파일을 열었었으면 닫아주는 로직이 필요하지만, requests files parameter handles file closing usually if passed simply, 
    # but open(image_data,'rb') creates a file handle that might leak if not closed. 
    # However, for simplicity and matching user snippet, we keep it. 
    # Better to read bytes and pass bytes.
    
    if response.status_code != 200:
        raise Exception(f"Naver OCR Error: {response.status_code}, {response.text}")

    return response.json()

def call_naver_ocr_numpy(image_np, api_url=None, secret_key=None):
    """
    OpenCV Numpy 이미지를 받아 네이버 OCR에 전송
    """
    # 이미지를 JPG 바이너리로 인코딩
    is_success, buffer = cv2.imencode(".jpg", image_np)
    if not is_success:
        raise ValueError("Could not encode image to JPG")
    
    return call_naver_ocr(buffer.tobytes(), api_url, secret_key)
