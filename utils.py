import cv2
import numpy as np
import os

def extract_objects(img):
    """
    윤곽선 기반으로 객체 검출하고, 간단한 분류(텍스트/로고 등)도 시도.
    OCR 매칭 기반으로 type을 정확히 정리하도록 개선 가능
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    objects = []
    img_area = img.shape[0] * img.shape[1]

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        aspect_ratio = w / h if h != 0 else 0
        rel_area = area / img_area

        if area < 1000 or rel_area < 0.0005 or rel_area > 0.7:
            continue  # 너무 작거나 너무 큰 경우 제외

        if rel_area < 0.005 and 0.8 < aspect_ratio < 1.2:
            obj_type = "로고"
        elif aspect_ratio > 4.0 or aspect_ratio < 0.25:
            obj_type = "선형요소"
        elif 0.3 <= aspect_ratio <= 4.0 and rel_area > 0.003:
            obj_type = "텍스트"
        else:
            obj_type = "기타"

        objects.append({
            "bbox": (x, y, x + w, y + h),
            "type": obj_type,
            "area": area,
            "aspect_ratio": round(aspect_ratio, 2)
        })

    return objects

def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    return interArea / float(boxAArea + boxBArea - interArea + 1e-5)

def compare_colors(img1, img2):
    hsv1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
    hsv2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)

    hist1 = cv2.calcHist([hsv1], [0, 1], None, [50, 60], [0, 180, 0, 256])
    hist2 = cv2.calcHist([hsv2], [0, 1], None, [50, 60], [0, 180, 0, 256])

    cv2.normalize(hist1, hist1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(hist2, hist2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

def get_roi_mean_color(img, bbox):
    x1, y1, x2, y2 = bbox
    roi = img[y1:y2, x1:x2]
    mean_color = cv2.mean(roi)[:3]
    return tuple(map(int, mean_color))

def group_text_lines(texts, y_threshold=30, x_gap_threshold=10, height_variation_ratio=0.8):
    """
    회전 중심과 높이 정보를 사용해서 같은 줄의 텍스트들을 그룹화합니다.
    
    Args:
        texts (list): 텍스트 요소들. 각 요소는 dict로, 'bbox', 'center', 'height', 'angle' 등을 포함.
        y_threshold (int): 수직 거리 허용값
        x_gap_threshold (int): 수평 간격 허용값
        height_variation_ratio (float): 높이 차이 비율 허용값

    Returns:
        list: 줄 단위로 묶인 텍스트 리스트 (list of list of dicts)
    """
    texts = sorted(texts, key=lambda x: x['center'][1])  # y 기준 정렬
    lines = []

    for text in texts:
        cx, cy = text['center']
        th = text['height']
        x1, _, x2, _ = text['bbox']

        placed = False
        for line in lines:
            centers_y = [t['center'][1] for t in line]
            heights = [t['height'] for t in line]
            x2s = [t['bbox'][2] for t in line]

            avg_cy = sum(centers_y) / len(centers_y)
            avg_height = sum(heights) / len(heights)

            y_dist = abs(cy - avg_cy)
            height_diff_ratio = abs(th - avg_height) / max(th, avg_height)
            x_gap = min([max(0.1, x1 - x2p) for x2p in x2s])  # 여러 요소와 비교

            if y_dist < y_threshold and x_gap < x_gap_threshold and height_diff_ratio < height_variation_ratio:
                line.append(text)
                placed = True
                break

        if not placed:
            lines.append([text])

    # 같은 줄 내 텍스트를 x1 좌표 기준으로 정렬
    for line in lines:
        line.sort(key=lambda t: t['bbox'][0])

    return lines



def match_ocr_to_objects(ocr_elements, object_elements, iou_threshold=0.5):
    matched = []
    for ocr in ocr_elements:
        best_match = None
        best_iou = 0
        for obj in object_elements:
            if obj['type'] != '텍스트':
                continue
            iou = calculate_iou(ocr['bbox'], obj['bbox'])
            if iou > best_iou and iou >= iou_threshold:
                best_iou = iou
                best_match = obj
        if best_match:
            matched.append({
                "text": ocr['text'],
                "ocr_bbox": ocr['bbox'],
                "obj_bbox": best_match['bbox'],
                "iou": best_iou
            })
    return matched

def detect_tables(img, min_table_area=10000, min_w=80, min_h=30, min_aspect=1.2, max_aspect=10.0):
    # 1. Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. GaussianBlur → Adaptive Threshold
    height, width = gray.shape
    kernel_size = max(int(height * 0.005), int(width * 0.005))
    if kernel_size % 2 == 0:
        kernel_size += 1
    blur = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 1)

    block_size = kernel_size * 2 - 1
    threshold = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size, 2
    )

    # 3. 라플라시안 테두리 강조
    laplacian = cv2.Laplacian(threshold, cv2.CV_64F)
    laplacian = np.uint8(np.absolute(laplacian))

    # 4. 컨투어 전체 추출
    contours, _ = cv2.findContours(laplacian, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 5. 사각형 모양만 추출
    table_boxes = []
    for cnt in contours:
        perimeter = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)

        # 4개 점을 가진 다각형만 (사각형 후보)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            area = w * h
            aspect = w / (h + 1e-5)
            box = (x, y, x + w, y + h)

            # ✅ 좌표값에 0이 두 개 이상 포함되면 제거
            zero_count = sum([1 for v in box if 0 <= v <= 3])
            if zero_count >= 2:
                continue

            if area > min_table_area and w > min_w and h > min_h and min_aspect <= aspect <= max_aspect:
                table_boxes.append((x, y, x + w, y + h))

    return table_boxes

def get_adaptive_params(img_shape):
    h, w = img_shape[:2]

    # 이미지 크기에 따라 scale, min_area 자동 조정
    scale = max(10, w // 110)                # scale은 너비 기준으로 설정
    min_area = (w * h) // 2000               # 전체 면적%
    min_area = max(min_area, 3000)           # 너무 작아지지 않게

    kernel_pad = max(1, w // 800)           # 선의 두께 기반 dilation kernel 조절

    return scale, min_area, kernel_pad

def detect_table_cells_morph(img, is_table_extractor=False):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)

    scale, min_area, pad = get_adaptive_params(img.shape)
    # 1. 이진화 (선 강조)
    bin_img = cv2.adaptiveThreshold(~gray, 255,
                                    cv2.ADAPTIVE_THRESH_MEAN_C,
                                    cv2.THRESH_BINARY,
                                    15, -2)

    # 2. 수평선 추출
    h_scale = max(1, bin_img.shape[1] // scale)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_scale, 1))
    horizontal = cv2.erode(bin_img, h_kernel, iterations=2)
    horizontal = cv2.dilate(horizontal, h_kernel, iterations=pad)  # 💡 1 → 2

    # 3. 수직선 추출
    v_scale = max(1, bin_img.shape[0] // scale)
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_scale))
    
    # [수정] 용도에 따른 침식 강도 조절
    # table_extractor용은 얇은 선도 탐지해야 하므로 1, main.py(기존)은 노이즈 제거를 위해 2 유지
    v_iterations = 1 if is_table_extractor else 2
    vertical = cv2.erode(bin_img, v_kernel, iterations=v_iterations)
    vertical = cv2.dilate(vertical, v_kernel, iterations=pad)

    # 4. 그리드 결합
    mask = cv2.add(horizontal, vertical)

    # 5. 틈 메우기
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=2)

    # 6. 컨투어 추출 (계층 구조까지 추적)
    contours, hierarchy = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cells = []
    for i, c in enumerate(contours):
        x, y, w, h = cv2.boundingRect(c)
        area = w * h

        # [개선] table_extractor 모드일 때는 독립적인 선(얇은 박스)도 허용하도록 필터 완화
        current_min_area = min_area if not is_table_extractor else 500
        current_min_w = 20 if not is_table_extractor else 3
        
        if area > current_min_area and w > current_min_w and h > 20:
            # 가장 바깥 외곽선 무시 (이미지 전체 크기와 유사한 경우)
            if w > img.shape[1] * 0.90 and h > img.shape[0] * 0.90:
                continue
            cells.append((x-10, y-10, x + w+10, y + h+10))

    return sorted(cells, key=lambda b: (b[1], b[0]))  # (y, x) 정렬

def get_color_regions(img):
    """
    이미지를 색상 기준으로 거칠게(coarse) 구역화합니다.
    서로 다른 배경색을 가진 구역 간의 무분별한 병합을 막기 위함입니다.
    """
    # 처리 속도와 노이즈 제거를 위해 이미지 축소 및 블러
    h, w = img.shape[:2]
    small = cv2.resize(img, (w // 8, h // 8), interpolation=cv2.INTER_AREA)
    blurred = cv2.medianBlur(small, 15)
    
    # Lab 색공간으로 변환 (인간의 색 인지와 유사한 거리 측정)
    lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
    data = lab.reshape((-1, 3)).astype(np.float32)
    
    # K-Means 클러스터링으로 주요 색상 구역 분리 (K=5 정도면 충분히 거친 구역화 가능)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, _ = cv2.kmeans(data, 5, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # 원래 크기로 복원 (Label Map)
    label_img = labels.reshape((lab.shape[0], lab.shape[1])).astype(np.uint8)
    label_img_full = cv2.resize(label_img, (w, h), interpolation=cv2.INTER_NEAREST)
    
    return label_img_full

def get_vertical_mask(img):
    """
    HoughLinesP와 Morphology를 결합하여 수직/곡선 구분선을 극한으로 탐지합니다.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h_img, w_img = gray.shape[:2]
    
    # 1. 전처리: 강력한 대비 향상 및 노이즈 제거
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 2. Canny Edge Detection (선 성분 추출)
    edges = cv2.Canny(blurred, 30, 150, apertureSize=3)
    
    # 3. 확률적 허프 변환 (HoughLinesP)으로 명확한 직선 찾기
    line_mask = np.zeros_like(edges)
    min_line_len = h_img // 50
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, 
                            minLineLength=min_line_len, maxLineGap=10)
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # 수직에 가까운 선만 마스크에 그림 (사선 포함)
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            if 45 < angle < 135:
                cv2.line(line_mask, (x1, y1), (x2, y2), 255, 3)
                
    # 4. 기존 Morphology 방식과 결합 (불완전한 선 보완)
    bin_img = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                    cv2.THRESH_BINARY_INV, 21, 5)
    
    v_len = max(25, h_img // 30)
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_len))
    morph_mask = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, v_kernel, iterations=1)
    
    # 5. 최종 결합: 허프 변환 선 + 형태학적 마스크
    final_mask = cv2.bitwise_or(line_mask, morph_mask)
    final_mask = cv2.dilate(final_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
    
    return final_mask

def has_vertical_separator(v_mask, box1, box2, color_label_map=None):
    """
    두 박스 사이에 물리적인 선 또는 색상 구역의 경계가 있는지 체크합니다.
    """
    b1, b2 = (box1, box2) if box1[0] < box2[0] else (box2, box1)
    
    gap_x1, gap_x2 = b1[2], b2[0]
    gap_y1, gap_y2 = min(b1[1], b2[1]), max(b1[3], b2[3])
    char_height = gap_y2 - gap_y1
    
    if char_height <= 0: return False

    # ── 1. 색상 구역(Color Zone) 체크 ──
    if color_label_map is not None:
        # 박스 A와 박스 B의 중심 좌표에서 구역 라벨 추출
        c1x, c1y = (b1[0] + b1[2]) // 2, (b1[1] + b1[3]) // 2
        c2x, c2y = (b2[0] + b2[2]) // 2, (b2[1] + b2[3]) // 2
        
        # 좌표 범위 안전 처리
        c1x, c1y = min(max(0, c1x), color_label_map.shape[1]-1), min(max(0, c1y), color_label_map.shape[0]-1)
        c2x, c2y = min(max(0, c2x), color_label_map.shape[1]-1), min(max(0, c2y), color_label_map.shape[0]-1)
        
        label1 = color_label_map[c1y, c1x]
        label2 = color_label_map[c2y, c2x]
        
        # 두 박스가 서로 다른 색상 구역에 있다면 병합 차단
        if label1 != label2:
            return True

    # ── 2. 물리적 구분선 체크 ──
    check_x1 = max(0, gap_x1 - 2)
    check_x2 = min(v_mask.shape[1], gap_x2 + 2)
    roi = v_mask[gap_y1:gap_y2, check_x1:check_x2]
    
    if roi.size > 0:
        v_projection = np.sum(roi > 0, axis=0)
        max_line_component = np.max(v_projection)
        # 탐지력을 높이기 위해 50%로 더 완화
        if max_line_component > char_height * 0.5:
            return True
            
    return False

def filter_overlapping_boxes(table_boxes, iou_thresh=0.3, containment_thresh=2):
    """
    큰 박스 중에서 다른 박스와 겹침(IoU)이 높고 여러 박스를 포함하면 제거
    """
    def is_contained(inner, outer):
        return (
            inner[0] >= outer[0] and inner[1] >= outer[1] and
            inner[2] <= outer[2] and inner[3] <= outer[3]
        )

    filtered = []
    for i, box in enumerate(table_boxes):
        contained_count = 0
        high_iou_count = 0
        for j, other in enumerate(table_boxes):
            if i == j:
                continue
            if is_contained(other, box):
                contained_count += 1
            if iou(box, other) > iou_thresh:
                high_iou_count += 1

        # 포함하는 박스 수 + 겹침 박스 수 기준 필터링
        if contained_count < containment_thresh and high_iou_count < containment_thresh:
            filtered.append(box)

    return filtered


def iou(boxA, boxB):
    # box = (x1, y1, x2, y2)
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH

    if interArea == 0:
        return 0.0

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou_value = interArea / float(boxAArea + boxBArea - interArea)
    return iou_value
