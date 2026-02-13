import cv2
import numpy as np
import os

# ============================================================
# UVDoc(docuwarp) 기반 문서 왜곡 보정
# 설치: pip install "docuwarp[cpu]"
# ============================================================

_unwarp_model = None

def _get_unwarp_model():
    """UVDoc 모델을 싱글턴으로 로드 (최초 호출 시 1회만 로드)"""
    global _unwarp_model
    if _unwarp_model is None:
        try:
            from docuwarp.unwarp import Unwarp
            _unwarp_model = Unwarp()
            print("✅ UVDoc 모델 로드 완료")
        except ImportError:
            raise ImportError(
                "docuwarp 패키지가 설치되지 않았습니다.\n"
                "설치 명령어: pip install \"docuwarp[cpu]\"\n"
                "GPU 사용 시: pip install \"docuwarp[gpu]\""
            )
    return _unwarp_model


def expand_document_fake(img, pad_ratio=0.15):
    """
    Docuwarp용 가짜 문서 확장 (Fake Document Expansion)
    거울(Reflect) 모드 대신 픽셀 연장(Replicate)을 사용하여 
    기존 굴곡의 방향성과 색상 흐름을 유지하며 캔버스를 확장합니다.
    """
    h, w = img.shape[:2]
    top = int(h * pad_ratio)
    bottom = int(h * pad_ratio)
    left = int(w * pad_ratio)
    right = int(w * pad_ratio)

    # BORDER_REPLICATE: 가장자리 마지막 픽셀을 그대로 바깥으로 밀어내어 확장합니다.
    # 이 방식은 굴곡의 흐름(기울기)을 반전시키지 않아 왜곡 보정 모델 인식 결과가 더 안정적입니다.
    expanded = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_REPLICATE)
    
    print(f"  [Expand] 방향성 유지를 위한 Replicate 확장 적용 (Padding: {top}px)")
    return expanded, (top, bottom, left, right)


def enhance_surface_for_ai(img):
    """
    딥러닝 모델이 문서의 굴곡과 경계를 더 잘 인식하도록 
    표면 디테일을 극대화하는 전처리를 수행합니다.
    """
    # 1. 가우시안 블러로 노이즈 제거
    blur = cv2.GaussianBlur(img, (3, 3), 0)
    
    # 2. LAB 색공간에서 밝기 채널(L)에 CLAHE 적용 (대비 극대화)
    lab = cv2.cvtColor(blur, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    
    return enhanced


def unwarp_document(img, use_expand=True):
    """
    개선된 왜곡 보정 파이프라인:
    1. 고해상도 이미지의 경우 메모리 부족 방지를 위한 리사이징 (Max 2000px)
    2. Fake Expansion (가장자리 잘림 방지, 선택 사항)
    3. Surface Enhancement (AI 인식률 향상용 전처리)
    4. AI Mapping (분석은 전처리본으로, 적용은 원본으로)
    """
    from PIL import Image
    import numpy as np
    
    # [메모리 오류 해결] 너무 큰 이미지는 리사이징하여 처리 (OOM 방지)
    h, w = img.shape[:2]
    max_dim = 2000
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        print(f"  [Unwarp] 이미지 크기 조정 (OOM 방지): {w}x{h} -> {int(w*scale)}x{int(h*scale)}")

    print(f"  [Unwarp Pipeline] 고정밀 왜곡 보정 시작... (Expand: {use_expand})")

    # 1. 가짜 문서 확장 (원본용)
    if use_expand:
        img_expanded_raw, pads = expand_document_fake(img)
    else:
        img_expanded_raw = img
        pads = (0, 0, 0, 0)
    
    # 2. AI 분석용 표면 강화 이미지 생성
    img_for_ai = enhance_surface_for_ai(img_expanded_raw)
    
    result_dl = None

    # 3. Docuwarp DL 모델 실행
    try:
        model = _get_unwarp_model()
        
        # 분석용(Enhanced)과 적용용(Raw) PIL 변환
        pil_for_ai = Image.fromarray(cv2.cvtColor(img_for_ai, cv2.COLOR_BGR2RGB))
        pil_raw = Image.fromarray(cv2.cvtColor(img_expanded_raw, cv2.COLOR_BGR2RGB))
        
        # 저수준 API를 사용하여 '분석'과 '적용'을 분리
        try:
            # A. 강화된 이미지로 왜곡 포인트를 분석 (AI 분석)
            resized_input, _, original_size = model.prepare_input(pil_for_ai)
            points, _ = model.session.run(None, {"input": resized_input.astype(np.float16)})
            
            # B. 분석된 포인트를 '원본 확장 이미지'에 적용 (화질 보존)
            _, original_input_raw, _ = model.prepare_input(pil_raw)
            # ONNX Runtime은 연속된 메모리 배열(contiguous array)을 요구함
            warped_data = np.ascontiguousarray(original_input_raw.astype(np.float32))
            point_data = np.ascontiguousarray(points.astype(np.float32))
            
            unwarped = model.bilinear_unwarping.run(
                None,
                {
                    "warped_img": warped_data,
                    "point_positions": point_data,
                    "img_size": np.array(original_size, dtype=np.int64),
                },
            )[0][0]
            
            unwarped_array = (unwarped.transpose(1, 2, 0) * 255).astype(np.uint8)
            result_dl = cv2.cvtColor(unwarped_array, cv2.COLOR_RGB2BGR)
            print("  [Unwarp] 고정밀 매핑 보정 완료 (Surface Enhanced Analysis)")
            
        except Exception as e:
            print(f"  [Unwarp] 정밀 매핑 실패, 표준 API 사용: {e}")
            unwarped_pil = model.inference(pil_raw)
            result_dl = cv2.cvtColor(np.array(unwarped_pil), cv2.COLOR_RGB2BGR)
                
    except Exception as e:
        print(f"  [Unwarp] 모델 실행 실패: {e}")
        result_dl = img_expanded_raw

    print("  [Unwarp Pipeline] 모든 단계 완료")
    return result_dl


def check_significant_distortion(design_img, scan_img, inlier_thresh=0.5):
    """
    디자인(기준) 이미지와 스캔 이미지를 매칭하여 왜곡 정도를 판단합니다.
    디자인 파일은 '아주 반듯한' 기준이므로, 스캔본이 이와 단일 Homography(평면 변환)로 
    잘 대응되는지(Inlier 비율)를 확인합니다.
    
    Inlier 비율이 이 임계값(default: 0.65)보다 낮으면 보행 왜곡이 있다고 판단합니다.
    
    Returns:
        True: 왜곡이 심함 (보정 필요)
        False: 평면적임 (보정 불필요)
    """
    # 1. 속도를 위해 리사이징 (긴 변 800px)
    h1, w1 = design_img.shape[:2]
    h2, w2 = scan_img.shape[:2]
    
    scale1 = 800 / max(h1, w1) if max(h1, w1) > 800 else 1.0
    scale2 = 800 / max(h2, w2) if max(h2, w2) > 800 else 1.0
    
    img1 = cv2.resize(design_img, None, fx=scale1, fy=scale1)
    img2 = cv2.resize(scan_img, None, fx=scale2, fy=scale2)
    
    # 2. ORB 특징점 검출
    orb = cv2.ORB_create(nfeatures=5000)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    
    if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
        print("  [Distortion Check] 특징점 부족 -> 보정 진행")
        return True 
        
    # 3. 매칭 (Hamming Distance)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    
    if len(matches) < 50:
        print("  [Distortion Check] 매칭 점 부족 -> 보정 진행")
        return True
        
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    
    # 4. RANSAC으로 Homography 계산 및 Inlier 확인
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    
    if M is None:
        print("  [Distortion Check] 호모그래피 산출 불가 -> 보정 진행")
        return True
        
    inliers = mask.ravel().tolist().count(1)
    ratio = inliers / len(matches)
    
    print(f"  [Distortion Check] Matches: {len(matches)}, Inliers: {inliers}, Good Ratio: {ratio:.0%}")
    
    # Inlier 비율이 50% 미만이면, 평면적인 변환으로 설명이 안 됨 -> 굴곡(왜곡)이 심함
    # 반대로 Inlier가 높으면, 이미 평평하거나 단순 원근 변환 상태임 -> 보정 불필요
    return ratio < inlier_thresh


def flat_illumination(img):
    """
    OpenCV morphological operation을 사용하여 불균일한 조명을 평탄화합니다.
    """
    yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    y_channel = yuv[:, :, 0]

    kernel_size = 50
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    background = cv2.morphologyEx(y_channel, cv2.MORPH_CLOSE, kernel)

    background = background.astype(np.float32)
    y_channel = y_channel.astype(np.float32)

    normalized_y = y_channel / (background + 1e-6) * 200
    normalized_y = np.clip(normalized_y, 0, 255).astype(np.uint8)

    yuv[:, :, 0] = normalized_y
    output = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    return output


def preprocess_image(img):
    """이진화 전처리 (외곽선 검출용)"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    adapt_thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    morph = cv2.morphologyEx(adapt_thresh, cv2.MORPH_CLOSE, kernel)
    return morph


def align_images(im1, im2, debug=False, debug_dir="debug_output"):
    """
    두 이미지를 외곽선 기반 Homography로 정렬합니다.

    Args:
        im1: 기준 이미지 (OpenCV BGR)
        im2: 정렬할 이미지 (OpenCV BGR)
        debug: 디버그 이미지 저장 여부
        debug_dir: 디버그 이미지 저장 경로

    Returns:
        정렬된 im2 이미지
    """
    im1_thresh = preprocess_image(im1)
    im2_thresh = preprocess_image(im2)

    contours1, _ = cv2.findContours(im1_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours2, _ = cv2.findContours(im2_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contour1 = max(contours1, key=cv2.contourArea)
    contour2 = max(contours2, key=cv2.contourArea)

    rect1 = cv2.minAreaRect(contour1)
    rect2 = cv2.minAreaRect(contour2)
    box1 = cv2.boxPoints(rect1)
    box2 = cv2.boxPoints(rect2)
    box1 = np.array(box1, dtype=np.float32)
    box2 = np.array(box2, dtype=np.float32)

    def sort_box_points(pts):
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)
        return np.array([
            pts[np.argmin(s)],
            pts[np.argmin(diff)],
            pts[np.argmax(s)],
            pts[np.argmax(diff)]
        ], dtype=np.float32)

    box1 = sort_box_points(box1)
    box2 = sort_box_points(box2)

    h, _ = cv2.findHomography(box2, box1)
    
    # ── [개선] 캔버스 확장: 변환 후 이미지가 잘리지 않도록 함 ──
    h_t, w_t = im2.shape[:2]
    corners = np.float32([[0,0], [w_t,0], [w_t,h_t], [0,h_t]]).reshape(-1,1,2)
    warped_corners = cv2.perspectiveTransform(corners, h)
    
    # 기준 이미지(im1)의 크기와 변환된 corners 중 최대 범위를 고려
    x_coords = warped_corners[:,0,0]
    y_coords = warped_corners[:,0,1]
    max_w = int(max(im1.shape[1], x_coords.max()))
    max_h = int(max(im1.shape[0], y_coords.max()))

    im2_aligned = cv2.warpPerspective(im2, h, (max_w, max_h))

    if debug:
        os.makedirs(debug_dir, exist_ok=True)
        debug1 = im1.copy()
        debug2 = im2.copy()
        cv2.drawContours(debug1, [np.int32(box1)], -1, (0, 255, 0), 2)
        cv2.drawContours(debug2, [np.int32(box2)], -1, (0, 255, 0), 2)
        cv2.imwrite(os.path.join(debug_dir, "im1_labeled.jpg"), debug1)
        cv2.imwrite(os.path.join(debug_dir, "im2_labeled.jpg"), debug2)
        cv2.imwrite(os.path.join(debug_dir, "aligned_image.jpg"), im2_aligned)

    return im2_aligned


def align_images_robust(base, target, debug=True, debug_dir="debug_align"):
    """
    ORB 특징점 + RANSAC Homography + ECC 정밀 보정으로 두 이미지를 정렬합니다.

    Args:
        base: 기준 이미지 (OpenCV BGR)
        target: 정렬할 이미지 (OpenCV BGR)
        debug: 디버그 이미지 저장 여부
        debug_dir: 디버그 이미지 저장 경로

    Returns:
        정렬된 target 이미지
    """
    os.makedirs(debug_dir, exist_ok=True)

    # 1. ORB 특징점 검출
    orb = cv2.ORB_create(5000)
    g1 = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    k1, d1 = orb.detectAndCompute(g1, None)
    k2, d2 = orb.detectAndCompute(g2, None)

    # 2. 매칭 (Lowe ratio test)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(d1, d2, k=2)
    good = [m for m, n in matches if m.distance < 0.75 * n.distance]

    if len(good) < 10:
        raise ValueError("매칭이 너무 적습니다!")

    # 3. Homography (RANSAC)
    src_pts = np.float32([k1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([k2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

    # 3-A. 변환된 코너 좌표로 캔버스 확장
    h_t, w_t = target.shape[:2]
    tgt_corners = np.float32([[0,0], [w_t,0], [w_t,h_t], [0,h_t]]).reshape(-1,1,2)
    warped_corners = cv2.perspectiveTransform(tgt_corners, H)
    x_coords = warped_corners[:,0,0]
    y_coords = warped_corners[:,0,1]
    min_x, min_y = np.floor([x_coords.min(), y_coords.min()])
    max_x, max_y = np.ceil([x_coords.max(), y_coords.max()])

    T = np.array([[1, 0, -min_x],
                   [0, 1, -min_y],
                   [0, 0,  1    ]], dtype=np.float32)
    H_shifted = T @ H
    new_w = int(max_x - min_x)
    new_h = int(max_y - min_y)

    aligned = cv2.warpPerspective(target, H_shifted, (new_w, new_h),
                                   borderMode=cv2.BORDER_CONSTANT,
                                   borderValue=(0,0,0))

    if debug:
        match_vis = cv2.drawMatches(base, k1, target, k2, good, None,
                                     flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv2.imwrite(os.path.join(debug_dir, "orb_matches.jpg"), match_vis)
        cv2.imwrite(os.path.join(debug_dir, "aligned_orb.jpg"), aligned)

    # 4. ECC로 추가 정밀 보정
    warp_matrix = np.eye(3, 3, dtype=np.float32)
    try:
        cc, warp_matrix = cv2.findTransformECC(
            cv2.cvtColor(base, cv2.COLOR_BGR2GRAY),
            cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY),
            warp_matrix,
            cv2.MOTION_HOMOGRAPHY,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 300, 1e-6))

        aligned_refined = cv2.warpPerspective(aligned, warp_matrix, (new_w, new_h),
                                               borderMode=cv2.BORDER_CONSTANT,
                                               borderValue=(0,0,0))
        if debug:
            cv2.imwrite(os.path.join(debug_dir, "aligned_refined.jpg"), aligned_refined)
        final_aligned = aligned_refined

    except cv2.error:
        print("⚠ ECC refinement failed, using initial alignment.")
        final_aligned = aligned

    if debug:
        cv2.imwrite(os.path.join(debug_dir, "aligned_final_resized.jpg"), final_aligned)

    return final_aligned