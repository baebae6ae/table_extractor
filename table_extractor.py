import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os
import cv2
import numpy as np
import threading
import pandas as pd
import json

import utils
import align
import naver_ocr_api

class TableExtractorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("식품 표시사항 표 추출기 (Table to Excel)")
        self.root.geometry("1200x1000") # 전체 창 크기 확대
        self.root.configure(bg="#f0f0f0")

        self.image_path = None
        self.last_excel_path = None
        self.preview_canvas = None

        # 메인 프레임
        main_frame = tk.Frame(root, bg="#f0f0f0")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # 상단 안내
        tk.Label(main_frame, text="이미지의 표 구조를 자동으로 인식하여 엑셀로 변환합니다.", 
                 font=("Malgun Gothic", 14, "bold"), bg="#f0f0f0", fg="#2c3e50").pack(pady=10)
        tk.Label(main_frame, text="(Docuwarp 왜곡 보정 및 Naver OCR 활용)", 
                 font=("Malgun Gothic", 10), bg="#f0f0f0", fg="#7f8c8d").pack(pady=(0, 15))

        # 이미지 선택 버튼
        btn_frame = tk.Frame(main_frame, bg="#f0f0f0")
        btn_frame.pack(fill=tk.X, pady=10)
        
        self.btn_load = tk.Button(btn_frame, text="📂 이미지 불러오기", command=self.load_image, 
                                  width=20, height=1, bg="#3498db", fg="white", font=("Malgun Gothic", 10, "bold"))
        self.btn_load.pack(side=tk.LEFT, padx=5)
        
        self.lbl_path = tk.Label(btn_frame, text="선택된 파일: 없음", bg="#f0f0f0", anchor="w", font=("Malgun Gothic", 9))
        self.lbl_path.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

        # 실행 버튼 영역
        control_frame = tk.Frame(main_frame, bg="#f0f0f0")
        control_frame.pack(pady=10)

        self.btn_extract = tk.Button(control_frame, text="🚀 표 추출 실행", command=self.run_extraction, 
                                     bg="#2ecc71", fg="white", font=("Malgun Gothic", 11, "bold"), width=20, height=1)
        self.btn_extract.pack(side=tk.LEFT, padx=10)

        self.btn_open_excel = tk.Button(control_frame, text="📊 엑셀 열기", command=self.open_excel, 
                                  state=tk.DISABLED, width=15, height=1, font=("Malgun Gothic", 10))
        self.btn_open_excel.pack(side=tk.LEFT, padx=10)

        # [추가] 시각화 확인 영역 (프레임 비중 확대)
        self.preview_frame = tk.LabelFrame(main_frame, text="추출 과정 실시간 확인 (Visual Preview)", bg="#f0f0f0", font=("Malgun Gothic", 9, "bold"))
        self.preview_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # 캔버스와 스크롤바 세트
        self.preview_canvas = tk.Canvas(self.preview_frame, bg="#d1d8e0", highlightthickness=0)
        self.v_scroll = tk.Scrollbar(self.preview_frame, orient="vertical", command=self.preview_canvas.yview)
        self.h_scroll = tk.Scrollbar(self.preview_frame, orient="horizontal", command=self.preview_canvas.xview)
        
        self.preview_canvas.configure(yscrollcommand=self.v_scroll.set, xscrollcommand=self.h_scroll.set)
        
        # 그리드 배치로 꽉 채우기
        self.preview_canvas.grid(row=0, column=0, sticky="nsew")
        self.v_scroll.grid(row=0, column=1, sticky="ns")
        self.h_scroll.grid(row=1, column=0, sticky="ew")
        
        self.preview_frame.grid_rowconfigure(0, weight=1)
        self.preview_frame.grid_columnconfigure(0, weight=1)

        # 마우스 휠 바인딩
        self.preview_canvas.bind_all("<MouseWheel>", self._on_mousewheel)

        # 로그 영역 (높이 조절)
        log_label = tk.Label(main_frame, text="실행 로그 (Processing Logs)", bg="#f0f0f0", font=("Malgun Gothic", 9, "bold"))
        log_label.pack(anchor="w", pady=(5, 0))

        self.log_text = tk.Text(main_frame, height=8, bg="#2c3e50", fg="#ecf0f1", 
                                font=("Consolas", 10), padx=10, pady=10)
        self.log_text.pack(fill=tk.X, expand=False)
        
        scrollbar = tk.Scrollbar(self.log_text)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.log_text.yview)

    def log(self, message):
        """UI 로그 영역에 메시지 출력"""
        self.log_text.insert(tk.END, f"{message}\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()

    def _on_mousewheel(self, event):
        """미리보기 캔버스 마우스 휠 스크롤"""
        self.preview_canvas.yview_scroll(int(-1*(event.delta/120)), "units")

    def show_preview(self, cv_img):
        """이미지를 프리뷰 캔버스에 출력 (사용자 요청에 맞춰 크기 최적화)"""
        if cv_img is None: return
        h, w = cv_img.shape[:2]
        
        # 대부분의 모니터 해상도에서 확인이 용이한 수준
        target_w = 1000
        scale = target_w / w
        new_w, new_h = int(w * scale), int(h * scale)
        
        # 고화질 보간법 사용
        resized = cv2.resize(cv_img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        img_tk = ImageTk.PhotoImage(Image.fromarray(rgb))
        
        # 캔버스 업데이트
        self.preview_canvas.delete("all")
        self.preview_canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
        self.preview_canvas.config(scrollregion=(0, 0, new_w, new_h))
        self.preview_canvas.image = img_tk # 가비지 컬렉션 방지

    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.jpeg *.bmp")])
        if path:
            self.image_path = path
            self.lbl_path.config(text=f"선택됨: {os.path.basename(path)}")
            self.log(f"📂 이미지 로드 완료: {os.path.basename(path)}")
            
            # 원본 프리뷰 표시
            img_array = np.fromfile(path, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            self.show_preview(img)
            self.btn_open_excel.config(state=tk.DISABLED)

    def run_extraction(self):
        if not self.image_path:
            messagebox.showerror("오류", "이미지를 먼저 선택해주세요.")
            return
            
        self.log_text.delete(1.0, tk.END)
        self.log("▶ 추출 프로세스 시작...")
        
        # 스레드로 실행 (UI 프리징 방지)
        threading.Thread(target=self._worker, daemon=True).start()

    def _worker(self):
        try:
            # 1. 이미지 로드 (한글 경로 대응)
            self.log("1. 이미지 로드 중...")
            img_array = np.fromfile(self.image_path, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
            # 2. Docuwarp 왜곡 보정 (무조건 실행)
            self.log("2. Docuwarp 왜곡 보정 및 표 영역 탐색 중...")
            
            # align.unwarp_document 실행 및 결과 즉시 반영
            unwarped = align.unwarp_document(img, use_expand=False)
            
            if unwarped is None:
                raise ValueError("Docuwarp 보정 결과가 비어있습니다. (이미지 분석 실패)")

            # [핵심] 보정된 이미지를 UI 프리뷰에 즉시 반영
            self.root.after(0, lambda: self.show_preview(unwarped))
            self.log(f"   - 왜곡 보정 완료 (크기: {unwarped.shape[1]}x{unwarped.shape[0]})")
            
            # 결과 저장용 폴더 생성
            out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output_table")
            if not os.path.exists(out_dir): os.makedirs(out_dir)
            
            # 보정된 이미지 임시 저장
            debug_path = os.path.join(out_dir, "unwarped_table.jpg")
            _, buf = cv2.imencode(".jpg", unwarped)
            buf.tofile(debug_path)
            self.log(f"   - 왜곡 보정 완료")

            # 3. Naver OCR 호출
            self.log("3. Naver OCR 분석 요청 중...")
            ocr_res = naver_ocr_api.call_naver_ocr(debug_path)
            
            if not ocr_res or 'images' not in ocr_res:
                raise ValueError("OCR API 응답이 올바르지 않습니다.")

            # 3-1. 자동 회전 보정 (OCR 결과를 바탕으로 90/180/270도 판단)
            self.log("3-1. 이미지 방향 체크 및 자동 회전...")
            rotation_needed = self._detect_rotation(ocr_res)
            if rotation_needed != 0:
                self.log(f"   - {rotation_needed}도 회전 감지, 재보정 중...")
                unwarped = self._rotate_image(unwarped, rotation_needed)
                
                # 회전 후 OCR 다시 수행 (좌표계 일치를 위해)
                _, buf = cv2.imencode(".jpg", unwarped)
                buf.tofile(debug_path)
                ocr_res = naver_ocr_api.call_naver_ocr(debug_path)
                self.root.after(0, lambda: self.show_preview(unwarped))

            # 3-2. 최종 보정 이미지에서 물리적 셀 구조 분석 (utils 활용)
            self.log("3-2. 보정본에서 표 물리 구조 분석 중...")
            
            # utils의 형태학적 셀(Box) 탐지 활용 (is_table_extractor=True로 설정하여 얇은 선 탐지 활성화)
            raw_cells = utils.detect_table_cells_morph(unwarped, is_table_extractor=True)
            # 중복 및 포함 관계 정제
            warped_cells = utils.filter_overlapping_boxes(raw_cells)
            
            self.log(f"   - {len(warped_cells)}개의 물리적 셀 구조 감지됨")

            # 4. 표 구조 재구성 (Reconstruct Table with Structural Hint)
            self.log("4. 표 데이터 구조화 시작 (구조 기반 매핑)...")
            # 시각화용 이미지 생성
            vis_img = unwarped.copy()
            
            # 구조 정보를 힌트로 사용하여 재구성
            table_data = self._reconstruct_table_with_structure(ocr_res, warped_cells, vis_img)
            
            # [추가] 분석 완료 후 최종 박스/그리드가 그려진 이미기 프리뷰 업데이트
            self.root.after(0, lambda: self.show_preview(vis_img))
            
            if not table_data:
                self.log("⚠️ 감지된 텍스트가 없습니다.")
                return

            # 5. 엑셀 저장
            self.log("5. 엑셀 파일 생성 중...")
            file_base = os.path.splitext(os.path.basename(self.image_path))[0]
            excel_path = os.path.join(out_dir, f"{file_base}_table.xlsx")
            
            df = pd.DataFrame(table_data)
            df.to_excel(excel_path, index=False, header=False)
            
            self.last_excel_path = excel_path
            self.log("-" * 40)
            self.log(f"✅ 추출 성공! 엑셀이 생성되었습니다.")
            self.log(f"📁 경로: {excel_path}")
            
            self.root.after(0, lambda: self.btn_open_excel.config(state=tk.NORMAL))
            self.root.after(0, lambda: messagebox.showinfo("완료", "표 데이터 추출이 완료되었습니다!"))

        except Exception as e:
            error_msg = str(e)
            self.log(f"❌ 오류 발생: {error_msg}")
            import traceback
            self.log(traceback.format_exc())
            self.root.after(0, lambda: messagebox.showerror("오류", f"추출 중 에러가 발생했습니다:\n{error_msg}"))

    def _rotate_image(self, img, angle):
        if angle == 90: return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        if angle == 180: return cv2.rotate(img, cv2.ROTATE_180)
        # 270 means 90 counter-clockwise
        if angle == 270: return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return img

    def _rotate_cells(self, cells, w, h, angle):
        """회전된 이미지의 좌표계에 맞춰 감지된 셀 정보들도 회전"""
        new_cells = []
        for (x1, y1, x2, y2) in cells:
            if angle == 90:
                # 시계방향 90도 (x, y) -> (h-y, x)
                nx1, ny1 = h - y2, x1
                nx2, ny2 = h - y1, x2
                if nx1 > nx2: nx1, nx2 = nx2, nx1
                if ny1 > ny2: ny1, ny2 = ny2, ny1
                new_cells.append((int(nx1), int(ny1), int(nx2), int(ny2)))
            elif angle == 180:
                # 180도 (x, y) -> (w-x, h-y)
                nx1, ny1 = w - x2, h - y2
                nx2, ny2 = w - x1, h - y1
                new_cells.append((int(nx1), int(ny1), int(nx2), int(ny2)))
            elif angle == 270:
                # 반시계 90도 (시계 270도) (x, y) -> (y, w-x)
                nx1, ny1 = y1, w - x2
                nx2, ny2 = y2, w - x1
                if nx1 > nx2: nx1, nx2 = nx2, nx1
                if ny1 > ny2: ny1, ny2 = ny2, ny1
                new_cells.append((int(nx1), int(ny1), int(nx2), int(ny2)))
            else:
                new_cells.append((x1, y1, x2, y2))
        return new_cells

    def _detect_rotation(self, ocr_res):
        """
        [고도화된 회전 감지]
        1. 각도를 4개의 주요 구간(0, 90, 180, 270)으로 분류
        2. 텍스트 박스의 너비를 가중치로 사용하여 '가장 지배적인 방향'을 투표로 결정
        """
        votes = {0: 0, 90: 0, 180: 0, 270: 0}
        
        for img_res in ocr_res.get('images', []):
            for field in img_res.get('fields', []):
                text = field.get('inferText', '').strip()
                if len(text) < 2: continue # 1글자는 방향 판단이 어려움
                
                verts = field.get('boundingPoly', {}).get('vertices', [])
                if len(verts) < 4: continue
                
                # 벡터 (v0 -> v1) 추출
                dx = verts[1].get('x', 0) - verts[0].get('x', 0)
                dy = verts[1].get('y', 0) - verts[0].get('y', 0)
                
                # 벡터의 길이 (텍스트의 너비 - 가중치로 사용)
                dist = np.sqrt(dx**2 + dy**2)
                if dist < 5: continue
                
                angle = np.degrees(np.arctan2(dy, dx))
                
                # 각도 구간별 투표 (가중치 적용)
                if -45 <= angle <= 45:
                    votes[0] += dist
                elif 45 < angle <= 135:
                    votes[270] += dist # 90도 시계방향 상태 -> 270도 회전 필요
                elif angle > 135 or angle <= -135:
                    votes[180] += dist
                elif -135 < angle < -45:
                    votes[90] += dist

        # 투표 결과 분석
        if not votes or max(votes.values()) == 0:
            return 0
            
        final_rotation = max(votes, key=votes.get)
        self.log(f"   - 방향 분석 결과: {votes} -> {final_rotation}도 회전 결정")
        return final_rotation

    def _reconstruct_table_with_structure(self, ocr_res, structure_cells, vis_img=None):
        """
        [지능형 하이브리드 표 복원]
        1. 텍스트 크기 통계 분석 (글자당 평균 너비/높이 계산)
        2. 사용자 정의 규칙 적용 (행: 70% 높이 차이 / 열: 10글자 공백 or 물리적 선 존재)
        3. 물리적 셀 정보(utils 결과)를 열 구분 결정적인 힌트로 사용
        """
        # 1. OCR 텍스트 파싱 및 통계 수집
        ocr_items = []
        total_w, total_h, char_count, item_count = 0, 0, 0, 0
        
        for img_res in ocr_res.get('images', []):
            for field in img_res.get('fields', []):
                text = field.get('inferText', '').strip()
                if not text: continue
                
                verts = field.get('boundingPoly', {}).get('vertices', [])
                x1 = min([v.get('x', 0) for v in verts])
                y1 = min([v.get('y', 0) for v in verts])
                x2 = max([v.get('x', 0) for v in verts])
                y2 = max([v.get('y', 0) for v in verts])
                
                w, h = x2 - x1, y2 - y1
                if char_count < 1000: # 통계용 샘플링
                    total_w += w
                    total_h += h
                    char_count += len(text)
                    item_count += 1
                
                ocr_items.append({
                    'text': text,
                    'bbox': (int(x1), int(y1), int(x2), int(y2)),
                    'center': ((x1 + x2) / 2, (y1 + y2) / 2),
                    'height': h
                })

        if not ocr_items: return []

        # 통계치 계산 (기본값 설정)
        avg_char_w = (total_w / char_count) if char_count > 0 else 15
        avg_char_h = (total_h / item_count) if item_count > 0 else 25
        
        self.log(f"   - 텍스트 통계: 평균 너비 {avg_char_w:.1f}px, 평균 높이 {avg_char_h:.1f}px")

        # [신규] 표 영역(Table Area) 감지 및 외부 텍스트 필터링
        # 사용자 요청: 표에 들어있지 않은 부분(제목 등)을 제외하여 표 판정 품질 향상
        if structure_cells:
            st_y1 = min(c[1] for c in structure_cells)
            st_y2 = max(c[3] for c in structure_cells)
            st_x1 = min(c[0] for c in structure_cells)
            st_x2 = max(c[2] for c in structure_cells)
            
            # 표 바깥 영역에서 표와 무관한 텍스트 제외
            # 마진: 글자 높이의 0.5배 정도로 축소 (제목 등 상단 텍스트 유입 차단)
            margin_y = avg_char_h * 0.5
            margin_x = avg_char_w * 3.0 # 가로로는 조금 더 넉넉하게
            
            table_filtered_items = []
            for item in ocr_items:
                ix1, iy1, ix2, iy2 = item['bbox']
                
                # '완전 외부' 판정: 표 박스들 전체 영역에서 마진 이상 벗어난 경우
                is_outside_y = (iy2 < st_y1 - margin_y) or (iy1 > st_y2 + margin_y)
                is_outside_x = (ix2 < st_x1 - margin_x) or (ix1 > st_x2 + margin_x)
                
                if is_outside_y or is_outside_x:
                    continue
                table_filtered_items.append(item)
            
            removed_count = len(ocr_items) - len(table_filtered_items)
            if removed_count > 0:
                self.log(f"   - [영역 필터] 표 외부 텍스트 {removed_count}개 제외 (제목 등)")
                ocr_items = table_filtered_items

        # -- 1. 행(Row) 탐지 (개선됨: 중앙값 클러스터링) --
        # 사용자 질문: "분명히 다른 행인데 왜 합쳐질까?"
        # 원인: 기존 로직은 '인접한 텍스트끼리' 비교하여 묶는 방식이라, 
        #      중간에 살짝 걸친 텍스트(다리 역할)가 있으면 위아래 행이 사슬처럼 엮여서(Chaining) 합쳐짐.
        # 해결: '행의 평균 중심선'을 기준으로 엄격하게 거리를 체크하는 방식(Clustering)으로 변경.
        
        rows = []
        # Y좌표 정렬
        sorted_items = sorted(ocr_items, key=lambda x: x['center'][1])
        
        # 행 분리 임계값 설정: 글자 높이의 60%
        # (이 값이 너무 크면 행이 합쳐지고, 너무 작으면 같은 행의 들쑥날쑥한 글자가 쪼개짐)
        y_threshold = avg_char_h * 0.6
        
        for item in sorted_items:
            best_row = None
            min_dist = float('inf')
            
            for row in rows:
                # 현재 형성된 행의 평균 Y 중심값 계산
                row_cy = sum(t['center'][1] for t in row) / len(row)
                dist = abs(item['center'][1] - row_cy)
                
                # 거리 조건 만족 시 후보 등록 (가장 가까운 행 찾기)
                if dist < y_threshold:
                    if dist < min_dist:
                        min_dist = dist
                        best_row = row
            
            if best_row is not None:
                best_row.append(item)
            else:
                rows.append([item])
        
        # 최종 행 정렬
        rows.sort(key=lambda r: sum(t['center'][1] for t in r)/len(r) if r else 0)

        # -- 2. 열(Column) 탐지 (점선/끊어진 선 통합 대응) --
        # [고도화] 테이블 전체 높이 계산 및 강력한 선(Strong Line) 기준 수립
        all_y_coords = [item['bbox'][1] for item in ocr_items] + [item['bbox'][3] for item in ocr_items]
        table_height = max(all_y_coords) - min(all_y_coords) if all_y_coords else 1000

        # 1. 모든 수직 선분 수집 (왼쪽/오른쪽 벽)
        v_segments = []
        for c in structure_cells:
            h = c[3] - c[1]
            if h > avg_char_h * 0.5: # 글자 높이 절반 이상의 선분 수집
                v_segments.append({'x': c[0], 'y1': c[1], 'y2': c[3]})
                v_segments.append({'x': c[2], 'y1': c[1], 'y2': c[3]})
        
        # 2. X좌표 기준 군집화 (근접한 선분끼리 그룹)
        v_segments.sort(key=lambda s: s['x'])
        physical_v_lines = []
        very_strong_v_lines = [] # 테이블 높이의 60% 이상을 차지하는 아주 강력한 선
        
        self.log(f"   - 수직 선분 조각 {len(v_segments)-1}개 분석 중...") # 더미 제외
        
        if v_segments:
            current_group = [v_segments[0]]
            # 마지막 그룹 처리를 위해 더미 데이터 추가
            v_segments.append({'x': v_segments[-1]['x'] + 9999, 'y1': 0, 'y2': 0})
            
            for i in range(1, len(v_segments)):
                curr = v_segments[i]
                prev = current_group[-1]
                
                # [개선] X좌표 허용 오차 확대 (5px -> 12px)
                # 이미지 노이즈나 약간의 기울기로 인해 세그먼트들이 일직선상에서 벗어날 수 있음
                if abs(curr['x'] - prev['x']) <= 12:
                    current_group.append(curr)
                else:
                    # -- 그룹 분석 --
                    avg_x = sum(s['x'] for s in current_group) / len(current_group)
                    
                    # Y축 커버 범위 계산 (Union Length)
                    y_intervals = sorted([(s['y1'], s['y2']) for s in current_group])
                    merged_len = 0
                    merged_segments = [] 
                    
                    if y_intervals:
                        start, end = y_intervals[0]
                        # 병합 로직: 15px 이내의 끊김은 하나로 합침
                        for next_start, next_end in y_intervals[1:]:
                            if next_start < end + 15: 
                                end = max(end, next_end)
                            else:
                                merged_len += (end - start)
                                merged_segments.append((start, end))
                                start, end = next_start, next_end
                        merged_len += (end - start)
                        merged_segments.append((start, end))
                    
                    # [지능적 필터] 이 선이 텍스트의 '중심부'를 관통하는지 체크
                    center_penetration_count = 0
                    for (my1, my2) in merged_segments:
                        for item in ocr_items:
                            iy1, iy2 = item['bbox'][1], item['bbox'][3]
                            if max(my1, iy1) < min(my2, iy2): # Y Overlap 존재
                                ix1, ix2 = item['bbox'][0], item['bbox'][2]
                                iw = ix2 - ix1
                                # 관통 판정: 선이 텍스트 좌우 25% 안쪽으로 들어오면 관통으로 간주
                                if ix1 + (iw * 0.25) < avg_x < ix2 - (iw * 0.25):
                                    center_penetration_count += 1
                    
                    # 판정 로직
                    coverage_ratio = merged_len / table_height
                    is_v_strong = coverage_ratio > 0.30
                    
                    # [개선] 물리적 선 신뢰도 강화: 점유율이 높을수록 텍스트 관통 허용치를 대폭 확대
                    # 1230px 같이 아주 긴 선은 OCR이 텍스트를 잘못 묶었을 가능성이 매우 높으므로 물리력을 우선함
                    if coverage_ratio > 0.50:
                        penetration_limit = len(rows) # 점유율 50%만 넘어도 거의 무제한 허용 (표 전체를 관통할 가능성 높음)
                    elif is_v_strong:
                        penetration_limit = max(10, int(len(rows) * 0.4)) # 30% 이상은 최소 10개 또는 40%
                    else:
                        penetration_limit = max(3, int(len(rows) * 0.1))
                    
                    if center_penetration_count <= penetration_limit:
                        if merged_len > avg_char_h * 1.5: # 최소 길이 조건 소폭 더 완화
                            physical_v_lines.append(avg_x)
                            if is_v_strong:
                                very_strong_v_lines.append(avg_x)
                                self.log(f"     [강력 선] X={int(avg_x)}, 길이={int(merged_len)}px (점유율 {coverage_ratio:.1%})")
                    else:
                        if is_v_strong:
                            self.log(f"     [강력 선 기각] X={int(avg_x)}, 관통수={center_penetration_count}, 길이={int(merged_len)}px (허용치={penetration_limit})")
                    
                    current_group = [curr]
            
            # [수정] 마지막 그룹 처리 루프 누락 수정
            final_groups_to_process = [current_group] if current_group else []
            for g in final_groups_to_process:
                avg_x = sum(s['x'] for s in g) / len(g)
                y_intervals = sorted([(s['y1'], s['y2']) for s in g])
                merged_len = 0
                merged_segments = [] 
                if y_intervals:
                    start, end = y_intervals[0]
                    for next_start, next_end in y_intervals[1:]:
                        if next_start < end + 15: 
                            end = max(end, next_end)
                        else:
                            merged_len += (end - start)
                            merged_segments.append((start, end))
                            start, end = next_start, next_end
                    merged_len += (end - start)
                    merged_segments.append((start, end))
                
                center_penetration_count = 0
                for (my1, my2) in merged_segments:
                    for item in ocr_items:
                        iy1, iy2 = item['bbox'][1], item['bbox'][3]
                        if max(my1, iy1) < min(my2, iy2):
                            ix1, ix2 = item['bbox'][0], item['bbox'][2]
                            iw = ix2 - ix1
                            if ix1 + (iw * 0.25) < avg_x < ix2 - (iw * 0.25):
                                center_penetration_count += 1
                
                coverage_ratio = merged_len / table_height
                is_v_strong = coverage_ratio > 0.30
                
                if coverage_ratio > 0.50:
                    penetration_limit = len(rows)
                elif is_v_strong:
                    penetration_limit = max(10, int(len(rows) * 0.4))
                else:
                    penetration_limit = max(3, int(len(rows) * 0.1))
                
                if center_penetration_count <= penetration_limit:
                    if merged_len > avg_char_h * 1.5:
                        physical_v_lines.append(avg_x)
                        if is_v_strong:
                            very_strong_v_lines.append(avg_x)
                            self.log(f"     [강력 선] X={int(avg_x)}, 길이={int(merged_len)}px (점유율 {coverage_ratio:.1%})")
                elif is_v_strong:
                    self.log(f"     [강력 선 기각] X={int(avg_x)}, 관통수={center_penetration_count}, 길이={int(merged_len)}px (허용치={penetration_limit})")
        
        # [디버그] 강력한 선 탐지 결과 요약 출력
        if very_strong_v_lines:
            v_strong_x_list = sorted([int(vx) for vx in very_strong_v_lines])
            self.log(f"   - [초기 감지] 아주 강력한 선 X좌표 ({len(v_strong_x_list)}개): {v_strong_x_list}")
        else:
            self.log("   - [초기 감지] 아주 강력한 선이 발견되지 않았습니다.")
            
        self.log(f"   - 물리적 구분선 전체 {len(physical_v_lines)}개 탐지됨")

        # 2. 텍스트 배치를 분석하여 '열 후보'를 행별로 먼저 클러스터링
        # [개선] 전체 텍스트 덩어리가 아닌, 각 행(Row) 내부에서 먼저 덩어리를 만들어 '계단형 병합' 방지
        text_clusters = [] # 시각화용 (각 행 내부의 텍스트 덩어리들)
        all_segment_starts = [] # 모든 행에서 발견된 덩어리 시작 X좌표들
        
        for row in rows:
            row_sorted = sorted(row, key=lambda t: t['bbox'][0])
            if not row_sorted: continue
            
            curr_seg = [row_sorted[0]]
            for i in range(1, len(row_sorted)):
                item = row_sorted[i]
                prev = curr_seg[-1]
                
                # [강화] 두 텍스트 사이에 물리적 구분선이 있는지 체크
                has_physical_divider = False
                for px in physical_v_lines:
                    # 1. 텍스트 박스 사이의 빈 공간에 선이 있는 경우
                    if prev['bbox'][2] < px < item['bbox'][0]:
                        has_physical_divider = True
                        break
                    # 2. 아주 강력한 선(V.strong)의 경우, 텍스트 박스가 선을 살짝 덮고 있더라도 
                    #    두 텍스트의 중심점 사이에 선이 있다면 분리 (사용자 요청)
                    is_very_strong = any(abs(vx - px) < 5 for vx in very_strong_v_lines)
                    if is_very_strong:
                        prev_cx = (prev['bbox'][0] + prev['bbox'][2]) / 2
                        curr_cx = (item['bbox'][0] + item['bbox'][2]) / 2
                        if prev_cx < px < curr_cx:
                            has_physical_divider = True
                            break
                
                # 같은 행 내에서 공백이 '1.5글자' 미만이고 사이에 선이 없으면 같은 셀 내용으로 묶음
                if item['bbox'][0] < prev['bbox'][2] + (avg_char_w * 1.5) and not has_physical_divider:
                    curr_seg.append(item)
                else:
                    text_clusters.append(curr_seg)
                    all_segment_starts.append(curr_seg[0]['bbox'][0])
                    curr_seg = [item]
            text_clusters.append(curr_seg)
            all_segment_starts.append(curr_seg[0]['bbox'][0])

        # 3. 전역 열 경계(Boundary) 확정
        # [개선] X-좌표 군집화(Support Voting)를 통해 노이즈에 강한 열 경계 생성
        # 물리적 선도 '강력한 후보'로서 초기부터 포함시킵니다.
        all_candidates_raw = all_segment_starts + physical_v_lines
        all_candidates_raw.sort()
        
        col_candidates = []
        if all_candidates_raw:
            curr_group = [all_candidates_raw[0]]
            for x in all_candidates_raw[1:]:
                # 이전 좌표와 1글자 너비 이내로 가깝다면 그룹화 (물리적 선이 텍스트 시작점과 겹칠 수 있음)
                if x - curr_group[-1] < avg_char_w:
                    curr_group.append(x)
                else:
                    col_candidates.append(curr_group)
                    curr_group = [x]
            col_candidates.append(curr_group)

        col_boundaries = []
        for group in col_candidates:
            # 그룹 대표값 (평균 대신 최소값 사용하거나, 물리적 선이 있으면 그 위치 우선)
            rep_x = min(group)
            
            # 이 그룹 내에 물리적 선이 포함되어 있는지 여부
            physical_line_in_group = None
            for gx in group:
                for px in physical_v_lines:
                    if abs(px - gx) < 5:
                        physical_line_in_group = px
                        break
                if physical_line_in_group is not None: break
            
            # 아주 강력한 선이 포함되어 있는지 확인
            is_very_strong = False
            if physical_line_in_group is not None:
                is_very_strong = any(abs(px - physical_line_in_group) < 5 for px in very_strong_v_lines)
            
            # 텍스트 시작점 개수 (Counting)
            text_start_count = sum(1 for gx in group if gx in all_segment_starts)
            
            # [결정 로직 - 1단계: 후보군 선정]
            # 1. 아주 강력한 물리적 선이면 무조건 채택
            # 2. 일반 물리적 선이 있거나 텍스트 시작점 투표수가 일정 이상이면 채택
            # [개선] Gap 선(물리적 선 없음)의 경우, 더 높은 지지율(행의 25% 이상)을 요구하여 
            #       단순히 한두 행에서 왼쪽/오른쪽 맞춤으로 인해 생긴 빈 공간을 무시합니다.
            min_support_for_gap = max(3, int(len(rows) * 0.25))
            min_support_for_phys = max(2, int(len(rows) * 0.1))
            
            if is_very_strong or physical_line_in_group is not None:
                is_selected = True
            elif text_start_count >= min_support_for_gap:
                is_selected = True
            else:
                is_selected = False

            if is_selected:
                # 물리적 선이 있으면 그 정확한 위치를 사용, 아니면 그룹 최소값 사용
                best_x = physical_line_in_group if physical_line_in_group is not None else rep_x
                col_boundaries.append(best_x - 5 if physical_line_in_group is None else best_x)
        
        col_boundaries.sort()

        # [기존 로직 유지하되 안전장치 강화] 경계선 검증 로직 고도화
        # 단순히 하나라도 관통하면 제거하는 것이 아니라, 전체적인 '관통 점수'를 계산합니다.
        safe_boundaries = []
        total_rows = len(rows) if rows else 1
        
        for x in col_boundaries:
            # [사용자 요청 1번] 아주 강력한 선이면 검증 단계를 대폭 완화하거나 통과
            is_very_strong_here = any(abs(vx - x) < 10 for vx in very_strong_v_lines)
            
            penetration_count = 0
            
            # 1. 이 후보 경계선이 '물리적 선' 근처인지 확인
            is_physical_supported = any(abs(px - x) < 15 for px in physical_v_lines)
            
            # 2. 텍스트 관통 횟수 계산
            for item in ocr_items:
                x1, _, x2, _ = item['bbox']
                width = x2 - x1
                core_margin = width * 0.20
                if x1 + core_margin < x < x2 - core_margin:
                    penetration_count += 1
            
            # 3. 결정 로직 (Thresholding)
            if is_very_strong_here:
                # [개선] 아주 강력한 선은 텍스트 관통 검사를 거의 하지 않음 (이미 위에서 더 정교하게 검사됨)
                allowable_penetration = total_rows 
            elif is_physical_supported:
                allowable_penetration = max(3, int(total_rows * 0.25)) 
            else:
                # [개선] 물리적 선이 '현저히' 없는 경우 (Gap 선) 필터링 강화
                # 단순히 텍스트를 관통하지 않는 것만으로는 부족하며, 
                # 주변에 아주 짧은 물리적 흔적(v_segments)이라도 있는지 확인합니다.
                has_any_physical_hint = any(abs(s['x'] - x) < 10 for s in v_segments[:-1]) # 더미 제외
                
                if not has_any_physical_hint:
                    # 물리적 흔적이 아예 없는 순수 공백은 무조건 제거 (사용자 요청)
                    continue 

                allowable_penetration = 0 
            
            if penetration_count <= allowable_penetration:
                safe_boundaries.append(x)
        
        col_boundaries = safe_boundaries

        # [추가 보정] 물리적 선이 존재하는데도 후보에 포함되지 않은 경우 강제 추가
        # (단, 텍스트를 대놓고 가르지 않는 경우에 한해)
        for px in physical_v_lines:
            # 이미 비슷한 위치에 경계가 있으면 스킵
            if any(abs(b - px) < 10 for b in col_boundaries):
                continue
                
            p_penetration_count = 0
            for item in ocr_items:
                ix1, _, ix2, _ = item['bbox']
                i_width = ix2 - ix1
                if ix1 + (i_width * 0.25) < px < ix2 - (i_width * 0.25):
                    p_penetration_count += 1
            
            # 물리적 선은 신뢰도가 높으므로 꽤 많이 관통해도(30%) 살림
            if p_penetration_count <= max(3, int(total_rows * 0.3)):
                col_boundaries.append(px)
        
        col_boundaries.sort()

        # [최종 필터: 열 점유율 검사] 
        # 사용자 요청: 열 구분선 사이에 텍스트 박스 영역이 대부분 들어오는 경우만 해당 열로 인정
        # 텍스트가 거의 없거나 조금만 걸쳐 있는 "가짜 열"들을 제거합니다.
        if col_boundaries:
            img_w = vis_img.shape[1] if vis_img is not None else 5000
            occupied_indices = set()
            
            for item in ocr_items:
                ix1, _, ix2, _ = item['bbox']
                # 텍스트의 중심점을 기준으로 어느 열에 속할지 1차 판정
                icx = (ix1 + ix2) / 2
                
                col_idx = -1
                for i in range(len(col_boundaries) - 1, -1, -1):
                    if icx >= col_boundaries[i]:
                        col_idx = i
                        break
                
                if col_idx != -1:
                    # 해당 열의 물리적 범위 [curr_b, next_b]
                    curr_b = col_boundaries[col_idx]
                    next_b = col_boundaries[col_idx + 1] if col_idx < len(col_boundaries) - 1 else img_w
                    
                    # 텍스트 박스가 이 열 구간과 겹치는 너비 계산
                    overlap_w = max(0, min(ix2, next_b) - max(ix1, curr_b))
                    # 텍스트 너비의 50% 이상이 이 열 안에 들어오면 "정상적으로 점유된 열"로 간주
                    if overlap_w > (ix2 - ix1) * 0.5:
                        occupied_indices.add(col_idx)
            
            # 실질적으로 텍스트를 "주로" 포함하고 있는 열 경계만 남김
            # [개선] 아주 강력한 선(V.Strong)은 텍스트 점유 여부와 상관없이 유지합니다. (물리적 실체 우선)
            final_boundaries = []
            for i, b in enumerate(col_boundaries):
                is_v_strong_here = any(abs(vx - b) < 10 for vx in very_strong_v_lines)
                if i in occupied_indices or is_v_strong_here:
                    final_boundaries.append(b)
            col_boundaries = final_boundaries

        # [최종 보정] 아주 강력한 선(V.Strong) 강제 복구
        # 필터링 과정에서 실수로 누락된 경우를 방지하기 위해, 탐지된 모든 V.Strong 선을 확인하여 추가합니다.
        for vx in very_strong_v_lines:
            if not any(abs(b - vx) < 15 for b in col_boundaries):
                self.log(f"   - [강제 복구] 누락된 아주 강력한 선 X={int(vx)}를 열 경계에 강제 추가합니다.")
                col_boundaries.append(vx)
        
        col_boundaries.sort()

        # [추가 개선] 너무 가까운 열 경계들 병합
        # 사용자 요청: '헤더 글자 길이 중 가장 작은 길이'보다 꽤나 작은 간격으로 그려진 애들은 하나로 병합
        
        # 1. 헤더(첫 행)에서 가장 작은 아이템 가로 너비 찾기
        min_header_item_w = avg_char_w * 2.0 # 기본값
        target_header_item = None
        
        # [추가] 표 구조의 상단 경계 파악 (제목 등 외부 텍스트가 헤더로 오인되는 것 방지)
        st_y1 = min(c[1] for c in structure_cells) if structure_cells else -1

        if rows:
            for r in rows:
                # [개선] 헤더 후보 조건: 
                # 1. 유의미한 데이터가 있는 행 (2개 이상 아이템)
                # 2. Y좌표가 표 상단 경계(st_y1)보다 아래에 있어야 함
                row_ay = sum(t['center'][1] for t in r) / len(r) if r else 0
                is_valid_pos = (st_y1 == -1) or (row_ay > st_y1 - 5)

                if len(r) >= 2 and is_valid_pos: # 유의미한 데이터가 있는 첫 행을 헤더로 간주
                    # 각 아이템의 정보를 담아 최소폭 찾기
                    items_with_info = []
                    for t in r:
                        w = t['bbox'][2] - t['bbox'][0]
                        items_with_info.append({'width': w, 'text': t['text'], 'bbox': t['bbox']})
                    
                    if items_with_info:
                        # 폭이 가장 작은 아이템 선택
                        min_item = min(items_with_info, key=lambda x: x['width'])
                        min_header_item_w = min_item['width']
                        target_header_item = min_item
                        break
        
        # 2. 병합 기준값 결정: 헤더 최소 너비의 60% 또는 최소 한 글자 너비 중 큰 값
        # '꽤나 작은 간격'을 약 60% 정도로 설정
        merge_threshold = max(avg_char_w, min_header_item_w * 0.6)
        
        if target_header_item:
            self.log(f"   - [헤더 분석] 기준 헤더: '{target_header_item['text']}' (폭: {int(min_header_item_w)}px, 위치: {target_header_item['bbox']})")
        
        self.log(f"   - [병합 기준] 병합 임계값={int(merge_threshold)}px (헤더 최소폭의 60% 적용)")

        if len(col_boundaries) > 1:
            merged_final = []
            i = 0
            while i < len(col_boundaries):
                group = [col_boundaries[i]]
                j = i + 1
                # 다음 선과의 간격이 merge_threshold보다 작으면 같은 그룹으로 묶음
                while j < len(col_boundaries) and (col_boundaries[j] - group[-1]) < merge_threshold:
                    group.append(col_boundaries[j])
                    j += 1
                
                if len(group) > 1:
                    # 그룹 내에서 가장 '품질이 좋은' 선 선택
                    best_x = group[0]
                    min_p_score = 9999
                    
                    for x in group:
                        # 관통 점수 계산
                        p_score = 0
                        for item in ocr_items:
                            ix1, _, ix2, _ = item['bbox']
                            iw = ix2 - ix1
                            if ix1 + (iw * 0.25) < x < ix2 - (iw * 0.25):
                                p_score += 10
                            elif ix1 < x < ix2:
                                p_score += 1
                        
                        # V.Strong 선은 우선권 부여
                        is_vx = any(abs(vx - x) < 5 for vx in very_strong_v_lines)
                        if is_vx: p_score -= 100 
                        
                        if p_score < min_p_score:
                            min_p_score = p_score
                            best_x = x
                    
                    self.log(f"   - [열 병합] 간격 {int(group[-1]-group[0])}px 내 {len(group)}개 경계 병합 -> X={int(best_x)}")
                    merged_final.append(best_x)
                else:
                    merged_final.append(group[0])
                i = j
            col_boundaries = merged_final

        # -- 3. 데이터 매핑 및 Grid 생성 --
        num_cols = len(col_boundaries)
        table_grid = []
        
        # 열 경계가 하나도 없는 경우 (전체가 하나의 열)
        if num_cols == 0:
            for r in rows:
                r_sorted = sorted(r, key=lambda t: t['bbox'][0])
                table_grid.append([" ".join([t['text'] for t in r_sorted])])
        else:
            for r in rows:
                # [개선] 각 행 내의 텍스트들을 왼쪽(X)부터 정렬하여 셀 내 텍스트 순서 보장
                r_sorted = sorted(r, key=lambda t: t['bbox'][0])
                row_cells = [[] for _ in range(num_cols)]
                
                for t in r_sorted:
                    # [개선] 시작점(bbox[0]) 대신 중심점(Center X)을 사용하여 더 정확한 열 판단
                    ctx = (t['bbox'][0] + t['bbox'][2]) / 2
                    
                    col_idx = 0
                    # 가장 오른쪽 열 경계부터 확인하여 정적 할당
                    for i in range(num_cols - 1, -1, -1):
                        if ctx >= col_boundaries[i]:
                            col_idx = i
                            break
                    row_cells[col_idx].append(t['text'])
                
                # 비어있는 셀은 빈 문자열로 처리하여 엑셀 정렬 유지
                table_grid.append([" ".join(c) for c in row_cells])

        # -- 4. 시각화 (격자선 정렬 최적화) --
        if vis_img is not None:
            # 1. 물리적 셀 (Cyan - 투명도 느낌의 얇은 선)
            for (bx1, by1, bx2, by2) in structure_cells:
                cv2.rectangle(vis_img, (bx1, by1), (bx2, by2), (255, 255, 0), 1)

            # 2. 행(Row) 구분선 (Blue - 파란색)
            for i in range(len(rows)):
                r = rows[i]
                if not r: continue
                max_y2 = max(t['bbox'][3] for t in r)
                if i < len(rows) - 1 and rows[i+1]:
                    next_min_y1 = min(t['bbox'][1] for t in rows[i+1])
                    boundary_y = int((max_y2 + next_min_y1) / 2)
                else:
                    boundary_y = max_y2 + 5
                cv2.line(vis_img, (0, boundary_y), (vis_img.shape[1], boundary_y), (255, 0, 0), 2)
            
            # 3. 열(Column) 구분선 (Red/Orange/Yellow - 디지별 시각화)
            for cb in col_boundaries:
                # [디버그 시각화] 선의 속성에 따라 색상/두께 차별화
                is_very_strong = any(abs(vx - cb) < 10 for vx in very_strong_v_lines)
                is_physical = any(abs(px - cb) < 10 for px in physical_v_lines)
                
                if is_very_strong:
                    color = (0, 0, 255) # 진한 빨강 (BGR)
                    thickness = 3
                elif is_physical:
                    color = (0, 165, 255) # 주황색 (물리적 선 기반)
                    thickness = 2
                else:
                    color = (0, 255, 255) # 노란색 (순수 공백 기반)
                    thickness = 2
                
                cv2.line(vis_img, (int(cb), 0), (int(cb), vis_img.shape[0]), color, thickness)
                # 텍스트로 속성 표시
                label = "V.Strong" if is_very_strong else ("Phys" if is_physical else "Gap")
                cv2.putText(vis_img, label, (int(cb)+2, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            # 4. 개별 텍스트 박스/클러스터 시각화 (기존 동일)
            for t in ocr_items:
                cv2.rectangle(vis_img, (t['bbox'][0], t['bbox'][1]), (t['bbox'][2], t['bbox'][3]), (0, 255, 0), 1)

            # 5. 텍스트 클러스터 (Magenta - 열 덩어리 확인용)
            for cluster in text_clusters:
                cl_x1 = min(t['bbox'][0] for t in cluster)
                cl_y1 = min(t['bbox'][1] for t in cluster)
                cl_x2 = max(t['bbox'][2] for t in cluster)
                cl_y2 = max(t['bbox'][3] for t in cluster)
                cv2.rectangle(vis_img, (cl_x1, cl_y1), (cl_x2, cl_y2), (255, 0, 255), 1)

        return table_grid

    def _reconstruct_table_grid_fallback(self, ocr_res, vis_img=None):
        """
        [복구 및 강화] 선 탐지 성능을 유지하면서 계단 현상을 제거하는 전역 그리드 알고리즘
        1. 선 탐지 능력이 좋았던 utils.group_text_lines의 로직을 사용하되
        2. x_gap_threshold를 무력화하여 수평으로 멀리 떨어진 글자들을 하나의 행으로 묶습니다.
        """
        texts = []
        for img_res in ocr_res.get('images', []):
            for field in img_res.get('fields', []):
                text = field.get('inferText', '').strip()
                if not text: continue
                
                verts = field.get('boundingPoly', {}).get('vertices', [])
                x_coords = [v.get('x', 0) for v in verts]
                y_coords = [v.get('y', 0) for v in verts]
                x1, y1, x2, y2 = min(x_coords), min(y_coords), max(x_coords), max(y_coords)
                
                texts.append({
                    'text': text,
                    'bbox': (int(x1), int(y1), int(x2), int(y2)),
                    'center': ((x1 + x2) / 2, (y1 + y2) / 2),
                    'height': y2 - y1
                })

        if not texts: return []

        # -- 1. 행(Row) 탐지 (수평 거리 무시 버전) --
        # 이미지 너비(w)를 고려하여 x_gap_threshold를 매우 크게 설정
        img_w = vis_img.shape[1] if vis_img is not None else 5000
        # y_threshold를 10으로 더 낮추어 초정밀 행 구분 (계단 현상 원천 차단)
        rows = utils.group_text_lines(texts, y_threshold=10, x_gap_threshold=img_w)

        # -- 2. 전역 열(Column) 추출 (X축 전체 분석) --
        # 모든 텍스트의 시작점을 모아 전역적인 열 경계 확정
        all_texts = [f for r in rows for f in r]
        x_starts = sorted(list(set([t['bbox'][0] for t in all_texts])))
        final_col_points = []
        if x_starts:
            final_col_points = [x_starts[0] - 5]
            last_p = x_starts[0]
            for x in x_starts[1:]:
                # 열 간격 민감도를 10으로 낮추어 미세한 칸도 모두 감지
                if x - last_p > 10: 
                    final_col_points.append(x - 5)
                    last_p = x
        
        # -- 3. 격자 채우기 (행/열 1:1 매핑) --
        num_cols = len(final_col_points) if final_col_points else 1
        table_grid = []
        for r in rows:
            # 텍스트들을 열 인덱스별로 분류
            row_cells = [[] for _ in range(num_cols)]
            for f in r:
                col_idx = 0
                for i in range(len(final_col_points)-1, -1, -1):
                    if f['bbox'][0] >= final_col_points[i]:
                        col_idx = i
                        break
                row_cells[col_idx].append(f['text'])
            
            # 한 리스트(행)를 엑셀의 한 줄로 구성
            table_grid.append([" ".join(c) for c in row_cells])

        # -- 4. 시각화 (사용자가 확인한 선 구조를 그대로 투영) --
        if vis_img is not None:
            # 행 구분선 (파란색)
            for r in rows:
                if not r: continue
                avg_y = int(sum(f['center'][1] for f in r) / len(r))
                cv2.line(vis_img, (0, avg_y), (vis_img.shape[1], avg_y), (255, 0, 0), 1)
            
            # 열 구분선 (빨간색)
            for cp in final_col_points:
                cv2.line(vis_img, (int(cp), 0), (int(cp), vis_img.shape[0]), (0, 0, 255), 1)
            
            # 텍스트 박스 (초록색)
            for t in texts:
                cv2.rectangle(vis_img, (t['bbox'][0], t['bbox'][1]), (t['bbox'][2], t['bbox'][3]), (0, 255, 0), 1)

        return table_grid

    def open_excel(self):
        if self.last_excel_path and os.path.exists(self.last_excel_path):
            import subprocess, platform
            if platform.system() == "Windows":
                os.startfile(self.last_excel_path)
            elif platform.system() == "Darwin":
                subprocess.call(["open", self.last_excel_path])
            else:
                subprocess.call(["xdg-open", self.last_excel_path])

if __name__ == "__main__":
    root = tk.Tk()
    app = TableExtractorApp(root)
    root.mainloop()
