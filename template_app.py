# traffic_violation_ocr.py
# ------------------------------------------------------------
# 교통법규 위반 고지서 OCR 프로그램 (Google Cloud Vision API + Streamlit)
# 요구 반영:
# 1) 이미지 최대 10장 동시 업로드 및 일괄 처리
# 2) 경찰서/기타 고지서 구분: "경찰청 고지" 문자열 유무로만 판단(공백 포함/미포함 모두)
# 3) 결과의 "사용자"는 차량배정리스트의 "성명" 컬럼 사용(없으면 "사용자"를 성명으로)
# 4) 날짜 인식 포맷 다양한 입력 → 출력은 항상 YYYY/MM/DD, 연도 없으면 당해년도
# 5) 첫 화면에서 "템플릿 좌표 재지정" 버튼으로 리셋 후 다시 설정 가능
# 6) 중복/복잡 코드 정리 및 잠재 오류 완화
# ------------------------------------------------------------

import os
import io
import json
import re
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd
import streamlit as st
st.set_page_config(page_title="교통법규 위반 고지서 OCR", page_icon="🚗", layout="wide")

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.font_manager as fm
from PIL import Image, ImageDraw, ImageFont

# Google Cloud Vision API
try:
    from google.cloud import vision
    from google.oauth2 import service_account
except ImportError:
    st.error("📦 Google Cloud Vision API가 필요합니다. 터미널에서 다음 명령을 실행하세요:\n\npip install google-cloud-vision")
    st.stop()

# 클릭 캡처 컴포넌트
try:
    from streamlit_image_coordinates import streamlit_image_coordinates
except ImportError:
    st.error("📦 추가 패키지가 필요합니다. 터미널에서 다음 명령을 실행하세요:\n\npip install streamlit-image-coordinates")
    st.stop()

# =========================
# 날짜 포맷 유틸 (항상 YYYY/MM/DD, 연도 없으면 당해년도)
# =========================
def fmt_date_uniform(s: str) -> str:
    """
    다양한 입력(yyyy.mm.dd, yyyy-m-d, mm.dd, mm-dd, yyyyMMdd, yyMMdd, yyyy년 m월 d일 등)을
    항상 YYYY/MM/DD로 변환. 연도 미포함이면 당해년도 사용.
    완전 파싱 불가 시 'YYYY/00/00'(YYYY=당해년도) 반환.
    """
    cur_year = datetime.now().year
    if not s or str(s).strip() == "":
        return f"{cur_year:04d}/00/00"

    src = str(s)
    cleaned = re.sub(r"[^\d./\-년월일]", " ", src)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    Y = None; M = None; D = None

    # 0) 붙은 숫자: YYYYMMDD / YYMMDD
    m = re.search(r'(?<!\d)(\d{4})(\d{2})(\d{2})(?!\d)', cleaned)
    if m:
        Y, M, D = m.group(1), m.group(2), m.group(3)
    else:
        m = re.search(r'(?<!\d)(\d{2})(\d{2})(\d{2})(?!\d)', cleaned)
        if m:
            Y, M, D = "20" + m.group(1), m.group(2), m.group(3)
        else:
            # 1) 구분자 있는 Y-M-D
            m = re.search(r'(\d{2,4})\s*[-./]\s*(\d{1,2})\s*[-./]\s*(\d{1,2})', cleaned)
            if m:
                Y, M, D = m.group(1), m.group(2), m.group(3)
            else:
                # 2) 한글 표기
                yk = re.search(r'(\d{2,4})\s*년', cleaned)
                mk = re.search(r'(\d{1,2})\s*월', cleaned)
                dk = re.search(r'(\d{1,2})\s*일', cleaned)
                if yk or mk or dk:
                    Y = yk.group(1) if yk else None
                    M = mk.group(1) if mk else None
                    D = dk.group(1) if dk else None
                else:
                    # 3) 연도 없는 M-D
                    m2 = re.search(r'(?<!\d)(\d{1,2})\s*[-./]\s*(\d{1,2})(?!\d)', cleaned)
                    if m2:
                        M, D = m2.group(1), m2.group(2)
                    else:
                        # 4) 숫자만 나열
                        nums = re.findall(r'\d+', cleaned)
                        if len(nums) >= 2:
                            a, b = nums[0], nums[1]
                            if len(a) == 4 and len(b) in (3, 4):
                                Y = a
                                if len(b) == 3:
                                    M, D = b[0], b[1:]
                                else:
                                    M, D = b[:2], b[2:]
                            elif len(a) == 4 and len(b) <= 2:
                                Y, M = a, b
                            elif len(a) <= 2 and len(b) in (3, 4):
                                if len(b) == 3:
                                    M, D = b[0], b[1:]
                                else:
                                    M, D = b[:2], b[2:]
                            elif len(a) <= 2 and len(b) <= 2:
                                M, D = a, b
                        elif len(nums) == 1:
                            n = nums[0]
                            if len(n) == 4:
                                M, D = n[:2], n[2:]
                            elif len(n) == 3:
                                M, D = n[0], n[1:]
                            else:
                                M = n

    # 보정/정형화
    def norm_year(v):
        if v is None: return f"{cur_year:04d}"
        if len(v) == 2: v = "20" + v
        return f"{int(v):04d}"

    def norm_md(v):
        if v is None: return "00"
        return f"{int(v):02d}"

    try:
        YY = norm_year(Y)
        MM = norm_md(M)
        DD = norm_md(D)
        mi, di = int(MM), int(DD)
        if mi < 1 or mi > 12: MM = "00"
        if di < 1 or di > 31: DD = "00"
        return f"{YY}/{MM}/{DD}"
    except Exception:
        return f"{cur_year:04d}/00/00"

# =========================
# Google Cloud Vision API 설정
# =========================
@st.cache_resource
def setup_google_cloud_vision():
    """Google Cloud Vision API 설정 및 클라이언트 초기화
    우선순위: 1) st.secrets 2) 환경변수 3) 하드코딩 경로
    """
    try:
        # 1) Streamlit secrets (Cloud에서 사용)
        creds_info = None
        try:
            # Cloud가 아니면 여기에서 KeyError가 날 수 있음
            creds_info = st.secrets.get("gcp_service_account", None)
        except Exception:
            creds_info = None

        if creds_info:
            from google.oauth2 import service_account
            credentials = service_account.Credentials.from_service_account_info(dict(creds_info))
            client = vision.ImageAnnotatorClient(credentials=credentials)
            st.success("✅ Vision 클라이언트(secrets) 초기화 완료")
            return client

        # 2) 환경변수 (로컬에서 추천)
        cred_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        if cred_path and os.path.exists(cred_path):
            client = vision.ImageAnnotatorClient()
            st.success(f"✅ Vision 클라이언트(env) 초기화 완료: {os.path.basename(cred_path)}")
            return client

        # 3) 하드코딩된 로컬 경로(최후 보루)
        fallback = r"C:\Users\150403\traffic ocr\traffic-ocr85b3ba21d821.json"
        if os.path.exists(fallback):
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = fallback
            client = vision.ImageAnnotatorClient()
            st.success(f"✅ Vision 클라이언트(local path) 초기화 완료: {os.path.basename(fallback)}")
            return client

        # 전부 실패
        st.error(
            "⚠️ Google Cloud Vision 인증 정보가 없습니다.\n"
            "다음 중 하나로 설정하세요:\n"
            "1) PowerShell에서 환경변수 설정\n"
            '   $env:GOOGLE_APPLICATION_CREDENTIALS="C:\\경로\\키파일.json"\n'
            "2) 프로젝트/.streamlit/secrets.toml 에 gcp_service_account 추가\n"
        )
        return None

    except Exception as e:
        st.error(f"❌ Google Cloud Vision API 초기화 실패: {e}")
        return None

# ↓↓↓↓↓ 반드시 '맨 왼쪽'에서 시작(들여쓰기 금지) ↓↓↓↓↓
@st.cache_resource
def get_vision_client():
    """Vision API 클라이언트 가져오기 (없으면 초기화 중지)"""
    client = setup_google_cloud_vision()
    if client is None:
        st.error("🚫 Google Cloud Vision API 클라이언트를 만들 수 없습니다.")
        st.stop()
    return client

# =========================
# 폰트 설정 (맑은 고딕 고정)
# =========================
@st.cache_resource
def setup_korean_font():
    malgun_paths = [
        "C:/Windows/Fonts/malgun.ttf",
        "C:/Windows/Fonts/malgunbd.ttf",
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    ]
    chosen_font = None
    for font_path in malgun_paths:
        if os.path.exists(font_path):
            chosen_font = font_path
            break
    pil_font = None
    font_prop = None
    try:
        if chosen_font:
            fm.fontManager.addfont(chosen_font)
            font_prop = fm.FontProperties(fname=chosen_font)
            mpl.rcParams["font.family"] = font_prop.get_name()
            mpl.rcParams["axes.unicode_minus"] = False
            pil_font = ImageFont.truetype(chosen_font, size=18)
        else:
            mpl.rcParams["axes.unicode_minus"] = False
            st.warning("⚠️ 맑은 고딕 폰트를 찾을 수 없습니다. 시스템 기본 폰트를 사용합니다.")
    except Exception as e:
        st.warning(f"맑은 고딕 폰트 로드 실패: {e}")
    return font_prop, pil_font, chosen_font

KFONT_PROP, PIL_FONT, KFONT_PATH = setup_korean_font()

# =========================
# 상태 초기화
# =========================
def init_session_state():
    defaults = {
        "current_step": "upload_files",  # upload_files, template_setup, ocr_process, results
        "vehicle_users_df": None,
        "uploaded_image": None,          # 대표 이미지(템플릿 설정용)
        "uploaded_images": [],           # 여러 장의 이미지
        "uploaded_image_names": [],      # 파일명 리스트
        "template_exists": False,
        "coordinates": {},
        "current_field_index": 0,
        "temp_coords": [],
        "click_step": 0,
        "display_width": 800,
        "last_click_sig": None,
        "ocr_results": {},
        "final_results": None,
        "batch_results": [],
        "full_ocr_text": "",
        "is_police_notice": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

# =========================
# 상수 정의
# =========================
FIELDS = ["차량번호", "일자", "장소", "과태료", "납기일", "내용"]
FIELD_COLORS = {
    "차량번호": "#FF0000",
    "일자": "#0000FF",
    "장소": "#00FF00",
    "과태료": "#FFA500",
    "납기일": "#800080",
    "내용": "#FF69B4",
}
FIELD_DESCRIPTIONS = {
    "차량번호": "차량번호는 '000가0000' 또는 '00가0000' 형식입니다.",
    "일자": "일자는 'YYYY/MM/DD' 형식의 날짜입니다.",
    "장소": "장소는 도로명, 지명 또는 'CCTV', 'IC' 등이 포함됩니다.",
    "과태료": "과태료는 10,000~500,000원 사이의 금액입니다.",
    "납기일": "납기일은 'YYYY/MM/DD' 형식의 납부 기한입니다.",
    "내용": "내용은 '주정차', '속도', '신호', '어린이보호구역' 등이 포함됩니다.",
}

# =========================
# 파일 업로드 섹션
# =========================
def file_upload_section():
    st.markdown('''
    <div style="padding:2rem;border:2px dashed #1f77b4;border-radius:10px;text-align:center;margin:2rem 0;">
    <h3>📁 1단계: 필수 파일 업로드</h3>
    <p>차량 사용자 파일과 교통법규 위반 고지서 이미지를 업로드하세요</p>
    </div>
    ''', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    # 좌: 차량 사용자 파일
    with col1:
        st.markdown("### 🚗 차량 사용자 파일")
        st.info("차량번호, 성명, 부서 정보가 포함된 엑셀 파일을 업로드하세요")
        vehicle_file = st.file_uploader(
            "차량 사용자 엑셀 파일",
            type=["xlsx", "xls"],
            key="vehicle_file",
            help="차량번호, 성명, 부서 컬럼이 포함된 엑셀 파일"
        )
        if vehicle_file is not None:
            try:
                df = pd.read_excel(vehicle_file)
                if "성명" not in df.columns and "사용자" in df.columns:
                    df["성명"] = df["사용자"]
                st.session_state.vehicle_users_df = df
                st.success(f"✅ 파일 로드 완료! (총 {len(df)}개 차량)")
                st.dataframe(df.head(), use_container_width=True)

                required = ["차량번호", "성명", "부서"]
                missing = [c for c in required if c not in df.columns]
                if missing:
                    st.error(f"❌ 필수 컬럼이 없습니다: {missing}")
                else:
                    st.success("✅ 모든 필수 컬럼이 확인되었습니다!")
            except Exception as e:
                st.error(f"❌ 파일 읽기 오류: {e}")

    # 우: 고지서 이미지 (최대 10장)
    with col2:
        st.markdown("### 📄 교통법규 위반 고지서 이미지")
        st.info("최대 10장까지 동시에 업로드 가능합니다.")
        image_files = st.file_uploader(
            "고지서 이미지 파일(최대 10장)",
            type=["png", "jpg", "jpeg"],
            key="image_files",
            accept_multiple_files=True,
            help="PNG, JPG, JPEG 형식을 지원합니다"
        )
        if image_files:
            if len(image_files) > 10:
                st.error("❌ 최대 10장까지만 업로드할 수 있어요.")
            else:
                try:
                    imgs, names = [], []
                    for f in image_files:
                        imgs.append(Image.open(f).convert("RGB"))
                        names.append(getattr(f, "name", ""))
                    st.session_state.uploaded_images = imgs
                    st.session_state.uploaded_image_names = names
                    st.session_state.uploaded_image = imgs[0]
                    st.success(f"✅ 이미지 {len(imgs)}장 로드 완료!")
                    st.image(imgs[0], caption=f"대표(1번) 이미지: {names[0] if names else ''}", use_container_width=True)
                    st.info(f"📊 대표 이미지 크기: {imgs[0].size[0]} × {imgs[0].size[1]} px")
                except Exception as e:
                    st.error(f"❌ 이미지 읽기 오류: {e}")

    st.markdown("---")

    # ✅ 첫 화면에서 템플릿 좌표 재지정 버튼
    st.markdown("### 🧭 템플릿 좌표 재지정")
    st.caption("저장된 좌표를 지우고 다시 지정합니다. (이미지 업로드 후 사용 가능)")
    can_reassign = bool(st.session_state.uploaded_images)
    if st.button("🧹 템플릿 좌표 재지정", type="primary", use_container_width=True,
                 disabled=not can_reassign, key="first_reset_reassign"):
        reset_template()
        st.session_state.current_step = "template_setup"
        st.rerun()

    # 다음 단계
    if st.session_state.vehicle_users_df is not None and st.session_state.uploaded_images:
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            if st.button("🎯 다음 단계: 템플릿 설정", type="primary", use_container_width=True):
                template_file = "police_template.json"
                if os.path.exists(template_file):
                    st.session_state.template_exists = True
                    try:
                        with open(template_file, 'r', encoding='utf-8') as f:
                            template_data = json.load(f)
                            st.session_state.coordinates = template_data.get("coordinates", {})
                        st.success("✅ 기존 템플릿을 로드했습니다!")
                        st.session_state.current_step = "ocr_process"
                    except Exception as e:
                        st.error(f"템플릿 로드 오류: {e}")
                        st.session_state.current_step = "template_setup"
                else:
                    st.session_state.template_exists = False
                    st.session_state.current_step = "template_setup"
                st.rerun()
    else:
        missing = []
        if st.session_state.vehicle_users_df is None:
            missing.append("차량 사용자 파일")
        if not st.session_state.uploaded_images:
            missing.append("고지서 이미지(최소 1장)")
        st.warning(f"⚠️ 다음 파일을 업로드해주세요: {', '.join(missing)}")

    if st.button("🔄 전체 초기화", type="secondary"):
        reset_all_states()
        st.rerun()

def reset_all_states():
    keys = [
        "current_step", "vehicle_users_df", "uploaded_image", "uploaded_images", "uploaded_image_names",
        "template_exists", "coordinates", "current_field_index", "temp_coords",
        "click_step", "display_width", "last_click_sig", "ocr_results", "final_results",
        "batch_results", "full_ocr_text", "is_police_notice"
    ]
    for k in keys:
        if k in st.session_state:
            if k == "current_step": st.session_state[k] = "upload_files"
            elif k == "display_width": st.session_state[k] = 800
            elif k in ["current_field_index", "click_step"]: st.session_state[k] = 0
            elif k in ["uploaded_images", "uploaded_image_names", "temp_coords", "batch_results"]: st.session_state[k] = []
            elif k in ["coordinates", "ocr_results"]: st.session_state[k] = {}
            else: st.session_state[k] = None

# =========================
# 템플릿 설정 섹션
# =========================
def template_setup_section():
    st.markdown('''
    <div style="padding:2rem;border:2px solid #ff7f0e;border-radius:10px;text-align:center;margin:2rem 0;">
    <h3>🎯 2단계: OCR 템플릿 설정</h3>
    <p>고지서 이미지에서 각 정보 영역의 좌표를 설정하세요</p>
    </div>
    ''', unsafe_allow_html=True)

    create_template_sidebar()
    if st.session_state.current_field_index < len(FIELDS):
        coordinate_selection_section()
    else:
        template_completion_section()

def create_template_sidebar():
    st.sidebar.title("📋 템플릿 설정 진행상황")

    progress = len(st.session_state.coordinates) / len(FIELDS) if FIELDS else 0
    st.sidebar.progress(progress)
    st.sidebar.write(f"**완료된 영역: {len(st.session_state.coordinates)}/{len(FIELDS)}**")

    if st.session_state.current_field_index < len(FIELDS):
        current_field = FIELDS[st.session_state.current_field_index]
        step_text = "좌상단 클릭 대기" if st.session_state.click_step == 0 else "우하단 클릭 대기"
        st.sidebar.markdown(f"**🎯 현재 설정: {current_field}**")
        st.sidebar.write(f"📍 단계: {step_text}")
        if current_field in FIELD_DESCRIPTIONS:
            st.sidebar.info(FIELD_DESCRIPTIONS[current_field])

    if st.session_state.uploaded_image:
        st.sidebar.markdown("### 🔍 이미지 크기 조절")
        original_width = st.session_state.uploaded_image.size[0]
        st.session_state.display_width = st.sidebar.slider(
            "표시 너비(px)", min_value=400, max_value=min(1200, original_width),
            value=st.session_state.display_width, step=50
        )
        scale = st.session_state.display_width / original_width
        st.sidebar.caption(f"스케일 비율: {scale:.2f}")
        st.sidebar.caption(f"원본 크기: {original_width} × {st.session_state.uploaded_image.size[1]}")

    if st.session_state.coordinates:
        st.sidebar.markdown("### ✅ 완료된 영역")
        for field, coords in st.session_state.coordinates.items():
            color = FIELD_COLORS[field]
            st.sidebar.markdown(f'<span style="color:{color}">●</span> **{field}**', unsafe_allow_html=True)
            st.sidebar.caption(f"   ({coords[0]}, {coords[1]}) → ({coords[2]}, {coords[3]})")

    st.sidebar.markdown("### 🛠️ 템플릿 관리")
    if st.sidebar.button("🔄 템플릿 초기화", type="secondary"):
        reset_template()
        st.rerun()
    if st.sidebar.button("⬅️ 파일 업로드로 돌아가기"):
        st.session_state.current_step = "upload_files"
        st.rerun()

def coordinate_selection_section():
    current_field = FIELDS[st.session_state.current_field_index]
    field_color = FIELD_COLORS[current_field]

    st.markdown(f'''
    <div style="font-size:1.3rem;padding:1rem;border:3px solid {field_color};border-radius:10px;
                background:linear-gradient(90deg, {field_color}20, transparent);text-align:center;margin:1rem 0;">
    📍 <b>{current_field}</b> 영역 설정 중... ({st.session_state.current_field_index + 1}/{len(FIELDS)})
    </div>
    ''', unsafe_allow_html=True)

    step_msg = "좌상단" if st.session_state.click_step == 0 else "우하단"
    st.markdown(f'''
    <div style="font-size:1.2rem;padding:1rem;border:2px solid #dc143c;border-radius:10px;
                background:#ffe4e1;text-align:center;margin:1rem 0;">
    🖱️ <b>{current_field}</b> 영역의 <b>{step_msg}</b>을 클릭하세요
    </div>
    ''', unsafe_allow_html=True)

    if current_field in FIELD_DESCRIPTIONS:
        st.info(f"💡 {FIELD_DESCRIPTIONS[current_field]}")

    create_clickable_image_with_coordinates(current_field, field_color)

    if st.session_state.temp_coords:
        txt = (f"✅ 좌상단 선택됨: {st.session_state.temp_coords[0]}"
               if len(st.session_state.temp_coords) == 1
               else f"✅ 선택 완료: {st.session_state.temp_coords[0]} → {st.session_state.temp_coords[1]}")
        st.markdown(f'''
        <div style="font-size:1.1rem;padding:1rem;border:2px solid #2e8b57;border-radius:8px;
                    background:#f0fff0;text-align:center;">{txt}</div>
        ''', unsafe_allow_html=True)

    create_control_buttons(current_field)

def create_image_with_overlays(current_field, field_color):
    image = st.session_state.uploaded_image.copy()
    draw = ImageDraw.Draw(image)

    # 기존 저장된 영역
    for field, coords in st.session_state.coordinates.items():
        x1, y1, x2, y2 = coords
        color = FIELD_COLORS[field]
        for i in range(4):
            draw.rectangle([x1 - i, y1 - i, x2 + i, y2 + i], outline=color)
        label = field
        if PIL_FONT:
            try:
                bbox = draw.textbbox((0, 0), label, font=PIL_FONT)
                tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
                bx1, by1 = x1, max(0, y1 - th - 10)
                bx2, by2 = x1 + tw + 10, max(0, y1 - 2)
                draw.rectangle([bx1, by1, bx2, by2], fill=color)
                draw.text((x1 + 5, by1 + 3), label, fill="white", font=PIL_FONT)
            except:
                draw.text((x1 + 5, y1 - 20), label, fill=color)

    # 현재 선택 중인 임시 영역
    if len(st.session_state.temp_coords) == 1:
        x, y = st.session_state.temp_coords[0]
        draw.line([x - 15, y, x + 15, y], fill=field_color, width=5)
        draw.line([x, y - 15, x, y + 15], fill=field_color, width=5)
        draw.ellipse([x - 10, y - 10, x + 10, y + 10], outline=field_color, width=3)
    elif len(st.session_state.temp_coords) == 2:
        x1, y1 = st.session_state.temp_coords[0]
        x2, y2 = st.session_state.temp_coords[1]
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        for i in range(6):
            draw.rectangle([x1 - i, y1 - i, x2 + i, y2 + i], outline=field_color)
        for px, py in [(x1, y1), (x2, y2)]:
            draw.ellipse([px - 8, py - 8, px + 8, py + 8], fill=field_color, outline="white", width=2)

    return image

def create_clickable_image_with_coordinates(current_field, field_color):
    display_image = create_image_with_overlays(current_field, field_color)
    original_width, original_height = st.session_state.uploaded_image.size
    display_width = st.session_state.display_width
    scale = display_width / original_width
    display_height = int(original_height * scale)
    display_image_resized = display_image.resize((display_width, display_height), Image.Resampling.LANCZOS)

    st.caption(f"🖱️ 이미지 크기: {display_width}×{display_height} (스케일: {scale:.2f}) — 영역을 클릭하세요")

    clicked = streamlit_image_coordinates(
        display_image_resized,
        key=f"img_coords_{st.session_state.current_field_index}_{st.session_state.click_step}_{len(st.session_state.temp_coords)}"
    )
    if clicked:
        sig = (clicked["x"], clicked["y"], st.session_state.current_field_index, st.session_state.click_step)
        if st.session_state.last_click_sig != sig:
            st.session_state.last_click_sig = sig
            ox = int(round(clicked["x"] / scale))
            oy = int(round(clicked["y"] / scale))
            if 0 <= ox < original_width and 0 <= oy < original_height:
                handle_image_click(ox, oy, current_field)

def handle_image_click(x, y, current_field):
    if st.session_state.click_step == 0:
        st.session_state.temp_coords = [(x, y)]
        st.session_state.click_step = 1
        st.success(f"✅ 좌상단 선택: ({x}, {y})")
        st.rerun()
    else:
        x1, y1 = st.session_state.temp_coords[0]
        x1, x2 = min(x1, x), max(x1, x)
        y1, y2 = min(y1, y), max(y1, y)
        st.session_state.temp_coords = [(x1, y1), (x2, y2)]
        st.success(f"✅ 우하단 선택: ({x}, {y})")
        st.success(f"🎯 **{current_field}** 영역 선택 완료: ({x1}, {y1}) → ({x2}, {y2})")
        st.rerun()

def create_control_buttons(current_field):
    st.markdown("### 🎮 제어 버튼")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("🔄 다시 선택", type="secondary", use_container_width=True):
            st.session_state.temp_coords = []
            st.session_state.click_step = 0
            st.rerun()
    with col2:
        if len(st.session_state.temp_coords) == 2:
            if st.button("✅ 좌표 적용", type="primary", use_container_width=True):
                apply_coordinates(current_field)
        else:
            st.button("✅ 좌표 적용", disabled=True, use_container_width=True)
    with col3:
        if st.button("⏭️ 건너뛰기", use_container_width=True):
            skip_current_field()
    with col4:
        if st.button("📋 수동 입력", use_container_width=True):
            show_manual_input_modal(current_field)

def show_manual_input_modal(current_field):
    st.markdown("### 📐 수동 좌표 입력")
    st.info(f"💡 {current_field} 영역의 좌표를 직접 입력할 수 있습니다.")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        x1 = st.number_input("X1 (좌상단 X)", min_value=0, value=50, key="manual_x1")
    with c2:
        y1 = st.number_input("Y1 (좌상단 Y)", min_value=0, value=50, key="manual_y1")
    with c3:
        x2 = st.number_input("X2 (우하단 X)", min_value=0, value=200, key="manual_x2")
    with c4:
        y2 = st.number_input("Y2 (우하단 Y)", min_value=0, value=100, key="manual_y2")
    if st.button("📐 수동 좌표 적용", type="primary", use_container_width=True):
        if x1 < x2 and y1 < y2:
            st.session_state.coordinates[current_field] = [x1, y1, x2, y2]
            st.session_state.current_field_index += 1
            st.session_state.temp_coords = []
            st.session_state.click_step = 0
            st.success(f"✅ {current_field} 수동 좌표 적용 완료!")
            st.rerun()
        else:
            st.error("❌ 좌표가 올바르지 않습니다! (X1 < X2, Y1 < Y2)")

def apply_coordinates(current_field):
    if len(st.session_state.temp_coords) == 2:
        x1, y1 = st.session_state.temp_coords[0]
        x2, y2 = st.session_state.temp_coords[1]
        st.session_state.coordinates[current_field] = [x1, y1, x2, y2]
        st.session_state.current_field_index += 1
        st.session_state.temp_coords = []
        st.session_state.click_step = 0
        st.success(f"✅ {current_field} 영역 저장 완료!")
        if st.session_state.current_field_index < len(FIELDS):
            next_field = FIELDS[st.session_state.current_field_index]
            st.info(f"➡️ 다음 영역: **{next_field}**")
        st.rerun()

def skip_current_field():
    current_field = FIELDS[st.session_state.current_field_index]
    st.session_state.current_field_index += 1
    st.session_state.temp_coords = []
    st.session_state.click_step = 0
    st.warning(f"⏭️ {current_field} 영역을 건너뛰었습니다.")
    st.rerun()

def template_completion_section():
    st.markdown('''
    <div style="text-align:center;padding:2rem;border:3px solid #28a745;border-radius:15px;
                background:linear-gradient(135deg, #d4edda, #c3e6cb);margin:2rem 0;">
    <h2 style="color:#155724;margin:0;">🎉 템플릿 설정 완료!</h2>
    <p style="color:#155724;margin:0.5rem 0 0;">OCR 템플릿이 성공적으로 생성되었습니다.</p>
    </div>
    ''', unsafe_allow_html=True)
    create_final_template_result()
    create_coordinates_table()
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        if st.button("💾 템플릿 저장 후 OCR 처리", type="primary", use_container_width=True):
            save_template()
            st.session_state.current_step = "ocr_process"
            st.rerun()

def create_final_template_result():
    st.markdown("### 📸 최종 템플릿 결과")
    img = st.session_state.uploaded_image
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.imshow(img)
    text_kwargs = {"fontproperties": KFONT_PROP} if KFONT_PROP else {}

    for field, coords in st.session_state.coordinates.items():
        x1, y1, x2, y2 = coords
        color = FIELD_COLORS[field]
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                 linewidth=4, edgecolor=color, facecolor=color, alpha=0.25)
        ax.add_patch(rect)
        ax.text(x1, y1 - 25, field, color="black", fontsize=16, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.8", facecolor="white", alpha=0.95,
                          edgecolor=color, linewidth=3), **text_kwargs)
        ax.text(x1 + 5, y1 + 5, f"({x1},{y1})", color="white", fontsize=10, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.4", facecolor=color, alpha=0.8), **text_kwargs)
        ax.text(x2 - 5, y2 - 5, f"({x2},{y2})", color="white", fontsize=10, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.4", facecolor=color, alpha=0.8),
                ha="right", va="bottom", **text_kwargs)
    ax.set_title("🎯 교통법규 위반 고지서 OCR 템플릿", fontsize=18, pad=30, fontweight="bold", **text_kwargs)
    ax.axis("off")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

def create_coordinates_table():
    st.markdown("### 📋 설정된 좌표 정보")
    rows = []
    for field, coords in st.session_state.coordinates.items():
        x1, y1, x2, y2 = coords
        rows.append({"영역": field, "색상": FIELD_COLORS[field],
                     "X1": x1, "Y1": y1, "X2": x2, "Y2": y2, "너비": x2 - x1, "높이": y2 - y1})
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True)

def save_template():
    template_data = {
        "template_name": "경찰서_고지서_템플릿",
        "created_at": datetime.now().isoformat(),
        "image_size": list(st.session_state.uploaded_image.size),
        "coordinates": st.session_state.coordinates,
    }
    try:
        with open("police_template.json", 'w', encoding='utf-8') as f:
            json.dump(template_data, f, ensure_ascii=False, indent=2)
        st.success("✅ 템플릿이 저장되었습니다!")
    except Exception as e:
        st.error(f"❌ 템플릿 저장 실패: {e}")

def reset_template():
    st.session_state.coordinates = {}
    st.session_state.current_field_index = 0
    st.session_state.temp_coords = []
    st.session_state.click_step = 0
    st.session_state.last_click_sig = None

# =========================
# OCR 처리 섹션 (일괄 처리)
# =========================
def ocr_process_section():
    st.markdown('''
    <div style="padding:2rem;border:2px solid #28a745;border-radius:10px;text-align:center;margin:2rem 0;">
    <h3>🔍 3단계: 지능형 OCR 처리</h3>
    <p>여러 장의 이미지를 자동으로 분석하여 정보를 추출합니다</p>
    </div>
    ''', unsafe_allow_html=True)

    if not st.session_state.batch_results:
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            if st.button("🚀 지능형 OCR 일괄 처리 시작", type="primary", use_container_width=True):
                process_all_images()
    else:
        show_ocr_results()

def process_all_images():
    images = st.session_state.uploaded_images or []
    names = st.session_state.uploaded_image_names or []
    if not images:
        st.error("이미지가 없습니다.")
        return

    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        status_text.text("🔧 Google Cloud Vision API 초기화 중...")
        vision_client = get_vision_client()
        if vision_client is None:
            st.error("❌ Google Cloud Vision API 초기화 실패")
            return

        all_results = []
        total = len(images)

        for idx, img in enumerate(images, start=1):
            st.session_state.uploaded_image = img
            progress_bar.progress(int((idx - 1) / total * 100))
            status_text.text(f"📄 ({idx}/{total}) 전체 이미지 텍스트 분석 중...")

            full_text = perform_full_image_ocr(vision_client)
            st.session_state.full_ocr_text = full_text

            # 분류: "경찰청 고지" 유무 (공백 제거 버전 포함)
            status_text.text(f"🕵️ ({idx}/{total}) 이미지 유형 분석 중...")
            is_police = classify_image_type(full_text)
            st.session_state.is_police_notice = is_police

            # 본 처리
            if is_police:
                status_text.text(f"🎯 ({idx}/{total}) 템플릿 기반 OCR 처리 중...")
                if st.session_state.coordinates:
                    ocr_results = process_template_based_ocr(vision_client)
                else:
                    st.error("❌ 저장된 템플릿이 없습니다. 템플릿을 먼저 설정해주세요.")
                    return
            else:
                status_text.text(f"🔍 ({idx}/{total}) 키워드 기반 정보 추출 중...")
                ocr_results = process_keyword_based_ocr(full_text)

            st.session_state.ocr_results = ocr_results

            # 차량번호 매칭
            status_text.text(f"🚗 ({idx}/{total}) 차량번호 매칭 중...")
            matched_vehicle = match_vehicle_number(ocr_results.get("차량번호", ""))

            # 결과 정리
            status_text.text(f"📊 ({idx}/{total}) 결과 정리 중...")
            final_result = compile_final_results(matched_vehicle)
            if names and len(names) >= idx:
                final_result["파일명"] = names[idx - 1]
            final_result["_index"] = idx
            final_result["분류"] = "경찰서(경찰청 고지)" if is_police else "일반 고지서"
            all_results.append(final_result)

        st.session_state.batch_results = all_results
        progress_bar.progress(100)
        status_text.text("✅ OCR 일괄 처리 완료!")
        st.success("🎉 모든 이미지 처리 완료!")
        st.rerun()

    except Exception as e:
        st.error(f"❌ OCR 일괄 처리 중 오류 발생: {e}")

def perform_full_image_ocr(vision_client):
    try:
        img_byte_arr = io.BytesIO()
        st.session_state.uploaded_image.save(img_byte_arr, format='PNG')
        image = vision.Image(content=img_byte_arr.getvalue())
        response = vision_client.text_detection(image=image)
        if response.error.message:
            raise Exception(f'{response.error.message}')
        texts = response.text_annotations
        return texts[0].description.strip() if texts else ""
    except Exception as e:
        st.warning(f"⚠️ 전체 이미지 OCR 실패: {e}")
        return ""

def classify_image_type(full_text):
    if not full_text:
        return False
    joined = re.sub(r"\s+", "", full_text)
    return ("경찰청 고지" in full_text) or ("경찰청고지" in joined)

def process_template_based_ocr(vision_client):
    ocr_results = {}
    for field, coords in st.session_state.coordinates.items():
        try:
            extracted_text = extract_text_from_region(vision_client, None, field, coords)
            ocr_results[field] = extracted_text
        except Exception as e:
            st.warning(f"⚠️ {field} 영역 처리 실패: {e}")
            ocr_results[field] = ""
    return ocr_results

def process_keyword_based_ocr(full_text):
    ocr_results = {
        "차량번호": extract_vehicle_number_from_text(full_text),
        "일자": extract_date_from_text(full_text, ["일시", "일자", "위반일", "발생일"]),
        "장소": extract_location_from_text(full_text),
        "과태료": extract_fine_amount_from_text(full_text),
        "납기일": extract_date_from_text(full_text, ["납부기한", "납기", "기한", "만료일"]),
        "내용": extract_violation_content_from_text(full_text),
    }
    return ocr_results

# =========================
# 필드별 텍스트 추출/후처리
# =========================
def extract_vehicle_number_from_text(text):
    """차량번호: 1) ROI 우선 2) 뒤 4자리만 사용"""
    try:
        if "coordinates" in st.session_state and "차량번호" in st.session_state.coordinates and st.session_state.uploaded_image is not None:
            client = get_vision_client()
            if client:
                roi_coords = st.session_state.coordinates["차량번호"]
                roi_text = extract_text_from_region(client, None, "차량번호", roi_coords) or ""
                m = re.search(r"(\d{4})(?!\d)", re.sub(r"\s+", "", roi_text))
                if m:
                    return m.group(1)
    except Exception:
        pass

    vehicle_keywords = ["차량번호", "차량", "대상"]
    lines = text.split("\n") if text else []
    for i, line in enumerate(lines):
        if any(k in line for k in vehicle_keywords):
            search_text = line + (" " + lines[i + 1] if i + 1 < len(lines) else "")
            m = re.search(r"(\d{4})(?!\d)", re.sub(r"\s+", "", search_text))
            if m:
                return m.group(1)

    penalty_words = ("원", "금액", "과태료", "범칙금")
    for line in lines:
        if any(p in line for p in penalty_words):
            continue
        m = re.search(r"(\d{4})(?!\d)", re.sub(r"\s+", "", line))
        if m:
            return m.group(1)

    m = re.search(r"(\d{4})(?!\d)", re.sub(r"\s+", "", text or ""))
    return m.group(1) if m else ""

def extract_date_from_text(text, keywords):
    """키워드 주변/전체에서 날짜를 찾아 fmt_date_uniform(YYYY/MM/DD)로 반환"""
    if not text:
        return fmt_date_uniform("")
    lines = text.split('\n')

    # 키워드 주변 탐색 (현재 줄 + 다음 줄)
    for kw in keywords:
        for i, line in enumerate(lines):
            if kw in line:
                seg = line + (" " + lines[i + 1] if i + 1 < len(lines) else "")
                return fmt_date_uniform(seg)

    # 전체 텍스트에서 탐색
    return fmt_date_uniform(text)

def extract_location_from_text(text):
    location_keywords = [
        "장소", "위치", "위반장소", "발생장소", "지점",
        "cctv", "ic", "앞", "방향", "뒤", "주변", "근처", "단지",
        "어린이보호구역", "로", "길", "대로", "교차로"
    ]
    lines = text.split('\n')
    for keyword in location_keywords:
        for line in lines:
            if keyword in line.lower():
                cleaned = re.sub(r'^\s*[-•]\s*', '', line.strip())
                if len(cleaned) > 3:
                    return cleaned
    return ""

def extract_fine_amount_from_text(text):
    """
    최종 납부 금액 추출 우선순위:
    1) '납부금액', '납기내금액' 줄에서 키워드 '뒤' 숫자
    2) '과태료' 줄(감경/가산 제외)에서 키워드 '뒤' 숫자
    3) 본문 어디서든 '숫자+원' 패턴
    4) 숫자 전체 검색 (문서번호 등 하이픈 코드 제거 후)
    """
    if not text:
        return ""
    lines = text.split("\n")

    def strip_hyphen_codes(s: str) -> str:
        # 예: 2025-063822-00 같은 코드 제거
        return re.sub(r"\d{2,4}\s*-\s*\d{3,6}\s*-\s*\d{1,4}", " ", s)

    def find_amount_after_keyword(segment: str):
        seg = strip_hyphen_codes(segment)
        # '숫자+원'
        m = re.search(r"(?<!\d)(\d{1,3}(?:,\d{3})+|\d{4,})\s*원", seg)
        if m:
            val = int(m.group(1).replace(",", ""))
            if 10_000 <= val <= 500_000:
                return val
        # 숫자 나열
        for n in re.findall(r"(?<!\d)(\d{1,3}(?:,\d{3})+|\d{4,})(?!\d)", seg):
            val = int(n.replace(",", ""))
            if 10_000 <= val <= 500_000:
                return val
        return None

    # 1) 납부금액/납기내금액
    for key in ("납부금액", "납기내금액"):
        for i, line in enumerate(lines):
            if key in line:
                start = line.find(key) + len(key)
                seg = line[start:] + (" " + lines[i + 1] if i + 1 < len(lines) else "")
                amt = find_amount_after_keyword(seg)
                if amt is not None:
                    return f"{amt:,}원"

    # 2) 과태료 (감경/가산 제외)
    for i, line in enumerate(lines):
        if "과태료" in line and "감경" not in line and "가산" not in line:
            start = line.find("과태료") + len("과태료")
            seg = line[start:] + (" " + lines[i + 1] if i + 1 < len(lines) else "")
            amt = find_amount_after_keyword(seg)
            if amt is not None:
                return f"{amt:,}원"

    # 3) 전체에서 '숫자+원'
    cleaned = strip_hyphen_codes("\n".join(lines))
    m = re.search(r"(?<!\d)(\d{1,3}(?:,\d{3})+|\d{4,})\s*원", cleaned)
    if m:
        val = int(m.group(1).replace(",", ""))
        if 10_000 <= val <= 500_000:
            return f"{val:,}원"

    # 4) 숫자 전체 검색
    for n in re.findall(r"(?<!\d)(\d{1,3}(?:,\d{3})+|\d{4,})(?!\d)", cleaned):
        val = int(n.replace(",", ""))
        if 10_000 <= val <= 500_000:
            return f"{val:,}원"
    return ""

def extract_violation_content_from_text(text):
    violation_keywords = [
        "주정차", "속도", "신호", "어린이보호", "중앙선", "끼어들기",
        "횡단보도", "안전거리", "진로변경", "통행금지", "일시정지"
    ]
    lines = text.split('\n')
    for keyword in violation_keywords:
        for line in lines:
            if keyword in line:
                cleaned = line.strip()
                if len(cleaned) > 2:
                    return cleaned
    content_keywords = ["내용", "위반", "사유", "항목"]
    for keyword in content_keywords:
        for i, line in enumerate(lines):
            if keyword in line and i + 1 < len(lines):
                nxt = lines[i + 1].strip()
                if len(nxt) > 2:
                    return nxt
    return ""

def extract_text_from_region(vision_client, image, field, coords):
    """특정 영역에서 텍스트 추출 (image 파라미터는 사용하지 않음)"""
    try:
        x1, y1, x2, y2 = coords
        pil_image = st.session_state.uploaded_image.crop((x1, y1, x2, y2))
        buf = io.BytesIO()
        pil_image.save(buf, format='PNG')
        cropped = vision.Image(content=buf.getvalue())
        response = vision_client.text_detection(image=cropped)
        if response.error.message:
            raise Exception(f'{response.error.message}')
        texts = response.text_annotations
        raw = texts[0].description.strip() if texts else ""
        return post_process_text(field, raw)
    except Exception as e:
        st.warning(f"⚠️ {field} 영역 OCR 실패: {e}")
        return ""

def post_process_text(field, raw_text):
    """필드별 후처리: 날짜는 fmt_date_uniform로 통일"""
    if not raw_text:
        return ""
    text = re.sub(r'\s+', ' ', raw_text).strip()

    if field == "차량번호":
        pats = [r'(\d{2,3}[가-힣]\d{4})', r'(\d{2,3}\s*[가-힣]\s*\d{4})']
        for p in pats:
            m = re.search(p, text)
            if m:
                return re.sub(r'\s+', '', m.group(1))
        return text

    if field in ("일자", "납기일"):
        return fmt_date_uniform(text)

    if field == "과태료":
        nums = re.findall(r'[\d,]+', text)
        for n in nums:
            try:
                val = int(n.replace(',', ''))
                if 10_000 <= val <= 500_000:
                    return f"{val:,}원"
            except:
                pass
        return text

    return text

# =========================
# 매칭/결과 정리/표시
# =========================
def match_vehicle_number(ocr_vehicle_number):
    """뒤 4자리 우선 매칭 + 유사도 보조"""
    if not ocr_vehicle_number or st.session_state.vehicle_users_df is None:
        return None
    ocr_digits = re.findall(r'\d+', ocr_vehicle_number)
    if not ocr_digits:
        return None
    last4 = ocr_digits[-1][-4:] if ocr_digits[-1] else ""
    if len(last4) != 4:
        return None

    best = None
    best_score = 0
    for _, row in st.session_state.vehicle_users_df.iterrows():
        vehicle_num = str(row.get('차량번호', ''))
        vd = re.findall(r'\d+', vehicle_num)
        if not vd:
            continue
        v_last4 = vd[-1][-4:] if vd[-1] else ""
        if v_last4 == last4:
            sim = SequenceMatcher(None, ocr_vehicle_number, vehicle_num).ratio()
            if sim > best_score:
                best_score = sim
                best = row
    return best

def compile_final_results(matched_vehicle):
    """최종 결과 정리 (사용자=성명, 감경금액 포함, 날짜는 YYYY/MM/DD)"""
    result = {}
    if matched_vehicle is not None:
        name_field = "성명" if "성명" in matched_vehicle else ("사용자" if "사용자" in matched_vehicle else None)
        result['부서'] = matched_vehicle.get('부서', '')
        result['사용자'] = matched_vehicle.get(name_field, '') if name_field else ''
        result['차량번호'] = matched_vehicle.get('차량번호', '')
    else:
        result['부서'] = '매칭 실패'
        result['사용자'] = '매칭 실패'
        result['차량번호'] = st.session_state.ocr_results.get('차량번호', '')

    # OCR 결과
    result['일자']   = fmt_date_uniform(st.session_state.ocr_results.get('일자', ''))
    result['장소']   = st.session_state.ocr_results.get('장소', '')
    result['내용']   = st.session_state.ocr_results.get('내용', '')
    result['과태료'] = st.session_state.ocr_results.get('과태료', '')
    result['납기일'] = fmt_date_uniform(st.session_state.ocr_results.get('납기일', ''))

    # 감경금액(20%)
    try:
        fine_text = result['과태료']
        fine_amount = int(re.sub(r'[^\d]', '', fine_text)) if fine_text else 0
        discounted = int(fine_amount * 0.8)
        result['감경금액'] = f"{discounted:,}원"
    except:
        result['감경금액'] = ''

    return result

def show_ocr_results():
    st.markdown("### 🎯 지능형 OCR 처리 결과")
    if st.session_state.get('is_police_notice', False):
        st.success("🚔 **경찰서 고지서**(“경찰청 고지” 감지)로 분류되어 템플릿 기반 OCR을 사용했습니다.")
    else:
        st.info("📋 **일반 고지서**(“경찰청 고지” 미감지)로 분류되어 키워드 기반 OCR을 사용했습니다.")

    tab1, tab2, tab3, tab4 = st.tabs(["📊 최종 결과(단일)", "🔍 OCR 상세결과", "🚗 차량 매칭", "📄 전체 텍스트"])
    with tab1:
        show_final_results()
    with tab2:
        show_detailed_ocr_results()
    with tab3:
        show_vehicle_matching_info()
    with tab4:
        show_full_text_analysis()

    st.markdown("---")
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        if st.button("📋 결과 확인 및 다운로드", type="primary", use_container_width=True):
            st.session_state.current_step = "results"
            st.rerun()

def show_final_results():
    result = st.session_state.final_results
    if not result:
        st.info("아직 단일 결과가 없습니다. 일괄 처리 후 '결과 확인 및 다운로드'로 이동하세요.")
        return
    cols = st.columns(3)
    with cols[0]:
        st.metric("부서", result.get("부서", ""))
        st.metric("사용자(성명)", result.get("사용자", ""))
    with cols[1]:
        st.metric("차량번호", result.get("차량번호", ""))
        st.metric("과태료", result.get("과태료", ""))
    with cols[2]:
        st.metric("감경금액(20%)", result.get("감경금액", ""))
        st.metric("납기일", result.get("납기일", ""))
    st.markdown("**장소**")
    st.code(result.get("장소", ""))
    st.markdown("**내용**")
    st.code(result.get("내용", ""))

def show_detailed_ocr_results():
    """상세 OCR 결과 (날짜는 YYYY/MM/DD로 통일 표시)"""
    method = "템플릿 기반" if st.session_state.get('is_police_notice', False) else "키워드 기반"
    st.markdown(f"#### 🔍 {method} OCR 결과")
    for field, text in st.session_state.ocr_results.items():
        if field in ("일자", "납기일") and text:
            text = fmt_date_uniform(text)
        color = FIELD_COLORS.get(field, "#000000")
        if text and str(text).strip():
            st.markdown(f'<span style="color:{color}">●</span> **{field}**: {text}', unsafe_allow_html=True)
        else:
            st.markdown(f'<span style="color:#888888">○</span> **{field}**: <span style="color:#888888">(인식 실패)</span>',
                        unsafe_allow_html=True)

def show_full_text_analysis():
    st.markdown("#### 📄 전체 이미지 텍스트 분석")
    full = st.session_state.full_ocr_text or ""
    if full:
        st.text_area("인식된 전체 텍스트", full, height=300,
                     help="Google Cloud Vision API로 인식된 전체 텍스트입니다.")
        stats = {"총 문자 수": len(full), "줄 수": len(full.split('\n')), "단어 수": len(full.split())}
        c1, c2, c3 = st.columns(3)
        for (k, v), col in zip(stats.items(), [c1, c2, c3]):
            with col:
                st.metric(k, v)
    else:
        st.warning("전체 텍스트 정보가 없습니다.")

def show_vehicle_matching_info():
    st.markdown("#### 🚗 차량번호 매칭 과정")
    ocr_vehicle = st.session_state.ocr_results.get('차량번호', '')
    final_vehicle = st.session_state.final_results.get('차량번호', '') if st.session_state.final_results else ''
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**OCR 인식 결과**")
        st.code(ocr_vehicle or "(인식 실패)")
    with c2:
        st.markdown("**매칭된 차량번호**")
        st.code(final_vehicle or "(매칭 실패)")
    if ocr_vehicle and final_vehicle and ocr_vehicle != final_vehicle:
        st.info("💡 뒤 4자리 숫자를 기준으로 차량 사용자 데이터에서 가장 유사한 차량번호를 찾았습니다.")
    elif not final_vehicle or final_vehicle == "매칭 실패":
        st.warning("⚠️ 차량 사용자 데이터에서 일치하는 차량번호를 찾을 수 없습니다.")
    else:
        st.success("✅ OCR 결과와 차량 사용자 데이터가 정확히 일치합니다.")

# =========================
# 결과 확인/다운로드
# =========================
def results_section():
    st.markdown("### 📦 일괄 결과 요약")
    batch = st.session_state.batch_results or []
    if not batch:
        st.info("아직 일괄 결과가 없습니다. 상단에서 OCR 일괄 처리를 먼저 실행하세요.")
        if st.button("처음으로 돌아가기"):
            reset_all_states()
            st.rerun()
        return
    df = pd.DataFrame(batch).sort_values(by=["_index"]).drop(columns=["_index"], errors="ignore")
    st.dataframe(df, use_container_width=True)

    csv = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label="⬇️ CSV 다운로드",
        data=csv,
        file_name=f"교통법규_OCR_결과_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        key="download_csv_results",
        use_container_width=True
    )

    if st.button("처음으로 돌아가기"):
        reset_all_states()
        st.rerun()

# =========================
# 메인
# =========================
def show_progress_indicator():
    steps = ["파일 업로드", "템플릿 설정", "OCR 처리", "결과 확인"]
    idx = {"upload_files": 0, "template_setup": 1, "ocr_process": 2, "results": 3}[st.session_state.current_step]
    st.progress((idx + 1) / len(steps))
    cols = st.columns(len(steps))
    for i, (col, step) in enumerate(zip(cols, steps)):
        with col:
            if i < idx: st.markdown(f"✅ **{step}**")
            elif i == idx: st.markdown(f"🔄 **{step}**")
            else: st.markdown(f"⏳ {step}")

def main():
    init_session_state()
    st.markdown('''
    <h1 style="text-align:center;color:#1f77b4;margin:0 0 1rem;">
    🚗 교통법규 위반 고지서 OCR 프로그램
    </h1>
    <p style="text-align:center;color:#666;margin-bottom:2rem;">
    Google Cloud Vision API를 활용한 지능형 OCR 시스템
    </p>
    ''', unsafe_allow_html=True)

    if KFONT_PATH:
        st.success(f"🖋 맑은 고딕 폰트 로드됨: {os.path.basename(KFONT_PATH)}")
    else:
        st.warning("🖋 맑은 고딕 폰트 없음 - 시스템 기본 폰트 사용")

    client = get_vision_client()
    if client is None:
        st.error("🚫 Google Cloud Vision API가 설정되지 않았습니다. 프로그램을 계속 사용할 수 없습니다.")
        st.stop()

    show_progress_indicator()

    if st.session_state.current_step == "upload_files":
        file_upload_section()
    elif st.session_state.current_step == "template_setup":
        template_setup_section()
    elif st.session_state.current_step == "ocr_process":
        ocr_process_section()
    elif st.session_state.current_step == "results":
        results_section()

if __name__ == "__main__":
    main()
