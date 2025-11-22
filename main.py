# ═══════════════════════════════════════════════════════
# 필수 패키지 설치 확인
# ═══════════════════════════════════════════════════════
# 설치 명령어: pip install -r requirements.txt
# 또는: pip install fastapi uvicorn pymysql google-generativeai pydantic
# ═══════════════════════════════════════════════════════


import sys

REQUIRED_PACKAGES = {
    'fastapi': 'fastapi',
    'uvicorn': 'uvicorn',
    'pymysql': 'pymysql',
    'google.genai': 'google-generativeai',
    'pydantic': 'pydantic'
}

missing_packages = []

for module_name, package_name in REQUIRED_PACKAGES.items():
    try:
        __import__(module_name)
    except ImportError:
        missing_packages.append(package_name)

if missing_packages:
    print("❌ 다음 패키지들이 설치되지 않았습니다:")
    for pkg in missing_packages:
        print(f"   - {pkg}")
    print("\n💡 다음 명령어로 설치하세요:")
    print(f"   pip install {' '.join(missing_packages)}")
    sys.exit(1)

import json
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pymysql
from contextlib import contextmanager

from google import genai
from google.genai import types

# ═══════════════════════════════════════════════════════
# 설정 
# ═══════════════════════════════════════════════════════
MY_GEMINI_API_KEY = "AIzaSyDGuN4D3ZDvFWxii5D0U_-pn420C_EAx-k"  # Gemini API 키 설정 필요

# DB 설정
DB_CONFIG = {
    "host": "localhost",
    "port": 3306,
    "user": "your_user",
    "password": "your_password",
    "database": "your_database",
    "charset": "utf8mb4"
}

app = FastAPI(title="AI Space Recommendation API")
client = genai.Client(api_key=MY_GEMINI_API_KEY)

# ═══════════════════════════════════════════════════════
# Pydantic 모델 정의
# ═══════════════════════════════════════════════════════
class NLPRequest(BaseModel):
    userText: str = Field(..., description="사용자의 자연어 입력")
    topN: int = Field(default=3, description="추천할 공간 개수")

class NLPCandidateRoom(BaseModel):
    spaceId: int
    spaceName: str
    purposeScore: float

class NLPResponse(BaseModel):
    candidateRooms: List[NLPCandidateRoom]
    placeFlag: int
    placeLat: float
    placeLng: float

# ═══════════════════════════════════════════════════════
# DB 연결 관리
# ═══════════════════════════════════════════════════════
@contextmanager
def get_db_connection():
    """DB 연결 컨텍스트 매니저"""
    conn = None
    try:
        conn = pymysql.connect(**DB_CONFIG)
        yield conn
    except Exception as e:
        if conn:
            conn.rollback()
        raise e
    finally:
        if conn:
            conn.close()

def fetch_all_spaces() -> List[Dict[str, Any]]:
    """DB에서 모든 공간 정보 조회"""
    with get_db_connection() as conn:
        with conn.cursor(pymysql.cursors.DictCursor) as cursor:
            query = """
                SELECT 
                    space_id,
                    space_name,
                    space_lat,
                    space_lon,
                    space_floor,
                    space_capacity,
                    quite_score,
                    talk_score,
                    study_score,
                    rest_score
                FROM Space
            """
            cursor.execute(query)
            return cursor.fetchall()

# ═══════════════════════════════════════════════════════
# Gemini NLP 모델 (모델 1)
# ═══════════════════════════════════════════════════════
GEMINI_SCHEMA: Dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "topSpaces": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "spaceId": {"type": "INTEGER"},
                    "purposeScore": {"type": "NUMBER"},
                },
                "required": ["spaceId", "purposeScore"],
            },
        },
        "placeFlag": {
            "type": "INTEGER",
            "description": "실제 장소 언급 여부 (1/0)",
        },
        "placeName": {
            "type": "STRING",
            "description": "사용자가 말한 실제 장소명 (없으면 빈 문자열)",
        },
    },
    "required": ["topSpaces", "placeFlag", "placeName"],
}

def _call_gemini(
    user_text: str,
    spaces: List[Dict[str, Any]],
    top_n: int,
) -> Dict[str, Any]:
    """Gemini API 호출"""
    spaces_for_llm = [
        {
            "spaceId": s["space_id"],
            "vector": [
                s["quite_score"],
                s["talk_score"],
                s["study_score"],
                s["rest_score"],
            ],
        }
        for s in spaces
    ]
    spaces_json = json.dumps(spaces_for_llm, ensure_ascii=False)

    prompt = f"""
너는 캠퍼스 공간 추천 모델이다.

- spaces: 각 공간은 spaceId와 vector를 가진다.
  vector는 ["조용한", "대화하는", "공부하는", "휴식하는"] 순서의 점수이다.
- user_text: 한국어 문장.

1. user_text를 분석해서 위 4차원에 대한 intent_vector를 마음속으로 만든다.
2. 각 공간의 vector와 intent_vector 사이의 코사인 유사도를 계산해서 purposeScore로 사용한다.
   cos_sim(a, b) = (Σ a_i * b_i) / (sqrt(Σ a_i^2) * sqrt(Σ b_i^2))
3. purposeScore를 기준으로 내림차순 정렬하여 상위 {top_n}개 공간만
   topSpaces 배열에 넣는다.
   각 항목은 {{ "spaceId", "purposeScore" }} 만 포함해야 한다.
4. user_text 안에 실제 장소명이 언급되었는지 보고,
   - 언급되면 placeFlag = 1, placeName 에 대표 장소명을 문자열로 넣는다.
   - 아니면 placeFlag = 0, placeName = "".

! 위도/경도(lat/lng)는 절대 생성하지 마라.
! 출력은 내가 제공한 GEMINI_SCHEMA에 정확히 맞는 순수 JSON만 포함한다.
   자연어 설명은 포함하지 않는다.

spaces(JSON):
{spaces_json}

user_text:
\"\"\"{user_text}\"\"\"
"""

    resp = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=GEMINI_SCHEMA,
        ),
    )
    return json.loads(resp.text)

def _find_space_by_partial_name(
    spaces: List[Dict[str, Any]],
    place_name: str,
) -> Optional[Dict[str, Any]]:
    """부분 문자열로 공간 검색"""
    norm_place = place_name.strip().lower()
    if not norm_place:
        return None

    for s in spaces:
        norm_space = str(s["space_name"]).strip().lower()
        if not norm_space:
            continue

        if norm_place in norm_space or norm_space in norm_place:
            return s

    return None

def run_nlp_model(
    user_text: str,
    spaces: List[Dict[str, Any]],
    top_n: int = 3,
) -> Dict[str, Any]:
    """NLP 모델 실행 (목적 점수 계산)"""
    gemini_res = _call_gemini(user_text, spaces, top_n)
    by_id = {s["space_id"]: s for s in spaces}

    candidate_rooms: List[Dict[str, Any]] = []
    for item in gemini_res["topSpaces"]:
        sid = item["spaceId"]
        score = item["purposeScore"]
        base = by_id.get(sid)
        if not base:
            continue

        candidate_rooms.append(
            {
                "spaceId": sid,
                "spaceName": base["space_name"],
                "purposeScore": score,
            }
        )

    place_flag = gemini_res["placeFlag"]
    place_name = gemini_res.get("placeName", "") or ""

    place_lat, place_lng = 0.0, 0.0
    if place_flag == 1 and place_name:
        space_row = _find_space_by_partial_name(spaces, place_name)
        if space_row is not None:
            place_lat = float(space_row.get("space_lat", 0.0))
            place_lng = float(space_row.get("space_lon", 0.0))

    return {
        "candidateRooms": candidate_rooms,
        "placeFlag": place_flag,
        "placeLat": place_lat,
        "placeLng": place_lng,
    }

# ═══════════════════════════════════════════════════════
# API 엔드포인트
# ═══════════════════════════════════════════════════════
@app.post("/api/internal/ai/nlp", response_model=NLPResponse)
async def nlp_endpoint(request: NLPRequest):
    """
    NLP 모델 API
    사용자 입력을 분석하여 목적 점수(purposeScore) 계산 및 장소 정보 반환
    
    Returns:
    {
      "candidateRooms": [
        { "spaceId": 201, "spaceName": "중앙도서관", "purposeScore": 0.9 },
        { "spaceId": 305, "spaceName": "카페", "purposeScore": 0.7 }
      ],
      "placeFlag": 1,
      "placeLat": 37.55,
      "placeLng": 126.94
    }
    """
    if not MY_GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="Gemini API 키가 설정되지 않았습니다")
    
    try:
        # DB에서 모든 공간 정보 조회
        spaces = fetch_all_spaces()
        
        if not spaces:
            raise HTTPException(status_code=404, detail="공간 정보를 찾을 수 없습니다")
        
        # NLP 모델 실행
        result = run_nlp_model(request.userText, spaces, request.topN)
        
        return NLPResponse(**result)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"NLP 모델 실행 중 오류: {str(e)}")

@app.get("/health")
async def health_check():
    """헬스 체크 엔드포인트"""
    return {"status": "healthy", "service": "AI Space Recommendation API"}

# ═══════════════════════════════════════════════════════
# 메인 실행
# ═══════════════════════════════════════════════════════
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)