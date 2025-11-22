# reco.py
# 공간 추천 점수 계산 모델 (AI 모델 2)

def calc_congestion_score(people, capacity):
    """혼잡도 계산 (한산할수록 점수↑)"""
    if capacity <= 0:
        return 0.0
    ratio = people / capacity
    score = 1 - ratio
    return max(0.0, min(1.0, score))


def calc_final_score(purpose, congestion, distance):
    """가중합 기반 최종 점수 계산"""
    WEIGHTS = {
        "purpose": 0.5,
        "congestion": 0.3,
        "distance": 0.2
    }
    return (
        WEIGHTS["purpose"] * purpose +
        WEIGHTS["congestion"] * congestion +
        WEIGHTS["distance"] * distance
    )


def recommend_rooms(candidate_rooms):
    """
    candidate_rooms: List[dict]
      각 원소 예시 (백엔드 spec):

      {
        "spaceId": 201,
        "spaceName": "중앙도서관",
        "purposeScore": 0.9,
        "distanceFeature": 0.88,
        "predictCount": 18,
        "capacity": 40
      }
    """
    results = []

    for c in candidate_rooms:
        space_id = c["spaceId"]
        space_name = c.get("spaceName", "")

        purpose_score = float(c.get("purposeScore", 0.0))
        distance_score = float(c.get("distanceFeature", 0.5))
        people = int(c.get("predictCount", 0))
        capacity = int(c.get("capacity", 1))

        congestion_score = calc_congestion_score(people, capacity)
        final_score = calc_final_score(purpose_score, congestion_score, distance_score)

        results.append({
            "spaceId": space_id,
            "spaceName": space_name,
            "purposeScore": purpose_score,
            "people": people,
            "capacity": capacity,
            "congestionScore": congestion_score,
            "distanceScore": distance_score,
            "finalScore": final_score,
            "finalRecommendScore": final_score
        })

    results.sort(key=lambda x: x["finalScore"], reverse=True)
    return results
