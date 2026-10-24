import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import normalize_evaluation, score_and_rank

def criteria():
    return [
        {"criterion_id": 1, "name": "Technical", "description": "", "weight": 60.0, "max_score": 10.0, "is_active": 1},
        {"criterion_id": 2, "name": "Commercial", "description": "", "weight": 40.0, "max_score": 10.0, "is_active": 1},
    ]

def supplier(name, scores, date, exp=5):
    cs = criteria()
    raw = {"supplier_name": name, "criteria": [
        {"criterion_id": c["criterion_id"], "score": scores[i], "max_score": 10,
         "justification": "test", "evidence": "test evidence"} for i,c in enumerate(cs)
    ], "risks": [], "overall_summary": "test"}
    ev, warnings = normalize_evaluation(raw, name, cs)
    return {"supplier_name": name, "submission_date": date, "experience_rating": exp,
            "evaluation": ev, "warnings": warnings}

def test_weighted_score_and_ppi():
    cs = criteria()
    result = score_and_rank([
        supplier("A", [10, 5], "2026-01-02"),
        supplier("B", [5, 10], "2026-01-01")
    ], cs)
    by_name = {x["supplier_name"]: x for x in result}
    assert by_name["A"]["absolute_score"] == 80.0
    assert by_name["B"]["absolute_score"] == 70.0
    assert by_name["A"]["ppi"] == 80.0
    assert by_name["B"]["ppi"] == 80.0
    # Equal PPI: earlier submission date wins.
    assert result[0]["supplier_name"] == "B"

def test_missing_and_out_of_range_normalized():
    cs = criteria()
    raw = {"criteria": [
        {"criterion_id": 1, "score": 99, "justification": "x", "evidence": "y"}
    ]}
    normalized, warnings = normalize_evaluation(raw, "A", cs)
    assert normalized["criteria"][0]["score"] == 10
    assert normalized["criteria"][1]["score"] == 0
    assert len(warnings) >= 2

def test_weights_must_total_100():
    cs = criteria()
    cs[0]["weight"] = 50
    try:
        score_and_rank([supplier("A", [5, 5], "2026-01-01")], cs)
    except ValueError as exc:
        assert "100%" in str(exc)
    else:
        assert False, "Expected invalid weight total to fail"
