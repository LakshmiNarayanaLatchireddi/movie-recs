from datetime import datetime
try:
    from datetime import UTC  
except Exception:  
    from datetime import timezone
    UTC = timezone.utc

import pytest

from recommender.schemas import validate_schema


def test_validate_schema_reco_response_length_mismatch_raises():
    data = {
        "user_id": 1,
        "movie_ids": [1, 2, 3],
        "scores": [0.9, 0.8],  
    }
    with pytest.raises(ValueError):
        validate_schema(data, "reco_responses")


def test_validate_schema_reco_request_adds_timestamp():
    data = {"user_id": 123}
    out = validate_schema(data, "reco_requests")

    assert out["user_id"] == 123
    _ = datetime.fromisoformat(out["timestamp"])
