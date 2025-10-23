from datetime import datetime
try:
    from datetime import UTC 
except Exception:  
    from datetime import timezone
    UTC = timezone.utc

import json

from recommender.schemas import validate_message


def test_validate_message_watch_adds_timestamp_and_fields():
    msg = json.dumps({"user_id": 1, "movie_id": 42})
    out = validate_message(msg, "watch")

    assert out["user_id"] == 1
    assert out["movie_id"] == 42
    ts = datetime.fromisoformat(out["timestamp"])
    assert ts.tzinfo is not None  


def test_validate_message_rate_ok():
    msg = json.dumps({"user_id": 7, "movie_id": 99, "rating": 3.5})
    out = validate_message(msg, "rate")

    assert out["user_id"] == 7
    assert out["movie_id"] == 99
    assert out["rating"] == 3.5
    _ = datetime.fromisoformat(out["timestamp"])
