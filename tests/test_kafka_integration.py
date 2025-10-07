import os
import pytest
from confluent_kafka.admin import AdminClient
from stream.producer import produce_test_message
from stream.consumer import consume_one_message

required_topics = [
    "myteam.watch",
    "myteam.rate",
    "myteam.reco_requests",
    "myteam.reco_responses",
]

@pytest.fixture(scope="module")
def kafka_admin():
    conf = {
        "bootstrap.servers": os.getenv("KAFKA_BOOTSTRAP"),
        "security.protocol": "SASL_SSL",
        "sasl.mechanism": "PLAIN",
        "sasl.username": os.getenv("KAFKA_API_KEY"),
        "sasl.password": os.getenv("KAFKA_API_SECRET"),
    }
    return AdminClient(conf)

def test_topics_exist(kafka_admin):
    metadata = kafka_admin.list_topics(timeout=10)
    existing = set(metadata.topics.keys())
    missing = [t for t in required_topics if t not in existing]
    assert not missing, f"Missing topics: {missing}"

def test_produce_consume_roundtrip():
    topic = required_topics[0]  # myteam.watch
    result = produce_test_message(topic)
    assert "Produced" in result
    msg = consume_one_message(topic)
    assert msg is not None
    assert "hello" in msg