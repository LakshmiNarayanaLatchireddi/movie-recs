# Optional: team-local event generator wrapper (implement if needed).
import os, json
from confluent_kafka import Producer

conf = {
  'bootstrap.servers': os.environ.get('KAFKA_BOOTSTRAP'),
  'security.protocol': 'SASL_SSL',
  'sasl.mechanisms': 'PLAIN',
  'sasl.username': os.environ.get('KAFKA_API_KEY'),
  'sasl.password': os.environ.get('KAFKA_API_SECRET'),
}

def main():
    topic = os.environ.get('WATCH_TOPIC','myteam.watch')
    p = Producer(conf)
    event = {'ts': 1, 'user_id': 1, 'movie_id': 50, 'minute': 1}
    p.produce(topic, json.dumps(event).encode('utf-8'))
    p.flush()
    print(f"Produced: {event} to {topic}")

if __name__ == '__main__':
    main()
