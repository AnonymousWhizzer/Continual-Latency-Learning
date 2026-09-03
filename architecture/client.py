import webbrowser
from fastapi import FastAPI
import threading
import time
import json
from datetime import datetime
import requests
import redis
import signal
import sys

degradations = {}

control_ip = '140.93.5.91'
provider_ip = '140.93.5.56'
db_ip = "localhost"

r = redis.StrictRedis(host=db_ip, port=6379, decode_responses=True)
r.set("location", "PoP1")

monitor_url = "http://" + control_ip + ":9000/monitor"
provider_url = "http://" + provider_ip + ":6000/provision"
adaptation_url = "http://194.199.113.66:9500/execute-command"

data = {
    "use_case": "adaptive_video_streaming",
    "name": "Picarx",
    "location": r.get("location")
}

r.set("prediction", 0)

# Start initial deployment
init_deployment = datetime.now()
response = requests.post(url=provider_url, json=data)
info = response.json()
end_deployment = datetime.now()

deployment_duration = (end_deployment - init_deployment).total_seconds()

if response.status_code == 200:
    print(info)
    print(deployment_duration)
    degradations["deployment"] = deployment_duration


def save_results_on_exit():
    """Save degradations data to a JSON file on exit."""
    with open("reactive_results.json", "w") as f:
        json.dump(degradations, f, indent=4)
    print("\nResults saved to 'reactive_results.json'. Exiting gracefully.")


def signal_handler(sig, frame):
    """Handle Ctrl+C to save results."""
    save_results_on_exit()
    sys.exit(0)


# Register the signal handler
signal.signal(signal.SIGINT, signal_handler)


def context_switcher():
    global r, degradations, adaptation_url
    data = {}
    data["name"] = "Picarx"
    i = 0
    count = 0
    while True:
        count+=1
        degradations["request_count"] = count
        try:
            location = r.get("location")
            delay = float(r.get("prediction"))
            if delay >= 900:
                t1 = datetime.now()
                print(f"Degradation occurred at: {t1}")
                if location == "PoP1":
                    data["location"] = "PoP2"
                    print(location)
                    response = requests.get(url=monitor_url, json=data)
                    print(response)
                    t2 = datetime.now()
                    r.set("location", "PoP2")
                    print(f"Adaptation process duration: {(t2 - t1).total_seconds()}")
                    rule = "tc qdisc add dev eth0 root netem delay 0ms"
                    print(rule)
                    response = requests.post(url=adaptation_url, json={'rule': rule})
                    i += 1
                elif location == "PoP2":
                    t1 = datetime.now()
                    print(f"Degradation occurred at: {t1}")
                    data["location"] = "PoP1"
                    print(location)
                    response = requests.get(url=monitor_url, json=data)
                    print(response)
                    t2 = datetime.now()
                    rule = "tc qdisc add dev eth0 root netem delay 0ms"
                    print(rule)
                    response = requests.post(url=adaptation_url, json={'rule': rule})
                    print(f"Adaptation process duration: {(t2 - t1).total_seconds()}")
                    r.set("location", "PoP1")

                degradations["degradation" + str(i)] = {
                    "start_time": t1.isoformat(),
                    "adaptation": t2.isoformat(),
                    "adaptation_time": (t2 - t1).total_seconds(),
                }
            time.sleep(5)
        except Exception as e:
            print(f"Error in context_switcher: {e}")
            json.dump(degradations,"degradations.json")



# Start the context_switcher
context_switcher()
