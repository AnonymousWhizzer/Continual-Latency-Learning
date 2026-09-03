import requests
import time
import json
import redis
from datetime import datetime


probe = "localhost"
probe_port = 6379
probe_db = redis.StrictRedis(host=probe, port=probe_port, decode_responses=True)

def scale (val: int):
    if val <= 100:
        duration = 5
    elif val <= 200:
        duration = 5
    elif val <= 1000:
        duration = 2
    elif val <= 2900:
        duration = 3
    elif val > 2900:
        duration = 15
    return duration

probe_db.set("active_pop", 0)

a = input()

#get profiles
# Chargement des données depuis le fichier nodes.json
with open("./profile_definition/profiles.json", "r") as f:
    profiles = json.load(f)

#define path
# cruise_path = [13, 9, 1, 5, 14, 3, 2]
cruise_path = [1,2]
# scenario = [1,2,3]
url = "http://194.199.113.66:9500/execute-command"
# url = "http://172.24.33.217:9500/execute-command"

time.sleep(5)

for PoP in cruise_path:
    probe_db.set("active_pop", PoP)
    values = profiles ["Profile"+str(PoP)]["latencies"]
    plan = profiles ["Profile"+str(PoP)]["occurence"]
    probe_db.set("min", min(values))
    probe_db.set("max", max(values))
    for RTT, duration in zip(values, plan):
        period = scale(RTT) 
        # rule = f"tc qdisc add dev eth0 root netem delay 0ms"
        rule = f"tc qdisc add dev eth0 root netem delay {RTT}ms"
        # print(datetime.now())
        print(rule)
        response = requests.post(url=url, json={'rule': rule})
        probe_db.set("RTT",RTT)
        print(response.json())
        # time.sleep(duration/100)
        time.sleep(period)

probe_db.set("active_pop", 0)
# while True:
#     #set duration to add
#     a = input ("enter duration to set on the interface:")
#     if a != "":
#         #complete command to execute to add duration
#         rule = f"tc qdisc add dev eth0 root netem duration {a}ms"
#     else:
#         rule = ""
#     #send command to the api
#     response = requests.post(url=url, json={'rule': rule})
#     print(response.json())