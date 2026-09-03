from fastapi import FastAPI
import requests
from tabulate import tabulate
import redis
import json
import uvicorn
import time

print('''
*******************************************************************
*    ___           _               _             _                *
*   / _ \ _ __ ___| |__   ___  ___| |_ _ __ __ _| |_ ___  _ __    *
*  | | | | '__/ __| '_ \ / _ \/ __| __| '__/ _` | __/ _ \| '__|   *
*  | |_| | | | (__| | | |  __/\__ \ |_| | | (_| | || (_) | |      *
*   \___/|_|  \___|_| |_|\___||___/\__|_|  \__,_|\__\___/|_|      *
*             ____            _             _ _                   *
*            / ___|___  _ __ | |_ _ __ ___ | | | ___ _ __         *
* _____     | |   / _ \| '_ \| __| '__/ _ \| | |/ _ \ '__|  _____ *
*|_____|    | |__| (_) | | | | |_| | | (_) | | |  __/ |    |_____|*
*            \____\___/|_| |_|\__|_|  \___/|_|_|\___|_|           *
*******************************************************************
''')
#clusters status
with open('clusters.json','r') as f:
    clusters_endpoint = json.load(f)

#setting up controller db
redis_host = "localhost"
redis_port = 6379
controller_db = redis.StrictRedis(host=redis_host, port=redis_port, decode_responses=True)

#setting up data intelligence provide
monitoring_ip = "localhost"

#setting up API
app = FastAPI()
selection_url = "http://"+monitoring_ip+":8000/node_selection"
deployment_url = "http://194.199.113.87:9000/deploy"
integration_url = "http://194.199.113.87:9000/integrate"
release_url = "http://194.199.113.87:9000/release"

#Setting up values:
data = {
    "service_list": ["proxy"]
}
data_string  = json.dumps(data)
controller_db.set('adaptive_video_streaming', data_string)

def interpreter(use_case: str):
    required_services = {}
    global controller_db
    string_output = controller_db.get(use_case)
    required_services = json.loads(string_output)
    print(f"Retrieved {use_case} service list: {required_services}")
    return required_services

def broker (required_services: dict):
    global selection_url
    services_location = {}
    #sending services to broker for location selection
    response = requests.post(selection_url, json=required_services)
    if response.status_code == 200:
        print("Requête réussie:")
        services_location = response.json()
        print(services_location)
        table = []
        for service, data in services_location.items():
            table.append([
                f"{service}",
                f"{data['node']}",
                f"{data['repo']}"
            ])
        headers = ["Service", "Location", "Repository"]
        print(tabulate(table, headers=headers))
    else:
        print(f"La requête a échoué avec le statut : {response.status_code}")
    return services_location

def deployer(services_location: dict):
    global deployment_url
    # global integration_url
    # print(f"Source code processing for integration")
    # response  = requests.get(integration_url, json=services_location)
    # if response.status_code == 200:
    #     print(f"Integration Successful")
    # else:
    #     print(f"Error while integrating source code {response.status_code}")
    print(f"Contacting Jenkins Master Node for placement")
    response = requests.get(deployment_url, json=services_location)
    if response.status_code == 200:
        print(f"Deployment Successful")
    else:
        print(f"Error while deploying services {response.status_code}")

def migrator(services_location: dict):
    global deployment_url
    print(f"Contacting Jenkins Master Node for placement")
    response = requests.get(deployment_url, json=services_location)
    if response.status_code == 200:
        print(f"Deployment Successful")
    else:
        print(f"Error while deploying services {response.status_code}")

def release(service_location: dict):
    #time.sleep(10)
    global release_url
    print(f"Contacting previous cluster for release")
    response = requests.get(release_url, json=service_location)
    if response.status_code == 200:
        print(f"Deployment Successful")
    else:
        print(f"Error while deploying services {response.status_code}")


def post_deployment_actions(data:dict, name: str, location: str):
    endpoints={}
    #Returning service location to user:
    for item in data:
        if "PoP1" in data[item]["node"]:
            endpoints["IP"] = clusters_endpoint["cluster_1"]
        else:
            endpoints["IP"] = clusters_endpoint["cluster_2"]
        endpoints["rtmp_port"] = "31935"
        endpoints["http_port"] = "31555"
    #Creating Service consumer profile within Orchestrator
    print(f"Creating Service consumer profile within controller")
    user = {
        "name": name,
        "location": location
    }
    user_data = json.dumps(user)
    controller_db.set(user["name"], user_data)
    return endpoints


@app.post("/service_init")
async def provision(data: dict):
    # time.sleep(5)
    # t1 = time.time()
    global controller_db
    #retrieve service list required for use case
    print(f"Establishing required services")
    broker_input = data
    broker_output = {}
    endpoints = {}
    temp = interpreter(data["use_case"])
    #a = input()
    broker_input ["service_list"] = temp['service_list']
    broker_input ["case"] = "deployment"
    print(broker_input)

    #selecting service location through broker
    print(f"Services location definition process according to Ressource catalog")
    broker_output = broker(broker_input)#contain service location
    #a = input()

    #CI/CD
    print(f"Service integration and deployment process through Jenkins")
    deployer(broker_output)
    #a = input()
    #Returning service location to user:
    endpoints = post_deployment_actions(broker_output, data['name'], data["location"])
    print(f"Returning endpoints for services consumption")
    # t2 = time.time()
    # endpoints["processing_time": (t2-t1)*1000]
    return endpoints

@app.get("/monitor")
async def perf_monitor(data: dict):
    broker_output = {}
    new_endpoints = {}
    old_location = {}
    global controller_db
    #migration following client trajectory
    current_info = json.loads(controller_db.get(data["name"]))
    if current_info["location"]!=data["location"]:
        print(f"{current_info['name']} about to leave {current_info['location']} for {data['location']} ")
        old_location["previous_location"] = current_info["location"]
        #selecting next host for migration
        broker_input = {
            "service_list":["proxy"],
            "location": data['location'],
            "case":"deployment"
        }
        broker_output = broker(broker_input)
        #service migration
        migrator(broker_output)
        #update provider
        new_endpoints = post_deployment_actions(broker_output, current_info['name'], data["location"])
        print(new_endpoints)
        # response = requests.get(url="http://"+monitoring_ip+":6000/update", json=new_endpoints)
        #release previous service
        release(old_location)

    # param = {"input_data":data["rtt"]}
    # if data["location"]=="PoP1":
    #     i = "9001"
    # elif data["location"]=="PoP2":
    #     i = "9002"
    # else:
    #     i = "9003"
    # url = "http://"+monitoring_ip+":"+i+"/predict"
    # response = requests.get(url, params=param)
    # print(response.text)



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9000)
    print("server running")