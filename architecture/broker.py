from fastapi import FastAPI
import requests
import json
import time
from tabulate import tabulate
import uvicorn
import subprocess
import re
import concurrent.futures

print('''
********************************************************************
* ____                  _            ____            _             *
*/ ___|  ___ _ ____   _(_) ___ ___  | __ ) _ __ ___ | | _____ _ __ *
*\___ \ / _ \ '__\ \ / / |/ __/ _ \ |  _ \| '__/ _ \| |/ / _ \ '__|*
* ___) |  __/ |   \ V /| | (_|  __/ | |_) | | | (_) |   <  __/ |   *
*|____/ \___|_|    \_/ |_|\___\___| |____/|_|  \___/|_|\_\___|_|   *
********************************************************************''')


app = FastAPI()
best_delay = 10000
cluster = {}
services = {}
name = ''
selected = ''


# Chargement des données depuis le fichier nodes.json
with open('clusters.json','r') as f:
    clusters_endpoint = json.load(f)
with open('nodes.json','r') as f:
    nodes = json.load(f)
# Chargement des données depuis le fichier services.json
with open('services.json', 'r') as f:
    services = json.load(f)


def ping_ip(ip):
    try:
        # Exécute la commande ping
        output = subprocess.check_output(f"ping {ip} -n 4", universal_newlines=True)
        return output
    except subprocess.CalledProcessError as e:
        return None


def parse_ping_output(output):
    min_time = max_time = avg_time = None
    if output:
        for line in output.splitlines():
            # Recherche de la ligne contenant les statistiques de temps
            if 'Minimum' in line or 'Minimun' in line:
                match = re.search(r'Minimum = (\d+)ms, Maximum = (\d+)ms, Average = (\d+)ms', line)
                if match:
                    min_time = int(match.group(1))
                    max_time = int(match.group(2))
                    avg_time = int(match.group(3))
                break
    return min_time, max_time, avg_time


def ping_and_collect(node, ip):
    output = ping_ip(ip)
    min_time, max_time, avg_time = parse_ping_output(output)
    return (node, ip, min_time, max_time, avg_time)


def host_selection(location: str):
    # Filtrer les nœuds pour ne garder que ceux contenant "PoP2"
    candidate_nodes = {node: ip for node, ip in nodes.items() if location in node}

    # Ping les IPs en parallèle pour les nœuds filtrés
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(lambda item: ping_and_collect(item[0], item[1]), candidate_nodes.items()))

    # Prépare les données pour le tableau
    table = []
    valid_results = []  # Pour stocker les résultats valides (ceux sans échec)
    for result in results:
        node, ip, min_time, max_time, avg_time = result
        if min_time is not None:
            table.append([node, ip, f"{min_time}ms", f"{max_time}ms", f"{avg_time}ms"])
            valid_results.append(result)  # Ajoute les résultats valides à une nouvelle liste
        else:
            table.append([node, ip, "Ping failed", "Ping failed", "Ping failed"])

    # Affiche les résultats sous forme de tableau
    print(tabulate(table, headers=["Node", "IP", "Minimum", "Maximum", "Average"], tablefmt="grid"))

    # Trouver le nœud avec le ping moyen le plus bas
    if valid_results:
        best_result = min(valid_results, key=lambda x: x[4])  # Trier par avg_time
        best_node, best_ip, best_min, best_max, best_avg = best_result

        # Affiche le meilleur résultat
        print(f"\nMeilleur résultat:")
        print(f"Nœud: {best_node}, IP: {best_ip}, Minimum = {best_min}ms, Maximum = {best_max}ms, Average = {best_avg}ms")
        
        return best_node
    else:
        print("\nAucun résultat valide trouvé.")
        return None  # Retourner None s'il n'y a pas de résultat valide



def match_service(services, service_class, case, delay=None):
    # Prépare les données pour le tableau
    table = []
    for service_name, service_data in services.items():
        table.append([
            service_name,
            service_data.get('type', 'N/A'),
            service_data.get('class', 'N/A'),
            service_data.get('use_case', 'N/A'),
            service_data.get('supported_delay', 'N/A'),
            ", ".join(service_data.get('supported_service', [])),
            service_data.get('location', 'N/A')
        ])


    # Affiche les résultats sous forme de tableau
    print(tabulate(table, headers=["Service", "Type", "Class", "Use Case", "Supported Delay", "Supported Service", "Location"], tablefmt="grid"))




    # Définir le use_case en fonction du paramètre 'case'
    if case == "deployment":
        use_case = "normal_conditions"
    elif case == "adaptation":
        use_case = "degraded_conditions"
    else:
        return None  # Si 'case' n'est ni 'deployment' ni 'adaptation', on retourne None


    # Recherche du service correspondant en fonction des paramètres
    for service_name, service_data in services.items():
        # Vérification de la correspondance du use_case
        if service_data.get("use_case") == use_case:
            # Vérification de la classe de service
            if service_data.get("class") == service_class:
                # Si le use_case est "normal_conditions", vérifier aussi le délai
                if use_case == "normal_conditions":
                    supported_delay = service_data.get("supported_delay")
                    if supported_delay is not None and delay is not None and supported_delay <= delay:
                        return service_data.get("location")  # Retourner le link si trouvé
                # Si le use_case est "degraded_conditions", ne pas vérifier le délai
                elif use_case == "degraded_conditions":
                    return service_data.get("location")  # Retourner le link si trouvé


    # Aucun service trouvé qui correspond aux critères
    return None


def catalog(input_data:dict):
    table = []
    for node, data in input_data.items():
        table.append([node,
                    f"{data['CPU_allocatable']}m",
                    f"{data['Memory_allocatable']}Ki",
                    f"{data['CPU_request%']}%",
                    f"{data['Memory_request%']}%",
                    f"{data['CPU_limit%']}%",
                    f"{data['Memory_limit%']}%",
                    f"{data['CPU_usage%']}%",
                    f"{data['Memory_usage%']}%"])


    headers = ["NODE", "CPU_allocatable", "Memory_allocatable", "CPU_request%", "Memory_request%", "CPU_limit%", "Memory_limit%", "CPU_usage%", "Memory_usage%"]


    print(tabulate(table, headers=headers, tablefmt="grid"))


def cluster_selection(data: dict):
    global clusters_endpoint, cluster, best_delay, name
    if data['location'] == 'PoP1':
            val = clusters_endpoint['cluster_1']
            print("""
 .----------------.  .----------------.  .----------------.  .----------------.  .----------------.  .----------------.  .----------------.   .----------------.
| .--------------. || .--------------. || .--------------. || .--------------. || .--------------. || .--------------. || .--------------. | | .--------------. |
| |     ______   | || |   _____      | || | _____  _____ | || |    _______   | || |  _________   | || |  _________   | || |  _______     | | | |     __       | |
| |   .' ___  |  | || |  |_   _|     | || ||_   _||_   _|| || |   /  ___  |  | || | |  _   _  |  | || | |_   ___  |  | || | |_   __ \    | | | |    /  |      | |
| |  / .'   \_|  | || |    | |       | || |  | |    | |  | || |  |  (__ \_|  | || | |_/ | | \_|  | || |   | |_  \_|  | || |   | |__) |   | | | |    `| |      | |
| |  | |         | || |    | |   _   | || |  | '    ' |  | || |   '.___`-.   | || |     | |      | || |   |  _|  _   | || |   |  __ /    | | | |     | |      | |
| |  \ `.___.'\  | || |   _| |__/ |  | || |   \ `--' /   | || |  |`\____) |  | || |    _| |_     | || |  _| |___/ |  | || |  _| |  \ \_  | | | |    _| |_     | |
| |   `._____.'  | || |  |________|  | || |    `.__.'    | || |  |_______.'  | || |   |_____|    | || | |_________|  | || | |____| |___| | | | |   |_____|    | |
| |              | || |              | || |              | || |              | || |              | || |              | || |              | | | |              | |
| '--------------' || '--------------' || '--------------' || '--------------' || '--------------' || '--------------' || '--------------' | | '--------------' |
 '----------------'  '----------------'  '----------------'  '----------------'  '----------------'  '----------------'  '----------------'   '----------------'
""")
    else:
            val = clusters_endpoint['cluster_2']
            print("""


 .----------------.  .----------------.  .----------------.  .----------------.  .----------------.  .----------------.  .----------------.   .----------------.
| .--------------. || .--------------. || .--------------. || .--------------. || .--------------. || .--------------. || .--------------. | | .--------------. |
| |     ______   | || |   _____      | || | _____  _____ | || |    _______   | || |  _________   | || |  _________   | || |  _______     | | | |    _____     | |
| |   .' ___  |  | || |  |_   _|     | || ||_   _||_   _|| || |   /  ___  |  | || | |  _   _  |  | || | |_   ___  |  | || | |_   __ \    | | | |   / ___ `.   | |
| |  / .'   \_|  | || |    | |       | || |  | |    | |  | || |  |  (__ \_|  | || | |_/ | | \_|  | || |   | |_  \_|  | || |   | |__) |   | | | |  |_/___) |   | |
| |  | |         | || |    | |   _   | || |  | '    ' |  | || |   '.___`-.   | || |     | |      | || |   |  _|  _   | || |   |  __ /    | | | |   .'____.'   | |
| |  \ `.___.'\  | || |   _| |__/ |  | || |   \ `--' /   | || |  |`\____) |  | || |    _| |_     | || |  _| |___/ |  | || |  _| |  \ \_  | | | |  / /____     | |
| |   `._____.'  | || |  |________|  | || |    `.__.'    | || |  |_______.'  | || |   |_____|    | || | |_________|  | || | |____| |___| | | | |  |_______|   | |
| |              | || |              | || |              | || |              | || |              | || |              | || |              | | | |              | |
| '--------------' || '--------------' || '--------------' || '--------------' || '--------------' || '--------------' || '--------------' | | '--------------' |
 '----------------'  '----------------'  '----------------'  '----------------'  '----------------'  '----------------'  '----------------'   '----------------'
""")
        #Check ressource catalog
    endpoint = "http://"+val+":9001/ressources"
    resp = requests.get(url=endpoint)
    #Affichage du catalog
    catalog(resp.json())
    #evaluate delay

@app.post("/node_selection")
async def node_selection(data: dict):
    # print(data)
    selected_node = {}
    global cluster
    global services
    #retrieving list of microservices to deploy
    service_list = data['service_list']
    print(f" Matchmaking & Negociation for {service_list}")
    selected_node = {}
    cluster_selection(data)
    #selecting nodes for deployment
    if data["case"]=="deployment":
        for item in service_list:
            service_host = host_selection(data["location"])
            if data["location"]=="PoP1" or data["location"]=="PoP2":
                service_repo = match_service(services,item,case='deployment',delay=100)
            else:
                service_repo = match_service(services,item,case='deployment',delay=70)
            selected_node[item]={
                "node":service_host,
                "repo":service_repo
            }
    else:
        for item in service_list:
            service_host = host_selection(data["location"])
            service_repo = match_service(services, item, case='adaptation')
            selected_node["item"] = {
                "node":service_host,
                "repo":service_repo
            }
    return selected_node


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
    print("server running")

