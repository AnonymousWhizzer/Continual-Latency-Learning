from fastapi import FastAPI
import subprocess
import uvicorn

print('''
****************************************************************
*    ___           _               _             _             *
*   / _ \ _ __ ___| |__   ___  ___| |_ _ __ __ _| |_ ___  _ __ *
*  | | | | '__/ __| '_ \ / _ \/ __| __| '__/ _` | __/ _ \| '__|*
*  | |_| | | | (__| | | |  __/\__ \ |_| | | (_| | || (_) | |   *
*   \___/|_| _\___|_| |_|\___||___/\__|_|  \__,_|\__\___/|_|   *
*           |  _ \  ___ _ __ | | ___  _   _  ___ _ __          *
* _____     | | | |/ _ \ '_ \| |/ _ \| | | |/ _ \ '__|  _____  *
*|_____|    | |_| |  __/ |_) | | (_) | |_| |  __/ |    |_____| *
*           |____/ \___| .__/|_|\___/ \__, |\___|_|            *
*                      |_|            |___/                    *
****************************************************************
''')

app = FastAPI()

def jenkins_deployment(service: str, node: str, link: str):
    print(f'Deploying service {service}')
    cluster = node[3]
    worker = node[len(node)-1]
    param_1='node=cluster-'+cluster
    param_2='ID='+cluster
    param_3='selector=space_'+worker
    param_4='repo_link='+link
    job = "Service_Deployment_Pipeline"
    command = ['java', '-jar', 'jenkins-cli.jar', '-s', 'http://"+deployer_ip+":8080/', '-auth', 'login', 'build' ,'-p', param_1,'-p', param_2, '-p', param_3,'-p', param_4, job, '-s', '-v']
    try:
        # Exécution de la commande
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        # Affichage de la sortie
        print("Liste des jobs Jenkins:")
        print(result.stdout)

    except subprocess.CalledProcessError as e:
        # Affichage de l'erreur en cas d'échec de la commande
        print("Erreur lors de l'exécution de la commande:")
        print(e.stderr)

def jenkins_release(node: str):
    try:
        if node!=None:
            param_5 = 'node=cluster-'+node[3]
            job = "Service_Release_Pipeline"
            release_command = ['java', '-jar', 'jenkins-cli.jar', '-s', 'http://"+deployer_ip+":8080/', '-auth', 'login', 'build' ,'-p', param_5, job, '-s', '-v']
        # Exécution de la commande
        result = subprocess.run(release_command, capture_output=True, text=True, check=True)
        # Affichage de la sortie
        print("Liste des jobs Jenkins:")
        print(result.stdout)

    except subprocess.CalledProcessError as e:
        # Affichage de l'erreur en cas d'échec de la commande
        print("Erreur lors de l'exécution de la commande:")
        print(e.stderr)

def jenkins_reconfigure(node: str):
    try:
        if node!=None:
            param_5 = 'node=cluster-'+node[3]
            job = "Service_Reconfiguration_Pipeline"
            release_command = ['java', '-jar', 'jenkins-cli.jar', '-s', 'http://"+deployer_ip+":8080/', '-auth', 'login', 'build' ,'-p', param_5, job, '-s', '-v']
        # Exécution de la commande
        result = subprocess.run(release_command, capture_output=True, text=True, check=True)
        # Affichage de la sortie
        print("Liste des jobs Jenkins:")
        print(result.stdout)

    except subprocess.CalledProcessError as e:
        # Affichage de l'erreur en cas d'échec de la commande
        print("Erreur lors de l'exécution de la commande:")
        print(e.stderr)

def jenkins_integration(service: str, node: str, link: str):
    print(f'Compiling image for {service} service')
    cluster = node[3]
    worker = node[len(node)-1]
    param_1='node=cluster-'+cluster
    param_2='repo_link='+link
    job = "Service_Integration_Pipeline"
    command = ['java', '-jar', 'jenkins-cli.jar', '-s', 'http://"+deployer_ip+":8080/', '-auth', 'login', 'build' ,'-p', param_1, '-p', param_2, job]
    try:
        # Exécution de la commande
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        # Affichage de la sortie
        print("Liste des jobs Jenkins:")
        print(result.stdout)

    except subprocess.CalledProcessError as e:
        # Affichage de l'erreur en cas d'échec de la commande
        print("Erreur lors de l'exécution de la commande:")
        print(e.stderr)


@app.get("/deploy")
async def service_deployment(data: dict):
    for item in data:
        jenkins_deployment(item, data[item]['node'], data[item]['repo'])

@app.get("/integrate")
async def continuous_integration(data: dict):
    for item in data:
        jenkins_integration(item, data[item]['node'], data[item]['repo'])

@app.get("/release")
async def service_deployment(data: dict):
    if data != None:
        jenkins_release(data["previous_location"])

@app.get("/reconfigure")
async def service_deployment(data: dict):
    if data != None:
        jenkins_reconfigure(data["target_PoP"])


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9000)
    print("server running")