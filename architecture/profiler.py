from fastapi import FastAPI, HTTPException
import subprocess

app = FastAPI()
base_command = "docker exec proxy "
clear_argument = "tc qdisc del dev eth0 root"

@app.post("/execute-command")
async def execute_command(data: dict):
    global base_command, clear_argument
    try:
        # Execute the command
        print("Updating profile")

        #clear old rule
        result = subprocess.run(base_command+clear_argument, shell=True, capture_output=True, text=True)

        # applying new rule
        if data["rule"]!="":
            print(data["rule"])
            result = subprocess.run(base_command+data['rule'], shell=True, capture_output=True, text=True)

            # Check if the command failed
            if result.returncode != 0:
                raise HTTPException(
                    status_code=500,
                    detail={
                        "error": f"Command failed with error: {result.stderr.strip()}",
                        "output": result.stdout.strip(),
                    },
                )

            # Return the successful output
            return {"output": result.stdout.strip()}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Run the application
# Use this command in the terminal to run the app: uvicorn filename:app --reload