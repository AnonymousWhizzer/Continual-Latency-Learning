from .auto_configure import *
from .auto_predict import *

import pathlib
import os
import json
import joblib
import pandas as pd
import numpy as np
import asyncio

from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import load_model

RMSE_PATIENCE = 10000000
RMSE_THRESHOLD = 1
TRIALS = 50

auto_app = FastAPI()

BASE_DIR = pathlib.Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR.parent / "dataset"
HISTORY_PATH = BASE_DIR.parent / "history.csv"
MODEL_DIR = BASE_DIR.parent / "saved_models"

os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

processed_dataset = None
selected_columns = None
selected_horizon = None
selected_pop_name = None
hyperparameters = None
model = None
predictions = []
prediction_index = 0
rmse_table = []
is_task_running = False


# =========================
# MODEL PERSISTENCE HELPERS
# =========================
def get_model_paths(algo_name: str, pop_name: str):
    base_name = f"{algo_name}_{pop_name}"
    keras_path = MODEL_DIR / f"{base_name}.keras"
    pkl_path = MODEL_DIR / f"{base_name}.pkl"
    metadata_path = MODEL_DIR / f"{base_name}_metadata.json"
    return keras_path, pkl_path, metadata_path


def save_model_and_metadata(model_obj, hyperparams: dict, pop_name: str, target_column: str, forecasting_horizon: int):
    algo_name = hyperparams["forecasting_model"]
    keras_path, pkl_path, metadata_path = get_model_paths(algo_name, pop_name)

    if algo_name in ["RNN", "LSTM", "GRU"]:
        model_obj.save(keras_path)
        model_path = str(keras_path)
    elif algo_name == "ESN":
        joblib.dump(model_obj, pkl_path)
        model_path = str(pkl_path)
    else:
        raise ValueError(f"Unsupported model type for persistence: {algo_name}")

    metadata = {
        "pop_name": pop_name,
        "algo_name": algo_name,
        "target_column": target_column,
        "forecasting_horizon": int(forecasting_horizon),
        "hyperparameters": hyperparams,
        "model_path": model_path
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    return metadata


def load_saved_model(algo_name: str, pop_name: str):
    keras_path, pkl_path, metadata_path = get_model_paths(algo_name, pop_name)

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found for {algo_name}_{pop_name}")

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    if algo_name in ["RNN", "LSTM", "GRU"]:
        if not keras_path.exists():
            raise FileNotFoundError(f"Model file not found: {keras_path}")
        loaded_model = load_model(keras_path)

    elif algo_name == "ESN":
        if not pkl_path.exists():
            raise FileNotFoundError(f"Model file not found: {pkl_path}")
        loaded_model = joblib.load(pkl_path)

    else:
        raise ValueError(f"Unsupported model type for loading: {algo_name}")

    # support both metadata formats:
    # 1) new nested format: {"hyperparameters": {...}}
    # 2) old flat format: {"forecasting_model": ..., "look_back": ..., ...}
    if "hyperparameters" in metadata and isinstance(metadata["hyperparameters"], dict):
        loaded_hyperparameters = metadata["hyperparameters"]
    else:
        loaded_hyperparameters = dict(metadata)

        # normalize old keys to the shape expected downstream
        if "algo_name" in loaded_hyperparameters and "forecasting_model" not in loaded_hyperparameters:
            loaded_hyperparameters["forecasting_model"] = loaded_hyperparameters["algo_name"]

        if "model_path" in loaded_hyperparameters and "saved_model_path" not in loaded_hyperparameters:
            loaded_hyperparameters["saved_model_path"] = loaded_hyperparameters["model_path"]

        loaded_hyperparameters["pop_name"] = pop_name

    return loaded_model, loaded_hyperparameters

def list_available_models_for_pop(pop_name: str):
    available = []

    for meta_file in MODEL_DIR.glob("*_metadata.json"):
        try:
            with open(meta_file, "r", encoding="utf-8") as f:
                metadata = json.load(f)

            if metadata.get("pop_name") == pop_name:
                available.append({
                    "pop_name": metadata.get("pop_name"),
                    "algo_name": metadata.get("algo_name"),
                    "target_column": metadata.get("target_column"),
                    "forecasting_horizon": metadata.get("forecasting_horizon")
                })
        except Exception:
            pass

    return available


# =========================
# HISTORY
# =========================
def load_history_file():
    global processed_dataset

    if HISTORY_PATH.exists():
        try:
            processed_dataset = pd.read_csv(HISTORY_PATH)
            print(f"[INFO] history.csv loaded: {processed_dataset.shape[0]} rows, {processed_dataset.shape[1]} columns")
        except Exception as e:
            print(f"[WARNING] Impossible de charger history.csv: {e}")
            processed_dataset = pd.DataFrame()
    else:
        print("[INFO] history.csv not found, starting with empty history")
        processed_dataset = pd.DataFrame()


def save_history_file():
    global processed_dataset
    try:
        if processed_dataset is None:
            pd.DataFrame().to_csv(HISTORY_PATH, index=False)
        else:
            processed_dataset.to_csv(HISTORY_PATH, index=False)
    except Exception as e:
        print(f"[WARNING] Impossible de sauvegarder history.csv: {e}")


@auto_app.on_event("startup")
async def startup_event():
    load_history_file()


# =========================
# RETRAIN
# =========================
async def find_and_retrain_model():
    global processed_dataset
    global selected_columns
    global selected_horizon
    global selected_pop_name
    global model
    global hyperparameters
    global is_task_running

    is_task_running = True
    try:
        model, hyperparameters, _ = find_best_model(
            processed_dataset,
            selected_columns,
            selected_horizon,
            TRIALS,
            pop_name=selected_pop_name
        )

        save_model_and_metadata(
            model_obj=model,
            hyperparams=hyperparameters,
            pop_name=selected_pop_name,
            target_column=selected_columns[0],
            forecasting_horizon=selected_horizon
        )

        print(
            f"[RETRAIN] model re-saved for pop={selected_pop_name}, "
            f"algo={hyperparameters['forecasting_model']}"
        )

    finally:
        is_task_running = False


async def check_and_retrain_model():
    global rmse_table
    global is_task_running

    if not is_task_running:
        if len(rmse_table) >= RMSE_PATIENCE and all(
            rmse > RMSE_THRESHOLD for rmse in rmse_table[-RMSE_PATIENCE:]
        ):
            await find_and_retrain_model()


# =========================
# ROOT
# =========================
@auto_app.get("/")
async def root():
    history_rows = 0 if processed_dataset is None else len(processed_dataset)
    return {
        "message": "QoS Predictor API is running",
        "history_file": str(HISTORY_PATH),
        "history_rows": history_rows
    }


# =========================
# TRAIN
# =========================
@auto_app.post("/main")
async def main_function(
    file: UploadFile,
    target_columns: str = Form(...),
    forecasting_horizon: int = Form(...),
):
    global processed_dataset
    global selected_columns
    global selected_horizon
    global selected_pop_name
    global hyperparameters
    global model
    global predictions
    global prediction_index
    global rmse_table

    if not file.filename.endswith((".csv", ".xlsx")):
        return JSONResponse(
            content={"error": "Unsupported file format. Please upload a CSV or Excel file."},
            status_code=400
        )

    dataset_path = os.path.join(DATASET_DIR, file.filename)
    pop_name = os.path.splitext(os.path.basename(file.filename))[0]

    with open(dataset_path, "wb") as f:
        f.write(await file.read())

    try:
        df = pd.read_csv(dataset_path) if dataset_path.endswith(".csv") else pd.read_excel(dataset_path)
        processed_dataset = df.copy()
        save_history_file()

        numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()
        target_columns = [col.strip() for col in target_columns.split(",")]

        if len(target_columns) > 1:
            return JSONResponse(content={"error": "Select only one column please."}, status_code=400)

        if not all(column in numeric_columns for column in target_columns):
            return JSONResponse(
                content={
                    "error": "Not all target columns are numerical.",
                    "numeric_columns": numeric_columns
                },
                status_code=400
            )

        if forecasting_horizon is None or int(forecasting_horizon) <= 0:
            return JSONResponse(
                content={"error": "Forecasting horizon must be a positive integer."},
                status_code=400
            )

        selected_columns = target_columns
        selected_horizon = int(forecasting_horizon)
        selected_pop_name = pop_name

        predictions = []
        prediction_index = 0
        rmse_table = []

        model, hyperparameters, training_elapsed_time = find_best_model(
            processed_dataset,
            selected_columns,
            selected_horizon,
            TRIALS,
            pop_name=selected_pop_name
        )

        metadata = save_model_and_metadata(
            model_obj=model,
            hyperparams=hyperparameters,
            pop_name=selected_pop_name,
            target_column=selected_columns[0],
            forecasting_horizon=selected_horizon
        )

        return JSONResponse(
            content={
                "message": "Training completed successfully.",
                "pop_name": selected_pop_name,
                "algo_name": hyperparameters["forecasting_model"],
                "Best Params": hyperparameters,
                "training_elapsed_time": training_elapsed_time,
                "saved_model": metadata["model_path"]
            },
            status_code=200
        )

    except Exception as e:
        return JSONResponse(
            content={"error": f"Error processing the dataset: {str(e)}"},
            status_code=400
        )


# =========================
# LOAD MODEL
# =========================
@auto_app.post("/load_model")
async def load_model_endpoint(
    pop_name: str = Form(...),
    algo_name: str = Form(...),
    target_column: str = Form(...),
    forecasting_horizon: int = Form(...)
):
    global selected_columns
    global selected_horizon
    global selected_pop_name
    global hyperparameters
    global model
    global predictions
    global prediction_index
    global rmse_table
    global processed_dataset

    try:
        loaded_model, loaded_hyperparameters = load_saved_model(algo_name, pop_name)

        model = loaded_model
        hyperparameters = loaded_hyperparameters
        selected_columns = [target_column]
        selected_horizon = int(forecasting_horizon)
        selected_pop_name = pop_name

        predictions = []
        prediction_index = 0
        rmse_table = []

        if processed_dataset is None:
            processed_dataset = pd.DataFrame()

        history_rows = len(processed_dataset)

        return JSONResponse(
            content={
                "message": f"Model {algo_name} for {pop_name} loaded successfully.",
                "hyperparameters": hyperparameters,
                "history_loaded": True,
                "history_rows": history_rows
            },
            status_code=200
        )

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)


# =========================
# SWITCH MODEL
# =========================
@auto_app.post("/switch_model")
async def switch_model_endpoint(
    pop_name: str = Form(...),
    algo_name: str = Form(...),
    target_column: str = Form(...),
    forecasting_horizon: int = Form(...)
):
    global selected_columns
    global selected_horizon
    global selected_pop_name
    global hyperparameters
    global model
    global predictions
    global prediction_index
    global rmse_table
    global processed_dataset

    try:
        loaded_model, loaded_hyperparameters = load_saved_model(algo_name, pop_name)

        model = loaded_model
        hyperparameters = loaded_hyperparameters
        selected_columns = [target_column]
        selected_horizon = int(forecasting_horizon)
        selected_pop_name = pop_name

        predictions = []
        prediction_index = 0
        rmse_table = []

        if processed_dataset is None:
            processed_dataset = pd.DataFrame()

        history_rows = len(processed_dataset)

        return JSONResponse(
            content={
                "message": f"Active model switched to {algo_name} for {pop_name}.",
                "hyperparameters": hyperparameters,
                "history_loaded": True,
                "history_rows": history_rows
            },
            status_code=200
        )

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)


# =========================
# AVAILABLE MODELS
# =========================
@auto_app.get("/available_models")
async def available_models(pop_name: str):
    try:
        available = list_available_models_for_pop(pop_name)
        return JSONResponse(
            content={
                "pop_name": pop_name,
                "available_models": available
            },
            status_code=200
        )
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)


class InputData(BaseModel):
    input_data: str


# =========================
# PREDICT
# =========================
@auto_app.get("/predict")
async def predict_endpoint(input_data: float):
    global processed_dataset
    global selected_columns
    global selected_horizon
    global model
    global hyperparameters
    global predictions
    global prediction_index
    global rmse_table
    global selected_pop_name

    try:
        input_data = float(input_data)
    except ValueError:
        raise HTTPException(status_code=400, detail="Please enter numerical value.")

    if selected_columns is None or len(selected_columns) == 0:
        raise HTTPException(status_code=400, detail="Target column not set.")
    if selected_horizon is None:
        raise HTTPException(status_code=400, detail="Forecasting horizon not set.")
    if hyperparameters is None:
        raise HTTPException(status_code=400, detail="No model or hyperparameters loaded.")
    if model is None:
        raise HTTPException(status_code=400, detail="No model loaded.")

    target_col = selected_columns[0]

    if processed_dataset is None:
        processed_dataset = pd.DataFrame()

    if target_col not in processed_dataset.columns:
        processed_dataset[target_col] = pd.Series(dtype=float)

    new_row = {target_col: input_data}
    processed_dataset = pd.concat([processed_dataset, pd.DataFrame([new_row])], ignore_index=True)
    save_history_file()

    in_data = processed_dataset[[target_col]]
    forecasting_model = hyperparameters["forecasting_model"]

    if forecasting_model in ["RNN", "LSTM", "GRU", "ESN"]:
        if "look_back" not in hyperparameters:
            raise HTTPException(status_code=400, detail="look_back missing in hyperparameters.")

        required_history = hyperparameters["look_back"]
        in_data = in_data[target_col].tail(required_history)

        if len(in_data) < required_history:
            raise HTTPException(
                status_code=400,
                detail=f"Not enough historical data for {forecasting_model}. Need {required_history} values, got {len(in_data)}."
            )
    else:
        if "window_length" not in hyperparameters:
            raise HTTPException(status_code=400, detail="window_length missing in hyperparameters.")

        required_history = hyperparameters["window_length"]
        in_data = in_data[target_col].tail(required_history)

        if len(in_data) < required_history:
            raise HTTPException(
                status_code=400,
                detail=f"Not enough historical data for {forecasting_model}. Need {required_history} values, got {len(in_data)}."
            )

    pred = auto_forecast(model, in_data.values, selected_horizon, hyperparameters)
    predictions.append(pred[0])

    if len(predictions) >= selected_horizon + 1 and prediction_index < len(predictions):
        current_pred = predictions[prediction_index]
        if len(in_data.tail(selected_horizon)) == len(current_pred):
            rmse = np.sqrt(mean_squared_error(in_data.tail(selected_horizon), current_pred))
            prediction_index += 1
            rmse_table.append(rmse)

    asyncio.create_task(check_and_retrain_model())

    return JSONResponse(
        content={
            "prediction": str(pred[0]),
            "forecasting_model": forecasting_model,
            "pop_name": selected_pop_name
        },
        status_code=200
    )


# =========================
# CONFIG UPDATE
# =========================
@auto_app.put("/update_configs")
async def update_constants(new_rmse_patience: int, new_rmse_threshold: float, new_trials: int):
    global RMSE_PATIENCE, RMSE_THRESHOLD, TRIALS
    RMSE_PATIENCE = new_rmse_patience
    RMSE_THRESHOLD = new_rmse_threshold
    TRIALS = new_trials

    return {
        "message": "Constants updated successfully",
        "RMSE_PATIENCE": RMSE_PATIENCE,
        "RMSE_THRESHOLD": RMSE_THRESHOLD,
        "TRIALS": TRIALS
    }