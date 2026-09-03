import os
import json
import pickle
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
import keras
import optuna
from optuna.samplers import TPESampler
from keras.callbacks import EarlyStopping

from .ASAP import *
from .esn import *

MODEL_DIR = "saved_models"


def ensure_model_dir():
    os.makedirs(MODEL_DIR, exist_ok=True)


def get_model_filename(model_name, pop_name):
    if model_name in ["RNN", "LSTM", "GRU"]:
        return os.path.join(MODEL_DIR, f"{model_name}_{pop_name}.keras")
    elif model_name == "ESN":
        return os.path.join(MODEL_DIR, f"{model_name}_{pop_name}.pkl")
    elif model_name == "ARIMA":
        return os.path.join(MODEL_DIR, f"{model_name}_{pop_name}_config.json")
    else:
        return os.path.join(MODEL_DIR, f"{model_name}_{pop_name}.pkl")


def get_metadata_filename(model_name, pop_name):
    return os.path.join(MODEL_DIR, f"{model_name}_{pop_name}_metadata.json")


def save_metadata(best_params, pop_name, model_path):
    ensure_model_dir()
    model_name = best_params["forecasting_model"]
    metadata_path = get_metadata_filename(model_name, pop_name)

    metadata = dict(best_params)
    metadata["pop_name"] = pop_name
    metadata["saved_model_path"] = model_path

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return metadata_path


def save_trained_model(model, best_params, pop_name):
    ensure_model_dir()

    model_name = best_params["forecasting_model"]
    model_path = get_model_filename(model_name, pop_name)

    if model_name in ["RNN", "LSTM", "GRU"]:
        model.save(model_path)

    elif model_name == "ESN":
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

    elif model_name == "ARIMA":
        with open(model_path, "w", encoding="utf-8") as f:
            json.dump(best_params, f, indent=2)

    else:
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

    metadata_path = save_metadata(best_params, pop_name, model_path)
    return model_path, metadata_path


def load_saved_model(algo_name, pop_name):
    ensure_model_dir()

    model_path = get_model_filename(algo_name, pop_name)
    metadata_path = get_metadata_filename(algo_name, pop_name)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if algo_name != "ARIMA" and not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    if algo_name in ["RNN", "LSTM", "GRU"]:
        model = load_model(model_path)
        with open(metadata_path, "r", encoding="utf-8") as f:
            hyperparameters = json.load(f)
        return model, hyperparameters

    elif algo_name == "ESN":
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        with open(metadata_path, "r", encoding="utf-8") as f:
            hyperparameters = json.load(f)
        return model, hyperparameters

    elif algo_name == "ARIMA":
        with open(model_path, "r", encoding="utf-8") as f:
            hyperparameters = json.load(f)
        hyperparameters["saved_model_path"] = model_path
        hyperparameters["pop_name"] = pop_name
        return None, hyperparameters

    else:
        raise ValueError(f"Unsupported model type: {algo_name}")


def list_available_models_for_pop(pop_name):
    ensure_model_dir()
    available = {}

    for algo_name in ["RNN", "LSTM", "GRU", "ESN", "ARIMA"]:
        model_path = get_model_filename(algo_name, pop_name)
        if os.path.exists(model_path):
            available[algo_name] = model_path

    return available


def sarima_multistep_forecast(history, config, window_size, n_steps):
    order, sorder, trend = config
    new_hist = history[:]
    yhat = []
    total_train_time = 0
    total_prediction_time = 0

    for _ in range(n_steps):
        model = SARIMAX(
            new_hist[-window_size:],
            order=order,
            seasonal_order=sorder,
            trend=trend,
            enforce_stationarity=False,
            enforce_invertibility=False
        )

        training_start_time = time.time()
        model_fit = model.fit(disp=False)
        training_end_time = time.time()
        total_train_time += (training_end_time - training_start_time)

        predictions_start_time = time.time()
        prediction = model_fit.predict(
            start=len(history[-window_size:]),
            end=len(history[-window_size:])
        )
        predictions_end_time = time.time()
        total_prediction_time += (predictions_end_time - predictions_start_time)

        yhat = np.append(yhat, prediction)
        new_hist = np.append(new_hist, prediction)
        new_hist = new_hist[1:]

    return yhat, total_train_time, total_prediction_time


def create_multistep_dataset(data, n_input, n_out=1):
    X, y = list(), list()
    in_start = 0

    for _ in range(len(data)):
        in_end = in_start + n_input
        out_end = in_end + n_out
        if out_end <= len(data):
            x_input = data[in_start:in_end]
            X.append(x_input)
            y.append(data[in_end:out_end])
        in_start += 1

    return np.array(X), np.array(y)


def find_best_model(data, selected_columns, horizon, TRIALS, pop_name="unknown_pop"):
    if horizon == 1 and len(selected_columns) == 1:

        def objective(trial):
            forecasting_model = trial.suggest_categorical(
                "forecasting_model",
                ["RNN", "LSTM", "GRU", "ESN", "ARIMA"]
            )

            variable = data[[selected_columns[0]]]
            variable_dataset = variable.values
            window_size, _ = smooth_ASAP(variable_dataset, resolution=50)
            denoised_variable_dataset = moving_average(variable_dataset, window_size)


            train, test = denoised_variable_dataset[-1500:-200], denoised_variable_dataset[-200:]

            if forecasting_model in ["LSTM", "GRU", "RNN"]:
                num_hidden_layers = trial.suggest_int("num_hidden_layers", 1, 5)
                look_back = trial.suggest_int("look_back", 10, 150)
                learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
                batch_size = trial.suggest_int("batch_size", 10, 150)
                epochs = 3

                trainX, trainY = create_multistep_dataset(train, look_back, 1)
                validX, validY = create_multistep_dataset(test, look_back, 1)

                trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
                validX = np.reshape(validX, (validX.shape[0], 1, validX.shape[1]))

                model = Sequential()
                for i in range(num_hidden_layers):
                    num_units = trial.suggest_int(f"units_layer_{i}", 8, 256, log=True)
                    return_sequences = i < num_hidden_layers - 1

                    if forecasting_model == "RNN":
                        model.add(SimpleRNN(units=num_units, return_sequences=return_sequences))
                    elif forecasting_model == "LSTM":
                        model.add(LSTM(units=num_units, return_sequences=return_sequences))
                    elif forecasting_model == "GRU":
                        model.add(GRU(units=num_units, return_sequences=return_sequences))

                model.add(Dense(1))
                optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
                model.compile(loss="mean_squared_error", optimizer=optimizer)
                model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size, verbose=0)
                valid_predict = model.predict(validX, verbose=0)

                return np.sqrt(mean_squared_error(validY, valid_predict))

            elif forecasting_model == "ESN":
                n_reservoir = trial.suggest_int("n_reservoir", 10, 1000)
                sparsity = trial.suggest_categorical("sparsity", [0.01, 0.1, 0.2, 0.3, 0.4, 0.5])
                spectral_radius = trial.suggest_categorical(
                    "spectral_radius",
                    [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.25, 10.0]
                )
                noise = trial.suggest_categorical(
                    "noise",
                    [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009]
                )
                look_back = trial.suggest_int("look_back", 10, 150)

                trainX, trainY = create_multistep_dataset(train, look_back, 1)
                validX, validY = create_multistep_dataset(test, look_back, 1)

                trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1]))
                validX = np.reshape(validX, (validX.shape[0], validX.shape[1]))

                model = ESN(
                    n_inputs=look_back,
                    n_outputs=1,
                    n_reservoir=n_reservoir,
                    sparsity=sparsity,
                    random_state=1234,
                    spectral_radius=spectral_radius,
                    noise=noise,
                    teacher_scaling=10
                )

                model.fit(trainX, trainY)
                predictions = np.array(model.predict(validX))
                return np.sqrt(mean_squared_error(predictions, validY))

            else:  # ARIMA
                p = trial.suggest_int("p", 0, 3)
                d = trial.suggest_int("d", 0, 2)
                q = trial.suggest_int("q", 0, 3)
                window_length = trial.suggest_int("window_length", 5, 150)

                order = (p, d, q)
                seasonal_order = (0, 0, 0, 0)
                cfg = (order, seasonal_order, "c")

                predictions = []
                ys = []
                history = []
                history.extend(train)

                yhat, _, _ = sarima_multistep_forecast(np.array(history), cfg, window_length, horizon)
                predictions.append(yhat)
                ys.append(test[:horizon])
                history.extend(test[:horizon])

                for i in range(horizon, len(test)):
                    yhat, _, _ = sarima_multistep_forecast(np.array(history), cfg, window_length, horizon)
                    predictions.append(yhat)
                    ys.append(test[i:i + horizon])
                    history.append(test[i])

                ys_converted = [array.tolist() for array in ys if len(array) == horizon]
                predictions_converted = [array.tolist() for array in predictions]

                return np.sqrt(mean_squared_error(ys_converted, predictions_converted[:len(ys_converted)]))

        study = optuna.create_study(direction="minimize", sampler=TPESampler())
        study.optimize(objective, n_trials=TRIALS, n_jobs=-1)
        best_params = study.best_params

    elif horizon > 1 and len(selected_columns) == 1:

        def objective(trial):
            forecasting_model = trial.suggest_categorical(
                "forecasting_model",
                # ["RNN", "LSTM", "GRU", "ESN", "ARIMA"]
                ["ESN"]
            )

            variable = data[[selected_columns[0]]]
            variable_dataset = variable.values
            window_size, _ = smooth_ASAP(variable_dataset, resolution=50)
            denoised_variable_dataset = moving_average(variable_dataset, window_size)

            half_size = len(denoised_variable_dataset) // 2  # prendre la moitié
            train_size = int(0.8 * half_size)                 # 80% pour l'entraînement
            test_size = half_size - train_size                 # 20% pour le test
            subset = denoised_variable_dataset[-half_size:]    # utiliser la moitié la plus récente
            train = subset[:train_size]
            test = subset[train_size:]

            # train, test = denoised_variable_dataset[-1500:-200], denoised_variable_dataset[-200:]

            if forecasting_model in ["RNN", "LSTM", "GRU"]:
                look_back = trial.suggest_int("look_back", 10, 150)
                num_hidden_layers = trial.suggest_int("num_hidden_layers", 1, 10)
                learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
                batch_size = trial.suggest_int("batch_size", 10, 150)
                epochs = 3

                trainX, trainY = create_multistep_dataset(train, look_back, horizon)
                validX, validY = create_multistep_dataset(test, look_back, horizon)

                trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
                validX = np.reshape(validX, (validX.shape[0], 1, validX.shape[1]))

                model = Sequential()
                for i in range(num_hidden_layers):
                    num_units = trial.suggest_int(f"units_layer_{i}", 8, 256, log=True)
                    return_sequences = i < num_hidden_layers - 1

                    if forecasting_model == "RNN":
                        model.add(SimpleRNN(units=num_units, return_sequences=return_sequences))
                    elif forecasting_model == "LSTM":
                        model.add(LSTM(units=num_units, return_sequences=return_sequences))
                    elif forecasting_model == "GRU":
                        model.add(GRU(units=num_units, return_sequences=return_sequences))

                model.add(Dense(horizon))
                optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
                model.compile(loss="mean_squared_error", optimizer=optimizer)
                model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size, verbose=0)
                valid_predict = model.predict(validX, verbose=0)

                return np.sqrt(mean_squared_error(validY, valid_predict))

            elif forecasting_model == "ESN":
                def esn_recursive_strategy(model, X_row, n_steps):
                    forecasts = []
                    for _ in range(n_steps):
                        forecast = model.predict(np.array([X_row]))
                        forecasts.append(forecast[0, 0])
                        X_row = X_row.tolist()
                        X_row.append(forecast[0, 0])
                        X_row = X_row[1:]
                        X_row = np.array(X_row)
                    return forecasts

                def esn_make_predictions(model, X, n_steps):
                    predictions = []
                    for i in range(len(X)):
                        row_forecasts = esn_recursive_strategy(model, X[i, :], n_steps)
                        predictions.append(row_forecasts)
                    return predictions

                n_reservoir = trial.suggest_int("n_reservoir", 10, 1000)
                sparsity = trial.suggest_categorical("sparsity", [0.01, 0.1, 0.2, 0.3, 0.4, 0.5])
                spectral_radius = trial.suggest_categorical(
                    "spectral_radius",
                    [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.25, 10.0]
                )
                noise = trial.suggest_categorical(
                    "noise",
                    [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009]
                )
                look_back = trial.suggest_int("look_back", 10, 150)

                trainX, trainY = create_multistep_dataset(train, look_back, 1)
                testX, _ = create_multistep_dataset(test, look_back, 1)

                trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1]))
                testX = np.reshape(testX, (testX.shape[0], testX.shape[1]))

                model = ESN(
                    n_inputs=look_back,
                    n_outputs=1,
                    n_reservoir=n_reservoir,
                    sparsity=sparsity,
                    random_state=1234,
                    spectral_radius=spectral_radius,
                    noise=noise,
                    teacher_scaling=10
                )

                model.fit(trainX, trainY)
                test_predict = np.array(esn_make_predictions(model, testX, horizon))
                _, new_testY = create_multistep_dataset(test, look_back, horizon)

                return np.sqrt(mean_squared_error(new_testY, test_predict[:len(new_testY), :]))

            else:  # ARIMA
                p = trial.suggest_int("p", 0, 3)
                d = trial.suggest_int("d", 0, 2)
                q = trial.suggest_int("q", 0, 3)
                window_length = trial.suggest_int("window_length", 5, 15)

                order = (p, d, q)
                seasonal_order = (0, 0, 0, 0)
                cfg = (order, seasonal_order, "c")

                predictions = []
                ys = []
                history = []
                history.extend(train)

                yhat, _, _ = sarima_multistep_forecast(np.array(history), cfg, window_length, horizon)
                predictions.append(yhat)
                ys.append(test[:horizon])
                history.extend(test[:horizon])

                for i in range(horizon, len(test)):
                    yhat, _, _ = sarima_multistep_forecast(np.array(history), cfg, window_length, horizon)
                    predictions.append(yhat)
                    ys.append(test[i:i + horizon])
                    history.append(test[i])

                ys_converted = [array.tolist() for array in ys if len(array) == horizon]
                predictions_converted = [array.tolist() for array in predictions]

                return np.sqrt(mean_squared_error(ys_converted, predictions_converted[:len(ys_converted)]))

        study = optuna.create_study(direction="minimize", sampler=TPESampler())
        study.optimize(objective, n_trials=TRIALS, n_jobs=-1)
        best_params = study.best_params

    else:
        raise ValueError("Only univariate forecasting is supported in this version.")

    if best_params["forecasting_model"] == "ARIMA":
        model_path, metadata_path = save_trained_model(None, best_params, pop_name)
        best_params["saved_model_path"] = model_path
        best_params["metadata_path"] = metadata_path
        best_params["pop_name"] = pop_name
        return None, best_params, None

    variable = data[[selected_columns[0]]]
    variable_dataset = variable.values
    window_size, _ = smooth_ASAP(variable_dataset, resolution=50)
    denoised_variable_dataset = moving_average(variable_dataset, window_size)

    if horizon == 1 and len(selected_columns) == 1:
        if best_params["forecasting_model"] in ["RNN", "LSTM", "GRU"]:
            look_back = best_params["look_back"]
            num_hidden_layers = best_params["num_hidden_layers"]
            learning_rate = best_params["learning_rate"]
            batch_size = best_params["batch_size"]

            trainX, trainY = create_multistep_dataset(denoised_variable_dataset, look_back, 1)
            trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))

            model = Sequential()
            for i in range(num_hidden_layers):
                num_units = best_params[f"units_layer_{i}"]
                return_sequences = i < num_hidden_layers - 1
                if best_params["forecasting_model"] == "RNN":
                    model.add(SimpleRNN(units=num_units, return_sequences=return_sequences))
                elif best_params["forecasting_model"] == "LSTM":
                    model.add(LSTM(units=num_units, return_sequences=return_sequences))
                elif best_params["forecasting_model"] == "GRU":
                    model.add(GRU(units=num_units, return_sequences=return_sequences))

            model.add(Dense(1))
            optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
            model.compile(loss="mean_squared_error", optimizer=optimizer)

            training_start_time = time.time()
            early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
            model.fit(
                trainX,
                trainY,
                epochs=100,
                batch_size=batch_size,
                verbose=0,
                callbacks=[early_stopping],
                validation_split=0.1
            )
            training_elapsed_time = time.time() - training_start_time

            model_path, metadata_path = save_trained_model(model, best_params, pop_name)
            best_params["saved_model_path"] = model_path
            best_params["metadata_path"] = metadata_path
            best_params["pop_name"] = pop_name
            return model, best_params, training_elapsed_time

        elif best_params["forecasting_model"] == "ESN":
            n_reservoir = best_params["n_reservoir"]
            sparsity = best_params["sparsity"]
            spectral_radius = best_params["spectral_radius"]
            noise = best_params["noise"]
            look_back = best_params["look_back"]

            trainX, trainY = create_multistep_dataset(denoised_variable_dataset, look_back, 1)
            trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1]))

            model = ESN(
                n_inputs=look_back,
                n_outputs=1,
                n_reservoir=n_reservoir,
                sparsity=sparsity,
                random_state=1234,
                spectral_radius=spectral_radius,
                noise=noise,
                teacher_scaling=10
            )

            training_start_time = time.time()
            model.fit(trainX, trainY)
            training_elapsed_time = time.time() - training_start_time

            model_path, metadata_path = save_trained_model(model, best_params, pop_name)
            best_params["saved_model_path"] = model_path
            best_params["metadata_path"] = metadata_path
            best_params["pop_name"] = pop_name
            return model, best_params, training_elapsed_time

    elif horizon > 1 and len(selected_columns) == 1:
        if best_params["forecasting_model"] in ["RNN", "LSTM", "GRU"]:
            look_back = best_params["look_back"]
            num_hidden_layers = best_params["num_hidden_layers"]
            learning_rate = best_params["learning_rate"]
            batch_size = best_params["batch_size"]

            trainX, trainY = create_multistep_dataset(denoised_variable_dataset, look_back, horizon)
            trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))

            model = Sequential()
            for i in range(num_hidden_layers):
                num_units = best_params[f"units_layer_{i}"]
                return_sequences = i < num_hidden_layers - 1
                if best_params["forecasting_model"] == "RNN":
                    model.add(SimpleRNN(units=num_units, return_sequences=return_sequences))
                elif best_params["forecasting_model"] == "LSTM":
                    model.add(LSTM(units=num_units, return_sequences=return_sequences))
                elif best_params["forecasting_model"] == "GRU":
                    model.add(GRU(units=num_units, return_sequences=return_sequences))

            model.add(Dense(horizon))
            optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
            model.compile(loss="mean_squared_error", optimizer=optimizer)

            training_start_time = time.time()
            early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
            model.fit(
                trainX,
                trainY,
                epochs=100,
                batch_size=batch_size,
                verbose=0,
                callbacks=[early_stopping],
                validation_split=0.1
            )
            training_elapsed_time = time.time() - training_start_time

            model_path, metadata_path = save_trained_model(model, best_params, pop_name)
            best_params["saved_model_path"] = model_path
            best_params["metadata_path"] = metadata_path
            best_params["pop_name"] = pop_name
            return model, best_params, training_elapsed_time

        elif best_params["forecasting_model"] == "ESN":
            n_reservoir = best_params["n_reservoir"]
            sparsity = best_params["sparsity"]
            spectral_radius = best_params["spectral_radius"]
            noise = best_params["noise"]
            look_back = best_params["look_back"]

            trainX, trainY = create_multistep_dataset(denoised_variable_dataset, look_back, 1)
            trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1]))

            model = ESN(
                n_inputs=look_back,
                n_outputs=1,
                n_reservoir=n_reservoir,
                sparsity=sparsity,
                random_state=1234,
                spectral_radius=spectral_radius,
                noise=noise,
                teacher_scaling=10
            )

            training_start_time = time.time()
            model.fit(trainX, trainY)
            training_elapsed_time = time.time() - training_start_time

            model_path, metadata_path = save_trained_model(model, best_params, pop_name)
            best_params["saved_model_path"] = model_path
            best_params["metadata_path"] = metadata_path
            best_params["pop_name"] = pop_name
            return model, best_params, training_elapsed_time

    raise ValueError("Unable to train the selected model configuration.")

# import os
# import json
# import pickle
# import time
# import numpy as np
# import pandas as pd
# import tensorflow as tf
# from tensorflow.keras.models import Sequential, load_model
# from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU
# from sklearn.metrics import mean_squared_error, mean_absolute_error
# from statsmodels.tsa.statespace.sarimax import SARIMAX
# import keras
# import optuna
# from optuna.samplers import TPESampler
# from keras.callbacks import EarlyStopping

# from .ASAP import *
# from .esn import *

# MODEL_DIR = "saved_models"


# def ensure_model_dir():
#     os.makedirs(MODEL_DIR, exist_ok=True)


# def get_model_filename(model_name, pop_name):
#     if model_name in ["RNN", "LSTM", "GRU"]:
#         return os.path.join(MODEL_DIR, f"{model_name}_{pop_name}.keras")
#     elif model_name == "ESN":
#         return os.path.join(MODEL_DIR, f"{model_name}_{pop_name}.pkl")
#     elif model_name == "ARIMA":
#         return os.path.join(MODEL_DIR, f"{model_name}_{pop_name}_config.json")
#     else:
#         return os.path.join(MODEL_DIR, f"{model_name}_{pop_name}.pkl")


# def get_metadata_filename(model_name, pop_name):
#     return os.path.join(MODEL_DIR, f"{model_name}_{pop_name}_metadata.json")


# def save_metadata(best_params, pop_name, model_path):
#     ensure_model_dir()
#     model_name = best_params["forecasting_model"]
#     metadata_path = get_metadata_filename(model_name, pop_name)

#     metadata = dict(best_params)
#     metadata["pop_name"] = pop_name
#     metadata["saved_model_path"] = model_path

#     with open(metadata_path, "w", encoding="utf-8") as f:
#         json.dump(metadata, f, indent=2)

#     return metadata_path


# def save_trained_model(model, best_params, pop_name):
#     ensure_model_dir()

#     model_name = best_params["forecasting_model"]
#     model_path = get_model_filename(model_name, pop_name)

#     if model_name in ["RNN", "LSTM", "GRU"]:
#         model.save(model_path)

#     elif model_name == "ESN":
#         with open(model_path, "wb") as f:
#             pickle.dump(model, f)

#     elif model_name == "ARIMA":
#         with open(model_path, "w", encoding="utf-8") as f:
#             json.dump(best_params, f, indent=2)

#     else:
#         with open(model_path, "wb") as f:
#             pickle.dump(model, f)

#     metadata_path = save_metadata(best_params, pop_name, model_path)
#     return model_path, metadata_path


# def load_saved_model(algo_name, pop_name):
#     ensure_model_dir()

#     model_path = get_model_filename(algo_name, pop_name)
#     metadata_path = get_metadata_filename(algo_name, pop_name)

#     if not os.path.exists(model_path):
#         raise FileNotFoundError(f"Model file not found: {model_path}")

#     if algo_name != "ARIMA" and not os.path.exists(metadata_path):
#         raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

#     if algo_name in ["RNN", "LSTM", "GRU"]:
#         model = load_model(model_path)
#         with open(metadata_path, "r", encoding="utf-8") as f:
#             hyperparameters = json.load(f)
#         return model, hyperparameters

#     elif algo_name == "ESN":
#         with open(model_path, "rb") as f:
#             model = pickle.load(f)
#         with open(metadata_path, "r", encoding="utf-8") as f:
#             hyperparameters = json.load(f)
#         return model, hyperparameters

#     elif algo_name == "ARIMA":
#         with open(model_path, "r", encoding="utf-8") as f:
#             hyperparameters = json.load(f)
#         hyperparameters["saved_model_path"] = model_path
#         hyperparameters["pop_name"] = pop_name
#         return None, hyperparameters

#     else:
#         raise ValueError(f"Unsupported model type: {algo_name}")


# def list_available_models_for_pop(pop_name):
#     ensure_model_dir()
#     available = {}

#     for algo_name in ["RNN", "LSTM", "GRU", "ESN", "ARIMA"]:
#         model_path = get_model_filename(algo_name, pop_name)
#         if os.path.exists(model_path):
#             available[algo_name] = model_path

#     return available


# def sarima_multistep_forecast(history, config, window_size, n_steps):
#     order, sorder, trend = config
#     new_hist = history[:]
#     yhat = []
#     total_train_time = 0
#     total_prediction_time = 0

#     for _ in range(n_steps):
#         model = SARIMAX(
#             new_hist[-window_size:],
#             order=order,
#             seasonal_order=sorder,
#             trend=trend,
#             enforce_stationarity=False,
#             enforce_invertibility=False
#         )

#         training_start_time = time.time()
#         model_fit = model.fit(disp=False)
#         training_end_time = time.time()
#         total_train_time += (training_end_time - training_start_time)

#         predictions_start_time = time.time()
#         prediction = model_fit.predict(
#             start=len(history[-window_size:]),
#             end=len(history[-window_size:])
#         )
#         predictions_end_time = time.time()
#         total_prediction_time += (predictions_end_time - predictions_start_time)

#         yhat = np.append(yhat, prediction)
#         new_hist = np.append(new_hist, prediction)
#         new_hist = new_hist[1:]

#     return yhat, total_train_time, total_prediction_time


# def create_multistep_dataset(data, n_input, n_out=1):
#     X, y = list(), list()
#     in_start = 0

#     for _ in range(len(data)):
#         in_end = in_start + n_input
#         out_end = in_end + n_out
#         if out_end <= len(data):
#             x_input = data[in_start:in_end]
#             X.append(x_input)
#             y.append(data[in_end:out_end])
#         in_start += 1

#     return np.array(X), np.array(y)


# def split_train_test_all_data(data_array, train_ratio=0.8):
#     """
#     Découpage sur l'ensemble des données :
#     - 80% train
#     - 20% test
#     """
#     n = len(data_array)
#     if n < 2:
#         raise ValueError("Not enough data to split train/test.")

#     train_size = int(n * train_ratio)

#     # garde au moins 1 point en train et 1 en test
#     train_size = max(1, min(train_size, n - 1))

#     train = data_array[:train_size]
#     test = data_array[train_size:]
#     return train, test


# def find_best_model(data, selected_columns, horizon, TRIALS, pop_name="unknown_pop"):
#     if horizon == 1 and len(selected_columns) == 1:

#         def objective(trial):
#             forecasting_model = trial.suggest_categorical(
#                 "forecasting_model",
#                 ["RNN", "LSTM", "GRU", "ESN", "ARIMA"]
#             )

#             variable = data[[selected_columns[0]]]
#             variable_dataset = variable.values
#             window_size, _ = smooth_ASAP(variable_dataset, resolution=50)
#             denoised_variable_dataset = moving_average(variable_dataset, window_size)

#             train, test = split_train_test_all_data(denoised_variable_dataset, train_ratio=0.8)

#             if forecasting_model in ["LSTM", "GRU", "RNN"]:
#                 num_hidden_layers = trial.suggest_int("num_hidden_layers", 1, 5)
#                 look_back = trial.suggest_int("look_back", 10, 150)
#                 learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
#                 batch_size = trial.suggest_int("batch_size", 10, 150)
#                 epochs = 3

#                 trainX, trainY = create_multistep_dataset(train, look_back, 1)
#                 validX, validY = create_multistep_dataset(test, look_back, 1)

#                 trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
#                 validX = np.reshape(validX, (validX.shape[0], 1, validX.shape[1]))

#                 model = Sequential()
#                 for i in range(num_hidden_layers):
#                     num_units = trial.suggest_int(f"units_layer_{i}", 8, 256, log=True)
#                     return_sequences = i < num_hidden_layers - 1

#                     if forecasting_model == "RNN":
#                         model.add(SimpleRNN(units=num_units, return_sequences=return_sequences))
#                     elif forecasting_model == "LSTM":
#                         model.add(LSTM(units=num_units, return_sequences=return_sequences))
#                     elif forecasting_model == "GRU":
#                         model.add(GRU(units=num_units, return_sequences=return_sequences))

#                 model.add(Dense(1))
#                 optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
#                 model.compile(loss="mean_squared_error", optimizer=optimizer)
#                 model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size, verbose=0)
#                 valid_predict = model.predict(validX, verbose=0)

#                 return np.sqrt(mean_squared_error(validY, valid_predict))

#             elif forecasting_model == "ESN":
#                 n_reservoir = trial.suggest_int("n_reservoir", 10, 1000)
#                 sparsity = trial.suggest_categorical("sparsity", [0.01, 0.1, 0.2, 0.3, 0.4, 0.5])
#                 spectral_radius = trial.suggest_categorical(
#                     "spectral_radius",
#                     [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.25, 10.0]
#                 )
#                 noise = trial.suggest_categorical(
#                     "noise",
#                     [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009]
#                 )
#                 look_back = trial.suggest_int("look_back", 10, 150)

#                 trainX, trainY = create_multistep_dataset(train, look_back, 1)
#                 validX, validY = create_multistep_dataset(test, look_back, 1)

#                 trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1]))
#                 validX = np.reshape(validX, (validX.shape[0], validX.shape[1]))

#                 model = ESN(
#                     n_inputs=look_back,
#                     n_outputs=1,
#                     n_reservoir=n_reservoir,
#                     sparsity=sparsity,
#                     random_state=1234,
#                     spectral_radius=spectral_radius,
#                     noise=noise,
#                     teacher_scaling=10
#                 )

#                 model.fit(trainX, trainY)
#                 predictions = np.array(model.predict(validX))
#                 return np.sqrt(mean_squared_error(predictions, validY))

#             else:  # ARIMA
#                 p = trial.suggest_int("p", 0, 3)
#                 d = trial.suggest_int("d", 0, 2)
#                 q = trial.suggest_int("q", 0, 3)
#                 window_length = trial.suggest_int("window_length", 5, 150)

#                 order = (p, d, q)
#                 seasonal_order = (0, 0, 0, 0)
#                 cfg = (order, seasonal_order, "c")

#                 predictions = []
#                 ys = []
#                 history = []
#                 history.extend(train)

#                 yhat, _, _ = sarima_multistep_forecast(np.array(history), cfg, window_length, horizon)
#                 predictions.append(yhat)
#                 ys.append(test[:horizon])
#                 history.extend(test[:horizon])

#                 for i in range(horizon, len(test)):
#                     yhat, _, _ = sarima_multistep_forecast(np.array(history), cfg, window_length, horizon)
#                     predictions.append(yhat)
#                     ys.append(test[i:i + horizon])
#                     history.append(test[i])

#                 ys_converted = [array.tolist() for array in ys if len(array) == horizon]
#                 predictions_converted = [array.tolist() for array in predictions]

#                 return np.sqrt(mean_squared_error(ys_converted, predictions_converted[:len(ys_converted)]))

#         study = optuna.create_study(direction="minimize", sampler=TPESampler())
#         study.optimize(objective, n_trials=TRIALS, n_jobs=-1)
#         best_params = study.best_params

#     elif horizon > 1 and len(selected_columns) == 1:

#         def objective(trial):
#             forecasting_model = trial.suggest_categorical(
#                 "forecasting_model",
#                 # ["RNN", "LSTM", "GRU", "ESN", "ARIMA"]
#                 ["ESN"]
#             )

#             variable = data[[selected_columns[0]]]
#             variable_dataset = variable.values
#             window_size, _ = smooth_ASAP(variable_dataset, resolution=50)
#             denoised_variable_dataset = moving_average(variable_dataset, window_size)
            
#             half_size = len(denoised_variable_dataset) // 2  # prendre la moitié
#             train_size = int(0.8 * half_size)                 # 80% pour l'entraînement
#             test_size = half_size - train_size                 # 20% pour le test
#             subset = denoised_variable_dataset[-half_size:]    # utiliser la moitié la plus récente
#             train = subset[:train_size]
#             test = subset[train_size:]

#             # train, test = split_train_test_all_data(denoised_variable_dataset, train_ratio=0.8)

#             # train, test = denoised_variable_dataset[-1000:-200], denoised_variable_dataset[-200:]


#             if forecasting_model in ["RNN", "LSTM", "GRU"]:
#                 look_back = trial.suggest_int("look_back", 10, 150)
#                 num_hidden_layers = trial.suggest_int("num_hidden_layers", 1, 10)
#                 learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
#                 batch_size = trial.suggest_int("batch_size", 10, 150)
#                 epochs = 3

#                 trainX, trainY = create_multistep_dataset(train, look_back, horizon)
#                 validX, validY = create_multistep_dataset(test, look_back, horizon)

#                 trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
#                 validX = np.reshape(validX, (validX.shape[0], 1, validX.shape[1]))

#                 model = Sequential()
#                 for i in range(num_hidden_layers):
#                     num_units = trial.suggest_int(f"units_layer_{i}", 8, 256, log=True)
#                     return_sequences = i < num_hidden_layers - 1

#                     if forecasting_model == "RNN":
#                         model.add(SimpleRNN(units=num_units, return_sequences=return_sequences))
#                     elif forecasting_model == "LSTM":
#                         model.add(LSTM(units=num_units, return_sequences=return_sequences))
#                     elif forecasting_model == "GRU":
#                         model.add(GRU(units=num_units, return_sequences=return_sequences))

#                 model.add(Dense(horizon))
#                 optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
#                 model.compile(loss="mean_squared_error", optimizer=optimizer)
#                 model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size, verbose=0)
#                 valid_predict = model.predict(validX, verbose=0)

#                 return np.sqrt(mean_squared_error(validY, valid_predict))

#             elif forecasting_model == "ESN":
#                 def esn_recursive_strategy(model, X_row, n_steps):
#                     forecasts = []
#                     for _ in range(n_steps):
#                         forecast = model.predict(np.array([X_row]))
#                         forecasts.append(forecast[0, 0])
#                         X_row = X_row.tolist()
#                         X_row.append(forecast[0, 0])
#                         X_row = X_row[1:]
#                         X_row = np.array(X_row)
#                     return forecasts

#                 def esn_make_predictions(model, X, n_steps):
#                     predictions = []
#                     for i in range(len(X)):
#                         row_forecasts = esn_recursive_strategy(model, X[i, :], n_steps)
#                         predictions.append(row_forecasts)
#                     return predictions

#                 n_reservoir = trial.suggest_int("n_reservoir", 10, 1000)
#                 sparsity = trial.suggest_categorical("sparsity", [0.01, 0.1, 0.2, 0.3, 0.4, 0.5])
#                 spectral_radius = trial.suggest_categorical(
#                     "spectral_radius",
#                     [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.25, 10.0]
#                 )
#                 noise = trial.suggest_categorical(
#                     "noise",
#                     [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009]
#                 )
#                 look_back = trial.suggest_int("look_back", 10, 150)

#                 trainX, trainY = create_multistep_dataset(train, look_back, 1)
#                 testX, _ = create_multistep_dataset(test, look_back, 1)

#                 trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1]))
#                 testX = np.reshape(testX, (testX.shape[0], testX.shape[1]))

#                 model = ESN(
#                     n_inputs=look_back,
#                     n_outputs=1,
#                     n_reservoir=n_reservoir,
#                     sparsity=sparsity,
#                     random_state=1234,
#                     spectral_radius=spectral_radius,
#                     noise=noise,
#                     teacher_scaling=10
#                 )

#                 model.fit(trainX, trainY)
#                 test_predict = np.array(esn_make_predictions(model, testX, horizon))
#                 _, new_testY = create_multistep_dataset(test, look_back, horizon)

#                 return np.sqrt(mean_squared_error(new_testY, test_predict[:len(new_testY), :]))

#             else:  # ARIMA
#                 p = trial.suggest_int("p", 0, 3)
#                 d = trial.suggest_int("d", 0, 2)
#                 q = trial.suggest_int("q", 0, 3)
#                 window_length = trial.suggest_int("window_length", 5, 15)

#                 order = (p, d, q)
#                 seasonal_order = (0, 0, 0, 0)
#                 cfg = (order, seasonal_order, "c")

#                 predictions = []
#                 ys = []
#                 history = []
#                 history.extend(train)

#                 yhat, _, _ = sarima_multistep_forecast(np.array(history), cfg, window_length, horizon)
#                 predictions.append(yhat)
#                 ys.append(test[:horizon])
#                 history.extend(test[:horizon])

#                 for i in range(horizon, len(test)):
#                     yhat, _, _ = sarima_multistep_forecast(np.array(history), cfg, window_length, horizon)
#                     predictions.append(yhat)
#                     ys.append(test[i:i + horizon])
#                     history.append(test[i])

#                 ys_converted = [array.tolist() for array in ys if len(array) == horizon]
#                 predictions_converted = [array.tolist() for array in predictions]

#                 return np.sqrt(mean_squared_error(ys_converted, predictions_converted[:len(ys_converted)]))

#         study = optuna.create_study(direction="minimize", sampler=TPESampler())
#         study.optimize(objective, n_trials=TRIALS, n_jobs=-1)
#         best_params = study.best_params

#     else:
#         raise ValueError("Only univariate forecasting is supported in this version.")

#     if best_params["forecasting_model"] == "ARIMA":
#         model_path, metadata_path = save_trained_model(None, best_params, pop_name)
#         best_params["saved_model_path"] = model_path
#         best_params["metadata_path"] = metadata_path
#         best_params["pop_name"] = pop_name
#         return None, best_params, None

#     variable = data[[selected_columns[0]]]
#     variable_dataset = variable.values
#     window_size, _ = smooth_ASAP(variable_dataset, resolution=50)
#     denoised_variable_dataset = moving_average(variable_dataset, window_size)

#     if horizon == 1 and len(selected_columns) == 1:
#         if best_params["forecasting_model"] in ["RNN", "LSTM", "GRU"]:
#             look_back = best_params["look_back"]
#             num_hidden_layers = best_params["num_hidden_layers"]
#             learning_rate = best_params["learning_rate"]
#             batch_size = best_params["batch_size"]

#             trainX, trainY = create_multistep_dataset(denoised_variable_dataset, look_back, 1)
#             trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))

#             model = Sequential()
#             for i in range(num_hidden_layers):
#                 num_units = best_params[f"units_layer_{i}"]
#                 return_sequences = i < num_hidden_layers - 1
#                 if best_params["forecasting_model"] == "RNN":
#                     model.add(SimpleRNN(units=num_units, return_sequences=return_sequences))
#                 elif best_params["forecasting_model"] == "LSTM":
#                     model.add(LSTM(units=num_units, return_sequences=return_sequences))
#                 elif best_params["forecasting_model"] == "GRU":
#                     model.add(GRU(units=num_units, return_sequences=return_sequences))

#             model.add(Dense(1))
#             optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
#             model.compile(loss="mean_squared_error", optimizer=optimizer)

#             training_start_time = time.time()
#             early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
#             model.fit(
#                 trainX,
#                 trainY,
#                 epochs=100,
#                 batch_size=batch_size,
#                 verbose=0,
#                 callbacks=[early_stopping],
#                 validation_split=0.1
#             )
#             training_elapsed_time = time.time() - training_start_time

#             model_path, metadata_path = save_trained_model(model, best_params, pop_name)
#             best_params["saved_model_path"] = model_path
#             best_params["metadata_path"] = metadata_path
#             best_params["pop_name"] = pop_name
#             return model, best_params, training_elapsed_time

#         elif best_params["forecasting_model"] == "ESN":
#             n_reservoir = best_params["n_reservoir"]
#             sparsity = best_params["sparsity"]
#             spectral_radius = best_params["spectral_radius"]
#             noise = best_params["noise"]
#             look_back = best_params["look_back"]

#             trainX, trainY = create_multistep_dataset(denoised_variable_dataset, look_back, 1)
#             trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1]))

#             model = ESN(
#                 n_inputs=look_back,
#                 n_outputs=1,
#                 n_reservoir=n_reservoir,
#                 sparsity=sparsity,
#                 random_state=1234,
#                 spectral_radius=spectral_radius,
#                 noise=noise,
#                 teacher_scaling=10
#             )

#             training_start_time = time.time()
#             model.fit(trainX, trainY)
#             training_elapsed_time = time.time() - training_start_time

#             model_path, metadata_path = save_trained_model(model, best_params, pop_name)
#             best_params["saved_model_path"] = model_path
#             best_params["metadata_path"] = metadata_path
#             best_params["pop_name"] = pop_name
#             return model, best_params, training_elapsed_time

#     elif horizon > 1 and len(selected_columns) == 1:
#         if best_params["forecasting_model"] in ["RNN", "LSTM", "GRU"]:
#             look_back = best_params["look_back"]
#             num_hidden_layers = best_params["num_hidden_layers"]
#             learning_rate = best_params["learning_rate"]
#             batch_size = best_params["batch_size"]

#             trainX, trainY = create_multistep_dataset(denoised_variable_dataset, look_back, horizon)
#             trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))

#             model = Sequential()
#             for i in range(num_hidden_layers):
#                 num_units = best_params[f"units_layer_{i}"]
#                 return_sequences = i < num_hidden_layers - 1
#                 if best_params["forecasting_model"] == "RNN":
#                     model.add(SimpleRNN(units=num_units, return_sequences=return_sequences))
#                 elif best_params["forecasting_model"] == "LSTM":
#                     model.add(LSTM(units=num_units, return_sequences=return_sequences))
#                 elif best_params["forecasting_model"] == "GRU":
#                     model.add(GRU(units=num_units, return_sequences=return_sequences))

#             model.add(Dense(horizon))
#             optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
#             model.compile(loss="mean_squared_error", optimizer=optimizer)

#             training_start_time = time.time()
#             early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
#             model.fit(
#                 trainX,
#                 trainY,
#                 epochs=100,
#                 batch_size=batch_size,
#                 verbose=0,
#                 callbacks=[early_stopping],
#                 validation_split=0.1
#             )
#             training_elapsed_time = time.time() - training_start_time

#             model_path, metadata_path = save_trained_model(model, best_params, pop_name)
#             best_params["saved_model_path"] = model_path
#             best_params["metadata_path"] = metadata_path
#             best_params["pop_name"] = pop_name
#             return model, best_params, training_elapsed_time

#         elif best_params["forecasting_model"] == "ESN":
#             n_reservoir = best_params["n_reservoir"]
#             sparsity = best_params["sparsity"]
#             spectral_radius = best_params["spectral_radius"]
#             noise = best_params["noise"]
#             look_back = best_params["look_back"]

#             trainX, trainY = create_multistep_dataset(denoised_variable_dataset, look_back, 1)
#             trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1]))

#             model = ESN(
#                 n_inputs=look_back,
#                 n_outputs=1,
#                 n_reservoir=n_reservoir,
#                 sparsity=sparsity,
#                 random_state=1234,
#                 spectral_radius=spectral_radius,
#                 noise=noise,
#                 teacher_scaling=10
#             )

#             training_start_time = time.time()
#             model.fit(trainX, trainY)
#             training_elapsed_time = time.time() - training_start_time

#             model_path, metadata_path = save_trained_model(model, best_params, pop_name)
#             best_params["saved_model_path"] = model_path
#             best_params["metadata_path"] = metadata_path
#             best_params["pop_name"] = pop_name
#             return model, best_params, training_elapsed_time

#     raise ValueError("Unable to train the selected model configuration.")