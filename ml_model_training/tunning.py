import os
import sys
import numpy as np
import neptune
import pandas as pd
import torch.optim as optim
import torch
import torch.nn as nn

from dotenv import load_dotenv
from hyperopt import hp
from hyperopt import fmin, rand, tpe, Trials, STATUS_OK
from neptune.metadata_containers.run import Run
from petro_model import PetroModel
from sklearn.model_selection import train_test_split

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_engineering.data_utils import add_lags, download_data
from model_utils import train_one_epoch, validate_one_epoch, generate_loader

load_dotenv()

neptune_run = neptune.init_run(
    project=os.getenv("NEPTUNE_PROJECT"),
    api_token=os.getenv("NEPTUNE_API_TOKEN"),
)

# Space to search
search_space = {
    'learning_rate': hp.loguniform('learning_rate', -4, -1),
    #'hidden_size': hp.choice('hidden_size', [16, 32, 64]), # maior pode aumentar o risco de overfitting
    'num_stacked_layers': hp.choice('num_stacked_layers', [1]), # quanto maior, mais dificil de treinar e suscetiveis a problemas de gradiente
    #'dropout': hp.uniform('dropout', 0.1, 0.5),
    'batch_size': hp.choice('batch_size', [16, 32, 64, 128]), # menor mais ruido
    'epochs': hp.uniform('epochs', 10, 100)
}

def objective(params):
    
    # Deactivate dropout when there is only one layer
    if params['num_stacked_layers'] == 1:
        params['dropout'] = 0.0
    # Unpack parameters
    learning_rate = params['learning_rate']
    # hidden_size = int(params['hidden_size'])
    num_stacked_layers = int(params['num_stacked_layers'])
    # dropout = params['dropout']
    batch_size = int(params['batch_size'])
    epochs = int(params['epochs'])

    # Prepare Data
    data = download_data().sort_index()
    data = add_lags(data, 7, columns=["pbr"])
    pipeline_params = {"num_features": 7}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train, test = train_test_split(data, shuffle=False, test_size=0.2)
    train_loader = generate_loader(train, pipeline_params, batch_size)
    test_loader = generate_loader(test, pipeline_params, batch_size, shuffle=False)

    # Log dataset information
    neptune_run["data/train_size"] = len(train)
    neptune_run["data/test_size"] = len(test)

    # Model, criterion, optimizer
    model = PetroModel(
        input_size=1, hidden_size=7, num_stacked_layers=num_stacked_layers, 
        device=device).to(device)
    loss_function = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # # Setting early stopping parameters
    # early_stop_patience = 3
    # best_val_loss = float('inf')
    # epochs_without_improvement = 0

    losses_history = []
    for epoch in range(epochs):
        # Evaluate on train set
        train_loss = float(train_one_epoch(
            epoch, train_loader, model, loss_function, optimizer, device, neptune_run
        ))

        # Evaluate on validation set  
        validate_loss = float(validate_one_epoch(
            epoch, model, test_loader, device, loss_function, neptune_run
        ))
        losses_history.append(validate_loss)

        # # Early stopping condition
        # if validate_loss < best_val_loss:
        #     best_val_loss = validate_loss
        #     epochs_without_improvement = 0
        # else:
        #     epochs_without_improvement += 1
        #     if epochs_without_improvement >= early_stop_patience:
        #         print(f"Early stopping starts at epoch {epoch + 1}")
        #         break
    
    return {
        'loss': validate_loss, 
        'params': params,
        'history': losses_history, 
        'best_epoch': epoch + 1,
        'status': STATUS_OK
    }

# Initialize trials object to track optimization history
trials = Trials()

# Run hyperparameter optimization
best = fmin(
    fn=objective,                # Objective function to minimize
    space=search_space,          # Search space for hyperparameters
    algo=tpe.suggest,            # Tree of Parzen Estimators (TPE) algorithm for optimization
    max_evals=10,                # Number of evaluations to run
    trials=trials                # Track results
)

# Stop the Neptune run
neptune_run.stop()
print()
print("Best parameters:", best)

  