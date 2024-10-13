import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader
from petro_dataset import PetroDataset
from petro_model import PetroModel
from sklearn.model_selection import train_test_split


def train_one_epoch(epoch, train_loader, model, loss_function, optimizer, device):
  model.train(True)
  print(f'Epoch: {epoch+1}')
  running_loss = 0.0
  for batch_index, batch in enumerate(train_loader):
    x_batch, y_batch = batch[0].to(device), batch[1].to(device)

    outputs = model(x_batch)
    loss = loss_function(outputs, y_batch)
    running_loss += loss.item()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if batch_index % 100 == 99:
      print(f'Batch {batch_index + 1}, Loss: {running_loss / 100}')
      running_loss = 0.0
  print()


  ## TO-DO: We need to review this part if we are going to test with batches or always the same data 
def validate_one_epoch(model, test_loader, device, loss_function):
    model.train(False)
    running_loss = 0.0

    for batch_index, batch in enumerate(test_loader):
        x_batch, y_batch = batch[0].to(device), batch[1].to(device)

        with torch.no_grad():
            output = model(x_batch)
            loss = loss_function(output, y_batch)
            running_loss += loss.item()

    avg_loss_accross_batches = running_loss / len(test_loader)
    print(f'Validation Loss: {avg_loss_accross_batches}')
    print('*********************************************')
    print()

def main(num_epochs, train_loader, test_loader, model, loss_function, optimizer, device):
    for epoch in range(num_epochs):
        train_one_epoch(epoch, train_loader, model, loss_function, optimizer, device)
        #validate_one_epoch(epoch, model, test_loader, device, loss_function)

if __name__ == '__main__':
    data = pd.read_csv('ml_model_training\petro.csv', index_col=0, parse_dates=True).sort_index()
    pipeline_params = {
        'num_lags':7,
        'columns': ['pbr', 'usd'],
        'num_features': 5
    }
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train, test = train_test_split(data, shuffle=False, test_size=0.2)
    train_dataset = PetroDataset(train, pipeline_params)
    test_dataset = PetroDataset(test, pipeline_params)
    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    model = PetroModel(1, 4, 1, device).to(device)
    learning_rate = 0.001
    num_epochs = 10
    loss_function = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    main(num_epochs, train_loader, test_loader, model, loss_function, optimizer, device)


    