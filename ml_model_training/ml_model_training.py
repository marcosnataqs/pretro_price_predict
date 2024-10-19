import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt

# %%

df = pd.read_csv("petro.csv", index_col=0, parse_dates=True)
df.sort_index(inplace=True)

# %%

X = df.iloc[:, 2:]
y = df.iloc[:, :1]

print(X)
print(y)
# %%
# Scaling Features and outputs

mm_sca = MinMaxScaler()
ss_sca = StandardScaler()

X_ss = ss_sca.fit_transform(X)
y_mm = mm_sca.fit_transform(y)

# %%
# Spliting data

train_size = int(len(df) * 0.8)
train, test = df.iloc[:train_size], df.iloc[train_size:]

X_train = X_ss[:train_size, :]
X_test = X_ss[train_size:, :]

y_train = y_mm[:train_size, :]
y_test = y_mm[train_size:, :]

# %%
# Convert numpy arrays to Tensors

X_train_tensors = Variable(torch.Tensor(X_train))
X_test_tensors = Variable(torch.Tensor(X_test))

y_train_tensors = Variable(torch.Tensor(y_train))
y_test_tensors = Variable(torch.Tensor(y_test))

# %%
# Reshaping to have timestamp informations

X_train_tensors_final = torch.reshape(
    X_train_tensors, (X_train_tensors.shape[0], 1, X_train_tensors.shape[1])
)

X_test_tensors_final = torch.reshape(
    X_test_tensors, (X_test_tensors.shape[0], 1, X_test_tensors.shape[1])
)

print("Training Shape", X_train_tensors_final.shape, y_train_tensors.shape)
print("Testing Shape", X_test_tensors_final.shape, y_test_tensors.shape)

# %%                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   # %%
# Creating model


class LSTM(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
        super(LSTM, self).__init__()
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc_1 = nn.Linear(hidden_size, 128)
        self.fc = nn.Linear(128, num_classes)

        self.relu = nn.ReLU()

    def forward(self, x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))

        output, (hn, cn) = self.lstm(x, (h_0, c_0))
        hn = hn.view(-1, self.hidden_size)
        out = self.relu(hn)
        out = self.fc_1(out)
        out = self.relu(out)
        out = self.fc(out)
        return out


# %%
# Set up hyperparameters for train

num_epochs = 9000
learning_rate = 0.001

input_size = 4
hidden_size = 3
num_layers = 1

num_classes = 1

# %%
# Instantiate the model

lstm = LSTM(
    num_classes, input_size, hidden_size, num_layers, X_train_tensors_final.shape[1]
)

# %%
# Define the Loss function and optimizer

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)

# %%
# Training the model

for epoch in range(num_epochs):
    outputs = lstm.forward(X_train_tensors_final)
    optimizer.zero_grad()

    loss = criterion(outputs, y_train_tensors)

    loss.backward()

    optimizer.step()
    if epoch % 100 == 0:
        print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))

# %%

df_X_ss = ss_sca.transform(df.iloc[:, 2:])
df_y_mm = mm_sca.transform(df.iloc[:, :1])

df_X_ss = Variable(torch.Tensor(df_X_ss))
df_y_mm = Variable(torch.Tensor(df_y_mm))

# reshaping the dataset

df_X_ss = torch.reshape(df_X_ss, (df_X_ss.shape[0], 1, df_X_ss.shape[1]))

# %%

train_predict = lstm(df_X_ss)
data_predict = train_predict.data.numpy()
dataY_plot = df_y_mm.data.numpy()

data_predict = mm_sca.inverse_transform(data_predict)
dataY_plot = mm_sca.inverse_transform(dataY_plot)
plt.figure(figsize=(10, 6))
plt.axvline(x=X_train.shape[0], c="r", linestyle="--")

plt.plot(dataY_plot, label="Actual Data")
plt.plot(data_predict, label="Predicted Data")
plt.title("Time-Series Prediction")
plt.legend()
plt.show()
