import pandas as pd
import pickle
import torch


def inputdata(path):
    data = pd.read_csv(path, header=0, sep=",", encoding="utf-8")
    return data


feature = inputdata(r"C:\Users\27535\Desktop\大数据挑战赛\temp\feature.csv")


def process_data(npdf, stp=32):
    ret = []
    for i in range(npdf.shape[0] - stp):
        train_seq = npdf[i : i + stp]
        train_label = npdf[i + stp]
        train_seq = torch.FloatTensor(train_seq)
        train_label = torch.FloatTensor(train_label).view(-1)
        ret.append((train_seq, train_label))

    return ret


column_names = feature.columns.tolist()
stockcodes = feature["StockCode"].drop_duplicates().tolist()

train_data = []
for stockcode in stockcodes:
    stock_data = feature[feature["StockCode"] == stockcode]
    max_date = stock_data["Date"].max()
    min_date = stock_data["Date"].min()
    stock_data = stock_data.values
    if len(stock_data) < 32:
        continue
    train_data += process_data(stock_data, stp=32)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# 模型参数
input_size = 1
hidden_size = 500
num_layers = 4
output_size = 1
dropout = 0.2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 定义LSTM模型
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True, dropout=dropout
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def __del__(self):
        del self.hidden_cell

    def forward(self, x):
        out, self.hidden_cell = self.lstm(x, self.hidden_cell)
        out = self.fc(out[:, -1, :])
        return out


def train_model(train_data, i, num_epochs=50):
    if len(train_data) == 0:
        return LSTMModel(1, hidden_size, num_layers, output_size, dropout).to(device)

    train_data = [(x.to(device), y.to(device)) for x, y in train_data]

    # 调整每个输入序列的维度为3D
    X_train_tensor = torch.stack([x for x, _ in train_data])
    y_train_tensor = torch.stack([y[i] for _, y in train_data])

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    model = LSTMModel(
        len(train_data[0][0][0]), hidden_size, num_layers, output_size, dropout
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.7)

    for epoch in range(num_epochs):
        # print(epoch)
        tot_loss = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            model.hidden_cell = (
                torch.zeros(model.num_layers, batch_X.size(0), model.hidden_size).to(
                    device
                ),
                torch.zeros(model.num_layers, batch_X.size(0), model.hidden_size).to(
                    device
                ),
            )
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y.unsqueeze(1))
            tot_loss += loss.item()
            loss.backward()
            optimizer.step()
        scheduler.step()
        print("Epoch [{}/{}], Loss: {:.4f}".format(epoch + 1, num_epochs, tot_loss/len(train_loader)))
    return model


colname2index = {x: i for i, x in enumerate(column_names)}
model_i = train_model(train_data, colname2index["Close"] + 2, num_epochs=5)
model_name = r"C:\Users\27535\Desktop\大数据挑战赛\model\model_Close.bin"
pickle.dump(model_i, open(model_name, "wb"))
