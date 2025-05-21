import pandas as pd
import numpy as np
import pickle
import torch

pred_len = 32


def inputdata(path):
    data = pd.read_csv(path, header=0, sep=",", encoding="utf-8")
    return data


def outputdata(path, data, is_index=False):
    data.to_csv(path, index=is_index, header=True, sep=",", mode="w", encoding="utf-8")


def transcolname(df, column_mapping):
    df.rename(columns=column_mapping, inplace=True)
    return df


def trans_datetime(df):
    ret_df = pd.DataFrame()
    dt = df["Date"]
    ret_df["year"] = dt.transform(lambda x: int(x.split("-")[0]))
    ret_df["month"] = dt.transform(lambda x: int(x.split("-")[1]))
    ret_df["day"] = dt.transform(lambda x: int(x.split("-")[2][:2]))
    df = pd.concat([df, ret_df], axis=1)
    unique_dates = pd.Series(df["Date"].unique()).sort_values().reset_index(drop=True)
    date_mapping = {date: rank + 1 for rank, date in enumerate(unique_dates)}
    df["Date"] = df["Date"].map(date_mapping)
    # df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d")
    # minTime = df["Date"].min()
    # df["Date"] = ((df["Date"] - minTime) / pd.Timedelta(days=1)).astype(int)
    return df


def processing_feature_test():
    # 读取数据
    # data1 = inputdata("./data/train.csv")
    data = inputdata(r"C:\Users\27535\Desktop\大数据挑战赛\data\test.csv")
    column_mapping = {
        "股票代码": "StockCode",
        "日期": "Date",
        "开盘": "Open",
        "收盘": "Close",
        "最高": "High",
        "最低": "Low",
        "成交量": "Volume",
        "成交额": "Turnover",
        "振幅": "Amplitude",
        "涨跌额": "PriceChange",
        "换手率": "TurnoverRate",
        "涨跌幅": "PriceChangePercentage",
    }
    # data = pd.concat([data1, data2], axis=0)
    data = transcolname(data, column_mapping)
    data.drop(columns=["PriceChangePercentage"], inplace=True)
    # stockcodes = data["StockCode"].drop_duplicates().tolist()
    data = trans_datetime(data)
    max_date = data["Date"].max()
    data = data[data["Date"] > max_date - pred_len]

    return data


data = processing_feature_test()
max_date = data["Date"].max()


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


PREDICT_COLS = [
    "Open",
    "Close",
    "High",
    "Low",
    "Volume",
    "Turnover",
    "Amplitude",
    "PriceChange",
    "TurnoverRate",
]
column_names = data.columns.tolist()
colname2index = {x: i for i, x in enumerate(column_names)}
stockcodes = data["StockCode"].drop_duplicates().tolist()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
all_preds = []


def process_data_predict(data, stockcode):
    data = data[data["StockCode"] == stockcode]
    data = np.array([data.values])
    return torch.tensor(data, dtype=torch.float32).to(device)


for stockcode in stockcodes:

    temp_preds = {}

    # 预测当前变量
    predict_data = process_data_predict(data, stockcode).to(device)
    model_name = r"C:\Users\27535\Desktop\大数据挑战赛\model\model_Close_lgbm.bin"
    estimator = pickle.load(open(model_name, "rb"))
    estimator.hidden_cell = (
        torch.zeros(
            estimator.num_layers, predict_data.size(0), estimator.hidden_size
        ).to(device),
        torch.zeros(
            estimator.num_layers, predict_data.size(0), estimator.hidden_size
        ).to(device),
    )
    with torch.no_grad():
        predi = estimator(predict_data)
    pred = predi[-1].cpu().detach().numpy()

    all_preds.append((stockcode, pred))


pricechangerate = []
for i in range(len(all_preds)):
    stockcode, pred = all_preds[i]
    preClose = data[(data["StockCode"] == stockcode) & (data["Date"] == max_date)][
        "Close"
    ].values[0]
    pricechangerate.append((stockcode, (pred - preClose) / preClose * 100))
pricechangerate = sorted(pricechangerate, key=lambda x: x[1], reverse=True)
pred_top_10_max_target = [x[0] for x in pricechangerate[:10]]
pred_top_10_min_target = [x[0] for x in pricechangerate[-10:]]
data = {
    "涨幅最大股票代码": pred_top_10_max_target,
    "涨幅最小股票代码": pred_top_10_min_target,
}

df = pd.DataFrame(data)
df.to_csv(r"C:\Users\27535\Desktop\大数据挑战赛\output\result_lgbm.csv", index=False)
