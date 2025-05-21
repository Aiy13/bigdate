import pandas as pd


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


def processing_feature():
    # 读取数据
    data = inputdata("./data/train.csv")
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
    data = transcolname(data, column_mapping)
    data.drop(columns=["PriceChangePercentage"], inplace=True)
    data = trans_datetime(data)

    return data


feature = processing_feature()

outputdata("./temp/feature.csv", feature)
