import pandas as pd
import numpy as np
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

def inputdata(path):
    try:
        return pd.read_csv(path, header=0, sep=",", encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(path, header=0, sep=",", encoding="gbk")

def transcolname(df, column_mapping):
    return df.rename(columns=column_mapping)

def trans_datetime(df):
    df['Date_str'] = df['Date']
    dt = pd.to_datetime(df['Date_str'], format='%Y-%m-%d')

    df['year'] = dt.dt.year
    df['month'] = dt.dt.month
    df['day'] = dt.dt.day
    df['weekday'] = dt.dt.weekday

    unique_dates = pd.Series(dt.dt.strftime('%Y-%m-%d')).sort_values().unique()
    date_mapping = {date: i + 1 for i, date in enumerate(unique_dates)}
    df['Date'] = dt.dt.strftime('%Y-%m-%d').map(date_mapping)

    df.drop(columns=['Date_str'], inplace=True)
    return df

def augment_features(df):
    df = df.sort_values(['StockCode', 'Date'])

    # 滞后收盘价
    df['PrevClose'] = df.groupby('StockCode')['Close'].shift(1)
    # 日收益率
    df['Return'] = df.groupby('StockCode')['Close'].pct_change()
    # 高低价差、开收差
    df['HighLowDiff'] = df['High'] - df['Low']
    df['OpenCloseDiff'] = df['Open'] - df['Close']
    # 成交量变化率
    df['VolumePct'] = df.groupby('StockCode')['Volume'].pct_change()

    # 滑动均值与标准差
    for w in (5, 10, 20):
        df[f'MA_{w}'] = df.groupby('StockCode')['Close'].transform(lambda x: x.rolling(w, min_periods=w).mean())
        df[f'STD_{w}'] = df.groupby('StockCode')['Close'].transform(lambda x: x.rolling(w, min_periods=w).std())
        # 添加相对均线的偏离度
        df[f'MA_{w}_Dev'] = (df['Close'] - df[f'MA_{w}']) / df[f'MA_{w}']
        # 添加成交量相对均量
        df[f'Volume_MA_{w}'] = df.groupby('StockCode')['Volume'].transform(lambda x: x.rolling(w, min_periods=w).mean())
        df[f'Volume_MA_{w}_Dev'] = (df['Volume'] - df[f'Volume_MA_{w}']) / df[f'Volume_MA_{w}']

    # 计算每日收益率排名
    df['ReturnRank'] = df.groupby('Date')['Return'].transform(lambda x: pd.qcut(x, q=10, labels=False, duplicates='drop'))
    
    return df.reset_index(drop=True)

def filter_stocks(df, lookback=32):
    """统一处理股票数据长度"""
    valid_stocks = []
    for stock_code in df['StockCode'].unique():
        stock_data = df[df['StockCode'] == stock_code]
        if len(stock_data) > lookback:
            valid_stocks.append(stock_code)
    return df[df['StockCode'].isin(valid_stocks)]

def prepare_features(df, target_col='ReturnRank', lookback=32, is_training=True):
    """准备特征数据，包括目标变量"""
    features = []
    targets = []
    
    for stock_code in df['StockCode'].unique():
        stock_data = df[df['StockCode'] == stock_code].sort_values('Date')
        if len(stock_data) <= lookback:
            continue
            
        # 准备特征
        feature_cols = [col for col in stock_data.columns if col not in ['StockCode', 'Date', target_col, 'Return']]
        X = stock_data[feature_cols].values
        y = stock_data[target_col].values
        
        if is_training:
            # 训练时使用滑动窗口
            for i in range(len(stock_data) - lookback):
                features.append(X[i:i+lookback])
                targets.append(y[i+lookback])
        else:
            # 预测时只使用最后lookback天的数据
            features.append(X[-lookback:])
            targets.append(y[-1])
            
    return np.array(features), np.array(targets)

def train_lgb_model(train_data_path, model_save_path='model/lgb_model.txt'):
    """训练LightGBM模型"""
    # 1. 数据预处理
    data = inputdata(train_data_path)
    data = transcolname(data, {
        "股票代码": "StockCode", "日期": "Date", "开盘": "Open", "收盘": "Close",
        "最高": "High", "最低": "Low", "成交量": "Volume", "成交额": "Turnover",
        "振幅": "Amplitude", "涨跌额": "PriceChange", "换手率": "TurnoverRate",
        "涨跌幅": "PriceChangePercentage",
    })
    data.drop(columns=["PriceChangePercentage"], inplace=True)
    
    # 2. 特征工程
    data = trans_datetime(data)
    data = augment_features(data)
    data = filter_stocks(data)  # 过滤数据长度不足的股票
    data.replace([float('inf'), -float('inf')], 0, inplace=True)
    data.fillna(0, inplace=True)
    
    # 3. 准备训练数据
    X, y = prepare_features(data, target_col='ReturnRank', is_training=True)
    
    # 4. 设置LightGBM参数
    params = {
        'objective': 'regression',
        'metric': 'mse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1
    }
    
    # 5. 创建数据集
    train_data = lgb.Dataset(X.reshape(X.shape[0], -1), label=y)
    
    # 6. 设置callbacks
    callbacks = [
        lgb.early_stopping(stopping_rounds=300),
        lgb.log_evaluation(period=100)
    ]
    
    # 7. 训练模型
    print("开始训练LightGBM模型...")
    model = lgb.train(
        params,
        train_data,
        num_boost_round=10000,
        valid_sets=[train_data],
        callbacks=callbacks
    )
    
    # 8. 保存模型
    model.save_model(model_save_path)
    print(f"模型已保存到 {model_save_path}")
    
    return model

def predict_lgb_model(test_data_path, model_path='model/lgb_model.txt'):
    """使用训练好的模型进行预测"""
    # 1. 加载模型
    model = lgb.Booster(model_file=model_path)
    
    # 2. 处理测试数据
    test_data = inputdata(test_data_path)
    test_data = transcolname(test_data, {
        "股票代码": "StockCode", "日期": "Date", "开盘": "Open", "收盘": "Close",
        "最高": "High", "最低": "Low", "成交量": "Volume", "成交额": "Turnover",
        "振幅": "Amplitude", "涨跌额": "PriceChange", "换手率": "TurnoverRate",
        "涨跌幅": "PriceChangePercentage",
    })
    test_data.drop(columns=["PriceChangePercentage"], inplace=True)
    
    # 3. 特征工程
    test_data = trans_datetime(test_data)
    test_data = augment_features(test_data)
    test_data = filter_stocks(test_data)  # 过滤数据长度不足的股票
    test_data.replace([float('inf'), -float('inf')], 0, inplace=True)
    test_data.fillna(0, inplace=True)
    
    # 4. 准备预测数据
    X, _ = prepare_features(test_data, target_col='ReturnRank', is_training=False)
    X = X.reshape(X.shape[0], -1)
    
    # 5. 预测
    preds = model.predict(X)
    
    # 6. 获取预测结果
    results = []
    for i, stock_code in enumerate(test_data['StockCode'].unique()):
        stock_data = test_data[test_data['StockCode'] == stock_code].sort_values('Date')
        if len(stock_data) < 32:
            continue
        results.append((stock_code, preds[i]))
    
    # 7. 根据预测的排名选择股票
    results = sorted(results, key=lambda x: x[1], reverse=True)
    top10_up = [r[0] for r in results[:10]]  # 预测排名最高的10只股票
    top10_down = [r[0] for r in results[-10:]]  # 预测排名最低的10只股票
    
    # 8. 保存结果
    out = pd.DataFrame({
        "涨幅最大股票代码": top10_up,
        "涨幅最小股票代码": top10_down
    })
    print(out)
    out.to_csv("output/result_lgb.csv", index=False, encoding="utf-8")
    print("预测完毕，结果保存在 output/result_lgb.csv")

def evaluate_predictions(predictions_df, check_data_path):
    """评估预测结果"""
    check_data = inputdata(check_data_path)
    
    # 获取预测的股票代码
    pred_up_stocks = set(predictions_df['涨幅最大股票代码'].values)
    pred_down_stocks = set(predictions_df['涨幅最小股票代码'].values)
    
    # 获取实际涨幅最大的股票
    actual_up_stocks = set(check_data['涨幅最大股票代码'].head(10).values)
    actual_down_stocks = set(check_data['涨幅最小股票代码'].tail(10).values)
    
    # 计算命中率
    up_hit = len(pred_up_stocks & actual_up_stocks)
    down_hit = len(pred_down_stocks & actual_down_stocks)
    
    print("\n预测结果评估:")
    print(f"涨幅最大股票命中数: {up_hit}/10")
    print(f"涨幅最小股票命中数: {down_hit}/10")
    print(f"总命中率: {(up_hit + down_hit)/20:.2%}")
    
    # 详细评估每只预测股票
    print("\n涨幅最大股票详细评估:")
    for stock in predictions_df['涨幅最大股票代码']:
        actual_rank = check_data[check_data['涨幅最大股票代码'] == stock].index[0] + 1 if stock in check_data['涨幅最大股票代码'].values else "未上榜"
        print(f"股票 {stock}: 实际排名 {actual_rank}")
    
    print("\n涨幅最小股票详细评估:")
    for stock in predictions_df['涨幅最小股票代码']:
        actual_rank = check_data[check_data['涨幅最小股票代码'] == stock].index[0] + 1 if stock in check_data['涨幅最小股票代码'].values else "未上榜"
        print(f"股票 {stock}: 实际排名 {actual_rank}")

def save_evaluation_results(predictions_df, check_data_path, output_path="output/evaluation_results.csv"):
    """保存评估结果到CSV文件"""
    check_data = inputdata(check_data_path)
    
    # 准备评估结果数据
    evaluation_results = []
    
    # 评估涨幅最大的股票
    for stock in predictions_df['涨幅最大股票代码']:
        actual_rank = check_data[check_data['涨幅最大股票代码'] == stock].index[0] + 1 if stock in check_data['涨幅最大股票代码'].head(10).values else "未上榜"
        evaluation_results.append({
            "股票代码": stock,
            "预测类型": "涨幅最大",
            "实际排名": actual_rank,
            "是否命中": "是" if stock in check_data['涨幅最大股票代码'].head(10).values else "否"
        })
    
    # 评估涨幅最小的股票
    for stock in predictions_df['涨幅最小股票代码']:
        actual_rank = check_data[check_data['涨幅最小股票代码'] == stock].index[0] + 1 if stock in check_data['涨幅最小股票代码'].values else "未上榜"
        evaluation_results.append({
            "股票代码": stock,
            "预测类型": "涨幅最小",
            "实际排名": actual_rank,
            "是否命中": "是" if stock in check_data['涨幅最小股票代码'].tail(10).values else "否"
        })
    
    # 保存结果
    results_df = pd.DataFrame(evaluation_results)
    results_df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"\n评估结果已保存到 {output_path}")

if __name__ == "__main__":
    # 训练模型
    model = train_lgb_model("data/train.csv")
    
    # 预测
    predict_lgb_model("data/test.csv")
    
    # 评估预测结果
    predictions = pd.read_csv("output/result_lgb.csv")
    evaluate_predictions(predictions, "data/check.csv")
    
    # 保存评估结果
    save_evaluation_results(predictions, "data/check.csv") 