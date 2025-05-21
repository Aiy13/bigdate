import pandas as pd
import numpy as np
import lightgbm as lgb
import warnings
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import optuna
import os
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

    return df.reset_index(drop=True)

def filter_stocks(df, lookback=32):
    """统一处理股票数据长度"""
    valid_stocks = []
    for stock_code in df['StockCode'].unique():
        stock_data = df[df['StockCode'] == stock_code]
        if len(stock_data) > lookback:
            valid_stocks.append(stock_code)
    return df[df['StockCode'].isin(valid_stocks)]

def select_features_rf(X, y, feature_names, threshold=0.01):
    """
    使用随机森林进行特征筛选
    
    参数:
    X: 特征矩阵 (样本数, 时间步长, 特征数)
    y: 目标变量
    feature_names: 特征名称列表
    threshold: 特征重要性阈值，低于此阈值的特征将被剔除
    
    返回:
    selected_features: 选中的特征名称列表
    """
    print("开始使用随机森林进行特征筛选...")
    
    # 计算每个特征类型的重要性
    n_samples, n_timesteps, n_features = X.shape
    feature_importance = np.zeros(n_features)
    
    # 对每个时间步长分别计算特征重要性
    for t in range(n_timesteps):
        X_t = X[:, t, :]  # 获取当前时间步长的特征
        rf = RandomForestRegressor(
            n_estimators=60,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        rf.fit(X_t, y)
        feature_importance += rf.feature_importances_
    
    # 计算平均特征重要性
    feature_importance = feature_importance / n_timesteps
    
    # 创建特征重要性DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    })
    
    # 按重要性排序
    importance_df = importance_df.sort_values('importance', ascending=False)
    
    # 选择重要性大于阈值的特征
    selected_features = importance_df[importance_df['importance'] > threshold]['feature'].tolist()
    
    print(f"\n特征重要性排名:")
    print(importance_df)
    print(f"\n原始特征数量: {len(feature_names)}")
    print(f"筛选后特征数量: {len(selected_features)}")
    
    return selected_features

def prepare_features(df, target_col='Return', lookback=32, is_training=True):
    """准备特征数据，包括目标变量"""
    features = []
    targets = []
    feature_names = []
    
    for stock_code in df['StockCode'].unique():
        stock_data = df[df['StockCode'] == stock_code].sort_values('Date')
        if len(stock_data) <= lookback:
            continue
            
        # 准备特征
        feature_cols = [col for col in stock_data.columns if col not in ['StockCode', 'Date', target_col]]
        if not feature_names:  # 只在第一次循环时获取特征名称
            feature_names = feature_cols
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
            
    return np.array(features), np.array(targets), feature_names

def objective(trial, X, y, n_folds=3):
    """Optuna优化目标函数"""
    param = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': trial.suggest_int('num_leaves', 31, 255),  # 增加搜索范围
        'max_depth': trial.suggest_int('max_depth', 3, 12),  # 增加搜索范围
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),  # 扩大学习率范围
        'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),  # 增加随机性
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),  # 增加随机性
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 200),  # 增加搜索范围
        'min_gain_to_split': trial.suggest_float('min_gain_to_split', 0.0, 0.2),  # 增加搜索范围
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 1.0, log=True),  # 修复对数分布下限
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 1.0, log=True),  # 修复对数分布下限
        'drop_rate': trial.suggest_float('drop_rate', 0.0, 0.2),  # 添加dropout
        'top_rate': trial.suggest_float('top_rate', 0.2, 0.8),  # 添加GOSS
        'other_rate': trial.suggest_float('other_rate', 0.1, 0.5),  # 添加GOSS
        'verbose': -1
    }
    
    # 5折交叉验证
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=trial.suggest_int('random_state', 0, 1000))  # 随机种子也作为超参数
    scores = []
    
    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        # 添加学习率衰减
        callbacks = [
            lgb.early_stopping(stopping_rounds=100),
            lgb.log_evaluation(period=50),
            lgb.reset_parameter(
                learning_rate=lambda iter: param['learning_rate'] * (0.99 ** iter)  # 学习率衰减
            )
        ]
        
        model = lgb.train(
            param,
            train_data,
            num_boost_round=2000,
            valid_sets=[val_data],
            callbacks=callbacks
        )
        
        preds = model.predict(X_val)
        score = np.sqrt(mean_squared_error(y_val, preds))
        scores.append(score)
    
    return np.mean(scores)

def optimize_hyperparameters(X, y, n_trials=50):  # 增加优化次数
    """使用Optuna进行超参数优化"""
    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(
            n_startup_trials=10,  # 增加初始随机搜索次数
            multivariate=True  # 启用多变量采样
        )
    )
    study.optimize(lambda trial: objective(trial, X, y), n_trials=n_trials)
    
    print("最佳超参数:")
    for key, value in study.best_params.items():
        print(f"{key}: {value}")
    print(f"最佳RMSE: {study.best_value}")
    
    return study.best_params

def train_lgb_model(train_data_path, model_save_path='model/lgb_model.txt', force_feature_selection=False):
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
    X, y, feature_names = prepare_features(data, target_col='Return', is_training=True)
    
    # 4. 特征筛选
    feature_info_path = 'model/selected_features.csv'
    if not force_feature_selection and os.path.exists(feature_info_path):
        print("加载已保存的特征信息...")
        feature_info = pd.read_csv(feature_info_path)
        selected_features = feature_info['selected_features'].tolist()
        selected_indices = feature_info['feature_indices'].tolist()
    else:
        print("开始特征筛选...")
        selected_features = select_features_rf(X, y, feature_names)
        selected_indices = [feature_names.index(feature) for feature in selected_features]
        # 保存特征信息
        feature_info = pd.DataFrame({
            'selected_features': selected_features,
            'feature_indices': selected_indices
        })
        os.makedirs('model', exist_ok=True)
        feature_info.to_csv(feature_info_path, index=False)
        print(f"特征信息已保存到 {feature_info_path}")
    
    # 只保留选中的特征
    X = X[:, :, selected_indices]
    
    # 重塑特征矩阵
    n_samples, n_timesteps, n_features = X.shape
    X = X.reshape(n_samples, -1)
    
    # 5. 使用优化后的参数，增强正则化
    best_params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,     # 减少叶子节点数
        'max_depth': 4,       # 减少树的深度
        'learning_rate': 0.001,  # 降低学习率
        'feature_fraction': 0.6,  # 减少特征使用比例
        'bagging_fraction': 0.6,  # 减少样本使用比例
        'bagging_freq': 5,        # 增加bagging频率
        'min_data_in_leaf': 100,  # 增加最小叶子节点样本数
        'min_gain_to_split': 0.2, # 增加分裂阈值
        'lambda_l1': 1.0,         # 增加L1正则化
        'lambda_l2': 1.0,         # 增加L2正则化
        'drop_rate': 0.4,         # 增加dropout
        'top_rate': 0.2,          # 减少GOSS top_rate
        'other_rate': 0.1,        # 保持GOSS other_rate
        'max_bin': 63,            # 减少分箱数
        'min_data_in_bin': 20,    # 增加每个箱的最小样本数
        'verbose': -1
    }
    
    # 6. 使用交叉验证训练模型
    print("\n使用交叉验证进行训练...")
    n_folds = 5  # 使用5折交叉验证
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    models = []
    cv_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        print(f"\n开始训练第 {fold} 折...")
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        # 定义学习率调度器，使用更保守的策略
        def lr_schedule(iter):
            if iter < 1000:
                return 0.001
            elif iter < 2000:
                return 0.0005
            else:
                return 0.0001
        
        # 训练模型
        model = lgb.train(
            best_params,
            train_data,
            num_boost_round=10000,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'valid'],
            callbacks=[
                lgb.log_evaluation(period=50),
                lgb.reset_parameter(
                    learning_rate=lr_schedule
                )
            ]
        )
        
        # 保存模型
        models.append(model)
        
        # 计算验证集分数
        val_preds = model.predict(X_val)
        val_score = np.sqrt(mean_squared_error(y_val, val_preds))
        cv_scores.append(val_score)
        print(f"第 {fold} 折验证集RMSE: {val_score:.6f}")
    
    # 计算平均分数
    mean_cv_score = np.mean(cv_scores)
    std_cv_score = np.std(cv_scores)
    print(f"\n交叉验证结果:")
    print(f"平均RMSE: {mean_cv_score:.6f} (±{std_cv_score:.6f})")
    
    # 选择最佳模型（验证集分数最低的模型）
    best_model_idx = np.argmin(cv_scores)
    best_model = models[best_model_idx]
    
    # 保存最佳模型
    best_model.save_model(model_save_path)
    print(f"最佳模型（第 {best_model_idx + 1} 折）已保存到 {model_save_path}")
    
    return best_model, selected_features

def predict_lgb_model(test_data_path, model_path='model/lgb_model.txt'):
    """使用训练好的模型进行预测"""
    # 1. 加载模型和特征信息
    model = lgb.Booster(model_file=model_path)
    feature_info = pd.read_csv('model/selected_features.csv')
    selected_indices = feature_info['feature_indices'].tolist()
    
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
    test_data = filter_stocks(test_data)
    test_data.replace([float('inf'), -float('inf')], 0, inplace=True)
    test_data.fillna(0, inplace=True)
    
    # 4. 准备预测数据
    X, _, _ = prepare_features(test_data, target_col='Return', is_training=False)
    # 只保留选中的特征
    X = X[:, :, selected_indices]
    X = X.reshape(X.shape[0], -1)
    
    # 5. 预测
    preds = model.predict(X)
    
    # 6. 计算涨跌幅
    results = []
    for i, stock_code in enumerate(test_data['StockCode'].unique()):
        stock_data = test_data[test_data['StockCode'] == stock_code].sort_values('Date')
        if len(stock_data) < 32:
            continue
            
        last_close = stock_data['Close'].iloc[-1]
        pred_return = preds[i]
        pct_change = pred_return * 100
        results.append((stock_code, pct_change))
    print(results)
    # 7. 获取涨幅最大和最小的10只股票
    results = sorted(results, key=lambda x: x[1], reverse=True)
    top10_up = [r[0] for r in results[:10]]
    top10_down = [r[0] for r in results[-10:]]
    
    # 8. 保存结果
    out = pd.DataFrame({
        "涨幅最大股票代码": top10_up,
        "涨幅最小股票代码": top10_down
    })
    
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
    import os
    
    # 创建必要的目录
    os.makedirs('model', exist_ok=True)
    os.makedirs('output', exist_ok=True)
    
    # 训练模型（第一次运行时强制特征筛选）
    model, selected_features = train_lgb_model("data/train.csv")
    
    # 预测
    predict_lgb_model("data/test.csv")
    
    # 评估预测结果
    predictions = pd.read_csv("output/result_lgb.csv")
    evaluate_predictions(predictions, "data/check.csv")
    
    # 保存评估结果
    save_evaluation_results(predictions, "data/check.csv")
