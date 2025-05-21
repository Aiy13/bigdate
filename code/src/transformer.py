import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import time
import warnings
warnings.filterwarnings('ignore')

class StockDataset(Dataset):
    def __init__(self, features, targets):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model=256, nhead=8, num_layers=4, dropout=0.1):
        super(TransformerModel, self).__init__()
        
        # 1. 特征嵌入层
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 2. 位置编码
        self.pos_encoder = PositionalEncoding(d_model)
        
        # 3. Transformer编码器
        encoder_layers = nn.TransformerEncoderLayer(
            d_model, 
            nhead, 
            dim_feedforward=d_model*2,  # 使用较小的前馈网络维度
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        # 4. 解码器
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.LayerNorm(d_model//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model//2, 1)
        )
        
        # 5. 初始化权重
        self._init_weights()
        
    def _init_weights(self):
        """使用Xavier初始化权重"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
    def forward(self, src):
        # 1. 特征嵌入
        src = self.embedding(src)
        
        # 2. 位置编码
        src = self.pos_encoder(src)
        
        # 3. Transformer编码
        output = self.transformer_encoder(src)
        
        # 4. 取最后一个时间步的输出
        output = output[:, -1, :]
        
        # 5. 解码预测
        output = self.decoder(output)
        
        return output

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

def prepare_features(df, target_col='Return', lookback=32, is_training=True):
    """准备特征数据，包括目标变量"""
    features = []
    targets = []
    stock_codes = []
    
    # 使用所有构造的特征
    feature_cols = [col for col in df.columns if col not in ['StockCode', 'Date', target_col]]
    
    # 限制处理的股票数量
    max_stocks = 1000  # 限制处理的股票数量
    unique_stocks = df['StockCode'].unique()
    if len(unique_stocks) > max_stocks:
        unique_stocks = np.random.choice(unique_stocks, max_stocks, replace=False)
    
    for stock_code in unique_stocks:
        stock_data = df[df['StockCode'] == stock_code].sort_values('Date')
        if len(stock_data) <= lookback:
            continue
            
        X = stock_data[feature_cols].values
        y = stock_data[target_col].values
        
        if is_training:
            # 训练时使用滑动窗口，但减少窗口数量
            step = 5  # 每隔5天取一个窗口
            max_windows = 50  # 每个股票最多取50个窗口
            windows = range(0, len(stock_data) - lookback, step)
            if len(windows) > max_windows:
                windows = np.random.choice(windows, max_windows, replace=False)
            
            for i in windows:
                features.append(X[i:i+lookback])
                targets.append(y[i+lookback])
                stock_codes.append(stock_code)
        else:
            features.append(X[-lookback:])
            targets.append(y[-1])
            stock_codes.append(stock_code)
    
    # 转换为numpy数组时使用float32而不是float64
    features = np.array(features, dtype=np.float32)
    targets = np.array(targets, dtype=np.float32)
    
    return features, targets, stock_codes

def train_transformer_model(train_data_path, model_save_path='model/transformer_model.pth'):
    """训练Transformer模型"""
    # 1. 数据预处理
    print("开始数据预处理...")
    data = inputdata(train_data_path)
    data = transcolname(data, {
        "股票代码": "StockCode", "日期": "Date", "开盘": "Open", "收盘": "Close",
        "最高": "High", "最低": "Low", "成交量": "Volume", "成交额": "Turnover",
        "振幅": "Amplitude", "涨跌额": "PriceChange", "换手率": "TurnoverRate",
        "涨跌幅": "PriceChangePercentage",
    })
    data.drop(columns=["PriceChangePercentage"], inplace=True)
    
    # 2. 特征工程
    print("开始特征工程...")
    data = trans_datetime(data)
    data = augment_features(data)
    data = filter_stocks(data)
    data.replace([float('inf'), -float('inf')], 0, inplace=True)
    data.fillna(0, inplace=True)
    
    # 3. 准备训练数据
    print("准备训练数据...")
    X, y, _ = prepare_features(data, target_col='Return', is_training=True)
    print(f"训练数据形状: X={X.shape}, y={y.shape}")
    
    # 4. 创建数据加载器
    dataset = StockDataset(X, y)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # 5. 初始化模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerModel(
        input_dim=X.shape[2],
        d_model=256,
        nhead=8,
        num_layers=4,
        dropout=0.1
    ).to(device)
    
    # 6. 训练参数
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=0.001,
        weight_decay=0.05,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # 使用更复杂的学习率调度器
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.001,
        epochs=100,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        anneal_strategy='cos',
        div_factor=25.0,
        final_div_factor=1000.0
    )
    
    # 7. 训练模型
    print("\n开始训练Transformer模型...")
    print("模型参数:")
    print(f"batch_size: 32")
    print(f"learning_rate: 0.001")
    print(f"weight_decay: 0.05")
    print(f"device: {device}")
    
    num_epochs = 100
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    # 添加学习率预热
    warmup_epochs = 5
    warmup_factor = 0.1
    
    for epoch in tqdm(range(num_epochs), desc="训练进度"):
        epoch_start = time.time()
        
        # 训练阶段
        model.train()
        train_loss = 0.0
        
        # 学习率预热
        if epoch < warmup_epochs:
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['initial_lr'] * warmup_factor * (epoch + 1) / warmup_epochs
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            # 添加噪声增强
            if epoch < num_epochs * 0.8:
                noise = torch.randn_like(batch_X) * 0.01
                batch_X = batch_X + noise
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs.squeeze(), batch_y)
            
            # 添加L1正则化
            l1_lambda = 0.01
            l1_norm = sum(p.abs().sum() for p in model.parameters())
            loss = loss + l1_lambda * l1_norm
            
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)  # 确保验证数据也移动到GPU
                outputs = model(batch_X)
                val_loss += criterion(outputs.squeeze(), batch_y).item()
                val_preds.extend(outputs.cpu().numpy())
                val_targets.extend(batch_y.cpu().numpy())  # 确保在移动到CPU之前获取数据
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        avg_val_loss = val_loss / len(val_loader)
        val_preds = np.array(val_preds, dtype=np.float32)
        val_targets = np.array(val_targets, dtype=np.float32)
        val_mse = np.mean((val_preds - val_targets) ** 2)
        val_rmse = np.sqrt(val_mse)
        
        epoch_time = time.time() - epoch_start
        
        print(f"\n轮次 [{epoch + 1}/{num_epochs}]")
        print(f"训练损失: {avg_train_loss:.6f}")
        print(f"验证损失: {avg_val_loss:.6f}")
        print(f"验证MSE: {val_mse:.6f}")
        print(f"验证RMSE: {val_rmse:.6f}")
        print(f"学习率: {scheduler.get_last_lr()[0]:.6f}")
        print(f"耗时: {epoch_time:.2f}s")
        
        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"模型已保存到 {model_save_path}")
            patience_counter = 0
        else:
            patience_counter += 1
            
    
    return model

def predict_transformer_model(test_data_path, model_path='model/transformer_model.pth'):
    """使用训练好的模型进行预测"""
    # 1. 加载模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
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
    X, _, stock_codes = prepare_features(test_data, target_col='Return', is_training=False)
    
    # 5. 初始化模型（使用与训练时相同的架构）
    model = TransformerModel(
        input_dim=X.shape[2],
        d_model=256,  # 与训练时相同
        nhead=8,
        num_layers=4,  # 与训练时相同
        dropout=0.1    # 与训练时相同
    )
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()
    
    # 6. 创建数据加载器
    dataset = StockDataset(X, np.zeros(len(X)))  # 创建临时目标值
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False)
    
    # 7. 预测
    predictions = []
    with torch.no_grad():
        for batch_X, _ in dataloader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            predictions.extend(outputs.cpu().numpy())
    
    # 8. 计算涨跌幅
    results = []
    for i, (stock_code, pred) in enumerate(zip(stock_codes, predictions)):
        stock_data = test_data[test_data['StockCode'] == stock_code].sort_values('Date')
        if len(stock_data) < 32:
            continue
            
        last_close = stock_data['Close'].iloc[-1]
        pred_return = pred[0]
        pct_change = pred_return * 100  # 直接使用预测的收益率
        results.append((stock_code, pct_change))
    print(results)
    # 9. 获取涨幅最大和最小的10只股票
    results = sorted(results, key=lambda x: x[1], reverse=True)
    top10_up = [r[0] for r in results[:10]]
    top10_down = [r[0] for r in results[-10:]]
    
    # 10. 保存结果
    out = pd.DataFrame({
        "涨幅最大股票代码": top10_up,
        "涨幅最小股票代码": top10_down
    })
    out.to_csv("output/result_transformer.csv", index=False, encoding="utf-8")
    print("预测完毕，结果保存在 output/result_transformer.csv")

if __name__ == "__main__":
    # 训练模型
    model = train_transformer_model("data/train.csv")
    
    # 预测
    predict_transformer_model("data/test.csv") 