import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.signal import detrend
from PyEMD import VEMD  # 需安装pyemd库（pip install pyemd）


# ====================== 数据预处理======================
class VEMDDataset(Dataset):
    def __init__(self, data, window_size=9):
        self.data = data
        self.window_size = window_size
        self.n_samples = len(data) - window_size

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        x = self.data[idx:(idx + self.window_size), :]
        y = self.data[idx + self.window_size, -1]  # 预测最后一个分量（残差）
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


# ====================== LSTM-Attention模型 ======================
class LSTMAttention(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=24, dropout=0.2):
        super(LSTMAttention, self).__init__()
        self.hidden_dim = hidden_dim

        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
            return_sequences=True
        )

        # 注意力机制
        self.W = nn.Linear(hidden_dim, 1)

        # 输出层
        self.fc = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def attention(self, lstm_output):
        # lstm_output: (batch_size, seq_len, hidden_dim)
        attn_weights = torch.tanh(self.W(lstm_output)).squeeze(-1)
        attn_weights = nn.functional.softmax(attn_weights, dim=1).unsqueeze(1)
        context = torch.bmm(attn_weights, lstm_output).squeeze(1)
        return context

    def forward(self, x):
        # x: (batch_size, window_size, input_dim)
        lstm_out, _ = self.lstm(x)
        context = self.attention(lstm_out)
        context = self.dropout(context)
        out = self.fc(context)
        return out


# ====================== 主流程 ======================
if __name__ == "__main__":
    # 输入数据序列（HR或RR数据）
    data = np.array(
        [56, 33, 76, 68, 23, 8, 86, 5, 1, 16, 36, 95, 91, 73, 96, 55, 80, 22, 99, 30, 34, 85, 72, 82, 26, 63, 49, 14,
         32, 40, 87, 90, 69, 45, 17, 66, 31, 53, 65, 88, 58, 81, 64, 19, 75, 18, 21, 60, 92, 83])

    # 数据预处理：去趋势、归一化
    data = detrend(data)  # 去除线性趋势（噪声处理）
    data_mean, data_std = np.mean(data), np.std(data)
    data_normalized = (data - data_mean) / data_std

    # VEMD分解
    vemd = VEMD()
    imfs = vemd.emd(data_normalized, n_iter=50)  # n_iter对应多次平均处理
    residual = vemd.residue  # 残差序列
    components = np.hstack([imfs, residual.reshape(-1, 1)])  # 合并IMF和残差（3个IMF）

    # 数据集处理（滑动窗口，窗口长度=9）
    window_size = 9
    dataset = VEMDDataset(components, window_size=window_size)
    train_size = int(0.8 * len(dataset))
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size])

    # 数据加载器
    batch_size = 128  # batch_size的设置
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 模型初始化（input_dim=IMF数量+1，3个IMF+1残差=4维输入）
    input_dim = components.shape[1]  # 实际需根据分解结果调整，此处为4
    model = LSTMAttention(input_dim=input_dim, hidden_dim=24, dropout=0.2)  # 神经元数=24
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # 学习率=0.001
    criterion = nn.MSELoss()

    # 训练模型
    epochs = 100  # epoch设为100
    for epoch in range(epochs):
        model.train()
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch.unsqueeze(1))
            loss.backward()
            optimizer.step()

    # 预测并反归一化
    model.eval()
    true_values = []
    pred_values = []
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            outputs = model(x_batch)
            true_values.extend(y_batch.numpy())
            pred_values.extend(outputs.numpy())

    # 反归一化（仅对残差分量反归一化，实际需合并所有IMF预测结果）
    true_original = np.array(true_values) * data_std + data_mean
    pred_original = np.array(pred_values) * data_std + data_mean

    # 计算评价指标（使用RMSE、MAE、R²）
    rmse = np.sqrt(mean_squared_error(true_original, pred_original))
    mae = mean_absolute_error(true_original, pred_original)
    r2 = 1 - np.sum((true_original - pred_original) ** 2) / np.sum((true_original - np.mean(true_original)) ** 2)

    # 打印结果
    print("================= 评价指标 =================")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R²: {r2:.4f}")