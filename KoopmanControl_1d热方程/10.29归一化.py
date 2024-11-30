import numpy as np
from scipy.integrate import solve_ivp
import torch
import torch.nn as nn
import torch.optim as optim
import scipy.linalg
import torch.utils.data as data


class PDE:
    def __init__(self, nx, dx, alpha, M):
        self.nx = nx  # 空间节点数
        self.dx = dx  # 空间步长
        self.alpha = alpha  # 热扩散系数
        self.M = M  # 受控节点数

        # 控制点位的位置
        self.control_positions = np.arange(0, nx, 2)  # Positions 0, 2, ..., 48

        # 构建离散化的Laplacian矩阵
        self.L = self.build_laplacian(nx, dx)

        # 控制影响矩阵B，尺寸为(nx, M)
        self.B = np.zeros((nx, M))
        for k, i in enumerate(self.control_positions):
            self.B[i, k] = 1.0

    def build_laplacian(self, nx, dx):
        # 构建离散化的Laplacian矩阵
        L = np.zeros((nx, nx))
        for i in range(1, nx - 1):
            L[i, i - 1] = 1
            L[i, i] = -2
            L[i, i + 1] = 1
        # 边界条件：Neumann边界条件（导数为零）
        L[0, 0] = -2
        L[0, 1] = 2
        L[-1, -2] = 2
        L[-1, -1] = -2
        L = self.alpha * L / (dx ** 2)
        return L

    def simulate(self, T0, u_sequence, t_span):
        # T0: 初始温度分布，形状(nx,)
        # u_sequence: 控制输入序列，形状(time_steps, M)
        # t_span: 时间跨度，例如[0, dt * total_steps]
        # 返回温度分布序列，形状(time_steps + 1, nx)

        time_steps = len(u_sequence)
        dt = (t_span[1] - t_span[0]) / time_steps
        times = np.linspace(t_span[0], t_span[1], time_steps + 1)

        T = np.zeros((time_steps + 1, self.nx))
        T[0, :] = T0

        for t in range(time_steps):
            # 当前时间步的控制输入
            u_t = u_sequence[t, :]  # 形状(M,)

            # 时间积分
            sol = solve_ivp(
                fun=lambda t, T_flat: self.odefun(t, T_flat, u_t),
                t_span=[times[t], times[t + 1]],
                y0=T[t, :],
                method='RK45'
            )
            T[t + 1, :] = sol.y[:, -1]

        return T

    def odefun(self, t, T_flat, u_t):
        # T_flat: 温度分布，形状(nx,)
        # u_t: 当前时间步的控制输入，形状(M,)

        # 控制影响
        B_u = np.dot(self.B, u_t)  # 形状(nx,)

        # dT/dt = L * T + B * u
        dTdt = np.dot(self.L, T_flat) + B_u

        return dTdt


class Encoder(nn.Module):
    def __init__(self, nx, M, hidden_dim, P, control_positions):
        super(Encoder, self).__init__()
        self.nx = nx  # 原始状态维度
        self.M = M  # 受控节点数
        self.P = P  # 嵌入空间维度
        self.hidden_dim = hidden_dim
        self.control_positions = control_positions  # Control positions

        # 非线性映射的神经网络
        self.net = nn.Sequential(
            nn.Linear(nx - M, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, P - M)
        )

    def forward(self, x):
        # x: 形状(batch_size, nx)
        batch_size = x.size(0)
        y = torch.zeros(batch_size, self.P, device=x.device)

        # Identity mapping for control positions
        y[:, self.control_positions] = x[:, self.control_positions]

        # Non-control positions
        mask = np.ones(self.nx, dtype=bool)
        mask[self.control_positions] = False
        x_non_control = x[:, mask]  # Extract x at non-control positions

        # Nonlinear mapping for the rest
        y_non_control = self.net(x_non_control)
        y[:, mask] = y_non_control
        return y


class Decoder(nn.Module):
    def __init__(self, nx, M, hidden_dim, P, control_positions):
        super(Decoder, self).__init__()
        self.nx = nx  # 原始状态维度
        self.M = M  # 受控节点数
        self.P = P  # 嵌入空间维度
        self.hidden_dim = hidden_dim
        self.control_positions = control_positions  # Control positions

        # 非线性解码的神经网络
        self.net = nn.Sequential(
            nn.Linear(P - M, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, nx - M)
        )

    def forward(self, y):
        # y: 形状(batch_size, P)
        batch_size = y.size(0)
        x = torch.zeros(batch_size, self.nx, device=y.device)

        # Identity mapping for control positions
        x[:, self.control_positions] = y[:, self.control_positions]

        mask = np.ones(self.nx, dtype=bool)
        mask[self.control_positions] = False

        # Nonlinear decoding for the rest
        y_non_control = y[:, mask]
        x_non_control = self.net(y_non_control)

        x[:, mask] = x_non_control
        return x


class Koopman_Model:
    def __init__(self, nx, M, hidden_dim, P, control_positions):
        self.nx = nx  # Original state dimension
        self.M = M  # Number of controlled nodes
        self.P = P  # Embedding space dimension
        self.control_positions = control_positions  # Control positions

        # Encoder and Decoder
        self.encoder = Encoder(nx, M, hidden_dim, P, control_positions).to(device)
        self.decoder = Decoder(nx, M, hidden_dim, P, control_positions).to(device)

        # 线性动力学矩阵A
        self.A = nn.Parameter(torch.eye(P) + 0.01 * torch.randn(P, P), requires_grad=True).to(device)

        # 优化器
        self.optimizer = optim.Adam(
            list(self.encoder.parameters()) +
            list(self.decoder.parameters()) +
            [self.A],
            lr=1e-3
        )

    def train_step(self, x_t, x_t1_prime):
        self.optimizer.zero_grad()

        # 编码
        y_t = self.encoder(x_t)
        y_t1 = self.encoder(x_t1_prime)

        # 线性预测
        y_t1_pred = torch.matmul(y_t, self.A.T)

        # 预测损失
        pred_loss = nn.MSELoss()(y_t1_pred, y_t1)

        # Reconstruction loss
        recon_loss = 0.5 * (nn.MSELoss()(self.decoder(y_t), x_t) + nn.MSELoss()(self.decoder(y_t1), x_t1_prime))

        # 总损失
        loss = pred_loss + 0.7 * recon_loss

        loss.backward()
        self.optimizer.step()

        return loss.item()

    def compute_loss(self, x_t, x_t1_prime):

        # Encoding and predictions
        y_t = self.encoder(x_t)
        y_t1 = self.encoder(x_t1_prime)

        # Linear prediction with control input
        y_t1_pred = torch.matmul(y_t, self.A.T)

        # 预测损失
        pred_loss = nn.MSELoss()(y_t1_pred, y_t1)

        # Reconstruction loss
        recon_loss = 0.5 * (nn.MSELoss()(self.decoder(y_t), x_t) + nn.MSELoss()(self.decoder(y_t1), x_t1_prime))

        # 总损失
        loss = pred_loss + 0.7 * recon_loss
        return loss.item()

    def compute_control(self, y_t, y_target, K):
        # Compute the control input

        #  控制的产生这里有点问题，这里产生的是25维度的控制，但是是如何影响y的
        y_error = y_t - y_target  # State error
        u_bar = -np.matmul(K, y_error.detach().cpu().numpy().T).flatten()  # Shape (M,)
        return u_bar  # Numpy array, shape (M,)

    def design_lqr_controller(self):
        # Discrete-time LQR controller design
        A = self.A.detach().cpu().numpy()
        overline_B = np.zeros((self.P, self.M))
        for k, i in enumerate(self.control_positions):
            overline_B[i, k] = 1.0
        B = overline_B
        Q = np.eye(A.shape[0])  # State weight matrix
        R = np.eye(B.shape[1])  # Control input weight matrix

        # 求解离散代数Riccati方程
        P = scipy.linalg.solve_discrete_are(A, B, Q, R)

        # 计算LQR增益K
        K = np.linalg.inv(B.T @ P @ B + R) @ (B.T @ P @ A)

        return K  # Shape (M, P)



def main():
    # 设备配置
    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # PDE parameters
    nx = 50  # Number of spatial nodes
    dx = 1.0 / nx  # Spatial step size
    # 想要对换01状态的控制的话，需要较小的温度传播系数, 否则温度传播过快的话，会稀释。想让一个全局稳定状态的话，可以有较大的传播系数
    alpha = 0.01  # Thermal diffusivity coefficient
    M = 25  # Number of controlled nodes

    # Control positions: every other point
    control_positions = np.arange(0, nx, 2)  # Positions 0, 2, ..., 48

    pde = PDE(nx, dx, alpha, M)

    # 生成训练数据
    num_samples = 3000
    time_steps = 80
    dt = 0.01
    x_t_list = []
    x_t1_list = []
    u_t_list = []

    for _ in range(num_samples):
        # 随机初始温度分布
        T0 = np.random.rand(nx)

        # Generate training data with control inputs
        control_input_scale = 0.05  # Adjust the scale as needed
        u_sequence = np.random.rand(time_steps + 1, M) * control_input_scale
        # u_sequence = np.zeros((time_steps + 1, M)) * control_input_scale

        # Simulate the system
        t_span = [0, dt * time_steps]
        T_sequence = pde.simulate(T0, u_sequence, t_span)  # Shape (time_steps + 1, nx)

        # 构建训练样本
        x_t_list.append(T_sequence[1:-2, :])  # x(t)
        x_t1_list.append(T_sequence[2:-1, :])  # x(t+1)
        u_t_list.append(u_sequence[1:-1, :])

    # 转换为张量
    x_t = torch.tensor(np.concatenate(x_t_list, axis=0), dtype=torch.float32)
    x_t1 = torch.tensor(np.concatenate(x_t1_list, axis=0), dtype=torch.float32)
    u_t = torch.tensor(np.concatenate(u_t_list, axis=0), dtype=torch.float32)

    # 数据归一化
    x_mean = x_t.mean(dim=0)
    x_std = x_t.std(dim=0) + 1e-8  # 防止除以零
    x_t = (x_t - x_mean) / x_std
    x_t1 = (x_t1 - x_mean) / x_std

    #  这里需要对控制进行一个放缩，因为batch_x_t1是放缩过的，如果这里控制不放缩，会学习错误。
    control_mean = x_mean[control_positions].unsqueeze(0)  # Shape (1, M)
    control_std = x_std[control_positions].unsqueeze(0)  # Shape (1, M)
    # Standardize batch_u_t
    u_t = (u_t - control_mean) / control_std  # Shape (batch_size, M)

    # 将数据移动到设备上
    x_t = x_t.to(device)
    x_t1 = x_t1.to(device)
    u_t = u_t.to(device)

    # 数据集和数据加载器
    dataset = data.TensorDataset(x_t, x_t1, u_t)

    # 划分训练集和验证集
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = data.random_split(dataset, [train_size, val_size])

    train_dataloader = data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_dataloader = data.DataLoader(val_dataset, batch_size=64, shuffle=False)

    # 定义模型
    nx = pde.nx
    hidden_dim = 128
    P = nx  # Embedding space dimension
    model = Koopman_Model(nx, M, hidden_dim, P, control_positions)

    # 训练模型
    num_epochs = 30
    patience = 20
    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        # 设置训练模式
        model.encoder.train()
        model.decoder.train()
        total_loss = 0
        for batch_x_t, batch_x_t1, batch_u_t in train_dataloader:
            B_u_t = torch.zeros_like(batch_x_t, device=device)
            B_u_t[:, control_positions] = batch_u_t
            batch_x_t1_prime = batch_x_t1 - B_u_t
            loss = model.train_step(batch_x_t, batch_x_t1_prime)
            total_loss += loss
        avg_train_loss = total_loss / len(train_dataloader)

        # 验证
        model.encoder.eval()
        model.decoder.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x_t, batch_x_t1, batch_u_t in val_dataloader:
                B_u_t = torch.zeros_like(batch_x_t, device=device)
                B_u_t[:, control_positions] = batch_u_t
                batch_x_t1_prime = batch_x_t1 - B_u_t
                loss = model.compute_loss(batch_x_t, batch_x_t1_prime)
                val_loss += loss
        avg_val_loss = val_loss / len(val_dataloader)

        print(f"Epoch {epoch + 1}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")

        # 早停机制
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping!")
                break

    # 在训练后进行预测与真实值的比较
    with torch.no_grad():
        batch_x_t, batch_x_t1, batch_u_t = next(iter(val_dataloader))

        # 编码当前状态
        y_t = model.encoder(batch_x_t)

        # 线性预测
        y_t1_pred = torch.matmul(y_t, model.A.T)

        # 解码
        x_t1_pred_prime = model.decoder(y_t1_pred)

        B_u = torch.zeros_like(batch_x_t1, device=device)
        B_u[:, control_positions] = batch_u_t    # Control influence
        # Add control influence to get predicted x(t+1)
        x_t1_pred = x_t1_pred_prime + B_u

        # 反归一化
        x_t1_pred = x_t1_pred * x_std + x_mean
        batch_x_t1 = batch_x_t1 * x_std + x_mean

        # 计算预测误差
        pred_error = nn.MSELoss()(x_t1_pred, batch_x_t1)
        print(f"Prediction error on validation batch: {pred_error.item():.6f}")

        # 可视化
        import matplotlib.pyplot as plt

        idx = 0  # 可视化第一个样本
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(batch_x_t1[idx].cpu().numpy(), label='True x(t+1)')
        plt.plot(x_t1_pred[idx].cpu().numpy(), label='Predicted x(t+1)')
        plt.legend()
        plt.title('True vs Predicted x(t+1)')

        plt.subplot(1, 2, 2)
        plt.plot((batch_x_t1[idx] - x_t1_pred[idx]).cpu().numpy())
        plt.title('Prediction Error')
        plt.show()

    # 设计LQR控制器
    K = model.design_lqr_controller()

    # 控制目标
    T_target = np.zeros(nx)
    T_target[:nx // 2] = 1  # 目标温度分布

    # 初始状态
    T_init = np.zeros(nx)
    T_init[nx // 2:] = 1

    # 仿真控制过程
    time_steps = 600
    T = np.zeros((time_steps + 1, nx))
    T[0, :] = T_init
    u_sequence = np.zeros((time_steps, M))

    # 编码目标状态
    x_target = torch.tensor((T_target - x_mean.cpu().numpy()) / x_std.cpu().numpy(), dtype=torch.float32).unsqueeze(
        0).to(device)
    y_target = model.encoder(x_target)
    # Initialize

    for t in range(time_steps):
        x_t = torch.tensor((T[t, :] - x_mean.cpu().numpy()) / x_std.cpu().numpy(),
                           dtype=torch.float32).unsqueeze(0).to(device)

        # Encode the adjusted state
        y_t = model.encoder(x_t)

        # Compute control input
        u_bar = model.compute_control(y_t, y_target, K)  # Shape (M,)
        # Extract means and stds for control positions
        control_mean = x_mean.cpu().numpy()[control_positions]
        control_std = x_std.cpu().numpy()[control_positions]

        # 这里计算出来的u_bar是放缩后的系统下的控制量，如果要用在原系统中，需要进行反向的放缩。
        # Unstandardize u_bar
        u_bar = u_bar * control_std + control_mean

        u_sequence[t, :] = u_bar  # Record control input

        # Simulate next time step
        T_t1 = pde.simulate(T[t, :], np.array([u_bar]), [t * dt, (t + 1) * dt])
        T[t + 1, :] = T_t1[-1]

    # Visualization of results
    plt.figure(figsize=(10, 6))
    plt.imshow(T.T, aspect='auto', cmap='hot', origin='lower')
    plt.colorbar()
    plt.xlabel('Time Step')
    plt.ylabel('Spatial Position')
    plt.title('Temperature Distribution Over Time')
    plt.show()


if __name__ == "__main__":
    main()
