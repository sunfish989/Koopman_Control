import numpy as np
from scipy.integrate import solve_ivp
import torch
import torch.nn as nn
import torch.optim as optim
import scipy.linalg
import matplotlib.pyplot as plt
from tqdm import tqdm

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
        y[:, :self.M] = x[:, self.control_positions]

        # Non-control positions
        mask = np.ones(self.nx, dtype=bool)
        mask[self.control_positions] = False
        x_non_control = x[:, mask]  # Extract x at non-control positions

        # Nonlinear mapping for the rest
        y_non_control = self.net(x_non_control)
        y[:, self.M:] = y_non_control
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
        x[:, self.control_positions] = y[:, :self.M]

        # Nonlinear decoding for the rest
        y_non_control = y[:, self.M:]
        x_non_control = self.net(y_non_control)

        mask = np.ones(self.nx, dtype=bool)
        mask[self.control_positions] = False
        x[:, mask] = x_non_control
        return x


class Koopman_Control_Model:
    def __init__(self, nx, M, hidden_dim, P, control_positions):
        self.nx = nx  # 原始状态维度
        self.M = M  # 受控节点数
        self.P = P  # 嵌入空间维度
        self.control_positions = control_positions  # 控制位置

        # 编码器和解码器
        self.encoder = Encoder(nx, M, hidden_dim, P, control_positions).to(device)
        self.decoder = Decoder(nx, M, hidden_dim, P, control_positions).to(device)

        # 线性动力学矩阵A
        self.A = nn.Parameter(torch.eye(P) + 0.01 * torch.randn(P, P), requires_grad=True).to(device)

        # 优化器（将在训练中初始化）
        self.optimizer = None

        # 嵌入空间中的控制影响矩阵B_embed
        self.B_embed = np.zeros((self.P, self.M))
        for k, i in enumerate(self.control_positions):
            self.B_embed[i, k] = 1.0

    def train_step(self, x_t, x_hat_t1):
        self.optimizer.zero_grad()

        # 编码
        y_t = self.encoder(x_t)
        y_hat_t1 = self.encoder(x_hat_t1)

        # 线性预测
        y_t1_pred = torch.matmul(y_t, self.A.T)

        # 预测损失
        pred_loss = nn.MSELoss()(y_t1_pred, y_hat_t1)

        # 重建损失
        recon_loss = 0.5 * (nn.MSELoss()(self.decoder(y_t), x_t) +
                            nn.MSELoss()(self.decoder(y_hat_t1), x_hat_t1))

        # 总损失
        loss = pred_loss + recon_loss

        loss.backward()
        self.optimizer.step()

        return loss.item()

    def compute_control(self, y_t, y_target, K):
        # y_t: 当前嵌入状态，形状(1, P)
        # y_target: 目标嵌入状态，形状(1, P)
        # K: LQR增益矩阵，形状(M, P)

        y_error = y_t - y_target  # 状态误差

        # 计算控制输入
        u_bar = -np.matmul(K, y_error.detach().cpu().numpy().T).flatten()  # 形状(M,)

        return u_bar  # numpy数组，形状(M,)

    def design_lqr_controller(self):
        # 离散LQR控制器设计
        A_np = self.A.detach().cpu().numpy()
        B_np = self.B_embed  # 使用嵌入空间中的B

        # 权重矩阵
        q_scale = 1
        r_scale = 1
        Q = np.eye(self.P) * q_scale  # 状态权重矩阵
        R = np.eye(B_np.shape[1]) * r_scale  # 控制权重矩阵

        # 求解离散代数Riccati方程
        P = scipy.linalg.solve_discrete_are(A_np, B_np, Q, R)

        # 计算LQR增益K
        K = np.linalg.inv(B_np.T @ P @ B_np + R) @ (B_np.T @ P @ A_np)

        return K  # 形状(M, P)

    def train(self, pde, num_epochs=100, time_steps=1000, dt=0.01):
        # 初始化优化器
        self.optimizer = optim.Adam(
            list(self.encoder.parameters()) +
            list(self.decoder.parameters()) +
            [self.A],
            lr=1e-4  # 减小学习率
        )

        # 可以使用学习率调度器
        # self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=10, verbose=True)

        # 目标状态（所有样本相同）
        T_target = np.zeros(self.nx)
        T_target[:self.nx // 2] = 1.0  # 期望的温度分布
        x_target = T_target.copy()  # 作为numpy数组使用

        # 初始化全局的均值和标准差
        x_mean_np = np.zeros(self.nx)
        x_std_np = np.ones(self.nx)

        for epoch in range(num_epochs):
            self.encoder.train()
            self.decoder.train()
            total_loss = 0

            # 生成训练数据（我们需要在每个epoch生成它以使用更新的A进行控制）
            num_samples = 1  # 我们将每次只处理一个样本，以满足逐步训练和控制的需求

            # 数据生成循环
            for sample_idx in tqdm(range(num_samples), desc=f"Epoch {epoch + 1}", leave=False):
                T0 = np.zeros(self.nx)
                T0[self.nx // 2:] = 1.0
                T_sequence = np.zeros((time_steps + 1, self.nx))
                T_sequence[0, :] = T0.copy()
                u_sequence = np.zeros((time_steps, self.M))

                u_t_sample = np.zeros(self.M)  # 初始化控制输入

                x_t_batch = []
                x_hat_t1_batch = []
                batch_interval = 10  # 每隔多少时间步进行一次批量训练

                for t in range(time_steps):
                    x_t_sample = T_sequence[t, :]

                    # 模拟一个时间步
                    T_t1 = pde.simulate(T_sequence[t, :], np.array([u_t_sample]), [t * dt, (t + 1) * dt])
                    T_sequence[t + 1, :] = T_t1[-1]

                    x_t1_sample = T_sequence[t + 1, :]

                    # 计算 x_hat_t1_sample = x_t1_sample - B * u_t_sample
                    B_u_t_sample = np.dot(pde.B, u_t_sample)  # 使用正确的 B 矩阵
                    x_hat_t1_sample = x_t1_sample - B_u_t_sample

                    # 更新全局的均值和标准差
                    if t >= 1:
                        x_mean_np = np.mean(T_sequence[:t + 1, :], axis=0)
                        x_std_np = np.std(T_sequence[:t + 1, :], axis=0) + 1e-8  # 避免除以零

                    # 对 x_t_sample 和 x_hat_t1_sample 进行归一化
                    x_t_sample_norm = (x_t_sample - x_mean_np) / x_std_np
                    x_hat_t1_sample_norm = (x_hat_t1_sample - x_mean_np) / x_std_np

                    x_t_sample_norm_tensor = torch.tensor(x_t_sample_norm, dtype=torch.float32).unsqueeze(0).to(device)
                    x_hat_t1_sample_norm_tensor = torch.tensor(x_hat_t1_sample_norm, dtype=torch.float32).unsqueeze(0).to(device)

                    # 收集批量数据
                    x_t_batch.append(x_t_sample_norm_tensor)
                    x_hat_t1_batch.append(x_hat_t1_sample_norm_tensor)

                    # 每隔一定时间步，进行一次迷你批次训练
                    if (t + 1) % batch_interval == 0 or t == time_steps - 1:
                        x_t_batch_tensor = torch.cat(x_t_batch, dim=0)
                        x_hat_t1_batch_tensor = torch.cat(x_hat_t1_batch, dim=0)

                        # 进行一次迷你批次训练
                        loss = self.train_step(x_t_batch_tensor, x_hat_t1_batch_tensor)
                        total_loss += loss

                        # 清空批次数据
                        x_t_batch = []
                        x_hat_t1_batch = []

                    # 现在根据需要更新控制
                    if t < 200:
                        u_t_sample = np.zeros(self.M)
                    elif (t - 200) % 20 == 0:
                        # 计算 y_target
                        x_target_norm = (x_target - x_mean_np) / x_std_np
                        x_target_norm_tensor = torch.tensor(x_target_norm, dtype=torch.float32).unsqueeze(0).to(device)
                        y_target = self.encoder(x_target_norm_tensor)

                        # 更新 K
                        K = self.design_lqr_controller()

                        # 计算控制输入
                        # 在训练后重新编码 x_t_sample
                        y_t_sample = self.encoder(x_t_sample_norm_tensor)
                        u_t_sample = self.compute_control(y_t_sample, y_target, K)
                    # 否则，保持 u_t_sample 不变

                    u_sequence[t, :] = u_t_sample  # 存储控制输入

                # 在每个样本结束时进行可视化
                if (epoch + 1) % 1 == 0:
                    # 可视化温度分布随时间的变化
                    plt.figure(figsize=(10, 6))
                    plt.imshow(T_sequence.T, aspect='auto', cmap='hot', origin='lower')
                    plt.colorbar()
                    plt.xlabel('Time Step')
                    plt.ylabel('Spatial Position')
                    plt.title(f'Temperature Distribution Over Time at Epoch {epoch + 1}')
                    plt.show()

            avg_train_loss = total_loss / (num_samples * time_steps)

            print(f"Epoch {epoch + 1}, Train Loss: {avg_train_loss:.6f}")

            # 可选地调整学习率
            # self.scheduler.step(avg_train_loss)


def main():
    # PDE 参数
    nx = 50  # 空间节点数
    dx = 1.0 / nx  # 空间步长
    alpha = 0.1  # 热扩散系数
    M = 25  # 受控节点数

    # 控制位置：每隔一个点
    control_positions = np.arange(0, nx, 2)  # Positions 0, 2, ..., 48

    pde = PDE(nx, dx, alpha, M)

    # 定义模型
    hidden_dim = 128
    P = nx  # 嵌入空间维度
    model = Koopman_Control_Model(nx, M, hidden_dim, P, control_positions)

    num_epochs = 100
    time_steps = 1000
    dt = 0.01

    # 训练模型
    model.train(pde, num_epochs=num_epochs, time_steps=time_steps, dt=dt)


if __name__ == "__main__":
    main()
