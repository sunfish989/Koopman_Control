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
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
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
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
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


class KoopmanModel:
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

    def train_step(self, x_t, x_t1, u_t_minus1, u_t):
        self.optimizer.zero_grad()

        # 控制影响
        B_u_t_minus1 = torch.zeros_like(x_t, device=device)
        B_u_t_minus1[:, self.control_positions] = u_t_minus1  # Control influence at time t

        B_u_t = torch.zeros_like(x_t1, device=device)
        B_u_t[:, self.control_positions] = u_t  # Control influence at time t+1

        # 减去控制影响，得到纯粹的系统状态
        x_t_prime = x_t - B_u_t_minus1  # x'(t) = x(t) - B*u(t-1)
        x_t1_prime = x_t1 - B_u_t  # x'(t+1) = x(t+1) - B*u(t)

        # 编码
        y_t = self.encoder(x_t_prime)
        y_t1 = self.encoder(x_t1_prime)

        # 线性预测
        y_t1_pred = torch.matmul(y_t, self.A.T)

        # 预测损失
        pred_loss = nn.MSELoss()(y_t1_pred, y_t1)

        # 重建损失
        x_t_recon = self.decoder(y_t)
        recon_loss = nn.MSELoss()(x_t_recon, x_t_prime)

        # 总损失
        loss = pred_loss + 0.7 * recon_loss

        loss.backward()
        self.optimizer.step()

        return loss.item()

    def compute_loss(self, x_t, x_t1, u_t_minus1, u_t):
        # Control influence
        B_u_t_minus1 = torch.zeros_like(x_t, device=device)
        B_u_t_minus1[:, self.control_positions] = u_t_minus1

        B_u_t = torch.zeros_like(x_t1, device=device)
        B_u_t[:, self.control_positions] = u_t

        # Pure system states
        x_t_prime = x_t - B_u_t_minus1
        x_t1_prime = x_t1 - B_u_t

        # Encoding and predictions
        y_t = self.encoder(x_t_prime)
        y_t1 = self.encoder(x_t1_prime)
        y_t1_pred = torch.matmul(y_t, self.A.T)

        # Prediction and reconstruction loss
        pred_loss = nn.MSELoss()(y_t1_pred, y_t1)
        x_t_recon = self.decoder(y_t)
        recon_loss = nn.MSELoss()(x_t_recon, x_t_prime)

        # Total loss
        loss = pred_loss + 0.7 * recon_loss
        return loss.item()

    def compute_control_mpc(self, y_t, y_target, N, Q, R):
        # y_t: Current embedded state, shape (1, P)
        # y_target: Target embedded state, shape (1, P)
        # N: Prediction horizon
        # Q: State weighting matrix, shape (P, P)
        # R: Control weighting matrix, shape (M, M)

        import cvxpy as cp

        # Convert tensors to numpy arrays
        y_t_np = y_t.detach().cpu().numpy().flatten()
        y_target_np = y_target.detach().cpu().numpy().flatten()

        # Get dimensions
        P = self.A.shape[0]
        M = len(self.control_positions)

        # Define optimization variables
        u = cp.Variable((N, M))
        y = cp.Variable((N + 1, P))

        # Define the cost function
        cost = 0
        constraints = [y[0] == y_t_np]

        for k in range(N):
            # Dynamics constraint
            B_u_k = np.zeros(P)
            B_u_k[:M] = u[k].value
            constraints += [y[k + 1] == self.A.detach().cpu().numpy() @ y[k] + B_u_k]

            # Cost function
            cost += cp.quad_form(y[k + 1] - y_target_np, Q) + cp.quad_form(u[k], R)

            # Optionally, add control input constraints
            # constraints += [cp.norm(u[k], 'inf') <= u_max]

        # Solve the optimization problem
        prob = cp.Problem(cp.Minimize(cost), constraints)
        prob.solve(solver=cp.OSQP)

        # Extract the first control input
        u_optimal = u.value[0]

        return u_optimal  # numpy array, shape (M,)


def main():
    # 设备配置
    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # PDE parameters
    nx = 50  # Number of spatial nodes
    dx = 1.0 / nx  # Spatial step size
    alpha = 0.1  # Thermal diffusivity coefficient
    M = 25  # Number of controlled nodes

    # Control positions: every other point
    control_positions = np.arange(0, nx, 2)  # Positions 0, 2, ..., 48

    pde = PDE(nx, dx, alpha, M)

    # 生成训练数据
    num_samples = 3000
    time_steps = 30
    dt = 0.01
    x_t_list = []
    x_t1_list = []
    u_t_minus1_list = []
    u_t_list = []

    for _ in range(num_samples):
        # 随机初始温度分布
        T0 = np.random.rand(nx)

        # Random control sequence
        u_sequence = np.random.randn(time_steps + 1, M) * 0.1  # Control inputs in [-0.1, 0.1]

        # Simulate the system
        t_span = [0, dt * time_steps]
        T_sequence = pde.simulate(T0, u_sequence, t_span)  # Shape (time_steps + 1, nx)

        # 构建训练样本
        x_t_list.append(T_sequence[1:-2, :])  # x(t)
        x_t1_list.append(T_sequence[2:-1, :])  # x(t+1)
        u_t_minus1_list.append(u_sequence[:-2, :])
        u_t_list.append(u_sequence[1:-1, :])

    # 转换为张量
    x_t = torch.tensor(np.concatenate(x_t_list, axis=0), dtype=torch.float32)
    x_t1 = torch.tensor(np.concatenate(x_t1_list, axis=0), dtype=torch.float32)
    u_t_minus1 = torch.tensor(np.concatenate(u_t_minus1_list, axis=0), dtype=torch.float32)
    u_t = torch.tensor(np.concatenate(u_t_list, axis=0), dtype=torch.float32)

    # 数据归一化
    x_mean = x_t.mean(dim=0)
    x_std = x_t.std(dim=0) + 1e-8  # 防止除以零
    x_t = (x_t - x_mean) / x_std
    x_t1 = (x_t1 - x_mean) / x_std

    # 将数据移动到设备上
    x_t = x_t.to(device)
    x_t1 = x_t1.to(device)
    u_t_minus1 = u_t_minus1.to(device)
    u_t = u_t.to(device)

    # 数据集和数据加载器
    dataset = data.TensorDataset(x_t, x_t1, u_t_minus1, u_t)

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
    model = KoopmanModel(nx, M, hidden_dim, P, control_positions)

    # 训练模型
    num_epochs = 20
    patience = 20
    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        # 设置训练模式
        model.encoder.train()
        model.decoder.train()
        total_loss = 0
        for batch_x_t, batch_x_t1, batch_u_t_minus1, batch_u_t in train_dataloader:
            loss = model.train_step(batch_x_t, batch_x_t1, batch_u_t_minus1, batch_u_t)
            total_loss += loss
        avg_train_loss = total_loss / len(train_dataloader)

        # 验证
        model.encoder.eval()
        model.decoder.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x_t, batch_x_t1, batch_u_t_minus1, batch_u_t in val_dataloader:
                loss = model.compute_loss(batch_x_t, batch_x_t1, batch_u_t_minus1, batch_u_t)
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
        batch_x_t, batch_x_t1, batch_u_t_minus1, batch_u_t = next(iter(val_dataloader))

        B_u_t_minus1 = torch.zeros_like(batch_x_t, device=device)
        B_u_t_minus1[:, control_positions] = batch_u_t_minus1  # Control influence
        x_t1_prime = batch_x_t1 - B_u_t_minus1  # x'(t) = x(t) - B*u(t-1)

        # 编码当前状态
        y_t = model.encoder(x_t1_prime)

        # 线性预测
        y_t1_pred = torch.matmul(y_t, model.A.T)

        # 解码
        x_t1_pred_prime = model.decoder(y_t1_pred)

        B_u = torch.zeros_like(batch_x_t1, device=device)
        B_u[:, control_positions] = batch_u_t  # Control influence
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

    # 提取模型参数
    A_np = model.A.detach().cpu().numpy()
    overline_B = np.zeros((P, M))
    for k, i in enumerate(control_positions):
        overline_B[i, k] = 1.0

    # 控制目标
    T_target = np.zeros(nx)
    T_target[:nx // 2] = 1.0  # 目标温度分布

    # 初始状态
    T_init = np.zeros(nx)
    T_init[nx // 2:] = 1.0

    # 仿真控制过程
    time_steps = 150
    T = np.zeros((time_steps + 1, nx))
    T[0, :] = T_init
    u_sequence = np.zeros((time_steps, M))

    # 编码目标状态
    x_target = torch.tensor((T_target - x_mean.cpu().numpy()) / x_std.cpu().numpy(), dtype=torch.float32).unsqueeze(
        0).to(device)
    y_target = model.encoder(x_target)

    # Define MPC parameters
    N = 20  # Prediction horizon
    Q = np.eye(P)  # State weighting matrix
    R = np.eye(M)  # Control weighting matrix

    # Initialize u_t_minus1
    u_t_minus1 = np.zeros(M)  # Shape (M,)

    for t in range(time_steps):
        # Normalize and subtract control influence
        x_t = torch.tensor((T[t, :] - x_mean.cpu().numpy()) / x_std.cpu().numpy(),
                           dtype=torch.float32).unsqueeze(0).to(device)
        B_u_t_minus1 = torch.zeros_like(x_t, device=device)
        B_u_t_minus1[:, control_positions] = torch.tensor(u_t_minus1,
                                                          dtype=torch.float32)  # Control influence from previous time step
        x_t_prime = x_t - B_u_t_minus1

        # Encode the adjusted state
        y_t = model.encoder(x_t_prime)

        # Compute control input using MPC
        u_bar = model.compute_control_mpc(y_t, y_target, N, Q, R)  # Shape (M,)

        # Update u_t_minus1 for the next iteration
        u_t_minus1 = u_bar.copy()

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
    plt.title('Temperature Distribution Over Time with MPC Control')
    plt.show()


if __name__ == "__main__":
    main()
