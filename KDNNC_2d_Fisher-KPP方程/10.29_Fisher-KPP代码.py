import numpy as np
from scipy.integrate import solve_ivp
import torch
import torch.nn as nn
import torch.optim as optim
import scipy.linalg
import torch.utils.data as data
import matplotlib.pyplot as plt

class PDE2D:
    def __init__(self, nx, ny, dx, dy, D, r):
        self.nx = nx  # x方向的节点数
        self.ny = ny  # y方向的节点数
        self.dx = dx  # x方向的空间步长
        self.dy = dy  # y方向的空间步长
        self.D = D    # 扩散系数
        self.r = r    # 反应速率

        # 构建2D的Laplacian算子
        self.L = self.build_laplacian(nx, ny, dx, dy)

        # 控制位置，每隔2个节点设置一个控制点
        self.control_positions = []
        for i in range(0, nx, 2):
            for j in range(0, ny, 2):
                self.control_positions.append((i, j))
        self.M = len(self.control_positions)

        # 控制影响矩阵B，大小为 (nx*ny, M)
        self.B = np.zeros((self.nx * self.ny, self.M))
        for k, (i, j) in enumerate(self.control_positions):
            idx = i * self.ny + j
            self.B[idx, k] = 1.0

    def build_laplacian(self, nx, ny, dx, dy):
        N = nx * ny
        L = np.zeros((N, N))
        for i in range(nx):
            for j in range(ny):
                idx = i * ny + j
                # 中心元素
                L[idx, idx] = -2 * self.D * (1 / dx ** 2 + 1 / dy ** 2)
                # Neumann 边界条件（边界处导数为零）
                # 左邻点
                if i > 0:
                    idx_left = (i - 1) * ny + j
                    L[idx, idx_left] += self.D / dx ** 2
                else:
                    # 边界处的零通量
                    L[idx, idx] += self.D / dx ** 2
                # 右邻点
                if i < nx - 1:
                    idx_right = (i + 1) * ny + j
                    L[idx, idx_right] += self.D / dx ** 2
                else:
                    L[idx, idx] += self.D / dx ** 2
                # 下邻点
                if j > 0:
                    idx_down = i * ny + (j - 1)
                    L[idx, idx_down] += self.D / dy ** 2
                else:
                    L[idx, idx] += self.D / dy ** 2
                # 上邻点
                if j < ny - 1:
                    idx_up = i * ny + (j + 1)
                    L[idx, idx_up] += self.D / dy ** 2
                else:
                    L[idx, idx] += self.D / dy ** 2
        return L

    def simulate(self, u0, u_sequence, t_span):
        time_steps = len(u_sequence)
        dt = (t_span[1] - t_span[0]) / time_steps
        times = np.linspace(t_span[0], t_span[1], time_steps + 1)

        N = self.nx * self.ny
        u = np.zeros((time_steps + 1, N))
        u[0, :] = u0.flatten()

        for t in range(time_steps):
            u_t = u_sequence[t, :]  # Shape (M,)

            sol = solve_ivp(
                fun=lambda t, u_flat: self.odefun(t, u_flat, u_t),
                t_span=[times[t], times[t + 1]],
                y0=u[t, :],
                method='RK45'
            )
            u[t + 1, :] = sol.y[:, -1]

        return u  # Shape (time_steps + 1, nx*ny)

    def odefun(self, t, u_flat, u_t):
        # 控制影响
        B_u = np.dot(self.B, u_t)  # Shape (N,)
        # du/dt = L * u + r * u * (1 - u) + B_u
        dudt = np.dot(self.L, u_flat) + self.r * u_flat * (1 - u_flat) + B_u
        return dudt


class Encoder(nn.Module):
    def __init__(self, nxny, M, hidden_dim, P, control_indices):
        super(Encoder, self).__init__()
        self.nxny = nxny  # 原始状态维度
        self.M = M        # 控制节点数
        self.P = P        # 嵌入空间维度
        self.hidden_dim = hidden_dim
        self.control_indices = control_indices  # 控制位置的索引

        # 非线性映射的神经网络
        self.net = nn.Sequential(
            nn.Linear(nxny - M, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, P - M)
        )

    def forward(self, x):
        # x: shape (batch_size, nxny)
        batch_size = x.size(0)
        y = torch.zeros(batch_size, self.P, device=x.device)

        # 对控制位置执行身份映射
        y[:, self.control_indices] = x[:, self.control_indices]

        # 非控制位置
        mask = torch.ones(self.nxny, dtype=torch.bool, device=x.device)
        mask[self.control_indices] = False
        x_non_control = x[:, mask]  # 提取非控制位置的x

        # 对其余位置执行非线性映射
        y_non_control = self.net(x_non_control)
        y[:, mask] = y_non_control
        return y


class Decoder(nn.Module):
    def __init__(self, nxny, M, hidden_dim, P, control_indices):
        super(Decoder, self).__init__()
        self.nxny = nxny  # 原始状态维度
        self.M = M        # 控制节点数
        self.P = P        # 嵌入空间维度
        self.hidden_dim = hidden_dim
        self.control_indices = control_indices  # 控制位置的索引

        # 非线性解码的神经网络
        self.net = nn.Sequential(
            nn.Linear(P - M, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, nxny - M)
        )

    def forward(self, y):
        # y: shape (batch_size, P)
        batch_size = y.size(0)
        x = torch.zeros(batch_size, self.nxny, device=y.device)

        # 对控制位置执行身份映射
        x[:, self.control_indices] = y[:, self.control_indices]

        mask = torch.ones(self.nxny, dtype=torch.bool, device=y.device)
        mask[self.control_indices] = False

        # 对其余位置执行非线性解码
        y_non_control = y[:, mask]
        x_non_control = self.net(y_non_control)

        x[:, mask] = x_non_control
        return x


class Koopman_Model:
    def __init__(self, nxny, M, hidden_dim, P, control_indices):
        self.nxny = nxny  # 原始状态维度
        self.M = M        # 控制数
        self.P = P        # 嵌入空间维度
        self.control_indices = control_indices  # 控制位置

        # 编码器和解码器
        self.encoder = Encoder(nxny, M, hidden_dim, P, control_indices).to(device)
        self.decoder = Decoder(nxny, M, hidden_dim, P, control_indices).to(device)

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
        # 重构损失
        recon_loss = 0.5 * (nn.MSELoss()(self.decoder(y_t), x_t) + nn.MSELoss()(self.decoder(y_t1), x_t1_prime))
        # 总损失
        loss = pred_loss + 0.7 * recon_loss

        loss.backward()
        self.optimizer.step()

        return loss.item()

    def compute_loss(self, x_t, x_t1_prime):
        # 编码和预测
        y_t = self.encoder(x_t)
        y_t1 = self.encoder(x_t1_prime)
        # 线性预测
        y_t1_pred = torch.matmul(y_t, self.A.T)
        # 预测损失
        pred_loss = nn.MSELoss()(y_t1_pred, y_t1)
        # 重构损失
        recon_loss = 0.5 * (nn.MSELoss()(self.decoder(y_t), x_t) + nn.MSELoss()(self.decoder(y_t1), x_t1_prime))
        # 总损失
        loss = pred_loss + 0.7 * recon_loss
        return loss.item()

    def compute_control(self, y_t, y_target, K):
        # 计算控制输入
        y_error = y_t - y_target  # 状态误差
        u_bar = -np.matmul(K, y_error.detach().cpu().numpy().T).flatten()  # Shape (M,)
        return u_bar  # Numpy array, shape (M,)

    def design_lqr_controller(self):
        # Discrete-time LQR controller design
        A = self.A.detach().cpu().numpy()
        overline_B = np.zeros((self.P, self.M))
        for k, idx in enumerate(self.control_indices):
            overline_B[idx, k] = 1.0
        B = overline_B
        Q = np.eye(A.shape[0])  # State weight matrix
        R = np.eye(B.shape[1])  # Control input weight matrix

        # Solve discrete-time Algebraic Riccati equation
        P = scipy.linalg.solve_discrete_are(A, B, Q, R)

        # Compute LQR gain K
        K = np.linalg.inv(B.T @ P @ B + R) @ (B.T @ P @ A)

        return K  # Shape (M, P)


def main():
    # 配置设备
    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # PDE参数
    nx = 30  # x方向的节点数
    ny = 30  # y方向的节点数
    dx = 1.0 / nx  # x方向的空间步长
    dy = 1.0 / ny  # y方向的空间步长
    D = 0.1       # 扩散系数
    # r为0.1 看起来可以实现一定的控制，r为1时，反应速率太快，不适合做到合适的控制
    r = 0.5        # 反应速率

    pde = PDE2D(nx, ny, dx, dy, D, r)

    control_positions = pde.control_positions  # 列表 (i, j)
    M = len(control_positions)  # 控制数

    # 将控制位置转换为展平数组中的索引
    control_indices = [i * ny + j for i, j in control_positions]

    # 生成训练数据
    num_samples = 1000
    time_steps = 60
    dt = 0.01
    x_t_list = []
    x_t1_list = []
    u_t_list = []

    np.random.seed(0)  # 为了可重复性

    for _ in range(num_samples):
        # 随机初始状态u0，范围在[0,1]
        u0 = np.random.rand(nx, ny)

        # 生成控制输入
        control_input_scale = 0.05  # 可根据需要调整
        u_sequence = np.random.rand(time_steps + 1, M) * control_input_scale

        # 模拟系统
        t_span = [0, dt * time_steps]
        u_sequence_full = pde.simulate(u0, u_sequence, t_span)  # Shape (time_steps + 1, nx*ny)

        # 构建训练样本
        x_t_list.append(u_sequence_full[1:-2, :])  # x(t)
        x_t1_list.append(u_sequence_full[2:-1, :])  # x(t+1)
        u_t_list.append(u_sequence[1:-1, :])

    # 转换为张量
    x_t = torch.tensor(np.concatenate(x_t_list, axis=0), dtype=torch.float32)
    x_t1 = torch.tensor(np.concatenate(x_t1_list, axis=0), dtype=torch.float32)
    u_t = torch.tensor(np.concatenate(u_t_list, axis=0), dtype=torch.float32)

    # 将数据移到设备上
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
    nxny = nx * ny
    hidden_dim = 512
    P = nxny
    model = Koopman_Model(nxny, M, hidden_dim, P, control_indices)

    # 训练模型
    num_epochs = 100
    patience = 10
    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        # 设定为训练模式
        model.encoder.train()
        model.decoder.train()
        total_loss = 0
        for batch_x_t, batch_x_t1, batch_u_t in train_dataloader:
            B_u_t = torch.zeros_like(batch_x_t1, device=device)
            B_u_t[:, control_indices] = batch_u_t
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
                B_u_t = torch.zeros_like(batch_x_t1, device=device)
                B_u_t[:, control_indices] = batch_u_t
                batch_x_t1_prime = batch_x_t1 - B_u_t
                loss = model.compute_loss(batch_x_t, batch_x_t1_prime)
                val_loss += loss
        avg_val_loss = val_loss / len(val_dataloader)

        print(f"Epoch {epoch + 1}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")

        # 早停策略
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping!")
                break

    # After training, compare prediction with ground truth
    with torch.no_grad():
        batch_x_t, batch_x_t1, batch_u_t = next(iter(val_dataloader))

        # Encode current state
        y_t = model.encoder(batch_x_t)

        # Linear prediction
        y_t1_pred = torch.matmul(y_t, model.A.T)

        # Decode
        x_t1_pred_prime = model.decoder(y_t1_pred)

        B_u = torch.zeros_like(batch_x_t1, device=device)
        B_u[:, control_indices] = batch_u_t  # Control influence
        # Add control influence to get predicted x(t+1)
        x_t1_pred = x_t1_pred_prime + B_u

        # Compute prediction error
        pred_error = nn.MSELoss()(x_t1_pred, batch_x_t1)
        print(f"Prediction error on validation batch: {pred_error.item():.6f}")

        # Visualization
        idx = 0  # Visualize the first sample
        T_true = batch_x_t1[idx].cpu().numpy().reshape(nx, ny)
        T_pred = x_t1_pred[idx].cpu().numpy().reshape(nx, ny)

        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(T_true, cmap='jet', origin='lower')
        plt.colorbar()
        plt.title('True x(t+1)')

        plt.subplot(1, 3, 2)
        plt.imshow(T_pred, cmap='jet', origin='lower')
        plt.colorbar()
        plt.title('Predicted x(t+1)')

        plt.subplot(1, 3, 3)
        plt.imshow(T_true - T_pred, cmap='bwr', origin='lower')
        plt.colorbar()
        plt.title('Prediction Error')
        plt.show()

    # Design LQR controller
    K = model.design_lqr_controller()

    # TODO  从这里开始有区别  另外就是反应速率和扩散速率的问题
    # Control target
    # 控制目标
    # x_target = np.ones((nx, ny))  # 全1状态   稳态解
    # x_target = np.zeros((nx, ny))  # 全0状态  稳态解
    # x_target[nx//4:3*nx//4, ny//4:3*ny//4] = 1  # 局部高值区域
    # 行波解的波形函数
    def wave_solution(x, y, t, c=1.0):
        return np.exp(-c * (x + y - c * t))

    # 生成行波解作为目标状态
    x_coords = np.linspace(0, 1, nx)
    y_coords = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x_coords, y_coords)
    x_target = wave_solution(X, Y, t=0)  # 行波解

    # Initial state
    u0 = np.random.randn(nx, ny)*0.1
    # u0 = np.zeros((nx, ny))
    # u0[nx//4:3*nx//4, nx//2:] = 1

    # 模拟时间
    time_steps = 600
    N = nx * ny
    u = np.zeros((time_steps + 1, N))
    u[0, :] = u0.flatten()
    u_t_sequence = np.zeros((time_steps, M))

    # Encode target state
    x_target = torch.tensor(x_target.flatten(), dtype=torch.float32).unsqueeze(0).to(device)
    y_target = model.encoder(x_target)

    # 在模拟闭环系统的循环中，更改变量名
    for k in range(time_steps):
        x_t_np = u[k, :]
        x_t = torch.tensor(x_t_np, dtype=torch.float32).unsqueeze(0).to(device)

        # Encode the state
        y_t = model.encoder(x_t)

        # 计算控制输入
        u_bar = model.compute_control(y_t, y_target, K)  # Shape (M,)

        # 将控制输入存储
        u_t_sequence[k, :] = u_bar

        # Simulate next time step
        u_t1 = pde.simulate(x_t_np.reshape(nx, ny), np.array([u_bar]), [k * dt, (k + 1) * dt])
        u[k + 1, :] = u_t1[-1]

    # Visualization of results
    u_full = u.reshape(time_steps + 1, nx, ny)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(u_full[0], cmap='jet', origin='lower')
    plt.colorbar()
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('Temperature Distribution at Initial Time')

    plt.subplot(1, 2, 2)
    plt.imshow(u_full[-1], cmap='jet', origin='lower')
    plt.colorbar()
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('Temperature Distribution at Final Time')
    plt.show()


if __name__ == "__main__":
    main()


