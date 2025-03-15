from scipy.linalg import solve_discrete_are
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import scipy.linalg
import torch.utils.data as data
import matplotlib.pyplot as plt
from scipy.optimize import minimize

class PDE2D:
    def __init__(self, nx, ny, dx, dy, D, r, control_density):
        self.nx = nx  # x方向的节点数
        self.ny = ny  # y方向的节点数
        self.dx = dx  # x方向的空间步长
        self.dy = dy  # y方向的空间步长
        self.D = D    # 扩散系数
        self.r = r    # 反应速率
        self.dt = 0.001
        # 构建2D的Laplacian算子
        self.L = self.build_laplacian(nx, ny, dx, dy)
        # 控制位置，每隔一定间隔设置一个控制点
        self.control_positions = []
        for i in range(self.nx):
            for j in range(self.ny):
                if i % int(1 / control_density) == 0 and j % int(1 / control_density) == 0:
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

    def simulate(self, T0, u_sequence, t_span):
        """
        Simulate the system using explicit Euler discretization with safeguards.
        """
        time_steps = len(u_sequence)
        dt = (t_span[1] - t_span[0]) / time_steps
        N = self.nx * self.ny
        T = np.zeros((time_steps + 1, N))
        T[0, :] = T0.flatten()

        for t in range(time_steps):
            u_t = u_sequence[t, :]  # Shape (M,)

            # Free evolution: x(t+1) = x(t) + dt * L * x(t)
            laplacian_term = np.dot(self.L, T[t, :])
            growth_term = self.r * T[t, :] * (1 - T[t, :])  # Logistic growth

            # Combine terms
            free_evolution = T[t, :] + dt * (laplacian_term + growth_term)

            # Control influence: B * u(t)
            control_influence = np.dot(self.B, u_t)

            # Total state update
            T[t + 1, :] = free_evolution + dt * control_influence


        return T  # Shape (time_steps + 1, nx * ny)

class Encoder(nn.Module):
    def __init__(self, nxny, M, hidden_dim, P, control_indices):
        super(Encoder, self).__init__()
        self.nxny = nxny
        self.M = M
        self.P = P
        self.hidden_dim = hidden_dim
        self.control_indices = control_indices
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
        batch_size = x.size(0)
        y = torch.zeros(batch_size, self.P, device=x.device)
        y[:, self.control_indices] = x[:, self.control_indices]
        mask = torch.ones(self.nxny, dtype=torch.bool, device=x.device)
        mask[self.control_indices] = False
        x_non_control = x[:, mask]
        y_non_control = self.net(x_non_control)
        y[:, mask] = y_non_control
        return y

class Decoder(nn.Module):
    def __init__(self, nxny, M, hidden_dim, P, control_indices):
        super(Decoder, self).__init__()
        self.nxny = nxny
        self.M = M
        self.P = P
        self.hidden_dim = hidden_dim
        self.control_indices = control_indices
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
    def __init__(self, nxny, M, hidden_dim, P, control_indices, device):
        self.M = M
        self.P = P
        self.control_indices = control_indices
        self.encoder = Encoder(nxny, M, hidden_dim, P, control_indices).to(device)
        self.decoder = Decoder(nxny, M, hidden_dim, P, control_indices).to(device)
        self.A = torch.eye(P, device=device)  # 初始化Koopman矩阵
        self.B = torch.zeros(P, M, device=device)
        for k, idx in enumerate(control_indices):
            self.B[idx, k] = 1.0  # 控制矩阵
        self.optimizer = optim.Adam(
            list(self.encoder.parameters()) +
            list(self.decoder.parameters()),
            lr=1e-3
        )

    def train_step(self, x_t, x_t1_prime):
        self.optimizer.zero_grad()
        # 编码
        y_t = self.encoder(x_t)
        y_t1 = self.encoder(x_t1_prime)
        # 预测损失
        y_t1_pred = y_t @ self.A.t()
        pred_loss = nn.MSELoss()(y_t1_pred, y_t1)
        # 重构损失
        recon_loss = 0.5 * (
                nn.MSELoss()(self.decoder(y_t), x_t) +
                nn.MSELoss()(self.decoder(y_t1), x_t1_prime)
        )
        # 总损失
        loss = pred_loss + 0.7 * recon_loss
        loss.backward()
        self.optimizer.step()
        return loss.item()


class KoopmanMPCController:
    def __init__(self, koopman_model, horizon=10, Q=None, R=None):
        self.model = koopman_model
        self.horizon = horizon
        self.Q = Q if Q is not None else np.eye(koopman_model.encoder.P)
        self.R = R if R is not None else np.eye(koopman_model.M)
        self.A = koopman_model.A.detach().cpu().numpy()
        self.B = koopman_model.B.detach().cpu().numpy()
        self.control_indices = koopman_model.control_indices

    def cost_function(self, u_seq_flat, y_current, y_target):
        cost = 0.0
        y = y_current.copy()
        u_seq = u_seq_flat.reshape(self.horizon, self.model.M)

        for t in range(self.horizon):
            # 状态误差
            state_error = y - y_target
            cost += state_error.T @ self.Q @ state_error

            # 控制成本
            u = u_seq[t]
            cost += u.T @ self.R @ u

            # 线性预测
            y = self.A @ y + self.B @ u

        return cost

    def control(self, y_current, y_target):
        # 初始猜测（前一时刻的控制序列）
        u0 = np.zeros(self.horizon * self.model.M)

        # 约束条件（控制输入范围）
        bounds = [(-1.0, 1.0) for _ in range(self.horizon * self.model.M)]

        # 求解优化问题
        res = minimize(
            self.cost_function,
            u0,
            args=(y_current, y_target),
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 50}
        )

        return res.x[:self.model.M]  # 返回当前时刻控制输入


def run_kdnn_mpc_control(pde, mpc, x_target_flat, y_target_flat, time_steps):
    nxny = pde.nx * pde.ny
    T = np.zeros((time_steps + 1, nxny))
    T[0, :] = np.random.randn(nxny) * 0.1  # 与KDNNC-LQR相同的初始条件
    control_mses = []
    control_energies = []
    total_energy = 0.0

    for t in range(time_steps):
        print(t+1)
        # 当前状态
        current_state = T[t, :].reshape(pde.nx, pde.ny)

        # 编码到Koopman空间
        with torch.no_grad():
            y_current = mpc.model.encoder(
                torch.tensor(current_state.reshape(1, 900), dtype=torch.float32).to(device)
            ).cpu().numpy().flatten()

        # 计算控制输入
        u_bar = mpc.control(y_current, y_target_flat)


        # 计算能量
        total_energy += np.sum(u_bar ** 2)
        control_energies.append(total_energy)

        # 系统仿真
        T_next = pde.simulate(
            current_state,
            np.array([u_bar]),
            [t * pde.dt, (t + 1) * pde.dt]
        )[-1]
        T[t + 1, :] = T_next

        # 计算MSE
        mse = np.mean((T_next - x_target_flat) ** 2)
        control_mses.append(mse)

    return T, control_mses, control_energies


if __name__ == "__main__":
    # 统一随机种子
    np.random.seed(0)
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # PDE参数（与KDNNC-LQR完全一致）
    nx = 30
    ny = 30
    dx = 1.0 / nx
    dy = 1.0 / ny
    D = 0.1
    r = 0.1
    control_density = 0.5
    dt = 0.001

    pde = PDE2D(nx, ny, dx, dy, D, r, control_density)
    control_indices = [i * ny + j for i, j in pde.control_positions]
    M = pde.M

    # 创建并加载训练好的Koopman模型
    hidden_dim = 512
    P = nx * ny

    checkpoint = torch.load('trained_kdnn_model.pth', map_location=device)
    model = Koopman_Model(
        nxny=checkpoint['nxny'],
        M=checkpoint['M'],
        hidden_dim=checkpoint['hidden_dim'],
        P=checkpoint['P'],
        control_indices=checkpoint['control_indices'],
        device=device
    )
    model.encoder.load_state_dict(checkpoint['encoder'])
    model.decoder.load_state_dict(checkpoint['decoder'])
    model.A = torch.tensor(checkpoint['A']).to(device)
    model.B = torch.tensor(checkpoint['B']).to(device)

    # MPC控制器配置（与LQR使用相同模型）
    horizon = 3
    Q = np.eye(P) * 1.0
    R = np.eye(M) * 0.01
    mpc = KoopmanMPCController(model, horizon=horizon, Q=Q, R=R)


    # 生成目标状态（行波解）
    def wave_solution(x, y, t, c=0.5):
        return np.exp(-c * (x + y - c * t))


    x_coords = np.linspace(0, 1, nx)
    y_coords = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x_coords, y_coords)
    T_target = wave_solution(X, Y, t=0)
    x_target_flat = T_target.reshape(1, 900)
    # 编码到Koopman空间
    with torch.no_grad():
        y_target_flat = mpc.model.encoder(
            torch.tensor(T_target.reshape(1, 900), dtype=torch.float32).to(device)
        ).cpu().numpy().flatten()

    # 运行控制仿真
    time_steps = 2000
    T, mse, energy = run_kdnn_mpc_control(pde, mpc, x_target_flat, y_target_flat, time_steps)

    # 保存数据
    np.savetxt('kdnn_mpc_mses.txt', mse)
    np.savetxt('kdnn_mpc_energies.txt', energy)

    # 可视化结果
    plt.figure(figsize=(14, 6))

    # MSE曲线
    plt.subplot(1, 2, 1)
    plt.plot(mse, label='KDNN-MPC MSE')
    plt.yscale('log')
    plt.xlabel('Time Steps')
    plt.ylabel('MSE (log scale)')
    plt.title('DKNNC-MPC Control Performance')
    plt.legend()

    # 能量消耗曲线
    plt.subplot(1, 2, 2)
    plt.plot(energy, label='Accumulated Energy', color='purple')
    plt.xlabel('Time Steps')
    plt.ylabel('Total Energy Consumption')
    plt.title('DKNNC-MPC Energy Usage')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # 最终状态对比
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(T[-1].reshape(nx, ny), cmap='jet', origin='lower')
    plt.colorbar()
    plt.title('Final State (DKNNC-MPC)')

    plt.subplot(1, 2, 2)
    plt.imshow(T_target, cmap='jet', origin='lower')
    plt.colorbar()
    plt.title('Target State')
    plt.show()