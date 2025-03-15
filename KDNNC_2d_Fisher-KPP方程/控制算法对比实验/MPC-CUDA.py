import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

class PDE2D:
    def __init__(self, nx, ny, dx, dy, D, r, control_density):
        self.nx = nx  # x方向的节点数
        self.ny = ny  # y方向的节点数
        self.dx = dx  # x方向的空间步长
        self.dy = dy  # y方向的空间步长
        self.dt = 0.001
        self.D = D    # 扩散系数
        self.r = r    # 反应速率
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
        self.B = torch.zeros((self.nx * self.ny, self.M), device='cuda', dtype=torch.float64)  # 使用 double 类型
        for k, (i, j) in enumerate(self.control_positions):
            idx = i * self.ny + j
            self.B[idx, k] = 1.0

    def build_laplacian(self, nx, ny, dx, dy):
        N = nx * ny
        L = torch.zeros((N, N), device='cuda', dtype=torch.float64)  # 使用 double 类型
        for i in range(nx):
            for j in range(ny):
                idx = i * ny + j
                L[idx, idx] = -2 * self.D * (1 / dx ** 2 + 1 / dy ** 2)
                # Neumann 边界条件（边界处导数为零）
                # 左邻点
                if i > 0:
                    idx_left = (i - 1) * ny + j
                    L[idx, idx_left] += self.D / dx ** 2
                else:
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
        time_steps = len(u_sequence)
        dt = (t_span[1] - t_span[0]) / time_steps
        N = self.nx * self.ny
        T = torch.zeros((time_steps + 1, N), device='cuda', dtype=torch.float64)  # 使用 double 类型
        T[0, :] = torch.tensor(T0.flatten(), device='cuda', dtype=torch.float64)  # 初始条件转移到 GPU

        for t in range(time_steps):
            u_t = torch.tensor(u_sequence[t, :], device='cuda', dtype=torch.float64)  # 控制输入转移到 GPU
            laplacian_term = torch.matmul(self.L, T[t, :])
            growth_term = self.r * T[t, :] * (1 - T[t, :])
            free_evolution = T[t, :] + dt * (laplacian_term + growth_term)
            control_influence = torch.matmul(self.B, u_t)
            T[t + 1, :] = free_evolution + dt * control_influence

        return T.cpu().numpy()  # 返回 CPU 上的 NumPy 数组


class NonlinearMPC:
    def __init__(self, pde, horizon=10, Q=None, R=None):
        self.pde = pde
        self.horizon = horizon
        self.nxny = pde.nx * pde.ny
        self.M = pde.M

        # 状态和控制权重矩阵
        self.Q = Q if Q is not None else torch.eye(self.nxny, device='cuda', dtype=torch.float64)  # 使用 double 类型
        self.R = R if R is not None else torch.eye(self.M, device='cuda', dtype=torch.float64)     # 使用 double 类型

    def cost_function(self, u_seq_flat, x_current_flat, x_target_flat):
        cost = 0.0
        x = torch.tensor(x_current_flat, device='cuda', dtype=torch.float64)  # 转换为 PyTorch 张量
        u_seq = torch.tensor(u_seq_flat, device='cuda', dtype=torch.float64).reshape(self.horizon, self.M)

        for t in range(self.horizon):
            state_error = x.reshape(1, self.pde.nx * self.pde.ny) - torch.tensor(x_target_flat, device='cuda', dtype=torch.float64).reshape(1, self.pde.nx * self.pde.ny)
            cost += state_error.flatten() @ self.Q @ state_error.flatten()
            control_input = u_seq[t]
            cost += control_input @ self.R @ control_input
            x_next = self.pde.simulate(
                x.reshape(self.pde.nx, self.pde.ny).cpu().numpy(),
                u_seq[t].cpu().numpy()[None, :],
                [0, self.pde.dt]
            )[-1]
            x = torch.tensor(x_next.flatten(), device='cuda', dtype=torch.float64)

        return float(cost)  # 返回标量值

    def control(self, x_current, x_target):
        # 初始猜测（前一时刻的控制序列）
        u0 = torch.zeros(self.horizon * self.M, device='cuda', dtype=torch.float64)  # 使用 double 类型

        # 优化约束和边界
        bounds = [(-1.0, 1.0) for _ in range(self.horizon * self.M)]

        # 最小化成本函数
        result = minimize(
            self.cost_function,
            u0.cpu().double().numpy(),  # 将 PyTorch 张量转换为 NumPy 数组以供 SciPy 使用
            args=(x_current.flatten(), x_target.flatten()),
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 50}
        )

        return result.x[:self.M]  # 返回当前时刻的控制输入


def run_mpc_control(pde, mpc, x_target_flat, time_steps):
    nxny = pde.nx * pde.ny
    T = torch.zeros((time_steps + 1, nxny), device='cuda', dtype=torch.float64)  # 在 GPU 上创建张量
    T[0, :] = torch.randn(nxny, device='cuda', dtype=torch.float64) * 0.1  # 随机初始条件
    control_mses = []
    control_energies = []
    total_energy = 0.0

    for t in range(time_steps):
        print(f"Time step: {t+1}/{time_steps}")
        current_state = T[t, :].reshape(pde.nx, pde.ny)
        u = mpc.control(current_state.cpu().numpy(), x_target_flat.reshape(pde.nx, pde.ny).cpu().numpy())

        # 计算控制能量
        control_energy = torch.sum(torch.tensor(u, device='cuda', dtype=torch.float64) ** 2)
        total_energy += control_energy
        control_energies.append(total_energy.cpu().item())

        # 应用控制输入
        T_next = pde.simulate(current_state.cpu().numpy(), u[None, :], [t * pde.dt, (t + 1) * pde.dt])[-1]
        T[t + 1, :] = torch.tensor(T_next.flatten(), device='cuda', dtype=torch.float64)

        # 计算MSE
        mse = torch.mean((torch.tensor(T_next.flatten(), device='cuda', dtype=torch.float64) - torch.tensor(x_target_flat.flatten(), device='cuda', dtype=torch.float64)) ** 2)
        control_mses.append(mse.cpu().item())

    return T.cpu().numpy(), control_mses, control_energies


if __name__ == "__main__":
    # 初始化PDE系统
    torch.manual_seed(0)
    nx = 30
    ny = 30
    dx = 1.0 / nx
    dy = 1.0 / ny
    D = 0.1
    r = 0.1
    control_density = 0.5
    pde = PDE2D(nx, ny, dx, dy, D, r, control_density)

    # 生成目标状态（行波解）
    def wave_solution(x, y, t, c=0.5):
        return torch.exp(-c * (x + y - c * t))

    x_coords = torch.linspace(0, 1, nx, device='cuda', dtype=torch.float64)
    y_coords = torch.linspace(0, 1, ny, device='cuda', dtype=torch.float64)
    X, Y = torch.meshgrid(x_coords, y_coords, indexing='ij')
    T_target = wave_solution(X, Y, t=0)
    x_target_flat = T_target.flatten()

    # MPC参数
    horizon = 3  # 预测时域
    Q = torch.eye(nx * ny, device='cuda', dtype=torch.float64) * 1.0  # 状态权重
    R = torch.eye(pde.M, device='cuda', dtype=torch.float64) * 0.01   # 控制权重

    # 创建MPC控制器
    mpc = NonlinearMPC(pde, horizon=horizon, Q=Q, R=R)

    # 运行控制仿真
    time_steps = 4000
    T_mpc, mse_mpc, energy_mpc = run_mpc_control(pde, mpc, x_target_flat, time_steps)

    # 保存数据到文件（保存为 .txt 文件）
    np.savetxt('mpc_control_mses.txt', np.array(mse_mpc))  # MSE 数据
    np.savetxt('mpc_control_energies.txt', np.array(energy_mpc))  # 能量消耗数据

    # 可视化结果
    plt.figure(figsize=(12, 6))

    # MSE曲线
    plt.subplot(1, 2, 1)
    plt.plot(mse_mpc, label='MPC Control MSE')
    plt.yscale('log')
    plt.xlabel('Time Steps')
    plt.ylabel('MSE (log scale)')
    plt.title('Nonlinear MPC Performance')
    plt.legend()

    # 能量消耗曲线
    plt.subplot(1, 2, 2)
    plt.plot(energy_mpc, label='Accumulated Energy', color='green')
    plt.xlabel('Time Steps')
    plt.ylabel('Total Energy Consumption')
    plt.title('MPC Energy Usage')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # 最终状态可视化
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(T_mpc[-1].reshape(nx, ny), cmap='jet', origin='lower')
    plt.colorbar()
    plt.title('Final State (MPC Control)')

    plt.subplot(1, 2, 2)
    plt.imshow(T_target.cpu().numpy(), cmap='jet', origin='lower')  # 转换为 NumPy 数组进行可视化
    plt.colorbar()
    plt.title('Target State')
    plt.show()