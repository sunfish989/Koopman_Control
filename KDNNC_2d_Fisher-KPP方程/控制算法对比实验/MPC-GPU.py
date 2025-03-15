import cupy as cp  # 替代 NumPy
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
        self.B = cp.zeros((self.nx * self.ny, self.M))  # 使用 CuPy 数组
        for k, (i, j) in enumerate(self.control_positions):
            idx = i * self.ny + j
            self.B[idx, k] = 1.0

    def build_laplacian(self, nx, ny, dx, dy):
        N = nx * ny
        L = cp.zeros((N, N))  # 使用 CuPy 数组
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
        T = cp.zeros((time_steps + 1, N))  # 使用 CuPy 数组
        T[0, :] = cp.array(T0.flatten())  # 将输入转换为 CuPy 数组

        for t in range(time_steps):
            u_t = cp.array(u_sequence[t, :])  # 转换为 CuPy 数组
            laplacian_term = cp.dot(self.L, T[t, :])
            growth_term = self.r * T[t, :] * (1 - T[t, :])
            free_evolution = T[t, :] + dt * (laplacian_term + growth_term)
            control_influence = cp.dot(self.B, u_t)
            T[t + 1, :] = free_evolution + dt * control_influence

        return T.get()  # 返回 NumPy 数组以便后续操作


class NonlinearMPC:
    def __init__(self, pde, horizon=10, Q=None, R=None):
        self.pde = pde
        self.horizon = horizon
        self.nxny = pde.nx * pde.ny
        self.M = pde.M

        # 状态和控制权重矩阵
        self.Q = Q if Q is not None else cp.eye(self.nxny)  # 使用 CuPy 数组
        self.R = R if R is not None else cp.eye(self.M)     # 使用 CuPy 数组

    def cost_function(self, u_seq_flat, x_current_flat, x_target_flat):
        cost = 0.0
        x = cp.array(x_current_flat)  # 转换为 CuPy 数组
        u_seq = cp.array(u_seq_flat).reshape(self.horizon, self.M)

        for t in range(self.horizon):
            state_error = x.reshape(1, self.pde.nx * self.pde.ny) - cp.array(x_target_flat).reshape(1, self.pde.nx * self.pde.ny)
            cost += state_error.flatten() @ self.Q @ state_error.flatten()
            control_input = u_seq[t]
            cost += control_input @ self.R @ control_input
            x_next = self.pde.simulate(
                x.reshape(self.pde.nx, self.pde.ny),
                cp.array([control_input]),
                [0, self.pde.dt]
            )[-1]
            x = x_next.flatten()

        return float(cost)  # 返回标量值

    def control(self, x_current, x_target):
        # 初始猜测（前一时刻的控制序列）
        u0 = cp.zeros(self.horizon * self.M)

        # 优化约束和边界
        bounds = [(-1.0, 1.0) for _ in range(self.horizon * self.M)]

        # 最小化成本函数
        result = minimize(
            self.cost_function,
            u0.get(),  # 将 CuPy 数组转换为 NumPy 数组以供 SciPy 使用
            args=(x_current.flatten(), x_target.flatten()),
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 50}
        )

        return result.x[:self.M]  # 返回当前时刻的控制输入


def run_mpc_control(pde, mpc, x_target_flat, time_steps):
    nxny = pde.nx * pde.ny
    T = cp.zeros((time_steps + 1, nxny))  # 使用 CuPy 数组
    T[0, :] = cp.random.randn(nxny) * 0.1  # 随机初始条件
    control_mses = []
    control_energies = []
    total_energy = 0.0

    for t in range(time_steps):
        print(f"Time step: {t+1}/{time_steps}")
        current_state = T[t, :].reshape(pde.nx, pde.ny)
        u = mpc.control(current_state.get(), x_target_flat.reshape(pde.nx, pde.ny).get())

        # 计算控制能量
        control_energy = cp.sum(cp.array(u) ** 2)
        total_energy += control_energy
        control_energies.append(total_energy.get())

        # 应用控制输入
        T_next = pde.simulate(current_state.get(), cp.array([u]), [t * pde.dt, (t + 1) * pde.dt])[-1]
        T[t + 1, :] = cp.array(T_next)

        # 计算MSE
        mse = cp.mean((T_next.reshape(pde.nx, pde.ny) - x_target_flat.reshape(pde.nx, pde.ny)) ** 2)
        control_mses.append(mse.get())

    return T.get(), control_mses, control_energies


if __name__ == "__main__":
    # 初始化PDE系统
    cp.random.seed(0)
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
        return cp.exp(-c * (x + y - c * t))

    x_coords = cp.linspace(0, 1, nx)
    y_coords = cp.linspace(0, 1, ny)
    X, Y = cp.meshgrid(x_coords, y_coords)
    T_target = wave_solution(X, Y, t=0)
    x_target_flat = T_target.flatten()

    # MPC参数
    horizon = 3  # 预测时域
    Q = cp.eye(nx * ny) * 1.0  # 状态权重
    R = cp.eye(pde.M) * 0.01   # 控制权重

    # 创建MPC控制器
    mpc = NonlinearMPC(pde, horizon=horizon, Q=Q, R=R)

    # 运行控制仿真
    time_steps = 4000
    T_mpc, mse_mpc, energy_mpc = run_mpc_control(pde, mpc, x_target_flat, time_steps)

    # 保存数据到文件
    cp.savetxt('mpc_control_mses.txt', cp.array(mse_mpc))
    cp.savetxt('mpc_control_energies.txt', cp.array(energy_mpc))

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
    plt.imshow(T_target.get(), cmap='jet', origin='lower')  # 转换为 NumPy 数组进行可视化
    plt.colorbar()
    plt.title('Target State')
    plt.show()