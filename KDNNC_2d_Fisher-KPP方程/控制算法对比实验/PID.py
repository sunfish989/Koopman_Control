import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import scipy.linalg
import torch.utils.data as data
import matplotlib.pyplot as plt

class PDE2D:
    def __init__(self, nx, ny, dx, dy, D, r, control_density):
        self.nx = nx  # x方向的节点数
        self.ny = ny  # y方向的节点数
        self.dx = dx  # x方向的空间步长
        self.dy = dy  # y方向的空间步长
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



class LocalizedPIDController:
    def __init__(self, control_indices, Kp, Ki, Kd, dt, nxny):
        """
        control_indices: 控制点的全局索引列表
        nxny: 全局状态维度
        """
        self.control_indices = control_indices
        self.nxny = nxny
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.dt = dt
        self.M = len(control_indices)
        self.prev_errors = np.zeros(self.M)  # 仅存储控制点误差
        self.integral_errors = np.zeros(self.M)  # 仅积分控制点误差

    def compute_control(self, global_error):
        """
        global_error: (nxny,) 全局误差向量
        返回: (nxny,) 控制输入，仅控制点有值
        """
        # 提取控制点误差
        local_errors = global_error[self.control_indices]

        # PID计算（仅基于控制点误差）
        self.integral_errors += local_errors * self.dt
        derivative = (local_errors - self.prev_errors) / self.dt
        output = self.Kp * local_errors + self.Ki * self.integral_errors + self.Kd * derivative
        self.prev_errors = local_errors.copy()


        np.clip(output, -0.5, 0.5, out=output)  # 限制控制输入范围
        u = output
        return u


def run_pid_control(pde, pid, x_target_flat, time_steps, dt):
    nxny = pde.nx * pde.ny
    T = np.zeros((time_steps + 1, nxny))
    T[0, :] = np.random.randn(nxny) * 0.1  # 与KDNNC相同的初始条件

    control_mses = []
    control_energies = []
    total_energy = 0.0

    for t in range(time_steps):
        current_state = T[t, :].reshape(pde.nx, pde.ny)
        error = current_state.flatten() - x_target_flat
        u = pid.compute_control(error)

        # 计算控制能量
        control_energy = np.sum(u ** 2)  # L2范数平方
        total_energy += control_energy
        control_energies.append(total_energy)

        # 应用控制输入
        T_next = pde.simulate(current_state, np.array([u]),
                              [t * dt, (t + 1) * dt])[-1]
        T[t + 1, :] = T_next

        # 计算MSE
        mse = np.mean((T_next.reshape(pde.nx, pde.ny) - x_target_flat.reshape(pde.nx, pde.ny)) ** 2)
        control_mses.append(mse)

    return T, control_mses, control_energies


if __name__ == "__main__":
    # 统一随机种子
    np.random.seed(42)

    # 初始化PDE系统（与您的原始参数一致）
    nx = 30
    ny = 30
    dx = 1.0 / nx
    dy = 1.0 / ny
    D = 0.1
    r = 0.1
    control_density = 0.5
    dt = 0.001  # 显式时间步长

    pde = PDE2D(nx, ny, dx, dy, D, r, control_density)

    # 生成控制点索引
    control_indices = [i * pde.ny + j for i, j in pde.control_positions]


    # 生成目标状态（与KDNNC相同）
    def wave_solution(x, y, t, c=0.5):
        return np.exp(-c * (x + y - c * t))


    x_coords = np.linspace(0, 1, nx)
    y_coords = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x_coords, y_coords)
    T_target = wave_solution(X, Y, t=0)
    x_target_flat = T_target.flatten()

    # PID控制器参数（需要调参）
    pid = LocalizedPIDController(
        control_indices=control_indices,
        Kp=10.0,
        Ki=1,
        Kd=1,
        dt=dt,
        nxny=nx * ny
    )

    # 运行控制仿真
    time_steps = 4000
    T_pid, mse_pid, energy_pid = run_pid_control(pde, pid, x_target_flat, time_steps, dt)

    # 保存数据到文件
    np.savetxt('pid_control_mses.txt', mse_pid)
    np.savetxt('pid_control_energies.txt', energy_pid)

    # 可视化结果
    plt.figure(figsize=(14, 6))

    # MSE曲线
    plt.subplot(1, 2, 1)
    plt.plot(mse_pid, label='PID Control MSE')
    plt.yscale('log')
    plt.xlabel('Time Steps')
    plt.ylabel('MSE (log scale)')
    plt.title('PID Control Performance on Fisher-KPP Equation')
    plt.legend()

    # 能量消耗曲线
    plt.subplot(1, 2, 2)
    plt.plot(energy_pid, label='Accumulated Energy', color='orange')
    plt.xlabel('Time Steps')
    plt.ylabel('Total Energy Consumption')
    plt.title('Control Energy Usage Comparison')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # 最终状态可视化
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(T_pid[-1].reshape(nx, ny), cmap='jet', origin='lower')
    plt.colorbar()
    plt.title('Final State (PID Control)')

    plt.subplot(1, 2, 2)
    plt.imshow(T_target, cmap='jet', origin='lower')
    plt.colorbar()
    plt.title('Target State')
    plt.show()