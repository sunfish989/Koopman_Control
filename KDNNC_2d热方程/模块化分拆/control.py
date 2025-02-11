# control.py
import numpy as np
import torch
import matplotlib.pyplot as plt
from model import PDE2D, Koopman_Model  # 公共模型组件


def simulate_control():
    # 加载保存的模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load('koopman_model.pth', map_location=device)
    config = checkpoint['config']

    # 重新创建PDE系统
    pde = PDE2D(nx=config['nx'], ny=config['ny'],
                dx=1.0 / config['nx'], dy=1.0 / config['ny'],
                alpha=config['alpha'],
                control_spacing=config['control_spacing'])

    # 重新创建模型
    nxny = config['nx'] * config['ny']
    model = Koopman_Model(nxny=nxny,
                          M=len(pde.control_positions),
                          hidden_dim=config['hidden_dim'],
                          P=nxny,
                          control_indices=checkpoint['control_indices'],
                          device=device)
    model.load_state_dict(checkpoint['model_state'])

    # 设计LQR控制器
    K = model.design_lqr_controller()

    # T_init = np.zeros((config['nx'], config['ny']))
    # Create a diagonal gradient
    # for i in range(config['nx']):
    #     for j in range(config['ny']):
    #         distance = abs(i - config['nx'] // 2) + abs(j - config['ny'] // 2)
    #         T_init[i, j] = 1 - (distance / (config['nx'] + config['ny']))  # Smooth gradient

    # T_init = np.random.randn(config['nx'], config['ny'])

    T_init = np.zeros((config['nx'], config['ny']))
    T_init[config['nx'] // 2:, :] = 1  # Bottom half is 1, top half is 0
    def create_target(nx, ny):
        """创建一个环形目标分布"""
        target = np.zeros((nx, ny))
        radius = nx // 3
        sigma = 5  # 控制平滑过渡区域的宽度

        for i in range(nx):
            for j in range(ny):
                x = i - nx // 2
                y = j - ny // 2
                dist = np.sqrt(x ** 2 + y ** 2)

                if dist < radius:
                    target[i, j] = 1.0
                elif dist < radius + sigma:
                    normalized_dist = (dist - radius) / sigma
                    target[i, j] = np.exp(-0.5 * normalized_dist ** 2)
                else:
                    target[i, j] = 0.0

        return np.clip(target, 0, 1)

    # T_target = create_target(config['nx'], config['ny'])

    T_target = np.zeros((config['nx'], config['ny']))
    T_target[:config['nx'] // 2, :] = 1  # Top half is 1, bottom half is 0

    # 控制仿真代码与原始main函数相同...
    # Simulation of control process
    time_steps = 600
    N = nxny
    dt = 0.01
    T = np.zeros((time_steps + 1, N))
    T[0, :] = T_init.flatten()
    u_sequence = np.zeros((time_steps, pde.M))

    # Encode target state
    x_target = torch.tensor(T_target.flatten(), dtype=torch.float32).unsqueeze(0).to(device)
    y_target = model.encoder(x_target)

    for t in range(time_steps):
        x_t_np = T[t, :]
        x_t = torch.tensor(x_t_np, dtype=torch.float32).unsqueeze(0).to(device)

        # Encode the state
        y_t = model.encoder(x_t)

        # Compute control input
        u_bar = model.compute_control(y_t, y_target, K)  # Shape (M,)

        u_sequence[t, :] = u_bar  # Record control input

        # Simulate next time step
        T_t1 = pde.simulate(x_t_np.reshape(config['nx'], config['ny']), np.array([u_bar]), [t * dt, (t + 1) * dt])
        T[t + 1, :] = T_t1[-1]

    # Visualization of results
    T_full = T.reshape(time_steps + 1, config['nx'], config['ny'])

    # 改进的可视化（包含动态收敛过程）
    def plot_convergence(T_full, T_target, time_steps):
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 5, 1)
        plt.imshow(T_full[0], cmap='hot', origin='lower')
        plt.title('Initial State')

        # 添加更多中间状态
        for i, t in enumerate([time_steps // 4, time_steps // 2, 3 * time_steps // 4]):
            plt.subplot(1, 5, 2 + i)
            plt.imshow(T_full[t], cmap='hot', origin='lower')
            plt.title(f'Step {t}')

        plt.subplot(1, 5, 5)
        plt.imshow(T_full[-1], cmap='hot', origin='lower')
        plt.title('Final State')
        plt.show()

    # Generate intermediate states at different time points
    time_steps_for_visualization = [0, 100, 200, 300, 400, 500, 600]
    plot_convergence(T_full, T_target, time_steps)


if __name__ == "__main__":
    simulate_control()