import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import scipy.linalg
import torch.utils.data as data
import matplotlib.pyplot as plt

class GrayScottModel:
    def __init__(self, nx, ny, dx, dy, Du, Dv, F, k, control_density):
        self.nx = nx  # x方向的节点数
        self.ny = ny  # y方向的节点数
        self.dx = dx  # x方向的空间步长
        self.dy = dy  # y方向的空间步长
        self.Du = Du  # u的扩散系数
        self.Dv = Dv  # v的扩散系数
        self.F = F    # 喂养率
        self.k = k    # 衰减速率
        # 构建2D的Laplacian算子
        self.L = self.build_laplacian(nx, ny, dx, dy)
        # 控制位置，每隔一定间隔设置一个控制点
        self.control_positions = []
        for i in range(self.nx):
            for j in range(self.ny):
                if i % int(1 / control_density) == 0 and j % int(1 / control_density) == 0:
                    self.control_positions.append((i, j))
        self.M = len(self.control_positions)
        # 控制影响矩阵 B，大小为 (2 * nx * ny, 2 * M)
        self.B = np.zeros((2 * self.nx * self.ny, 2 * self.M))
        for k, (i, j) in enumerate(self.control_positions):
            idx_u = i * self.ny + j
            idx_v = self.nx * self.ny + idx_u  # v 的索引偏移
            self.B[idx_u, k] = 1.0  # 控制对 u 的影响
            self.B[idx_v, self.M + k] = 1.0  # 控制对 v 的影响

    def build_laplacian(self, nx, ny, dx, dy):
        N = nx * ny
        L = np.zeros((N, N))
        for i in range(nx):
            for j in range(ny):
                idx = i * ny + j
                # 中心元素
                L[idx, idx] = -2 * (1 / dx ** 2 + 1 / dy ** 2)
                # 左邻点
                if i > 0:
                    idx_left = (i - 1) * ny + j
                    L[idx, idx_left] += 1 / dx ** 2
                else:
                    L[idx, idx] += 1 / dx ** 2
                # 右邻点
                if i < nx - 1:
                    idx_right = (i + 1) * ny + j
                    L[idx, idx_right] += 1 / dx ** 2
                else:
                    L[idx, idx] += 1 / dx ** 2
                # 下邻点
                if j > 0:
                    idx_down = i * ny + (j - 1)
                    L[idx, idx_down] += 1 / dy ** 2
                else:
                    L[idx, idx] += 1 / dy ** 2
                # 上邻点
                if j < ny - 1:
                    idx_up = i * ny + (j + 1)
                    L[idx, idx_up] += 1 / dy ** 2
                else:
                    L[idx, idx] += 1 / dy ** 2
        return L

    def simulate(self, U0, V0, u_sequence, t_span):
        """
        Simulate the system using explicit Euler discretization with safeguards.
        """
        time_steps = len(u_sequence)
        dt = (t_span[1] - t_span[0]) / time_steps
        N = self.nx * self.ny
        U = np.zeros((time_steps + 1, N))
        V = np.zeros((time_steps + 1, N))
        U[0, :] = U0.flatten()
        V[0, :] = V0.flatten()

        for t in range(time_steps):
            u_t = u_sequence[t, :self.M]  # Shape (M,)，对 u 的控制
            v_t = u_sequence[t, self.M:]  # Shape (M,)，对 v 的控制

            # Reaction terms
            uv2 = U[t, :] * V[t, :] ** 2
            reaction_u = -uv2 + self.F * (1 - U[t, :])
            reaction_v = uv2 - (self.F + self.k) * V[t, :]

            # Laplacian terms
            laplacian_u = np.dot(self.L, U[t, :])
            laplacian_v = np.dot(self.L, V[t, :])

            # Free evolution
            U[t + 1, :] = U[t, :] + dt * (self.Du * laplacian_u + reaction_u)
            V[t + 1, :] = V[t, :] + dt * (self.Dv * laplacian_v + reaction_v)

            # Control influence: B * u(t)
            control_influence_u = np.dot(self.B[:N, :self.M], u_t)  # 对 u 的控制
            control_influence_v = np.dot(self.B[N:, self.M:], v_t)  # 对 v 的控制

            # Total state update
            U[t + 1, :] += dt * control_influence_u
            V[t + 1, :] += dt * control_influence_v

        return U, V  # Shape (time_steps + 1, nx * ny)

def generate_sample(pde, nx, ny, M, time_steps, dt, control_input_scale):
    """Generate a single training sample."""
    # Random initial conditions
    U0 = np.ones((nx, ny))
    V0 = np.zeros((nx, ny))
    # Add random perturbation in the center
    U0[nx // 2 - 2:nx // 2 + 2, ny // 2 - 2:ny // 2 + 2] = 0
    V0[nx // 2 - 2:nx // 2 + 2, ny // 2 - 2:ny // 2 + 2] = 1
    u_sequence = -control_input_scale + 2 * control_input_scale * np.random.rand(time_steps + 1,
                                                                                 2 * M)  # Shape (time_steps + 1, 2 * M)
    t_span = [0, dt * time_steps]
    U_sequence, V_sequence = pde.simulate(U0, V0, u_sequence, t_span)
    # 合并 u 和 v 的状态
    X_sequence = np.hstack([U_sequence, V_sequence])  # Shape (time_steps + 1, 2 * nx * ny)
    return X_sequence[1:-2, :], X_sequence[2:-1, :], u_sequence[1:-1, :]

class Encoder(nn.Module):
    def __init__(self, nxny, M, hidden_dim, P, control_indices):
        super(Encoder, self).__init__()
        self.nxny = nxny
        self.M = 2*M
        self.P = P
        self.hidden_dim = hidden_dim
        self.control_indices = control_indices
        self.net = nn.Sequential(
            nn.Linear(2*nxny - 2*M, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, P - 2*M)
        )

    def forward(self, x):
        batch_size = x.size(0)
        y = torch.zeros(batch_size, self.P, device=x.device)

        # 确保 control_indices 不越界
        valid_control_indices = [idx for idx in self.control_indices if idx < 2 * self.nxny]
        y[:, valid_control_indices] = x[:, valid_control_indices]

        mask = torch.ones(2 * self.nxny, dtype=torch.bool, device=x.device)
        mask[valid_control_indices] = False
        x_non_control = x[:, mask]
        y_non_control = self.net(x_non_control)
        y[:, mask] = y_non_control
        return y


class Decoder(nn.Module):
    def __init__(self, nxny, M, hidden_dim, P, control_indices):
        super(Decoder, self).__init__()
        self.nxny = nxny
        self.M = 2*M
        self.P = P
        self.hidden_dim = hidden_dim
        self.control_indices = control_indices
        self.net = nn.Sequential(
            nn.Linear(P - 2*M, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2*nxny - 2*M)
        )

    def forward(self, y):
        batch_size = y.size(0)
        x = torch.zeros(batch_size, 2*self.nxny, device=y.device)

        # 确保 control_indices 不越界
        valid_control_indices = [idx for idx in self.control_indices if idx < 2 * self.nxny]
        x[:, valid_control_indices] = y[:, valid_control_indices]

        mask = torch.ones(2*self.nxny, dtype=torch.bool, device=y.device)
        mask[valid_control_indices] = False
        y_non_control = y[:, mask]
        x_non_control = self.net(y_non_control)
        x[:, mask] = x_non_control

        return x


class Koopman_Model:
    def __init__(self, nxny, M, hidden_dim, P, control_indices, device):
        self.nxny = nxny
        self.M = 2*M
        self.P = P
        self.control_indices = control_indices
        self.encoder = Encoder(nxny, M, hidden_dim, P, control_indices).to(device)
        self.decoder = Decoder(nxny, M, hidden_dim, P, control_indices).to(device)
        self.A = nn.Parameter(torch.eye(P) + 0.01 * torch.randn(P, P), requires_grad=True).to(device)
        self.optimizer = optim.Adam(
            list(self.encoder.parameters()) +
            list(self.decoder.parameters()) +
            [self.A],
            lr=1e-3
        )

    def train_step(self, x_t, x_t1_prime):
        self.optimizer.zero_grad()
        y_t = self.encoder(x_t)
        y_t1 = self.encoder(x_t1_prime)
        y_t1_pred = torch.matmul(y_t, self.A.T)
        pred_loss = nn.MSELoss()(y_t1_pred, y_t1)
        recon_loss = 0.5 * (nn.MSELoss()(self.decoder(y_t), x_t) + nn.MSELoss()(self.decoder(y_t1), x_t1_prime))
        loss = pred_loss + 0.7 * recon_loss
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def compute_loss(self, x_t, x_t1_prime):
        y_t = self.encoder(x_t)
        y_t1 = self.encoder(x_t1_prime)
        y_t1_pred = torch.matmul(y_t, self.A.T)
        pred_loss = nn.MSELoss()(y_t1_pred, y_t1)
        recon_loss = 0.5 * (nn.MSELoss()(self.decoder(y_t), x_t) + nn.MSELoss()(self.decoder(y_t1), x_t1_prime))
        loss = pred_loss + 0.7 * recon_loss
        return loss.item()

    def compute_control(self, y_t, y_target, K):
        y_error = y_t - y_target
        u_bar = -np.matmul(K, y_error.detach().cpu().numpy().T).flatten()
        return u_bar

    def design_lqr_controller(self):
        A = self.A.detach().cpu().numpy()
        overline_B = np.zeros((self.P, self.M))
        for k, idx in enumerate(self.control_indices):
            overline_B[idx, k] = 1.0
        B = overline_B
        Q = np.eye(A.shape[0])
        R = np.eye(B.shape[1])
        P = scipy.linalg.solve_discrete_are(A, B, Q, R)
        K = np.linalg.inv(B.T @ P @ B + R) @ (B.T @ P @ A)
        return K


def main():
    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # GS参数
    nx = 30
    ny = 30
    dx = 1.0 / nx
    dy = 1.0 / ny
    Du = 0.16
    Dv = 0.08
    F = 0.04
    k = 0.06
    control_densities = [0.5]
    train_errors_over_time = []
    val_errors_over_time = []
    control_mses_over_time = []

    num_samples = 200
    time_steps = 1000
    dt = 0.001
    control_input_scale = 0.05

    for control_density in control_densities:
        print(f"Running experiment with control density: {control_density}")
        gs = GrayScottModel(nx, ny, dx, dy, Du, Dv, F, k, control_density)

        # 计算控制索引
        control_indices_u = [i * ny + j for i, j in gs.control_positions]  # u 的控制索引
        control_indices_v = [nx * ny + idx for idx in control_indices_u]  # v 的控制索引
        control_indices = control_indices_u + control_indices_v  # 合并索引

        from joblib import Parallel, delayed
        # 并行生成训练数据
        results = Parallel(n_jobs=-1)(
            delayed(generate_sample)(gs, nx, ny, gs.M, time_steps, dt, control_input_scale)
            for _ in range(num_samples)
        )
        # 解包结果
        x_t_list, x_t1_list, u_t_list = zip(*results)

        # 转换为张量
        x_t = torch.tensor(np.concatenate(x_t_list, axis=0), dtype=torch.float32).to(device)
        x_t1 = torch.tensor(np.concatenate(x_t1_list, axis=0), dtype=torch.float32).to(device)
        u_t = torch.tensor(np.concatenate(u_t_list, axis=0), dtype=torch.float32).to(device)

        # Datasets and data loaders
        dataset = data.TensorDataset(x_t, x_t1, u_t)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = data.random_split(dataset, [train_size, val_size])
        train_dataloader = data.DataLoader(train_dataset, batch_size=128, shuffle=True)
        val_dataloader = data.DataLoader(val_dataset, batch_size=128, shuffle=False)

        # 定义模型
        nxny = nx * ny
        hidden_dim = 2048
        P = 2*nxny
        model = Koopman_Model(nxny, gs.M, hidden_dim, P, control_indices, device)

        # 训练模型
        num_epochs = 100
        patience = 7
        best_val_loss = float('inf')
        epochs_no_improve = 0
        train_errors = []
        val_errors = []
        for epoch in range(num_epochs):
            model.encoder.train()
            model.decoder.train()
            total_loss = 0
            for batch_x_t, batch_x_t1, batch_u_t in train_dataloader:
                B_u_t = torch.zeros_like(batch_x_t1, device=device)
                B_u_t[:, control_indices] = batch_u_t
                batch_x_t1_prime = batch_x_t1 - dt * B_u_t
                loss = model.train_step(batch_x_t, batch_x_t1_prime)
                total_loss += loss
            avg_train_loss = total_loss / len(train_dataloader)
            train_errors.append(avg_train_loss)

            # 验证
            model.encoder.eval()
            model.decoder.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_x_t, batch_x_t1, batch_u_t in val_dataloader:
                    B_u_t = torch.zeros_like(batch_x_t1, device=device)
                    B_u_t[:, control_indices] = batch_u_t
                    batch_x_t1_prime = batch_x_t1 - dt * B_u_t
                    loss = model.compute_loss(batch_x_t, batch_x_t1_prime)
                    val_loss += loss
            avg_val_loss = val_loss / len(val_dataloader)
            val_errors.append(avg_val_loss)
            print(f"Epoch {epoch + 1}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping!")
                break

        train_errors_over_time.append(train_errors)
        val_errors_over_time.append(val_errors)

        # 设计LQR控制器
        K = model.design_lqr_controller()

        def generate_target_pattern(nx, ny, dx, dy):
            """
            Generate target patterns U_target and V_target for the Gray-Scott model.
            """
            # Create a grid
            x = np.linspace(0, nx * dx, nx)
            y = np.linspace(0, ny * dy, ny)
            X, Y = np.meshgrid(x, y)

            # Define a circular pattern for U_target
            radius = 0.25  # Radius of the circle (relative to domain size)
            center_x, center_y = nx * dx / 2, ny * dy / 2  # Center of the domain
            distance_from_center = np.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)
            U_target = np.where(distance_from_center < radius, 0.3, 1.0)  # Low value inside the circle, high outside

            # Define V_target as complementary to U_target
            V_target = 1.0 - U_target  # Ensure U + V ≈ 1

            return U_target, V_target

        def ring_distribution(nx, ny, dx, dy, radius=10, thickness=2):
            # Create a grid
            x = np.linspace(0, nx * dx, nx)
            y = np.linspace(0, ny * dy, ny)
            X, Y = np.meshgrid(x, y)

            center_x = nx // 2
            center_y = ny // 2
            # 初始化 U_target 和 V_target
            U_target = np.zeros((nx, ny))
            V_target = np.zeros((nx, ny))

            for i in range(nx):
                for j in range(ny):
                    distance = np.sqrt((i - center_x) ** 2 + (j - center_y) ** 2)
                    if radius - thickness <= distance <= radius + thickness:
                        U_target[i, j] = 0.4
                    else:
                        U_target[i, j] = 0
            # Define V_target as complementary to U_target
            V_target = 1.0 - U_target  # Ensure U + V ≈ 1
            return U_target, V_target

        U_init, V_init = generate_target_pattern(nx, ny, dx, dy)
        X_init = np.hstack([U_init.flatten(), V_init.flatten()])  # Shape (2 * nx * ny,)

        U_target, V_target = ring_distribution(nx, ny, dx, dy)

        X_target = np.hstack([U_target.flatten(), V_target.flatten()])  # Shape (2 * nx * ny,)
        X_target_tensor = torch.tensor(X_target, dtype=torch.float32).unsqueeze(0).to(device)
        y_target = model.encoder(X_target_tensor)

        '''
        # Initial state
        U_init = np.ones((nx, ny))
        V_init = np.zeros((nx, ny))
        U_init[nx // 2 - 2:nx // 2 + 2, ny // 2 - 2:ny // 2 + 2] = 0
        V_init[nx // 2 - 2:nx // 2 + 2, ny // 2 - 2:ny // 2 + 2] = 1

        # Initial state
        U_init = np.ones((nx, ny))
        V_init = np.zeros((nx, ny))
        U_init[nx // 2 - 2:nx // 2 + 2, ny // 2 - 2:ny // 2 + 2] = 0
        V_init[nx // 2 - 2:nx // 2 + 2, ny // 2 - 2:ny // 2 + 2] = 1
        '''

        # 模拟闭环系统
        time_steps = 3000
        X = np.zeros((time_steps + 1, 2 * nx * ny))  # State history
        X[0, :] = X_init  # Initial state
        u_t_sequence = np.zeros((time_steps, 2 * gs.M))  # Control input history
        control_mses = []  # MSE between current state and target state

        for t in range(time_steps):
            # Current state
            x_t_np = X[t, :]
            x_t = torch.tensor(x_t_np, dtype=torch.float32).unsqueeze(0).to(device)

            # Encode current state
            y_t = model.encoder(x_t)

            # Compute control input
            u_bar = model.compute_control(y_t, y_target, K)  # Shape (2M,)
            u_t_sequence[t, :] = u_bar

            # Simulate next state
            U_current = x_t_np[:nx * ny].reshape(nx, ny)
            V_current = x_t_np[nx * ny:].reshape(nx, ny)
            U_t1, V_t1 = gs.simulate(U_current, V_current, np.array([u_bar]), [t * dt, (t + 1) * dt])

            # Update state
            X[t + 1, :nx * ny] = U_t1[-1]
            X[t + 1, nx * ny:] = V_t1[-1]

            # Compute control MSE
            control_mse = np.mean((X[t + 1, :nx * ny].reshape(nx, ny) - U_target) ** 2)
            control_mses.append(control_mse)

        control_mses_over_time.append(control_mses)

        # Visualization of results
        X = np.array(X).reshape(time_steps + 1, 2*nx*ny)
        U_full = X[:, :nx * ny].reshape(time_steps + 1, nx, ny)
        V_full = X[:, nx * ny:].reshape(time_steps + 1, nx, ny)
        # Visualization of results
        plt.rcParams['font.sans-serif'] = ['Heiti TC']  # 设置中文字体为黑体
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
        # Plot intermediate states
        plt.figure(figsize=(18, 6))

        # 绘制不同时间段的目标状态和实际状态
        time_steps_for_visualization = [0, 300, 500, 1000, 2000]

        # 绘制实际状态和目标状态
        for i, t in enumerate(time_steps_for_visualization):
            # 第一行：绘制 U 的实际状态
            plt.subplot(2, 6, i + 1)
            plt.imshow(U_full[t], cmap='jet', origin='lower')
            plt.title(f"U at t={t}")

            # 第二行：绘制 V 的实际状态
            plt.subplot(2, 6, i + 7)
            plt.imshow(V_full[t], cmap='jet', origin='lower')
            plt.title(f"V at t={t}")

            # 如果是每行的最后一个子图，绘制目标状态
            if (i + 1) % 5 == 0:
                plt.subplot(2, 6, i + 2)  # 目标状态绘制在下一个子图
                plt.imshow(U_target, cmap='jet', origin='lower', alpha=0.5)
                plt.title("U Target")
                plt.subplot(2, 6, i + 8)  # 目标状态绘制在下一个子图
                plt.imshow(V_target, cmap='jet', origin='lower', alpha=0.5)
                plt.title("V Target")

        plt.tight_layout()
        plt.savefig('change_system.png')
        plt.show()
        plt.close()

        # 可视化结果
        plt.figure(figsize=(12, 6))

        # 训练误差随时间变化
        plt.subplot(1, 2, 1)
        for i, control_density in enumerate(control_densities):
            plt.plot(range(len(train_errors_over_time[i])), train_errors_over_time[i],
                     label=f"控制点密度={control_density}")
        plt.xlabel("迭代次数")
        plt.ylabel("训练误差")
        plt.title("模型训练误差随时间变化情况")
        plt.legend()

        # 控制MSE随时间变化
        plt.subplot(1, 2, 2)
        for i, control_density in enumerate(control_densities):
            plt.plot(range(len(control_mses_over_time[i])), control_mses_over_time[i],
                     label=f"控制点密度={control_density}")
        plt.xlabel("时间步数")
        plt.ylabel("控制MSE（对数刻度）")
        plt.title("控制MSE随时间变化情况（对数刻度）")
        plt.yscale('log')
        plt.legend()

        plt.tight_layout()
        plt.savefig("results_compare_gs.png")
        plt.show()


if __name__ == "__main__":
    main()