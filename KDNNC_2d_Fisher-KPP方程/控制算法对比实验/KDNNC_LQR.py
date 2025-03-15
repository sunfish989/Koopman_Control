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
        self.nxny = nxny
        self.M = M
        self.P = P
        self.control_indices = control_indices
        self.encoder = Encoder(nxny, M, hidden_dim, P, control_indices).to(device)
        self.decoder = Decoder(nxny, M, hidden_dim, P, control_indices).to(device)

        B = np.zeros((self.P, self.M))
        for k, idx in enumerate(self.control_indices):
            B[idx, k] = 1.0
        self.B = B

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
    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # PDE参数
    nx = 30
    ny = 30
    dx = 1.0 / nx
    dy = 1.0 / ny
    D = 0.1
    r = 0.1
    control_densities = [0.5]
    train_errors_over_time = []
    val_errors_over_time = []
    control_mses_over_time = []

    for control_density in control_densities:
        print(f"Running experiment with control density: {control_density}")
        pde = PDE2D(nx, ny, dx, dy, D, r, control_density)
        control_indices = [i * ny + j for i, j in pde.control_positions]
        M = pde.M
        print(M)

        # 生成训练数据
        num_samples = 200
        time_steps = 600
        dt = 0.001
        x_t_list = []
        x_t1_list = []
        u_t_list = []
        np.random.seed(0)
        for _ in range(num_samples):
            T0 = np.random.rand(nx, ny)
            control_input_scale = 0.05
            u_sequence = -control_input_scale + 2 * control_input_scale * np.random.rand(time_steps + 1, M)
            t_span = [0, dt * time_steps]
            T_sequence = pde.simulate(T0, u_sequence, t_span)
            x_t_list.append(T_sequence[1:-2, :])
            x_t1_list.append(T_sequence[2:-1, :])
            u_t_list.append(u_sequence[1:-1, :])

        # 转换为张量
        x_t = torch.tensor(np.concatenate(x_t_list, axis=0), dtype=torch.float32).to(device)
        x_t1 = torch.tensor(np.concatenate(x_t1_list, axis=0), dtype=torch.float32).to(device)
        u_t = torch.tensor(np.concatenate(u_t_list, axis=0), dtype=torch.float32).to(device)

        # Datasets and data loaders
        dataset = data.TensorDataset(x_t, x_t1, u_t)
        # Split into training and validation sets
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = data.random_split(dataset, [train_size, val_size])
        train_dataloader = data.DataLoader(train_dataset, batch_size=128, shuffle=True)
        val_dataloader = data.DataLoader(val_dataset, batch_size=128, shuffle=False)

        # 定义模型
        nxny = nx * ny
        hidden_dim = 512
        P = nxny
        model = Koopman_Model(nxny, M, hidden_dim, P, control_indices, device)

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
            x_t1_pred = x_t1_pred_prime + dt * B_u

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
        # 在训练代码末尾添加：
        torch.save({
            'encoder': model.encoder.state_dict(),
            'decoder': model.decoder.state_dict(),
            'A': model.A,
            'B': model.B,
            'control_indices': model.control_indices,
            'nxny': model.encoder.nxny,
            'hidden_dim': model.encoder.hidden_dim,
            'M': model.M,
            'P': model.P
        }, f'trained_kdnn_model.pth')

        # 设计LQR控制器
        K = model.design_lqr_controller()

        # TODO  从这里开始有区别  另外就是反应速率和扩散速率的问题
        # Control target
        # 控制目标
        # x_target = np.ones((nx, ny))  # 全1状态   稳态解
        # x_target = np.zeros((nx, ny))  # 全0状态  稳态解
        # x_target[nx//4:3*nx//4, ny//4:3*ny//4] = 1  # 局部高值区域
        # 行波解的波形函数
        def wave_solution(x, y, t, c=0.5):
            return np.exp(-c * (x + y - c * t))

        # 生成行波解作为目标状态
        x_coords = np.linspace(0, 1, nx)
        y_coords = np.linspace(0, 1, ny)
        X, Y = np.meshgrid(x_coords, y_coords)
        T_target = wave_solution(X, Y, t=0)  # 行波解

        # Initial state
        T_init = np.random.randn(nx, ny)*0.1
        # u0 = np.zeros((nx, ny))
        # u0[nx//4:3*nx//4, nx//2:] = 1

        # 模拟闭环系统
        time_steps = 4000
        N = nx * ny
        T = np.zeros((time_steps + 1, N))
        T[0, :] = T_init.flatten()
        u_t_sequence = np.zeros((time_steps, M))
        x_target_tensor = torch.tensor(T_target.flatten(), dtype=torch.float32).unsqueeze(0).to(device)
        y_target = model.encoder(x_target_tensor)
        control_mses = []

        # 新增能量存储变量
        control_energies = []
        total_energy = 0.0

        for t in range(time_steps):
            x_t_np = T[t, :]
            x_t = torch.tensor(x_t_np, dtype=torch.float32).unsqueeze(0).to(device)
            y_t = model.encoder(x_t)
            u_bar = model.compute_control(y_t, y_target, K)
            # u_bar = u_bar / dt
            u_t_sequence[t, :] = u_bar
            T_t1 = pde.simulate(x_t_np.reshape(nx, ny), np.array([u_bar]), [t * dt, (t + 1) * dt])
            T[t + 1, :] = T_t1[-1]
            # 计算MSE
            control_mse = np.mean((T[t + 1].reshape(nx, ny) - T_target) ** 2)
            control_mses.append(control_mse)
            # 计算控制能量
            control_input_norm = np.sum(u_bar ** 2)  # L2范数平方
            total_energy += control_input_norm
            control_energies.append(total_energy)

        # 保存到文件（新增）
        # np.savetxt(f'KDNNC_LQR_MSE.txt', control_mses)
        # np.savetxt(f'KDNNC_LQR_Energies.txt', control_energies)


        control_mses_over_time.append(control_mses)
        # Visualization of results
        T_full = T.reshape(time_steps + 1, nx, ny)
        # Generate intermediate states at different time points
        time_steps_for_visualization = [300, 500, 1000, 2000]
        # Visualization of results
        plt.rcParams['font.sans-serif'] = ['Heiti TC']  # 设置中文字体为黑体
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
        # Plot intermediate states
        plt.figure(figsize=(15, 10))
        plt.subplot(2, 3, 1)
        plt.imshow(T_full[0], cmap='jet', origin='lower')
        plt.title(f'初始波动分布Init')
        for i, t in enumerate(time_steps_for_visualization):
            plt.subplot(2, 3, i + 2)
            plt.imshow(T_full[t], cmap='jet', origin='lower')
            plt.title(f'{t / 1000}s时波动分布')
        plt.subplot(2, 3, 6)
        plt.imshow(T_target, cmap='jet', origin='lower')
        plt.title(f'目标行波解Target')

        plt.tight_layout()
        # plt.savefig('change_system.png')
        plt.show()
        plt.close()
    '''
    print(train_errors_over_time[0])
    print(val_errors_over_time[0])
    print(control_mses_over_time[0][0], control_mses_over_time[0][100], control_mses_over_time[0][300],
          control_mses_over_time[0][500],
          control_mses_over_time[0][1000], control_mses_over_time[0][1500], control_mses_over_time[0][2000],
          control_mses_over_time[0][-1])
    # 可视化结果
    plt.rcParams['font.sans-serif'] = ['Heiti TC']
    plt.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    for i, control_density in enumerate(control_densities):
        plt.plot(range(len(train_errors_over_time[i])), train_errors_over_time[i], label=f"控制点密度={control_density}")
    plt.xlabel("迭代次数")
    plt.ylabel("训练误差")

    plt.title("模型训练误差随时间变化情况")
    plt.legend()

    plt.subplot(1, 2, 2)
    for i, control_density in enumerate(control_densities):
        plt.plot(range(len(control_mses_over_time[i])), control_mses_over_time[i], label=f"控制点密度={control_density}")
    plt.xlabel("时间步数")
    plt.ylabel("控制MSE（对数刻度）")
    plt.title("控制MSE随时间变化情况（对数刻度）")
    plt.yscale('log')
    plt.legend()
    plt.tight_layout()
    plt.savefig("results_compare.png")
    plt.show()
    '''

if __name__ == "__main__":
    main()