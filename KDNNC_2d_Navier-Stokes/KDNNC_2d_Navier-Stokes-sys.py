import numpy as np
from scipy.integrate import solve_ivp
import torch
import torch.nn as nn
import torch.optim as optim
import scipy.linalg
import torch.utils.data as data
import matplotlib.pyplot as plt
from joblib import Parallel, delayed


class PDE2D_NS:
    def __init__(self, nx, ny, Lx, Ly, viscosity, control_density):
        self.nx = nx  # Number of grid points in x
        self.ny = ny  # Number of grid points in y
        self.Lx = Lx  # Domain size in x
        self.Ly = Ly  # Domain size in y
        self.dx = Lx / (nx - 1)
        self.dy = Ly / (ny - 1)
        self.viscosity = viscosity

        self.control_positions = []
        for i in range(self.nx):
            for j in range(self.ny):
                if i % int(1 / control_density) == 0 and j % int(1 / control_density) == 0:
                    self.control_positions.append((i, j))
        self.M = len(self.control_positions)

        # 控制影响矩阵B，大小为 (nx*ny, M)
        self.B = np.zeros((nx * ny, self.M))
        for k, (i, j) in enumerate(self.control_positions):
            idx = i * self.ny + j
            self.B[idx, k] = 1.0

    def laplacian(self, f):
        """Compute the Laplacian of f using central differences."""
        f = f.reshape(self.nx, self.ny)
        lap = np.zeros_like(f)
        lap[1:-1, 1:-1] = (
                (f[2:, 1:-1] - 2 * f[1:-1, 1:-1] + f[:-2, 1:-1]) / self.dx ** 2 +
                (f[1:-1, 2:] - 2 * f[1:-1, 1:-1] + f[1:-1, :-2]) / self.dy ** 2
        )
        # Neumann BCs (zero normal derivative at boundaries)
        lap[0, :] = lap[1, :]
        lap[-1, :] = lap[-2, :]
        lap[:, 0] = lap[:, 1]
        lap[:, -1] = lap[:, -2]
        return lap.flatten()

    def streamfunction_poisson(self, omega):
        """Solve Poisson equation for streamfunction: ∇²ψ = -ω."""
        omega = omega.reshape(self.nx, self.ny)
        psi = np.zeros_like(omega)
        dx2 = self.dx ** 2
        dy2 = self.dy ** 2
        dx2dy2 = dx2 * dy2
        denom = 2 * (dx2 + dy2)

        for iteration in range(5000):
            psi_old = psi.copy()
            psi[1:-1, 1:-1] = (
                    (dy2 * (psi[2:, 1:-1] + psi[:-2, 1:-1]) +
                     dx2 * (psi[1:-1, 2:] + psi[1:-1, :-2]) +
                     dx2dy2 * (-omega[1:-1, 1:-1])) / denom
            )
            # Apply boundary conditions
            psi[0, :] = 0
            psi[-1, :] = 0
            psi[:, 0] = 0
            psi[:, -1] = 0

            max_diff = np.max(np.abs(psi - psi_old))
            if max_diff < 1e-6:
                break
        return psi.flatten()

    def compute_velocity(self, psi):
        """Compute velocities u and v from streamfunction ψ."""
        psi = psi.reshape(self.nx, self.ny)
        u = np.zeros_like(psi)
        v = np.zeros_like(psi)

        # Central differences for interior points
        u[1:-1, 1:-1] = (psi[1:-1, 2:] - psi[1:-1, :-2]) / (2 * self.dy)
        v[1:-1, 1:-1] = -(psi[2:, 1:-1] - psi[:-2, 1:-1]) / (2 * self.dx)

        # Neumann BCs (zero normal derivative at boundaries)
        u[0, :] = u[1, :]
        u[-1, :] = u[-2, :]
        u[:, 0] = u[:, 1]
        u[:, -1] = u[:, -2]
        v[0, :] = v[1, :]
        v[-1, :] = v[-2, :]
        v[:, 0] = v[:, 1]
        v[:, -1] = v[:, -2]

        return u, v

    def simulate(self, omega0, u_sequence, t_span):
        """
        Simulate the PDE using explicit finite difference method.
        """
        time_steps = len(u_sequence)
        dt = (t_span[1] - t_span[0]) / time_steps
        N = self.nx * self.ny
        omega = np.zeros((time_steps + 1, N))
        omega[0, :] = omega0.flatten()

        for t in range(time_steps):
            omega_t = omega[t, :]
            u_t = u_sequence[t, :]  # Shape (M,)
            omega_t1 = self.step(omega_t, u_t, dt)
            omega[t + 1, :] = omega_t1

        return omega  # Shape (time_steps + 1, N)

    def step(self, omega, u_t, dt):
        """
        Perform one time step using explicit finite difference method.
        """
        # Add control input
        B_u = self.B @ u_t  # Shape (N,)

        # Reshape to 2D for computation
        omega_2d = omega.reshape(self.nx, self.ny)
        B_u_2d = B_u.reshape(self.nx, self.ny)

        # Solve for streamfunction ψ
        psi = self.streamfunction_poisson(omega)

        # Compute velocities
        u, v = self.compute_velocity(psi)
        u = u.reshape(self.nx, self.ny)
        v = v.reshape(self.nx, self.ny)

        # Compute Laplacian of ω
        lap_omega = self.laplacian(omega).reshape(self.nx, self.ny)

        # Compute advection term
        conv_omega = np.zeros_like(omega_2d)
        conv_omega[1:-1, 1:-1] = (
                u[1:-1, 1:-1] * (omega_2d[1:-1, 2:] - omega_2d[1:-1, :-2]) / (2 * self.dx) +
                v[1:-1, 1:-1] * (omega_2d[2:, 1:-1] - omega_2d[:-2, 1:-1]) / (2 * self.dy)
        )

        # Time derivative ∂ω/∂t
        domega_dt = -conv_omega + self.viscosity * lap_omega + B_u_2d

        # Update ω using explicit Euler method
        omega_new = omega_2d + dt * domega_dt
        return omega_new.flatten()

def generate_sample(pde, nx, ny, M, time_steps, dt, control_input_scale):
    """Generate a single training sample."""
    T0 = np.random.rand(nx, ny)
    u_sequence = -control_input_scale + 2 * control_input_scale * np.random.rand(time_steps + 1, M)
    t_span = [0, dt * time_steps]
    T_sequence = pde.simulate(T0, u_sequence, t_span)
    return T_sequence[1:-2, :], T_sequence[2:-1, :], u_sequence[1:-1, :]

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
    nx = 32
    ny = 32
    Lx = np.pi  # Domain size in x
    Ly = np.pi  # Domain size in y
    viscosity = 0.01  # Viscosity coefficient

    control_densities = [0.5]
    train_errors_over_time = []
    val_errors_over_time = []
    control_mses_over_time = []

    num_samples = 200
    time_steps = 2000
    dt = 0.001
    control_input_scale = 0.05

    for control_density in control_densities:
        np.random.seed(0)
        print(f"Running experiment with control density: {control_density}")
        pde = PDE2D_NS(nx, ny, Lx, Ly, viscosity, control_density)

        control_indices = [i * ny + j for i, j in pde.control_positions]
        M = pde.M
        print(M)
        # 并行生成训练数据
        results = Parallel(n_jobs=-1)(
            delayed(generate_sample)(pde, nx, ny, pde.M, time_steps, dt, control_input_scale)
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

        # 设计LQR控制器
        K = model.design_lqr_controller()
        '''设置目标状态'''
        def traveling_wave_solution(x, y, t, c=0.5):
            return np.exp(-c * ((x - c * t) ** 2 + (y - c * t) ** 2))

        # 定义网格和参数
        x_coords = np.linspace(0, Lx, nx)
        y_coords = np.linspace(0, Ly, ny)
        X, Y = np.meshgrid(x_coords, y_coords)
        # 设置波速和时间
        c = 0.5
        t1 = 0
        t2 = 2
        t3 = 4
        # 计算行波解  多目标动态控制
        T_target1 = traveling_wave_solution(X, Y, t1, c)
        T_target2 = traveling_wave_solution(X, Y, t2, c)
        T_target3 = traveling_wave_solution(X, Y, t3, c)

        '''设置系统的初始状态'''
        # 随机初始化，并平滑处理 作为系统的初始状态
        import scipy.ndimage as ndimage
        T_init = np.random.randn(nx, ny)

        # 使用高斯滤波进行平滑
        sigma = 2  # 平滑程度
        T_init = ndimage.gaussian_filter(T_init, sigma=sigma)

        # 模拟闭环系统
        time_steps = 7000
        # Store omega fields for visualization
        T_history = [T_init.flatten()]

        u_t_sequence = np.zeros((time_steps, M))
        x_target_tensor1 = torch.tensor(T_target1.flatten(), dtype=torch.float32).unsqueeze(0).to(device)
        y_target1 = model.encoder(x_target_tensor1)
        x_target_tensor2 = torch.tensor(T_target2.flatten(), dtype=torch.float32).unsqueeze(0).to(device)
        y_target2 = model.encoder(x_target_tensor2)
        x_target_tensor3 = torch.tensor(T_target3.flatten(), dtype=torch.float32).unsqueeze(0).to(device)
        y_target3 = model.encoder(x_target_tensor3)
        control_mses = []

        def smooth_transition(t, T_target1, T_target2):
            if t <= 2000:
                return T_target1
            elif 2000 < t <= 4000:
                alpha = (t - 2000) / 2000.0  # Linear interpolation factor
                return (1 - alpha) * T_target1 + alpha * T_target2
            else:
                return T_target3

        for t in range(time_steps):
            x_t_np = T_history[t]
            x_t = torch.tensor(x_t_np, dtype=torch.float32).unsqueeze(0).to(device)
            y_t = model.encoder(x_t)

            # 使用平滑过渡函数计算当前的目标状态
            current_target = smooth_transition(t, T_target1, T_target2)
            y_target = model.encoder(
                torch.tensor(current_target.flatten(), dtype=torch.float32).unsqueeze(0).to(device))

            u_bar = model.compute_control(y_t, y_target, K)
            u_t_sequence[t, :] = u_bar
            T_t1 = pde.simulate(np.array(x_t_np).reshape(nx, ny), np.array([u_bar]), [t * dt, (t + 1) * dt])
            T_history.append(T_t1[-1].flatten())
            control_mse = np.mean((T_t1[-1].reshape(nx, ny) - current_target) ** 2)
            control_mses.append(control_mse)



        control_mses_over_time.append(control_mses)
        # Visualization of results
        T_full = np.array(T_history).reshape(time_steps + 1, nx, ny)
        # Visualization of results
        plt.rcParams['font.sans-serif'] = ['Heiti TC']  # 设置中文字体为黑体
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
        # Plot intermediate states
        plt.figure(figsize=(18, 9))

        # 绘制不同时间段的目标状态和实际状态
        time_steps_for_visualization = [0, 300, 500, 1000, 2000,
                                        2000, 2300, 2500, 3000, 4000,
                                        4000, 4300, 4500, 5000, 6000]

        tmp = 1  # 子图索引
        for i, t in enumerate(time_steps_for_visualization):
            if tmp > 18:  # 确保 tmp 不超过子图总数
                break

            # 绘制实际状态
            plt.subplot(3, 6, tmp)
            plt.imshow(T_full[t], cmap='jet', origin='lower')
            tmp += 1

            # 检查是否需要绘制目标状态（每行的最后一个子图）
            if (i + 1) % 5 == 0:  # 每行有5个实际状态，第6个子图绘制目标状态
                if tmp <= 18:
                    plt.subplot(3, 6, tmp)
                    if i // 5 == 0:  # 第一行
                        plt.imshow(T_target1, cmap='jet', origin='lower', alpha=0.5)
                    elif i // 5 == 1:  # 第二行
                        plt.imshow(T_target2, cmap='jet', origin='lower', alpha=0.5)
                    elif i // 5 == 2:  # 第三行
                        plt.imshow(T_target3, cmap='jet', origin='lower', alpha=0.5)
                    tmp += 1

        plt.tight_layout()
        plt.savefig('change_system.png')
        plt.show()
        plt.close()
    # print(train_errors_over_time[0])
    # print(val_errors_over_time[0])
    # print(control_mses_over_time[0][0], control_mses_over_time[0][100], control_mses_over_time[0][300],
    #       control_mses_over_time[0][500],
    #       control_mses_over_time[0][1000], control_mses_over_time[0][1500], control_mses_over_time[0][2000],
    #       control_mses_over_time[0][-1])
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
    plt.show()

if __name__ == "__main__":
    main()