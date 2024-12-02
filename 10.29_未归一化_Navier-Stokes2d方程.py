import numpy as np
from scipy.integrate import solve_ivp
import torch
import torch.nn as nn
import torch.optim as optim
import scipy.linalg
import torch.utils.data as data
import matplotlib.pyplot as plt

class PDE2D_NS:
    def __init__(self, nx, ny, Lx, Ly, viscosity):
        self.nx = nx  # Number of grid points in x
        self.ny = ny  # Number of grid points in y
        self.Lx = Lx  # Domain size in x
        self.Ly = Ly  # Domain size in y
        self.dx = Lx / (nx - 1)
        self.dy = Ly / (ny - 1)
        self.viscosity = viscosity

        # 控制位置，每隔2个节点设置一个控制点
        self.control_positions = []
        for i in range(0, nx, 2):
            for j in range(0, ny, 2):
                self.control_positions.append((i, j))
        self.M = len(self.control_positions)

        # 控制影响矩阵B，大小为 (nx*ny, M)
        self.B = np.zeros((nx*ny, self.M))
        for k, (i, j) in enumerate(self.control_positions):
            idx = i * self.ny + j
            self.B[idx, k] = 1.0

    def laplacian(self, f):
        """Compute the Laplacian of f using central differences."""
        f = f.reshape(self.nx, self.ny)
        lap = np.zeros_like(f)
        lap[1:-1, 1:-1] = (
                (f[2:, 1:-1] - 2 * f[1:-1, 1:-1] + f[0:-2, 1:-1]) / self.dx ** 2 +
                (f[1:-1, 2:] - 2 * f[1:-1, 1:-1] + f[1:-1, 0:-2]) / self.dy ** 2
        )
        # Neumann BCs (zero normal derivative at boundaries)
        # Left and right boundaries
        lap[0, :] = lap[1, :]
        lap[-1, :] = lap[-2, :]
        # Bottom and top boundaries
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
                    (dy2 * (psi[2:, 1:-1] + psi[0:-2, 1:-1]) +
                     dx2 * (psi[1:-1, 2:] + psi[1:-1, 0:-2]) +
                     dx2dy2 * (-omega[1:-1, 1:-1])) / denom
            )
            # Apply boundary conditions
            psi[0, :] = psi[1, :]
            psi[-1, :] = psi[-2, :]
            psi[:, 0] = psi[:, 1]
            psi[:, -1] = psi[:, -2]
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
        u[1:-1, 1:-1] = (psi[1:-1, 2:] - psi[1:-1, 0:-2]) / (2 * self.dy)
        v[1:-1, 1:-1] = -(psi[2:, 1:-1] - psi[0:-2, 1:-1]) / (2 * self.dx)
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

    def simulate(self, omega0, u_sequence, t_span, time_steps):
        dt = (t_span[1] - t_span[0]) / time_steps
        times = np.linspace(t_span[0], t_span[1], time_steps + 1)

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
        # omega is a flattened array of shape (N,)
        # Add control input
        B_u = self.B @ u_t  # Shape (N,)

        # Reshape to 2D for computation
        omega_2d = omega.reshape(self.nx, self.ny)
        B_u_2d = B_u.reshape(self.nx, self.ny)

        # Solve for streamfunction ψ
        psi = self.streamfunction_poisson(omega)

        # Compute velocities
        u, v = self.compute_velocity(psi)

        # Reshape velocities to 2D
        u = u.reshape(self.nx, self.ny)
        v = v.reshape(self.nx, self.ny)

        # Compute Laplacian of ω
        lap_omega = self.laplacian(omega)

        # Reshape laplacian to 2D
        lap_omega_2d = lap_omega.reshape(self.nx, self.ny)

        # Compute advection term
        conv_omega = np.zeros_like(omega_2d)
        # Central differences for advection
        conv_omega[1:-1, 1:-1] = (
                u[1:-1, 1:-1] * (omega_2d[1:-1, 2:] - omega_2d[1:-1, 0:-2]) / (2 * self.dx) +
                v[1:-1, 1:-1] * (omega_2d[2:, 1:-1] - omega_2d[0:-2, 1:-1]) / (2 * self.dy)
        )

        # Time derivative ∂ω/∂t
        domega_dt = -conv_omega + self.viscosity * lap_omega_2d + B_u_2d

        # Update ω
        omega_new = omega_2d + dt * domega_dt

        return omega_new.flatten()


class Encoder(nn.Module):
    def __init__(self, nxny, M, hidden_dim, P, control_indices):
        super(Encoder, self).__init__()
        self.nxny = nxny  # Original state dimension
        self.M = M  # Number of controlled nodes
        self.P = P  # Embedding space dimension
        self.hidden_dim = hidden_dim
        self.control_indices = control_indices  # Control indices

        # Nonlinear mapping neural network
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

        # Identity mapping for control positions
        y[:, self.control_indices] = x[:, self.control_indices]

        # Non-control positions
        mask = torch.ones(self.nxny, dtype=torch.bool, device=x.device)
        mask[self.control_indices] = False
        x_non_control = x[:, mask]  # Extract x at non-control positions

        # Nonlinear mapping for the rest
        y_non_control = self.net(x_non_control)
        y[:, mask] = y_non_control
        return y


class Decoder(nn.Module):
    def __init__(self, nxny, M, hidden_dim, P, control_indices):
        super(Decoder, self).__init__()
        self.nxny = nxny  # Original state dimension
        self.M = M  # Number of controlled nodes
        self.P = P  # Embedding space dimension
        self.hidden_dim = hidden_dim
        self.control_indices = control_indices  # Control indices

        # Nonlinear decoding neural network
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

        # Identity mapping for control positions
        x[:, self.control_indices] = y[:, self.control_indices]

        mask = torch.ones(self.nxny, dtype=torch.bool, device=y.device)
        mask[self.control_indices] = False

        # Nonlinear decoding for the rest
        y_non_control = y[:, mask]
        x_non_control = self.net(y_non_control)

        x[:, mask] = x_non_control
        return x


class Koopman_Model:
    def __init__(self, nxny, M, hidden_dim, P, control_indices):
        self.nxny = nxny  # Original state dimension
        self.M = M  # Number of controlled nodes
        self.P = P  # Embedding space dimension
        self.control_indices = control_indices  # Control indices

        # Encoder and Decoder
        self.encoder = Encoder(nxny, M, hidden_dim, P, control_indices).to(device)
        self.decoder = Decoder(nxny, M, hidden_dim, P, control_indices).to(device)

        # Linear dynamics matrix A
        self.A = nn.Parameter(torch.eye(P) + 0.01 * torch.randn(P, P), requires_grad=True).to(device)

        # Optimizer
        self.optimizer = optim.Adam(
            list(self.encoder.parameters()) +
            list(self.decoder.parameters()) +
            [self.A],
            lr=1e-3
        )

    def train_step(self, x_t, x_t1_prime):
        self.optimizer.zero_grad()
        # Encode
        y_t = self.encoder(x_t)
        y_t1 = self.encoder(x_t1_prime)
        # Linear prediction
        y_t1_pred = torch.matmul(y_t, self.A.T)
        # Prediction loss
        pred_loss = nn.MSELoss()(y_t1_pred, y_t1)
        # Reconstruction loss
        recon_loss = 0.5 * (nn.MSELoss()(self.decoder(y_t), x_t) + nn.MSELoss()(self.decoder(y_t1), x_t1_prime))
        # Total loss
        loss = pred_loss + 0.7 * recon_loss

        loss.backward()
        self.optimizer.step()

        return loss.item()

    def compute_loss(self, x_t, x_t1_prime):
        # Encoding and predictions
        y_t = self.encoder(x_t)
        y_t1 = self.encoder(x_t1_prime)
        # Linear prediction
        y_t1_pred = torch.matmul(y_t, self.A.T)
        # Prediction loss
        pred_loss = nn.MSELoss()(y_t1_pred, y_t1)
        # Reconstruction loss
        recon_loss = 0.5 * (nn.MSELoss()(self.decoder(y_t), x_t) + nn.MSELoss()(self.decoder(y_t1), x_t1_prime))
        # Total loss
        loss = pred_loss + 0.7 * recon_loss
        return loss.item()

    def compute_control(self, y_t, y_target, K):

        # 计算控制输入
        y_error = y_t - y_target
        u_bar = -np.matmul(K, y_error.detach().cpu().numpy().T).T.flatten()
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

    # PDE parameters
    nx = 32  # Number of grid points in x
    ny = 32  # Number of grid points in y
    Lx = np.pi  # Domain size in x
    Ly = np.pi  # Domain size in y
    viscosity = 0.01  # Viscosity coefficient

    pde = PDE2D_NS(nx, ny, Lx, Ly, viscosity)

    control_positions = pde.control_positions  # 列表 (i, j)
    M = len(control_positions)  # 控制数

    # 将控制位置转换为展平数组中的索引
    control_indices = [i * ny + j for i, j in control_positions]

    # Generate training data
    num_samples = 300
    time_steps = 20
    dt = 0.01
    x_t_list = []
    x_t1_list = []
    u_t_list = []

    np.random.seed(0)  # For reproducibility

    for _ in range(num_samples):
        # Random initial vorticity distribution
        omega0 = np.random.randn(nx, ny) * 0.1

        # Generate control inputs
        control_input_scale = 0.1
        u_sequence = np.random.randn(time_steps, M) * control_input_scale

        # Simulate the system
        t_span = [0, dt * time_steps]
        omega_sequence = pde.simulate(omega0, u_sequence, t_span, time_steps)

        # Build training samples
        x_t_list.append(omega_sequence[:-1, :])  # x(t)
        x_t1_list.append(omega_sequence[1:, :])  # x(t+1)
        u_t_list.append(u_sequence[:])

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
            B = torch.tensor(pde.B, dtype=torch.float32).to(device)
            B_u_t = torch.matmul(B, batch_u_t.unsqueeze(-1)).squeeze(-1)
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
                B = torch.tensor(pde.B, dtype=torch.float32).to(device)
                B_u_t = torch.matmul(B, batch_u_t.unsqueeze(-1)).squeeze(-1)
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

        B = torch.tensor(pde.B, dtype=torch.float32).to(device)
        B_u = torch.matmul(B, batch_u_t.unsqueeze(-1)).squeeze(-1)
        # Add control influence to get predicted x(t+1)
        x_t1_pred = x_t1_pred_prime + B_u

        # Compute prediction error
        pred_error = nn.MSELoss()(x_t1_pred, batch_x_t1)
        print(f"Prediction error on validation batch: {pred_error.item():.6f}")

        # Visualization
        idx = 0  # Visualize the first sample
        omega_true = batch_x_t1[idx].cpu().numpy().reshape(nx, ny)
        omega_pred = x_t1_pred[idx].cpu().numpy().reshape(nx, ny)

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(omega_true, cmap='jet')
        plt.title('True Omega at t+1')
        plt.colorbar()
        plt.subplot(1, 2, 2)
        plt.imshow(omega_pred, cmap='jet')
        plt.title('Predicted Omega at t+1')
        plt.colorbar()
        plt.show()

    # Design LQR controller
    K = model.design_lqr_controller()

    # Control target
    x_target = np.zeros((nx, ny))
    x_target[nx // 4:3 * nx // 4, ny // 4:3 * ny // 4] = 1

    # Initial condition for simulation
    omega = np.random.randn(nx, ny) * 0.1
    omega_tensor = torch.tensor(omega.reshape(1, -1).flatten(), dtype=torch.float32).unsqueeze(0).to(device)

    # Control simulation parameters
    time_steps = 300

    # Store omega fields for visualization
    omega_history = [omega.copy()]

    # Encode target state
    x_target = torch.tensor(x_target.flatten(), dtype=torch.float32).unsqueeze(0).to(device)
    y_target = model.encoder(x_target)

    for t in range(time_steps):
        # Encode current state
        y_t = model.encoder(omega_tensor)

        # Compute control input
        u_t = model.compute_control(y_t, y_target, K)  # Shape (M,)

        # Apply control input to the PDE simulator
        u_t_full = np.zeros((nx, ny))
        for idx_ctrl, (i, j) in enumerate(control_positions):
            u_t_full[i, j] = u_t[idx_ctrl]

        # Simulate one time step with control input
        omega = pde.simulate(omega, np.array([u_t]), [0, dt], time_steps=1)[-1]

        # Update omega_tensor
        omega_tensor = torch.tensor(omega.reshape(1, -1), dtype=torch.float32).to(device)

        # Store omega for visualization
        omega_history.append(omega.reshape(nx, ny))

    print(len(omega_history))
    print(omega_history[0].shape)
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(omega_history[0], cmap='jet', origin='lower')
    plt.colorbar()
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('Distribution at Initial Time')

    plt.subplot(1, 2, 2)
    plt.imshow(omega_history[-1], cmap='jet', origin='lower')
    plt.colorbar()
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('Distribution at Final Time')
    plt.show()


if __name__ == "__main__":
    main()


