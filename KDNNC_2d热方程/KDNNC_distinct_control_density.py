import numpy as np
from scipy.integrate import solve_ivp
import torch
import torch.nn as nn
import torch.optim as optim
import scipy.linalg
import torch.utils.data as data
import matplotlib.pyplot as plt


class PDE2D:
    def __init__(self, nx, ny, dx, dy, alpha, x_interval, y_interval):
        self.nx = nx  # Number of spatial nodes in x
        self.ny = ny  # Number of spatial nodes in y
        self.dx = dx  # Spatial step size in x
        self.dy = dy  # Spatial step size in y
        self.alpha = alpha  # Thermal diffusivity coefficient

        # Build the Laplacian operator for 2D
        self.L = self.build_laplacian(nx, ny, dx, dy)

        # Control positions based on x_interval and y_interval
        self.control_positions = []
        for i in range(0, nx, x_interval):
            for j in range(0, ny, y_interval):
                self.control_positions.append((i, j))
        self.M = len(self.control_positions)  # Number of controls

        # Control influence matrix B, size (nx*ny, M)
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
                # Center element
                L[idx, idx] = -2 * self.alpha * (1 / dx ** 2 + 1 / dy ** 2)
                # Neumann BC (zero derivative at boundaries)
                if i > 0:
                    idx_left = (i - 1) * ny + j
                    L[idx, idx_left] += self.alpha / dx ** 2
                else:
                    L[idx, idx] += self.alpha / dx ** 2
                if i < nx - 1:
                    idx_right = (i + 1) * ny + j
                    L[idx, idx_right] += self.alpha / dx ** 2
                else:
                    L[idx, idx] += self.alpha / dx ** 2
                if j > 0:
                    idx_down = i * ny + (j - 1)
                    L[idx, idx_down] += self.alpha / dy ** 2
                else:
                    L[idx, idx] += self.alpha / dy ** 2
                if j < ny - 1:
                    idx_up = i * ny + (j + 1)
                    L[idx, idx_up] += self.alpha / dy ** 2
                else:
                    L[idx, idx] += self.alpha / dy ** 2
        return L

    def simulate(self, T0, u_sequence, t_span):
        """
        Simulate the system using explicit Euler discretization.
        Args:
            T0: Initial temperature distribution (2D array or flattened array).
            u_sequence: Control input sequence (shape: (time_steps, M)).
            t_span: Time span [t_start, t_end].
        Returns:
            T: Temperature evolution over time (shape: (time_steps + 1, nx * ny)).
        """
        time_steps = len(u_sequence)
        dt = (t_span[1] - t_span[0]) / time_steps
        N = self.nx * self.ny
        T = np.zeros((time_steps + 1, N))
        T[0, :] = T0.flatten()
        for t in range(time_steps):
            u_t = u_sequence[t, :]  # Shape (M,)
            free_evolution = T[t, :] + dt * np.dot(self.L, T[t, :])

            # Control influence: B * u(t)
            control_influence = np.dot(self.B, u_t)

            # Total state update
            T[t + 1, :] = free_evolution + dt * control_influence

        return T  # Shape (time_steps + 1, nx * ny)

    def odefun(self, t, T_flat, u_t):
        """
        Define the ODE function for continuous-time simulation.
        This is no longer used in the discrete-time version.
        """
        B_u = np.dot(self.B, u_t)  # Control influence
        dTdt = np.dot(self.L, T_flat) + B_u
        return dTdt


class Encoder(nn.Module):
    def __init__(self, nxny, M, hidden_dim, P, control_indices):
        super(Encoder, self).__init__()
        self.nxny = nxny  # Original state dimension
        self.M = M  # Number of controlled nodes
        self.P = P  # Embedding space dimension
        self.hidden_dim = hidden_dim
        self.control_indices = control_indices  # Control indices in flattened array

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
        mask = torch.ones(self.nxny, dtype=torch.bool)
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
        self.control_indices = control_indices  # Control indices in flattened array

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

        mask = torch.ones(self.nxny, dtype=torch.bool)
        mask[self.control_indices] = False

        # Nonlinear decoding for the rest
        y_non_control = y[:, mask]
        x_non_control = self.net(y_non_control)

        x[:, mask] = x_non_control
        return x


class Koopman_Model:
    def __init__(self, nxny, M, hidden_dim, P, control_indices, device):
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
        # Linear prediction with explicit time step
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
        # Linear prediction with explicit time step
        y_t1_pred = torch.matmul(y_t, self.A.T)
        # Prediction loss
        pred_loss = nn.MSELoss()(y_t1_pred, y_t1)
        # Reconstruction loss
        recon_loss = 0.5 * (nn.MSELoss()(self.decoder(y_t), x_t) + nn.MSELoss()(self.decoder(y_t1), x_t1_prime))
        # Total loss
        loss = pred_loss + 0.7 * recon_loss
        return loss.item()

    def compute_control(self, y_t, y_target, K):
        # Compute the control input
        y_error = y_t - y_target  # State error
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
    # Device configuration
    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # PDE parameters
    nx = 30  # Number of spatial nodes in x
    ny = 30  # Number of spatial nodes in y
    dx = 1.0 / nx  # Spatial step size in x
    dy = 1.0 / ny  # Spatial step size in y
    alpha = 0.005  # Thermal diffusivity coefficient

    # Parameters for control intervals experiment
    control_intervals = [(2, 2), (3, 2), (3, 3), (3, 4)]  # (x_interval, y_interval)
    train_errors_over_time = []  # Store training errors over epochs for different control intervals
    control_mses_over_time = []  # Store control MSEs over time steps for different control intervals

    for x_interval, y_interval in control_intervals:
        print(f"Running experiment with x_interval={x_interval}, y_interval={y_interval}")

        # Initialize PDE with different control positions
        pde = PDE2D(nx, ny, dx, dy, alpha, x_interval, y_interval)
        control_positions = pde.control_positions
        M = len(control_positions)  # Number of controls
        control_indices = [i * ny + j for i, j in control_positions]
        print(pde.control_positions)
        print(pde.B)
        print(M)
        # Generate training data
        num_samples = 1000
        time_steps = 60
        dt = 0.01
        x_t_list = []
        x_t1_list = []
        u_t_list = []
        np.random.seed(0)  # For reproducibility
        for _ in range(num_samples):
            T0 = np.random.rand(nx, ny)
            control_input_scale = 0.05
            u_sequence = -control_input_scale * np.ones((time_steps + 1, M)) + 2 * control_input_scale * np.random.rand(
                time_steps + 1, M)
            t_span = [0, dt * time_steps]
            T_sequence = pde.simulate(T0, u_sequence, t_span)
            x_t_list.append(T_sequence[1:-2, :])
            x_t1_list.append(T_sequence[2:-1, :])
            u_t_list.append(u_sequence[1:-1, :])

        # Convert to tensors and move to device
        x_t = torch.tensor(np.concatenate(x_t_list, axis=0), dtype=torch.float32).to(device)
        x_t1 = torch.tensor(np.concatenate(x_t1_list, axis=0), dtype=torch.float32).to(device)
        u_t = torch.tensor(np.concatenate(u_t_list, axis=0), dtype=torch.float32).to(device)

        # Define model
        nxny = nx * ny
        hidden_dim = 1024
        P = nxny
        model = Koopman_Model(nxny, M, hidden_dim, P, control_indices, device)

        # Training model
        num_epochs = 60
        train_errors = []  # Record training errors over epochs
        for epoch in range(num_epochs):
            model.encoder.train()
            model.decoder.train()
            total_loss = 0
            for batch_x_t, batch_x_t1, batch_u_t in data.DataLoader(data.TensorDataset(x_t, x_t1, u_t), batch_size=128,
                                                                    shuffle=True):
                B_u_t = torch.zeros_like(batch_x_t1, device=device)
                B_u_t[:, control_indices] = batch_u_t
                batch_x_t1_prime = batch_x_t1 - dt * B_u_t
                loss = model.train_step(batch_x_t, batch_x_t1_prime)
                total_loss += loss
            avg_train_loss = total_loss / len(x_t_list)
            train_errors.append(avg_train_loss)

            # Validation (same as before)
            model.encoder.eval()
            model.decoder.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_x_t, batch_x_t1, batch_u_t in data.DataLoader(data.TensorDataset(x_t, x_t1, u_t),
                                                                        batch_size=128, shuffle=False):
                    B_u_t = torch.zeros_like(batch_x_t1, device=device)
                    B_u_t[:, control_indices] = batch_u_t
                    batch_x_t1_prime = batch_x_t1 - dt * B_u_t
                    loss = model.compute_loss(batch_x_t, batch_x_t1_prime)
                    val_loss += loss
            avg_val_loss = val_loss / len(x_t_list)

            print(f"Epoch {epoch + 1}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")

        # Record training errors over epochs
        train_errors_over_time.append(train_errors)

        # Design LQR controller
        K = model.design_lqr_controller()

        # Initial state
        # T_init = np.zeros((nx, ny))
        # T_init[nx // 2:, :] = 1  # Bottom half is 1, top half is 0
        T_init = np.zeros((nx, ny))
        # Create a diagonal gradient
        for i in range(nx):
            for j in range(ny):
                distance = abs(i - nx // 2) + abs(j - ny // 2)
                T_init[i, j] = 1 - (distance / (nx + ny))  # Smooth gradient

        # Target state
        # T_target = np.zeros((nx, ny))
        # T_target[:nx // 2, :] = 1  # Top half is 1, bottom half is 0
        # T_target[nx // 8:3 * nx // 8, ny // 8:3 * ny // 8] = 1
        # Target state
        T_target = np.zeros((nx, ny))
        # Create a ring-shaped target with smooth transition
        radius = nx // 3
        sigma = 5  # Controls the smoothness of the transition

        for i in range(nx):
            for j in range(ny):
                x = i - nx // 2
                y = j - ny // 2
                dist = np.sqrt(x ** 2 + y ** 2)

                # Smooth transition using a Gaussian-like function
                if dist < radius:
                    T_target[i, j] = 0.1
                elif dist < radius + 5 * sigma:
                    # Transition region
                    normalized_dist = (dist - radius) / sigma
                    T_target[i, j] = 0.8 * np.exp(-0.5 * normalized_dist ** 2)
                else:
                    T_target[i, j] = 0.1

        # Optional: Add a small gradient to make the background not completely zero
        background_gradient = 0.1
        for i in range(nx):
            for j in range(ny):
                # Add a slight radial gradient to the background
                T_target[i, j] += background_gradient * (1 - np.sqrt(i ** 2 + j ** 2) / (nx * np.sqrt(2)))
                # Clip values to ensure they stay within [0, 1]
                T_target[i, j] = np.clip(T_target[i, j], 0, 1)

        time_steps = 100
        N = nx * ny
        T = np.zeros((time_steps + 1, N))
        T[0, :] = T_init.flatten()
        u_sequence = np.zeros((time_steps, M))
        x_target = torch.tensor(T_target.flatten(), dtype=torch.float32).unsqueeze(0).to(device)
        y_target = model.encoder(x_target)
        control_mses = []  # Record control MSEs over time steps
        for t in range(time_steps):
            x_t_np = T[t, :]
            x_t = torch.tensor(x_t_np, dtype=torch.float32).unsqueeze(0).to(device)
            y_t = model.encoder(x_t)
            u_bar = model.compute_control(y_t, y_target, K)
            u_bar = u_bar / dt
            u_sequence[t, :] = u_bar

            T_t1 = pde.simulate(x_t_np.reshape(nx, ny), np.array([u_bar]), [t * dt, (t + 1) * dt])
            T[t + 1, :] = T_t1[-1]
            # Compute control MSE at this time step
            control_mse = np.mean((T[t + 1].reshape(nx, ny) - T_target) ** 2)
            control_mses.append(control_mse)

        # Record control MSEs over time steps
        control_mses_over_time.append(control_mses)

    # Visualization of results
    plt.rcParams['font.sans-serif'] = ['Heiti TC']  # 设置中文字体为黑体
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    # Plot results
    plt.figure(figsize=(12, 6))

    # Plot training errors over epochs
    plt.subplot(1, 2, 1)
    for i, (x_interval, y_interval) in enumerate(control_intervals):
        label = f"x_interval={x_interval}, y_interval={y_interval}"
        plt.plot(range(len(train_errors_over_time[i])), train_errors_over_time[i], label=label)
    plt.xlabel("迭代次数")
    plt.ylabel("训练误差")
    plt.title("模型训练误差随时间变化情况")
    plt.legend()

    # Plot control MSEs over time steps
    plt.subplot(1, 2, 2)
    for i, (x_interval, y_interval) in enumerate(control_intervals):
        label = f"x_interval={x_interval}, y_interval={y_interval}"
        plt.plot(range(len(control_mses_over_time[i])), control_mses_over_time[i], label=label)
    plt.xlabel("时间步数")
    plt.ylabel("控制MSE（对数刻度）")
    plt.title("控制MSE随时间变化情况（对数刻度）")
    plt.yscale('log')  # 设置y轴为对数刻度
    plt.legend()

    plt.tight_layout()
    plt.savefig("4.png")
    plt.show()
    plt.close()


if __name__ == "__main__":
    main()