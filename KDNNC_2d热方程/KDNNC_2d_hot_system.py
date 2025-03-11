import numpy as np
from scipy.integrate import solve_ivp
import torch
import torch.nn as nn
import torch.optim as optim
import scipy.linalg
import torch.utils.data as data
import matplotlib.pyplot as plt


class PDE2D:
    def __init__(self, nx, ny, dx, dy, alpha):
        self.nx = nx  # Number of spatial nodes in x
        self.ny = ny  # Number of spatial nodes in y
        self.dx = dx  # Spatial step size in x
        self.dy = dy  # Spatial step size in y
        self.alpha = alpha  # Thermal diffusivity coefficient

        # Build the Laplacian operator for 2D
        self.L = self.build_laplacian(nx, ny, dx, dy)

        # Control positions every 2 positions in both x and y
        self.control_positions = []
        for i in range(0, nx, 2):
            for j in range(0, ny, 2):
                self.control_positions.append((i, j))
        self.M = len(self.control_positions)

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
                # Left neighbor
                if i > 0:
                    idx_left = (i - 1) * ny + j
                    L[idx, idx_left] += self.alpha / dx ** 2
                else:
                    L[idx, idx] += self.alpha / dx ** 2

                # Right neighbor
                if i < nx - 1:
                    idx_right = (i + 1) * ny + j
                    L[idx, idx_right] += self.alpha / dx ** 2
                else:
                    L[idx, idx] += self.alpha / dx ** 2

                # Down neighbor
                if j > 0:
                    idx_down = i * ny + (j - 1)
                    L[idx, idx_down] += self.alpha / dy ** 2
                else:
                    L[idx, idx] += self.alpha / dy ** 2

                # Up neighbor
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

            # Free evolution: x(t+1) = x(t) + dt * L * x(t)
            free_evolution = T[t, :] + dt * np.dot(self.L, T[t, :])

            # Control influence: B * u(t)
            control_influence = np.dot(self.B, u_t)

            # Total state update
            T[t + 1, :] = free_evolution + dt * control_influence


        return T  # Shape (time_steps + 1, nx * ny)

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

    pde = PDE2D(nx, ny, dx, dy, alpha)

    control_positions = pde.control_positions  # List of (i, j)
    M = len(control_positions)  # Number of controls
    print(M)

    # Convert control positions to indices in flattened array
    control_indices = [i * ny + j for i, j in control_positions]

    # Generate training data
    num_samples = 200
    time_steps = 600
    global dt
    dt = 0.001
    x_t_list = []
    x_t1_list = []
    u_t_list = []

    np.random.seed(0)  # 为了可重复性

    for _ in range(num_samples):
        # Random initial temperature distribution
        T0 = np.random.rand(nx, ny)

        # Generate training data with control inputs
        control_input_scale = 0.05  # Adjust the scale as needed
        # u_sequence = np.zeros((time_steps + 1, M)) * control_input_scale
        u_sequence = -control_input_scale * np.ones((time_steps+1, M)) + 2 * control_input_scale * np.random.rand(time_steps + 1, M)

        # Simulate the system
        t_span = [0, dt * time_steps]
        T_sequence = pde.simulate(T0, u_sequence, t_span)  # Shape (time_steps + 1, nx*ny)

        # Build training samples
        x_t_list.append(T_sequence[1:-2, :])  # x(t)
        x_t1_list.append(T_sequence[2:-1, :])  # x(t+1)
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

    # Define model
    nxny = nx * ny
    hidden_dim = 512
    P = nxny
    model = Koopman_Model(nxny, M, hidden_dim, P, control_indices, device)

    # Training model
    num_epochs = 100
    patience = 8
    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        # Set training mode
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

        # Validation
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

        print(f"Epoch {epoch + 1}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")

        # Early stopping
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

        # Visualization of results
        plt.rcParams['font.sans-serif'] = ['Heiti TC']  # 设置中文字体为黑体
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(T_true, cmap='hot', origin='lower')
        plt.colorbar()
        plt.title('t+1时刻真实分布')

        plt.subplot(1, 3, 2)
        plt.imshow(T_pred, cmap='hot', origin='lower')
        plt.colorbar()
        plt.title('t+1时刻预测分布')

        plt.subplot(1, 3, 3)
        plt.imshow(T_true - T_pred, cmap='bwr', origin='lower')
        plt.colorbar()
        plt.title('误差分布')
        plt.savefig('预测效果对比.png')
        plt.show()
        plt.close()



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



    # Simulation of control process
    time_steps = 600
    N = nx * ny
    T = np.zeros((time_steps + 1, N))
    T[0, :] = T_init.flatten()
    u_sequence = np.zeros((time_steps, M))

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

        u_bar = u_bar/dt

        u_sequence[t, :] = u_bar  # Record control input

        # Simulate next time step
        T_t1 = pde.simulate(x_t_np.reshape(nx, ny), np.array([u_bar]), [t * dt, (t + 1) * dt])
        T[t + 1, :] = T_t1[-1]

    # Visualization of results
    T_full = T.reshape(time_steps + 1, nx, ny)

    # Generate intermediate states at different time points
    time_steps_for_visualization = [100, 200, 300, 400, 500, 600]
    # Visualization of results
    plt.rcParams['font.sans-serif'] = ['Heiti TC']  # 设置中文字体为黑体
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    # Plot intermediate states
    plt.figure(figsize=(15, 10))
    plt.subplot(3, 3, 1)
    plt.imshow(T_full[0], cmap='hot', origin='lower')
    plt.colorbar()
    plt.title(f'初始温度分布Init')
    for i, t in enumerate(time_steps_for_visualization):
        plt.subplot(3, 3, i + 2)
        plt.imshow(T_full[t], cmap='hot', origin='lower')
        plt.colorbar()
        plt.title(f'{t/100}s时温度分布')
    plt.subplot(3, 3, 9)
    plt.imshow(T_target, cmap='hot', origin='lower')
    plt.title(f'目标温度分布Target')

    plt.tight_layout()
    plt.savefig('3.png')
    plt.show()
    plt.close()


if __name__ == "__main__":
    main()


