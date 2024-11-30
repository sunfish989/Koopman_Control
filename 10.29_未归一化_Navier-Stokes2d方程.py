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

        # Grid coordinates
        self.x = np.linspace(0, Lx, nx)
        self.y = np.linspace(0, Ly, ny)
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing='ij')

        # Control positions every 4 positions in both x and y
        self.control_positions = []
        for i in range(0, nx, 2):
            for j in range(0, ny, 2):
                self.control_positions.append((i, j))
        self.M = len(self.control_positions)

        # Control influence matrix B, size (nx*ny, M)
        self.B = np.zeros((nx, ny, self.M))
        for k, (i, j) in enumerate(self.control_positions):
            self.B[i, j, k] = 1.0

    def laplacian(self, f):
        """Compute the Laplacian of f using central differences."""
        lap = np.zeros_like(f)
        lap[1:-1, 1:-1] = (
            (f[2:, 1:-1] - 2*f[1:-1, 1:-1] + f[0:-2, 1:-1]) / self.dx**2 +
            (f[1:-1, 2:] - 2*f[1:-1, 1:-1] + f[1:-1, 0:-2]) / self.dy**2
        )
        # Neumann BCs (zero normal derivative at boundaries)
        # Left and right boundaries
        lap[0, 1:-1] = (
            (f[1, 1:-1] - f[0, 1:-1]) / self.dx**2 +
            (f[0, 2:] - 2*f[0, 1:-1] + f[0, 0:-2]) / self.dy**2
        )
        lap[-1, 1:-1] = (
            (f[-2, 1:-1] - f[-1, 1:-1]) / self.dx**2 +
            (f[-1, 2:] - 2*f[-1, 1:-1] + f[-1, 0:-2]) / self.dy**2
        )
        # Bottom and top boundaries
        lap[1:-1, 0] = (
            (f[2:, 0] - 2*f[1:-1, 0] + f[0:-2, 0]) / self.dx**2 +
            (f[1:-1, 1] - f[1:-1, 0]) / self.dy**2
        )
        lap[1:-1, -1] = (
            (f[2:, -1] - 2*f[1:-1, -1] + f[0:-2, -1]) / self.dx**2 +
            (f[1:-1, -2] - f[1:-1, -1]) / self.dy**2
        )
        return lap

    def advect(self, u, v, f):
        """Compute the advection term."""
        dudx = (f[1:-1, 1:-1] - f[0:-2, 1:-1]) / self.dx
        dvdy = (f[1:-1, 1:-1] - f[1:-1, 0:-2]) / self.dy
        advection = u[1:-1, 1:-1] * dudx + v[1:-1, 1:-1] * dvdy
        return advection

    def streamfunction_poisson(self, omega):
        """Solve Poisson equation for streamfunction: ∇²ψ = -ω."""
        psi = np.zeros_like(omega)
        # Use iterative solver; for simplicity, use Gauss-Seidel
        for iteration in range(1000):
            psi_old = psi.copy()
            psi[1:-1, 1:-1] = 0.25 * (
                psi[2:, 1:-1] + psi[0:-2, 1:-1] +
                psi[1:-1, 2:] + psi[1:-1, 0:-2] +
                self.dx * self.dy * (-omega[1:-1, 1:-1])
            )
            # Apply boundary conditions
            # For this example, assume psi = 0 at boundaries
            max_diff = np.max(np.abs(psi - psi_old))
            if max_diff < 1e-6:
                break
        return psi

    def compute_velocity(self, psi):
        """Compute velocities u and v from streamfunction ψ."""
        u = np.zeros_like(psi)
        v = np.zeros_like(psi)
        # Central differences for interior points
        u[1:-1, 1:-1] = (psi[1:-1, 2:] - psi[1:-1, 0:-2]) / (2 * self.dy)
        v[1:-1, 1:-1] = -(psi[2:, 1:-1] - psi[0:-2, 1:-1]) / (2 * self.dx)
        # Boundary points
        # Assuming no-slip boundaries (u = v = 0 at boundaries)
        return u, v

    def simulate(self, omega0, u_sequence, t_span, time_steps):
        dt = (t_span[1] - t_span[0]) / time_steps
        times = np.linspace(t_span[0], t_span[1], time_steps + 1)

        omega = np.zeros((time_steps + 1, self.nx, self.ny))
        omega[0, :, :] = omega0

        for t in range(time_steps):
            omega_t = omega[t, :, :]
            u_t = u_sequence[t, :]  # Shape (M,)

            omega_t1 = self.step(omega_t, u_t, dt)
            omega[t + 1, :, :] = omega_t1

        return omega  # Shape (time_steps + 1, nx, ny)

    def step(self, omega, u_t, dt):
        # Add control input
        B_u = np.sum(self.B * u_t[np.newaxis, np.newaxis, :], axis=2)  # Shape (nx, ny)

        # Solve for streamfunction ψ
        psi = self.streamfunction_poisson(omega)

        # Compute velocities
        u, v = self.compute_velocity(psi)

        # Compute Laplacian of ω
        lap_omega = self.laplacian(omega)

        # Compute advection term
        conv_omega = np.zeros_like(omega)
        # Central differences for advection
        conv_omega[1:-1, 1:-1] = (
            u[1:-1, 1:-1] * (omega[1:-1, 2:] - omega[1:-1, 0:-2]) / (2 * self.dx) +
            v[1:-1, 1:-1] * (omega[2:, 1:-1] - omega[0:-2, 1:-1]) / (2 * self.dy)
        )

        # Time derivative ∂ω/∂t
        domega_dt = -conv_omega + self.viscosity * lap_omega + B_u

        # Update ω
        omega_new = omega + dt * domega_dt

        return omega_new


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
    nx = 32  # Number of grid points in x
    ny = 32  # Number of grid points in y
    Lx = np.pi  # Domain size in x
    Ly = np.pi  # Domain size in y
    viscosity = 0.01  # Viscosity coefficient

    pde = PDE2D_NS(nx, ny, Lx, Ly, viscosity)

    control_positions = pde.control_positions  # List of (i, j)
    M = len(control_positions)  # Number of controls

    # Convert control positions to indices in flattened array
    control_indices = [i * ny + j for i, j in control_positions]

    # Generate training data
    num_samples = 500
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
        u_sequence = np.random.randn(time_steps + 1, M) * control_input_scale

        # Simulate the system
        t_span = [0, dt * time_steps]
        omega_sequence = pde.simulate(omega0, u_sequence, t_span, time_steps)

        # Build training samples
        x_t_list.append(omega_sequence[:-1, :, :].reshape(-1, nx * ny))     # x(t)
        x_t1_list.append(omega_sequence[1:, :, :].reshape(-1, nx * ny))     # x(t+1)
        u_t_list.append(u_sequence[:-1, :])

    # Convert to tensors
    x_t = torch.tensor(np.concatenate(x_t_list, axis=0), dtype=torch.float32)
    x_t1 = torch.tensor(np.concatenate(x_t1_list, axis=0), dtype=torch.float32)
    u_t = torch.tensor(np.concatenate(u_t_list, axis=0), dtype=torch.float32)

    # Move data to device
    x_t = x_t.to(device)
    x_t1 = x_t1.to(device)
    u_t = u_t.to(device)

    # Datasets and data loaders
    dataset = data.TensorDataset(x_t, x_t1, u_t)
    # Split into training and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = data.random_split(dataset, [train_size, val_size])

    train_dataloader = data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_dataloader = data.DataLoader(val_dataset, batch_size=64, shuffle=False)

    # Define model
    nxny = nx * ny
    hidden_dim = 512
    P = nxny
    model = Koopman_Model(nxny, M, hidden_dim, P, control_indices)

    # Training model
    num_epochs = 100
    patience = 10
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
            batch_x_t1_prime = batch_x_t1 - B_u_t
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
                batch_x_t1_prime = batch_x_t1 - B_u_t
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

        # Decode predicted y_t1 to get x_t1_pred
        x_t1_pred = model.decoder(y_t1_pred)

        # Compute prediction error
        prediction_error = nn.MSELoss()(x_t1_pred, batch_x_t1)

        print(f"Prediction Error: {prediction_error.item():.6f}")

        # Visualize the true and predicted omega fields
        import matplotlib.pyplot as plt

        idx = 0  # Index of sample to visualize
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

    # Control simulation parameters
    num_control_steps = 200
    dt = 0.01

    # Initial condition for simulation
    omega = np.random.randn(nx, ny) * 0.1
    omega_tensor = torch.tensor(omega.reshape(1, -1), dtype=torch.float32).to(device)

    # Target embedding (e.g., zero state)
    y_target = torch.zeros(model.P, device=device)

    # Store omega fields for visualization
    omega_history = [omega.copy()]

    for t in range(num_control_steps):
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
        omega_history.append(omega.copy())

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


