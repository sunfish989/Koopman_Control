import numpy as np
from scipy.integrate import solve_ivp
import torch
import torch.nn as nn
import torch.optim as optim
import scipy.linalg
import torch.utils.data as data
import matplotlib.pyplot as plt

class GrayScottModel:
    def __init__(self, grid_size, Du=0.02, Dv=0.03, F=0.035, k=0.065):
        self.grid_size = grid_size
        self.Du = Du
        self.Dv = Dv
        self.F = F
        self.k = k
        self.control_positions = [(i, j) for i in range(0, grid_size, 2) for j in range(0, grid_size, 2)]
        self.M = len(self.control_positions)
        self.N = grid_size * grid_size

    def laplacian(self, Z):
        return (
            np.roll(Z, +1, axis=0) + np.roll(Z, -1, axis=0) +
            np.roll(Z, +1, axis=1) + np.roll(Z, -1, axis=1) -
            4 * Z
        ) / (self.dx * self.dx)

    def reaction_diffusion(self, t, y, u_c, v_c):
        u = y[:self.grid_size ** 2].reshape(self.grid_size, self.grid_size)
        v = y[self.grid_size ** 2:].reshape(self.grid_size, self.grid_size)
        Lu = self.laplacian(u)
        Lv = self.laplacian(v)
        dudt = self.Du * Lu - u * v ** 2 + self.F * (1 - u) + u_c
        dvdt = self.Dv * Lv + u * v ** 2 - (self.F + self.k) * v + v_c
        return np.concatenate([dudt.flatten(), dvdt.flatten()])

    def step(self, u, v, dt, u_c, v_c):
        y0 = np.concatenate([u.flatten(), v.flatten()])
        sol = solve_ivp(lambda t, y: self.reaction_diffusion(t, y, u_c, v_c), [0, dt], y0, method='RK45')
        u = sol.y[:self.grid_size ** 2, -1].reshape(self.grid_size, self.grid_size)
        v = sol.y[self.grid_size ** 2:, -1].reshape(self.grid_size, self.grid_size)
        return u, v

    def simulate(self, U0, V0, u_sequence):
        time_steps = len(u_sequence)
        U_sequence = np.zeros((time_steps + 1, self.grid_size, self.grid_size))
        V_sequence = np.zeros((time_steps + 1, self.grid_size, self.grid_size))
        U_sequence[0] = U0
        V_sequence[0] = V0

        for t in range(time_steps):
            u_c = np.zeros((self.grid_size, self.grid_size))
            v_c = np.zeros((self.grid_size, self.grid_size))
            # Apply control input to u at control positions
            for idx, (i, j) in enumerate(self.control_positions):
                u_c[i, j] = u_sequence[t, idx]
            U_next, V_next = self.step(U_sequence[t], V_sequence[t], self.dt, u_c, v_c)
            U_sequence[t + 1] = U_next
            V_sequence[t + 1] = V_next

        return U_sequence, V_sequence  # Each is shape (time_steps + 1, grid_size, grid_size)

class Encoder(nn.Module):
    def __init__(self, nxny, M, hidden_dim, P, control_indices):
        super(Encoder, self).__init__()
        self.nxny = nxny  # Original state dimension (N)
        self.M = M  # Number of controls
        self.P = P  # Embedding space dimension
        self.hidden_dim = hidden_dim
        self.control_indices = control_indices  # Control indices in flattened array

        # Nonlinear mapping neural network
        self.net = nn.Sequential(
            nn.Linear(nxny - M, hidden_dim),
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
        self.nxny = nxny  # Original state dimension (N)
        self.M = M  # Number of controls
        self.P = P  # Embedding space dimension
        self.hidden_dim = hidden_dim
        self.control_indices = control_indices  # Control indices in flattened array

        # Nonlinear decoding neural network
        self.net = nn.Sequential(
            nn.Linear(P - M, hidden_dim),
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
        self.nxny = nxny  # Original state dimension (N)
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
    grid_size = 30  # Grid size
    Du = 0.08  # Diffusion coefficient for u
    Dv = 0.04  # Diffusion coefficient for v
    F = 0.02  # Feed rate
    k = 0.065  # Kill rate

    model_pde = GrayScottModel(Du=Du, Dv=Dv, F=F, k=k, grid_size=grid_size)

    control_positions = model_pde.control_positions  # List of (i, j)
    M = len(control_positions)  # Number of controls

    # Convert control positions to indices in the full state vector
    N = grid_size * grid_size
    control_indices_u = [i * grid_size + j for i, j in control_positions]
    control_indices = control_indices_u  # Indices in u variable

    # Generate training data
    num_samples = 200  # Reduce if computational resources are limited
    time_steps = 10
    dt = 0.5
    x_t_list = []
    x_t1_list = []
    u_t_list = []

    for _ in range(num_samples):
        # Random initial conditions
        U0 = np.ones((grid_size, grid_size))
        V0 = np.zeros((grid_size, grid_size))
        # Add random perturbation in the center
        U0[grid_size//2 - 2:grid_size//2 + 2, grid_size//2 - 2:grid_size//2 + 2] = 0
        V0[grid_size//2 - 2:grid_size//2 + 2, grid_size//2 - 2:grid_size//2 + 2] = 1

        # Generate control inputs
        control_input_scale = 0.005  # Adjust the scale as needed
        u_sequence = np.random.rand(time_steps + 1, M) * control_input_scale

        # Simulate the system
        U_sequence, V_sequence = model_pde.simulate(U0, V0, u_sequence)

        # Build training samples
        x_t = U_sequence[1:-2, :, :].reshape(time_steps-1, -1)
        x_t1 = U_sequence[2:-1, :, :].reshape(time_steps-1, -1)
        x_t_list.append(x_t)
        x_t1_list.append(x_t1)
        u_t_list.append(u_sequence[1:-1, :])

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
    nxny = N  # Total state size (only u variable)
    hidden_dim = 512
    P = nxny  # Embedding space dimension
    model = Koopman_Model(nxny, M, hidden_dim, P, control_indices)

    # Training model
    num_epochs = 200
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
            # Control influence applied to u variable
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

        # Decode
        x_t1_pred_prime = model.decoder(y_t1_pred)

        B_u = torch.zeros_like(batch_x_t1, device=device)
        B_u[:, control_indices] = batch_u_t  # Control influence
        # Add control influence to get predicted x(t+1)
        x_t1_pred = x_t1_pred_prime + B_u

        # Compute prediction error
        pred_error = nn.MSELoss()(x_t1_pred, batch_x_t1)
        print(f"Prediction error on validation batch: {pred_error.item():.6f}")

        # Visualization
        idx = 4  # Visualize the first sample
        U_true = batch_x_t1[idx].cpu().numpy().reshape(grid_size, grid_size)
        U_pred = x_t1_pred[idx].cpu().numpy().reshape(grid_size, grid_size)

        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(U_true, cmap='hot', origin='lower')
        plt.colorbar()
        plt.title('True U(x, y, t+1)')

        plt.subplot(1, 3, 2)
        plt.imshow(U_pred, cmap='hot', origin='lower')
        plt.colorbar()
        plt.title('Predicted U(x, y, t+1)')

        plt.subplot(1, 3, 3)
        plt.imshow(U_true - U_pred, cmap='bwr', origin='lower')
        plt.colorbar()
        plt.title('Prediction Error U')

        plt.show()

    # Design LQR controller
    K = model.design_lqr_controller()

    # Control target
    U_target = np.zeros((grid_size, grid_size))
    U_target[grid_size//8:3*grid_size//8, grid_size//8:3*grid_size//8] = 1  # Desired pattern

    # Initial state
    U_init = np.ones((grid_size, grid_size))
    V_init = np.zeros((grid_size, grid_size))
    U_init[grid_size//2 - 2:grid_size//2 + 2, grid_size//2 - 2:grid_size//2 + 2] = 0
    V_init[grid_size//2 - 2:grid_size//2 + 2, grid_size//2 - 2:grid_size//2 + 2] = 1

    # Simulation of control process
    time_steps = 500
    T = np.zeros((time_steps + 1, N))
    T[0, :] = U_init.flatten()
    u_sequence = np.zeros((time_steps, M))

    # Encode target state
    x_target = torch.tensor(U_target.flatten(), dtype=torch.float32).unsqueeze(0).to(device)
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
        U_t = x_t_np.reshape(grid_size, grid_size)
        V_t = V_init  # Assume V remains the same (simplification)
        u_c = np.zeros((grid_size, grid_size))
        v_c = np.zeros((grid_size, grid_size))
        for idx, (i, j) in enumerate(control_positions):
            u_c[i, j] = u_bar[idx]
        U_t1, V_t1 = model_pde.step(U_t, V_t, dt, u_c, v_c)
        T[t + 1, :] = U_t1.flatten()
        V_init = V_t1  # Update V for next step

    # Visualization of results
    U_full = T.reshape(time_steps + 1, grid_size, grid_size)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(U_full[0], cmap='hot', origin='lower')
    plt.colorbar()
    plt.title('U at Initial Time')

    plt.subplot(1, 2, 2)
    plt.imshow(U_full[-1], cmap='hot', origin='lower')
    plt.colorbar()
    plt.title('U at Final Time')
    plt.show()

if __name__ == "__main__":
    main()


# 我的代码虽然实现了控制趋势的对的，但是最后的值的区间不对，具体地，
#
# Control target
# U_target = np.zeros((grid_size, grid_size))
# U_target[grid_size//8:3*grid_size//8, grid_size//8:3*grid_size//8] = 1  # Desired pattern
# 最后控制的结果是grid_size//8:3grid_size//8, grid_size//8:3grid_size//8 这些区域的值在0.6附近向1靠近（但是差的确实远了点），其他区域在0.4-0.5附近，也是和0有很大差距。这是什么原因呢？
# 另外我还发现，当我增加 Simulation of control process的time_steps, U的图像数值是整体变小的，也就是说grid_size//8:3grid_size//8, grid_size//8:3grid_size//8 区域在0.4到0.5左右，
# 其他区域在0.3左右，这是什么原因？

# 会不会是我只对U施加了控制，并没有作用V的原因？ 也可能是我控制的时间步长不够长？你分析一下原因，给我对应的解决方案。