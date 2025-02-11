# model.py
import numpy as np
from scipy.integrate import solve_ivp
import torch
import torch.nn as nn
import scipy.linalg
import torch.optim as optim


class PDE2D:
    # 保持原始PDE2D类不变，增加control_spacing参数
    def __init__(self, nx, ny, dx, dy, alpha, control_spacing=2):
        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.dy = dy
        self.alpha = alpha
        # Build the Laplacian operator for 2D
        self.L = self.build_laplacian(nx, ny, dx, dy)
        self.control_spacing = control_spacing

        # Control positions every 2 positions in both x and y
        self.control_positions = []
        # 修改控制点生成逻辑
        self.control_positions = []
        for i in range(0, nx, control_spacing):
            for j in range(0, ny, control_spacing):
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
                    # Zero flux at boundary
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
        time_steps = len(u_sequence)
        dt = (t_span[1] - t_span[0]) / time_steps
        times = np.linspace(t_span[0], t_span[1], time_steps + 1)

        N = self.nx * self.ny
        T = np.zeros((time_steps + 1, N))
        T[0, :] = T0.flatten()

        for t in range(time_steps):
            u_t = u_sequence[t, :]  # Shape (M,)

            sol = solve_ivp(
                fun=lambda t, T_flat: self.odefun(t, T_flat, u_t),
                t_span=[times[t], times[t + 1]],
                y0=T[t, :],
                method='RK45'
            )
            T[t + 1, :] = sol.y[:, -1]

        return T  # Shape (time_steps + 1, nx*ny)

    def odefun(self, t, T_flat, u_t):
        # Control influence
        B_u = np.dot(self.B, u_t)  # Shape (N,)
        # dT/dt = L * T + B * u
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


# 保持原始Decoder类不变...
class Koopman_Model:
    def __init__(self, nxny, M, hidden_dim, P, control_indices, device):
        super().__init__()
        self.nxny = nxny  # Original state dimension
        self.M = M  # Number of controlled nodes
        self.P = P  # Embedding space dimension
        self.control_indices = control_indices  # Control indices
        self.device = device
        # 修改初始化方式
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

    def state_dict(self):
        return {
            'encoder': self.encoder.state_dict(),
            'decoder': self.decoder.state_dict(),
            'A': self.A.detach().cpu().numpy()
        }

    def load_state_dict(self, state_dict):
        self.encoder.load_state_dict(state_dict['encoder'])
        self.decoder.load_state_dict(state_dict['decoder'])
        with torch.no_grad():
            self.A.copy_(torch.tensor(state_dict['A']))