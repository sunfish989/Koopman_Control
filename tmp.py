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
        lap = np.roll(Z, 1, axis=0) + np.roll(Z, -1, axis=0) + np.roll(Z, 1, axis=1) + np.roll(Z, -1, axis=1) - 4 * Z
        return lap / (2 * np.square(1 / self.grid_size))

    def reaction_diffusion(self, t, y, u_c, v_c):
        u = y[:self.N].reshape(self.grid_size, self.grid_size)
        v = y[self.N:].reshape(self.grid_size, self.grid_size)
        Lu = self.laplacian(u)
        Lv = self.laplacian(v)
        dudt = self.Du * Lu - u * v**2 + self.F * (1 - u) + u_c
        dvdt = self.Dv * Lv + u * v**2 - (self.F + self.k) * v + v_c
        return np.concatenate([dudt.flatten(), dvdt.flatten()])

    def simulate(self, U0, V0, u_sequence, dt=0.5):
        time_steps = len(u_sequence)
        U_sequence = np.zeros((time_steps + 1, self.grid_size, self.grid_size))
        V_sequence = np.zeros((time_steps + 1, self.grid_size, self.grid_size))
        U_sequence[0] = U0
        V_sequence[0] = V0

        for t in range(time_steps):
            u_c = np.zeros((self.grid_size, self.grid_size))
            v_c = np.zeros((self.grid_size, self.grid_size))
            for idx, (i, j) in enumerate(self.control_positions):
                u_c[i, j] = u_sequence[t, idx]
            y0 = np.concatenate([U_sequence[t].flatten(), V_sequence[t].flatten()])
            sol = solve_ivp(self.reaction_diffusion, [0, dt], y0, args=(u_c, v_c), method='RK45')
            y = sol.y[:, -1]
            U_sequence[t + 1] = y[:self.N].reshape(self.grid_size, self.grid_size)
            V_sequence[t + 1] = y[self.N:].reshape(self.grid_size, self.grid_size)
        return U_sequence, V_sequence

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
            nn.Linear(hidden_dim, P - M)
        )

    def forward(self, x):
        batch_size = x.size(0)
        y = torch.zeros(batch_size, self.P, device=x.device)
        y[:, self.control_indices] = x[:, self.control_indices]

        # Non-control positions
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
            nn.Linear(hidden_dim, nxny - M)
        )

    def forward(self, y):
        batch_size = y.size(0)
        x = torch.zeros(batch_size, self.nxny, device=y.device)
        x[:, self.control_indices] = y[:, self.control_indices]

        mask = torch.ones(self.nxny, dtype=torch.bool, device=y.device)
        mask[self.control_indices] = False
        y_non_control = y[:, mask]
        x_non_control = self.net(y_non_control)
        x[:, mask] = x_non_control
        return x

class Koopman_Model:
    def __init__(self, nxny, M, hidden_dim, P, control_indices):
        self.nxny = nxny
        self.M = M
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

    grid_size = 30
    Du = 0.08
    Dv = 0.04
    F = 0.02
    k = 0.065

    model_pde = GrayScottModel(grid_size, Du, Dv, F, k)
    control_positions = model_pde.control_positions
    M = model_pde.M
    control_indices = [i * grid_size + j for i, j in control_positions]

    num_samples = 200
    time_steps = 10
    dt = 0.5
    x_t_list = []
    x_t1_list = []
    u_t_list = []

    for _ in range(num_samples):
        U0 = np.ones((grid_size, grid_size))
        V0 = np.zeros((grid_size, grid_size))
        U0[grid_size//2 - 2:grid_size//2 + 2, grid_size//2 - 2:grid_size//2 + 2] = 0
        V0[grid_size//2 - 2:grid_size//2 + 2, grid_size//2 - 2:grid_size//2 + 2] = 1
        u_sequence = np.random.rand(time_steps + 1, M) * 0.005
        U_sequence, V_sequence = model_pde.simulate(U0, V0, u_sequence, dt)
        x_t = U_sequence[1:-2].reshape(-1, grid_size**2)
        x_t1 = U_sequence[2:-1].reshape(-1, grid_size**2)
        x_t_list.append(x_t)
        x_t1_list.append(x_t1)
        u_t_list.append(u_sequence[1:-1])

    x_t = torch.tensor(np.concatenate(x_t_list, axis=0), dtype=torch.float32).to(device)
    x_t1 = torch.tensor(np.concatenate(x_t1_list, axis=0), dtype=torch.float32).to(device)
    u_t = torch.tensor(np.concatenate(u_t_list, axis=0), dtype=torch.float32).to(device)

    dataset = data.TensorDataset(x_t, x_t1, u_t)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = data.random_split(dataset, [train_size, val_size])
    train_dataloader = data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_dataloader = data.DataLoader(val_dataset, batch_size=64, shuffle=False)

    nxny = grid_size**2
    hidden_dim = 512
    P = grid_size**2
    model = Koopman_Model(nxny, M, hidden_dim, P, control_indices)

    num_epochs = 200
    patience = 10
    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        model.encoder.train()
        model.decoder.train()
        total_loss = 0
        for batch_x_t, batch_x_t1, batch_u_t in train_dataloader:
            B_u_t = torch.zeros_like(batch_x_t1)
            B_u_t[:, control_indices] = batch_u_t
            batch_x_t1_prime = batch_x_t1 - B_u_t
            loss = model.train_step(batch_x_t, batch_x_t1_prime)
            total_loss += loss
        avg_train_loss = total_loss / len(train_dataloader)

        model.encoder.eval()
        model.decoder.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x_t, batch_x_t1, batch_u_t in val_dataloader:
                B_u_t = torch.zeros_like(batch_x_t1)
                B_u_t[:, control_indices] = batch_u_t
                batch_x_t1_prime = batch_x_t1 - B_u_t
                loss = model.compute_loss(batch_x_t, batch_x_t1_prime)
                val_loss += loss
        avg_val_loss = val_loss / len(val_dataloader)
        print(f"Epoch {epoch + 1}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping!")
                break

    with torch.no_grad():
        batch_x_t, batch_x_t1, batch_u_t = next(iter(val_dataloader))
        y_t = model.encoder(batch_x_t)
        y_t1_pred = torch.matmul(y_t, model.A.T)
        x_t1_pred_prime = model.decoder(y_t1_pred)
        B_u = torch.zeros_like(batch_x_t1)
        B_u[:, control_indices] = batch_u_t
        x_t1_pred = x_t1_pred_prime + B_u
        pred_error = nn.MSELoss()(x_t1_pred, batch_x_t1)
        print(f"Prediction error on validation batch: {pred_error.item():.6f}")

        idx = 0
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

    K = model.design_lqr_controller()

    U_target = np.zeros((grid_size, grid_size))
    U_target[grid_size//8:3*grid_size//8, grid_size//8:3*grid_size//8] = 1

    U_init = np.ones((grid_size, grid_size))
    V_init = np.zeros((grid_size, grid_size))
    U_init[grid_size//2 - 2:grid_size//2 + 2, grid_size//2 - 2:grid_size//2 + 2] = 0
    V_init[grid_size//2 - 2:grid_size//2 + 2, grid_size//2 - 2:grid_size//2 + 2] = 1

    time_steps = 500
    T = np.zeros((time_steps + 1, grid_size**2))
    T[0, :] = U_init.flatten()
    u_sequence = np.zeros((time_steps, M))

    x_target = torch.tensor(U_target.flatten(), dtype=torch.float32).unsqueeze(0).to(device)
    y_target = model.encoder(x_target)

    for t in range(time_steps):
        x_t_np = T[t, :]
        x_t = torch.tensor(x_t_np, dtype=torch.float32).unsqueeze(0).to(device)
        y_t = model.encoder(x_t)
        u_bar = model.compute_control(y_t, y_target, K)
        u_sequence[t, :] = u_bar
        U_t = x_t_np.reshape(grid_size, grid_size)
        V_t = V_init
        u_c = np.zeros((grid_size, grid_size))
        for idx, (i, j) in enumerate(control_positions):
            u_c[i, j] = u_bar[idx]
        U_t1, V_t1 = model_pde.simulate(U_t, V_t, [u_bar], dt=dt)
        T[t + 1, :] = U_t1[1].flatten()
        V_init = V_t1[1]

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