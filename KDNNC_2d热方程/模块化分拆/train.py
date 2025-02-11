# train.py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import scipy.linalg
import torch.utils.data as data
from model import PDE2D, Encoder, Decoder, Koopman_Model  # 公共模型组件


def train_model():
    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # PDE参数（与后续控制脚本保持一致）
    config = {
        'nx': 30,
        'ny': 30,
        'alpha': 0.008,
        'control_spacing': 2,
        'hidden_dim': 1024,
        'num_epochs': 60,
        'batch_size': 128
    }

    # 初始化PDE系统
    pde = PDE2D(nx=config['nx'], ny=config['ny'],
                dx=1.0 / config['nx'], dy=1.0 / config['ny'],
                alpha=config['alpha'],
                control_spacing=config['control_spacing'])

    # 生成训练数据
    num_samples = 1024
    time_steps = 50
    dt = 0.01

    # 数据生成代码与原始main函数相同...
    x_t_list = []
    x_t1_list = []
    u_t_list = []

    np.random.seed(0)  # 为了可重复性

    for _ in range(num_samples):
        # Random initial temperature distribution
        T0 = np.random.rand(config['nx'], config['ny'])

        # Generate training data with control inputs
        control_input_scale = 0.05  # Adjust the scale as needed
        # u_sequence = np.zeros((time_steps + 1, M)) * control_input_scale
        u_sequence = np.random.rand(time_steps + 1, pde.M) * control_input_scale

        # Simulate the system
        t_span = [0, dt * time_steps]
        T_sequence = pde.simulate(T0, u_sequence, t_span)  # Shape (time_steps + 1, nx*ny)

        # Build training samples
        x_t_list.append(T_sequence[1:-2, :])  # x(t)
        x_t1_list.append(T_sequence[2:-1, :])  # x(t+1)
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

    # 创建模型
    nxny = config['nx'] * config['ny']
    model = Koopman_Model(nxny=nxny,
                          M=len(pde.control_positions),
                          hidden_dim=config['hidden_dim'],
                          P=nxny,
                          control_indices=[i * config['ny'] + j for i, j in pde.control_positions],
                          device=device)

    # 训练循环代码与原始main函数相同...
    patience = 10
    best_val_loss = float('inf')
    epochs_no_improve = 0
    # Convert control positions to indices in flattened array
    control_indices = [i * pde.ny + j for i, j in pde.control_positions]
    for epoch in range(config['num_epochs']):
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




    # 保存完整模型和配置
    torch.save({
        'model_state': model.state_dict(),
        'config': config,
        'control_indices': [i * config['ny'] + j for i, j in pde.control_positions]
    }, 'koopman_model.pth')


if __name__ == "__main__":
    train_model()