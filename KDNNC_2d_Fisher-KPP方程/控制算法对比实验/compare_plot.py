import numpy as np
import matplotlib.pyplot as plt

# plt.rcParams['font.sans-serif'] = ['Heiti TC']
# plt.rcParams['axes.unicode_minus'] = False

# 设置全局字体和图片格式
plt.rcParams['font.family'] = 'serif'  # 使用衬线字体
plt.rcParams['font.size'] = 12        # 字体大小
plt.rcParams['axes.linewidth'] = 1.5  # 坐标轴宽度
plt.rcParams['lines.linewidth'] = 2   # 线条宽度
plt.rcParams['figure.dpi'] = 300      # 图片分辨率

# 加载数据
kdnn_mpc_mses = np.loadtxt('kdnn_mpc_mses.txt')
kdnn_mpc_energies = np.loadtxt('kdnn_mpc_energies.txt')
kdnnc_lqr_mses = np.loadtxt('KDNNC_LQR_MSE.txt')[:2000]
kdnnc_lqr_energies = np.loadtxt('KDNNC_LQR_Energies.txt')[:2000]
mpc_mses = np.loadtxt('mpc_control_mses.txt')
mpc_energies = np.loadtxt('mpc_control_energies.txt')
pid_mses = np.loadtxt('pid_control_mses.txt')[:2000]
pid_energies = np.loadtxt('pid_control_energies.txt')[:2000]

# 创建时间轴（假设所有文件的时间步数相同）
time_steps = len(kdnn_mpc_mses)
time_axis = np.arange(time_steps)

# 绘制控制MSE对比图
plt.figure(figsize=(8, 6))
plt.plot(time_axis, kdnn_mpc_mses, label='KDNN-MPC', color='blue', linestyle='-')
plt.plot(time_axis, kdnnc_lqr_mses, label='KDNNC-LQR', color='red', linestyle='--')
plt.plot(time_axis, mpc_mses, label='MPC', color='green', linestyle='-.')
plt.plot(time_axis, pid_mses, label='PID', color='purple', linestyle=':')

# 添加图例、标题和坐标轴标签
plt.legend(loc='upper right', fontsize=10, frameon=True, edgecolor='black')
plt.title('Control MSE Comparison', fontsize=14, fontweight='bold')
plt.xlabel('Time Steps', fontsize=12)
plt.ylabel('Mean Squared Error (MSE)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)  # 添加网格线
plt.tight_layout()  # 自动调整布局
plt.savefig('control_mse_comparison.png', bbox_inches='tight')  # 保存图片
plt.show()

# 绘制控制能量消耗对比图
plt.figure(figsize=(8, 6))
plt.plot(time_axis, kdnn_mpc_energies, label='KDNN-MPC', color='blue', linestyle='-')
plt.plot(time_axis, kdnnc_lqr_energies, label='KDNNC-LQR', color='red', linestyle='--')
plt.plot(time_axis, mpc_energies, label='MPC', color='green', linestyle='-.')
plt.plot(time_axis, pid_energies, label='PID', color='purple', linestyle=':')

# 添加图例、标题和坐标轴标签
plt.legend(loc='upper left', fontsize=10, frameon=True, edgecolor='black')
plt.title('Control Energy Consumption Comparison', fontsize=14, fontweight='bold')
plt.xlabel('Time Steps', fontsize=12)
plt.ylabel('Energy Consumption', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)  # 添加网格线
plt.tight_layout()  # 自动调整布局
plt.savefig('control_energy_comparison.png', bbox_inches='tight')  # 保存图片
plt.show()
