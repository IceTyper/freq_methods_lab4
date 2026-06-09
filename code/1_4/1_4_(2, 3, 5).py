import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import os

# --- Общие параметры моделирования ---
t1, t2 = -17, 27
b = 0.8  # Амплитуда шума
np.random.seed(42)
t = np.linspace(t1 - 20, t2 + 20, 2000)
dt = t[1] - t[0]
xi = 2 * np.random.rand(len(t)) - 1

def plot_and_save_two_panels(filename, t, g, u, y_filt):
    """
    Генерация двухпанельного комплекса графиков (временная и частотная области)
    без внутренних заголовков и с корректными обозначениями по методичке.
    """
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))
    
    # Вычисление Фурье-образов (Спектральная плотность)
    fft_g = np.fft.fft(g) * dt
    fft_u = np.fft.fft(u) * dt
    fft_y = np.fft.fft(y_filt) * dt
    freqs = np.fft.fftfreq(len(t), d=dt)
    omega_fft = 2 * np.pi * freqs
    
    # Маска для отображения положительных частот в линейном масштабе
    mask = (omega_fft >= 0) & (omega_fft <= 25)
    w_plot = omega_fft[mask]

    # --- Левый субплот: Временные диаграммы сигналов ---
    axs[0].plot(t, g, label=r'Исходный $g(t)$', color='blue', lw=2)
    axs[0].plot(t, u, label=r'Зашумленный $u(t)$', color='orange', alpha=0.4, lw=1)
    axs[0].plot(t, y_filt, label=r'Фильтрованный $y(t)$', color='green', lw=2.5)
    axs[0].set_xlabel(r'$t$', fontsize=14)
    axs[0].set_ylabel(r'Амплитуда', fontsize=14)
    axs[0].legend(fontsize=14, loc='best')
    axs[0].grid(True, linestyle='--', alpha=0.6)
    axs[0].tick_params(labelsize=14)

    # --- Правый субплот: Модули Фурье-образов ---
    # Шум убираем на задний план (zorder=1) и делаем более прозрачным
    axs[1].plot(w_plot, np.abs(fft_u[mask]), label=r'$|\hat{u}(\omega)|$', color='orange', alpha=0.25, lw=1, zorder=1)
    # Исходный спектр делаем штриховым (zorder=2)
    axs[1].plot(w_plot, np.abs(fft_g[mask]), label=r'$|\hat{g}(\omega)|$', color='blue', lw=2, linestyle='--', zorder=2)
    # Фильтрованный спектр выводим на передний план жирной линией (zorder=3)
    axs[1].plot(w_plot, np.abs(fft_y[mask]), label=r'$|\hat{y}(\omega)|$', color='green', lw=2.5, linestyle='-', zorder=3)
    
    axs[1].set_xlabel(r'$\omega$', fontsize=14)
    axs[1].set_ylabel(r'Фурье-образ', fontsize=14)
    axs[1].legend(fontsize=14, loc='best')
    axs[1].grid(True, linestyle='--', alpha=0.6)
    axs[1].tick_params(labelsize=14)

    plt.tight_layout()
    
    # Сохранение по относительному пути начиная с папки images
    save_dir = os.path.join("images", "1_4", "1_4_2")
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, filename), dpi=300)
    plt.close(fig)


# --- 1. Полноценное исследование влияния T (при репрезентативной выборке, a=4) ---
a_fixed = 4
T_cases = [0.2, 0.5, 1.0, 2.0, 5.0, 15.0]  # Расширенный спектр параметров

g_fixed = np.where((t >= t1) & (t <= t2), a_fixed, 0)
u_fixed = g_fixed + b * xi

for i, T in enumerate(T_cases):
    system = signal.TransferFunction([1], [T, 1])
    _, y_filt, _ = signal.lsim(system, u_fixed, t - t[0])
    
    filename = f"filtration_T_case_{i+1}_T_{T}.png"
    plot_and_save_two_panels(filename, t, g_fixed, u_fixed, y_filt)


# --- 2. Полноценное исследование влияния амплитуды сигнала а (T=2.0) ---
T_fixed = 2.0
a_cases = [0.5, 1.5, 3.0, 5.0, 8.0]  # Расширенный набор амплитуд

system_fixed = signal.TransferFunction([1], [T_fixed, 1])

for i, a in enumerate(a_cases):
    g = np.where((t >= t1) & (t <= t2), a, 0)
    u = g + b * xi
    _, y_filt, _ = signal.lsim(system_fixed, u, t - t[0])

    filename = f"filtration_a_case_{i+1}_a_{a}.png"
    plot_and_save_two_panels(filename, t, g, u, y_filt)