import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, ifft, fftfreq, fftshift
import os

# --- Общие параметры моделирования ---
t1, t2 = -17, 27
b = 0.8  # Амплитуда шума (согласовано с пунктом 1.4.2)
np.random.seed(42)

# Сетка на N=4000 точек для высокой точности БПФ
N = 4000
t = np.linspace(t1 - 20, t2 + 20, N)
dt = t[1] - t[0]
xi = 2 * np.random.rand(len(t)) - 1
noise = b * xi

def plot_and_save_verification(filename, t, y_time, y_freq, omega, y_hat_abs, y_hat_prod_abs, save_dir):
    """
    Генерация двухпанельного комплекса графиков без внутренних заголовков.
    Левая панель (1.4.4) — Временная область.
    Правая панель (1.4.6) — Частотная область (линейный масштаб).
    """
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))
    
    # --- Левый субплот: Временная область (Пункт 1.4.4) ---
    axs[0].plot(t, y_time, label=r'y(t) = $W_1(p)u(t)$', color='navy', lw=4)
    axs[0].plot(t, y_freq, label=r'y(t) = $\mathcal{F}^{-1}\{W_1(i\omega)\hat{u}(\omega)\}$', 
                color='orange', lw=2, linestyle=(0, (5, 5)))
    axs[0].set_xlabel(r'$t$', fontsize=14)
    axs[0].set_ylabel(r'Амплитуда', fontsize=14)
    axs[0].legend(fontsize=14, loc='lower center')
    axs[0].grid(True, linestyle='--', alpha=0.6)
    axs[0].tick_params(labelsize=14)

    # --- Правый субплот: Частотная область (Пункт 1.4.6) ---
    omega_shifted = fftshift(omega)
    y_hat_abs_shifted = fftshift(y_hat_abs)
    y_hat_prod_abs_shifted = fftshift(y_hat_prod_abs)
    
    # Маска диапазона частот для наглядности отображения на линейной сетке
    mask = (omega_shifted >= -15) & (omega_shifted <= 15)
    
    axs[1].plot(omega_shifted[mask], y_hat_prod_abs_shifted[mask], 
                label=r'$|W_1(i\omega) \cdot \hat{u}(\omega)|$', color='navy', lw=4, alpha=0.8)
    axs[1].plot(omega_shifted[mask], y_hat_abs_shifted[mask], 
                label=r'$|\hat{y}(\omega)|$', color='orange', lw=1, linestyle=(0, (5, 5)))
    axs[1].set_xlabel(r'$\omega$', fontsize=14)
    axs[1].set_ylabel(r'Модуль Фурье-образа', fontsize=14)
    axs[1].legend(fontsize=14, loc='best')
    axs[1].grid(True, linestyle='--', alpha=0.6)
    axs[1].tick_params(labelsize=14)
    axs[1].set_xlim(-15, 15)

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, filename), dpi=300)
    plt.close(fig)


# --- 1. Исследование влияния T (при постоянной амплитуде a=4) ---
a_fixed = 4
T_cases = [0.2, 0.5, 1.0, 2.0, 5.0, 15.0]
g_fixed = np.where((t >= t1) & (t <= t2), a_fixed, 0)
u_fixed = g_fixed + noise

save_dir_T = os.path.join("images", "1_4", "1_4_(4, 6)", "verification_T")

for i, T in enumerate(T_cases):
    # 1. Сначала рассчитываем частотную характеристику и БПФ
    omega = 2 * np.pi * fftfreq(N, dt)
    W1_freq = 1 / (1 + 1j * omega * T)
    u_hat = fft(u_fixed)
    
    y_hat_prod = u_hat * W1_freq
    y_fft = ifft(y_hat_prod).real
    
    # 2. Выставляем корректное начальное условие на основе БПФ-решения
    x0_correct = [y_fft[0]]
    
    # 3. Моделируем во временной области через State-Space для устранения скрытого деления состояния на T
    system_ss = signal.StateSpace([[-1.0 / T]], [[1.0 / T]], [[1.0]], [[0.0]])
    _, y_lsim, _ = signal.lsim(system_ss, u_fixed, t - t[0], X0=x0_correct)
    
    y_hat = fft(y_lsim)
    
    filename = f"verification_T_case_{i+1}_T_{T}.png"
    plot_and_save_verification(filename, t, y_lsim, y_fft, omega, np.abs(y_hat), np.abs(y_hat_prod), save_dir_T)


# --- 2. Исследование влияния амплитуды сигнала а (при постоянной T=2.0) ---
T_fixed = 2.0
a_cases = [0.5, 1.5, 3.0, 5.0, 8.0]
save_dir_a = os.path.join("images", "1_4", "1_4_(4, 6)", "verification_a")

system_fixed_ss = signal.StateSpace([[-1.0 / T_fixed]], [[1.0 / T_fixed]], [[1.0]], [[0.0]])
omega = 2 * np.pi * fftfreq(N, dt)
W1_freq_fixed = 1 / (1 + 1j * omega * T_fixed)

for i, a in enumerate(a_cases):
    g = np.where((t >= t1) & (t <= t2), a, 0)
    u = g + noise
    
    # 1. Сначала рассчитываем частотную область
    u_hat = fft(u)
    y_hat_prod = u_hat * W1_freq_fixed
    y_fft = ifft(y_hat_prod).real
    
    # 2. Корректное начальное условие для lsim
    x0_correct = [y_fft[0]]
    
    # 3. Временное моделирование через State-Space
    _, y_lsim, _ = signal.lsim(system_fixed_ss, u, t - t[0], X0=x0_correct)
    
    y_hat = fft(y_lsim)
    
    filename = f"verification_a_case_{i+1}_a_{a}.png"
    plot_and_save_verification(filename, t, y_lsim, y_fft, omega, np.abs(y_hat), np.abs(y_hat_prod), save_dir_a)

print("Все совмещенные двухпанельные графики верификации успешно сформированы и сохранены.")