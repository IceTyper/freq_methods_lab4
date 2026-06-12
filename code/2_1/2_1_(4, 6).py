import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, ifft, fftfreq, fftshift
import os

# --- Общие параметры моделирования ---
t1, t2 = -17, 27
a = 4.0          # Амплитуда полезного прямоугольного импульса
c = 2.0          # Амплитуда гармонической помехи
omega_0 = 5.0    # Центральная частота режекции фильтра

# Сетка на N=4000 точек
N = 4000
t = np.linspace(t1 - 20, t2 + 20, N)
dt = t[1] - t[0]

def plot_and_save_verification(filename, t, y_time, y_freq, omega, y_hat_abs, y_hat_prod_abs, save_dir):
    """
    Генерация двухпанельного комплекса графиков без внутренних заголовков.
    Стиль оформления полностью скопирован из Задания 1.
    """
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))
    
    # --- Левый субплот: Временная область (Пункт 2.1.4) ---
    axs[0].plot(t, y_time, label=r'y(t) = $W_1(p)u(t)$', color='navy', lw=4, alpha=0.8)
    axs[0].plot(t, y_freq, label=r'$y(t) = \mathcal{F}^{-1}\{W_2(i\omega)\hat{u}(\omega)\}$', 
                color='orange', lw=2, linestyle='--')
    axs[0].set_xlabel(r'$t$', fontsize=14)
    axs[0].set_ylabel(r'Амплитуда', fontsize=14)
    axs[0].legend(fontsize=14, loc='best')
    axs[0].grid(True, linestyle='--', alpha=0.6)
    axs[0].tick_params(labelsize=14)
    axs[0].set_xlim(t[0], t[-1])
    
    # --- Правый субплот: Частотная область (Пункт 2.1.6) ---
    omega_shifted = fftshift(omega)
    y_hat_abs_shifted = fftshift(y_hat_abs)
    y_hat_prod_abs_shifted = fftshift(y_hat_prod_abs)
    
    mask = (omega_shifted >= -15) & (omega_shifted <= 15)
    
    axs[1].plot(omega_shifted[mask], y_hat_prod_abs_shifted[mask], 
                label=r'$|W_2(i\omega) \cdot \hat{u}(\omega)|$', color='navy', lw=4, alpha=0.8)
    axs[1].plot(omega_shifted[mask], y_hat_abs_shifted[mask], 
                label=r'$|\hat{y}(\omega)|$', 
                color='orange', lw=2, linestyle='--')
    
    axs[1].set_xlabel(r'$\omega$', fontsize=14)
    axs[1].set_ylabel(r'Модуль Фурье-образа', fontsize=14)
    axs[1].legend(fontsize=14, loc='best')
    axs[1].grid(True, linestyle='--', alpha=0.6)
    axs[1].tick_params(labelsize=14)
    axs[1].set_xlim(-15, 15)
    
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, filename), dpi=300)
    plt.close()

# --- Вычислительный эксперимент и верификация ---
cases = [
    {"b1": 0.1,  "d": 5.0,  "desc": "b1_0.1_d_5.0"},
    {"b1": 0.5,  "d": 5.0,  "desc": "b1_0.5_d_5.0"},
    {"b1": 1.0,  "d": 5.0,  "desc": "b1_1.0_d_5.0"},
    {"b1": 2.0,  "d": 5.0,  "desc": "b1_2.0_d_5.0"},
    {"b1": 5.0,  "d": 5.0,  "desc": "b1_5.0_d_5.0"},
    {"b1": 10.0, "d": 5.0,  "desc": "b1_10.0_d_5.0"},
    # Блок 2: Исследование влияния частоты помехи d (при b1 = 1.0)
    {"b1": 1.0, "d": 2.0,  "desc": "b1_1.0_d_2.0"},
    {"b1": 1.0, "d": 4.0,  "desc": "b1_1.0_d_4.0"},
    {"b1": 1.0, "d": 4.9,  "desc": "b1_1.0_d_4.9"},
    {"b1": 1.0, "d": 5.1,  "desc": "b1_1.0_d_5.1"},
    {"b1": 1.0, "d": 7.0,  "desc": "b1_1.0_d_7.0"},
]

save_dir = os.path.join("images", "2_1", "2_1_(4, 6)")
omega = 2 * np.pi * fftfreq(N, dt)

for case in cases:
    b1 = case["b1"]
    d = case["d"]
    desc = case["desc"]
    
    g = np.where((t >= t1) & (t <= t2), a, 0.0)
    u = g + c * np.sin(d * t)
    
    u_hat = fft(u) * dt
    
    num_tf = [1, 0, omega_0**2]
    den_tf = [1, b1, omega_0**2]
    numerator_w = omega_0**2 - omega**2
    denominator_w = (omega_0**2 - omega**2) + 1j * b1 * omega
    W2_freq = numerator_w / denominator_w
    
    y_hat_prod = u_hat * W2_freq
    y_fft = ifft(y_hat_prod / dt).real
    
    # ИСПРАВЛЕНИЕ: Выставляем честные физические начальные условия покоя для системы 2-го порядка
    A, B, C, D = signal.tf2ss(num_tf, den_tf)
    x0_physical = [0.0, 0.0]
    
    _, y_lsim, _ = signal.lsim((A, B, C, D), u, t - t[0], X0=x0_physical)
    
    y_hat = fft(y_lsim) * dt
    
    filename = f"verification_{desc}.png"
    plot_and_save_verification(filename, t, y_lsim, y_fft, omega, np.abs(y_hat), np.abs(y_hat_prod), save_dir)
    print(f"График успешно сохранен: {filename}")