# DEPRECATED, он заменён кодом 1_4_(4, 6).py


import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, ifft, fftfreq
import os

# --- Общие параметры ---
a, t1, t2 = 4, -17, 27
T = 2.0  # Фиксированная постоянная времени фильтра
np.random.seed(42)

# --- Временная ось ---
N = 4000 # Увеличим количество точек для большей точности FFT
t = np.linspace(t1 - 20, t2 + 20, N, endpoint=False)
dt = t[1] - t[0]
g = np.where((t >= t1) & (t <= t2), a, 0)

# --- Сетка параметров для исследования (такая же, как в 1.4.3) ---
param_combinations = [
    {'b': 0.5, 'c': 0,   'd': 0},      # Только слабый белый шум
    {'b': 2.0, 'c': 0,   'd': 0},      # Только сильный белый шум
]

# --- Функция для генерации и сохранения ---
def create_and_save_plot(filename, data):
    """Создает и сохраняет один сравнительный график."""
    t, y_lsim, y_fft, title = data
    
    plt.figure(figsize=(12, 6))
    
    # --- ИЗМЕНЕНИЯ ЗДЕСЬ ---
    plt.plot(t, y_lsim, label=r'$y(t) = W_1(p)u(t)$', linewidth=2.5, color='blue')
    plt.plot(t, y_fft, label=r'$y(t) = \mathcal{F}^{-1}\{W_1(i\omega)\hat{u}(\omega)\}$', linewidth=1.5, color='red', linestyle='--')

    plt.title(title, fontsize=16)
    plt.xlabel('$t$', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()

    save_dir = r"images\1_4\1_4_4"
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=300)
    plt.close()
    print(f"График сохранен: {filepath}")

# --- Основной цикл ---
# Фильтр во временной области
W1_system = signal.TransferFunction([1], [T, 1])
t_sim = t - t[0]

# Частотная ось для FFT
omega = 2 * np.pi * fftfreq(N, dt)

# Частотная характеристика фильтра
W1_freq = 1 / (1 + 1j * omega * T)

for i, params in enumerate(param_combinations):
    b, c, d = params['b'], params['c'], params['d']
    
    # 1. Генерируем зашумленный сигнал u(t)
    xi = 2 * np.random.rand(len(t)) - 1
    noise = b * xi + c * np.sin(d * t)
    u = g + noise
    
    # 2. Фильтрация во временной области (lsim)
    _, y_lsim, _ = signal.lsim(W1_system, u, t_sim)
    
    # 3. Фильтрация в частотной области (через FFT)
    # 3.1. Прямое преобразование Фурье от u(t)
    u_hat = fft(u)
    
    # 3.2. Умножение на АЧХ фильтра
    y_hat = u_hat * W1_freq
    
    # 3.3. Обратное преобразование Фурье
    y_fft = ifft(y_hat).real
    
    # --- Формирование заголовка и имени файла ---
    if c == 0:
        title = f'Сравнение методов фильтрации при $b={b}, c=0$'
        filename = f'task_1_4_4_case_{i+1}_b{b}_c0.png'
    else:
        title = f'Сравнение методов фильтрации при $b={b}, c={c}, d={d}$'
        filename = f'task_1_4_4_case_{i+1}_b{b}_c{c}_d{d}.png'
        
    plot_data = (t, y_lsim, y_fft, title)
    create_and_save_plot(filename, plot_data)

print("\nВсе графики для пункта 1.4.4 сгенерированы.")
