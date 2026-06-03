import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftshift, fftfreq
import os

# --- Общие параметры ---
a, t1, t2 = 4, -17, 27
T = 2.0  # Фиксированная постоянная времени фильтра
np.random.seed(42)

# --- Временная ось ---
N = 4000
t = np.linspace(t1 - 20, t2 + 20, N, endpoint=False)
dt = t[1] - t[0]
g = np.where((t >= t1) & (t <= t2), a, 0)

# --- Сетка параметров для исследования ---
param_combinations = [
    {'b': 0.5, 'c': 0,   'd': 0},      # Только слабый белый шум
    {'b': 2.0, 'c': 0,   'd': 0},      # Только сильный белый шум
]

# --- Функция для генерации и сохранения ---
def create_and_save_plot(filename, data):
    """Создает и сохраняет один сравнительный график спектров."""
    omega, g_hat_abs, u_hat_abs, y_hat_abs, title = data
    
    plt.figure(figsize=(12, 7))
    
    # Принцип "Матрёшки" для наглядности
    # Слой 1: Зашумленный (широкий, полупрозрачный фон)
    plt.plot(omega, u_hat_abs, label=r'$|\hat{u}(\omega)|$', linewidth=3.5, color='green', alpha=0.8)
    # Слой 3: Идеальный (самый узкий, контрастный, поверх всего)
    plt.plot(omega, g_hat_abs, label=r'$|\hat{g}(\omega)|$', linewidth=2, color='red', linestyle='--', alpha=1)
    # Слой 2: Отфильтрованный (уже, поверх фона)
    plt.plot(omega, y_hat_abs, label=r'$|\hat{y}(\omega)|$', linewidth=1.5, color='blue', alpha=0.7)


    plt.title(title, fontsize=16)
    plt.xlabel(r'$\omega$', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlim(-15, 15)
    plt.tight_layout()

    save_dir = r"images\1_4_5"
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=300)
    plt.close()
    print(f"График сохранен: {filepath}")

# --- Основной цикл ---
# Фильтр
W1_system = signal.TransferFunction([1], [T, 1])
t_sim = t - t[0]

# Частотная ось для FFT
omega = 2 * np.pi * fftfreq(N, dt)

# Спектр исходного сигнала g(t)
g_hat = fft(g)

for i, params in enumerate(param_combinations):
    b, c, d = params['b'], params['c'], params['d']
    
    # 1. Генерируем зашумленный сигнал u(t)
    xi = 2 * np.random.rand(len(t)) - 1
    noise = b * xi + c * np.sin(d * t)
    u = g + noise
    
    # 2. Фильтрация во временной области (lsim) для получения y(t)
    _, y_lsim, _ = signal.lsim(W1_system, u, t_sim)
    
    # 3. Вычисление спектров
    u_hat = fft(u)
    y_hat = fft(y_lsim)
    
    # 4. Получение модулей и сдвиг для центрирования на 0
    g_hat_abs = np.abs(fftshift(g_hat))
    u_hat_abs = np.abs(fftshift(u_hat))
    y_hat_abs = np.abs(fftshift(y_hat))
    omega_shifted = fftshift(omega)

    # --- Формирование заголовка и имени файла ---
    if c == 0:
        title = f'Модули фурье-образов сигналов при $b={b}, c=0$'
        filename = f'task_1_4_5_case_{i+1}_b{b}_c0.png'
    else:
        title = f'Модули фурье-образов сигналов при $b={b}, c={c}, d={d}$'
        filename = f'task_1_4_5_case_{i+1}_b{b}_c{c}_d{d}.png'
        
    plot_data = (omega_shifted, g_hat_abs, u_hat_abs, y_hat_abs, title)
    create_and_save_plot(filename, plot_data)

print("\nВсе графики для пункта 1.4.5 сгенерированы.")