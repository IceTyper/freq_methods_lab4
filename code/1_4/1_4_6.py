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
    {'b': 0.5, 'c': 0,   'd': 0},
    {'b': 2.0, 'c': 0,   'd': 0},
]

# --- Функция для генерации и сохранения ---
def create_and_save_plot(filename, data):
    """Создает и сохраняет один сравнительный график Фурье-образов."""
    omega, y_hat_abs, y_hat_prod_abs, title = data
    
    plt.figure(figsize=(12, 7))
    
    # Рисуем один график поверх другого для демонстрации совпадения
    # Широкая линия - теоретический результат
    plt.plot(omega, y_hat_prod_abs, label=r'$|W_1(i\omega) \cdot \hat{u}(\omega)|$', linewidth=3.5, color='blue', alpha=0.7)
    # Тонкая контрастная линия - практический результат
    plt.plot(omega, y_hat_abs, label=r'$|\hat{y}(\omega)|$', linewidth=1.5, color='red', linestyle='--', alpha=1)

    plt.title(title, fontsize=16)
    plt.xlabel(r'$\omega$', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlim(-15, 15)
    plt.tight_layout()

    save_dir = r"images\1_4\1_4_6"
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=300)
    plt.close()
    print(f"График сохранен: {filepath}")

# --- Основной цикл ---
# Фильтр во временной области
W1_system = signal.TransferFunction([1], [T, 1])
t_sim = t - t[0]

# Частотная ось и АЧХ фильтра
omega = 2 * np.pi * fftfreq(N, dt)
W1_freq = 1 / (1 + 1j * omega * T)

for i, params in enumerate(param_combinations):
    b, c, d = params['b'], params['c'], params['d']
    
    # 1. Генерируем зашумленный сигнал u(t)
    xi = 2 * np.random.rand(len(t)) - 1
    noise = b * xi + c * np.sin(d * t)
    u = g + noise
    
    # 2. Фильтрация во временной области для получения y(t)
    _, y_lsim, _ = signal.lsim(W1_system, u, t_sim)
    
    # 3. Вычисление Фурье-образа y_hat из y(t)
    y_hat = fft(y_lsim)
    
    # 4. Вычисление Фурье-образа y_hat_prod через умножение в частотной области
    u_hat = fft(u)
    y_hat_prod = u_hat * W1_freq
    
    # 5. Получение модулей и сдвиг для центрирования на 0
    y_hat_abs = np.abs(fftshift(y_hat))
    y_hat_prod_abs = np.abs(fftshift(y_hat_prod))
    omega_shifted = fftshift(omega)

    # --- Формирование заголовка и имени файла ---
    if c == 0:
        title = f'Сравнение Фурье-образов при $b={b}, c=0$'
        filename = f'task_1_4_6_case_{i+1}_b{b}_c0.png'
    else:
        title = f'Сравнение Фурье-образов при $b={b}, c={c}, d={d}$'
        filename = f'task_1_4_6_case_{i+1}_b{b}_c{c}_d{d}.png'
        
    plot_data = (omega_shifted, y_hat_abs, y_hat_prod_abs, title)
    create_and_save_plot(filename, plot_data)

print("\nВсе графики для пункта 1.4.6 сгенерированы.")