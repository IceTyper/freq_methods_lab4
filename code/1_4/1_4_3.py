import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import os

# --- Общие параметры ---
a, t1, t2 = 4, -17, 27
T = 2.0  # Фиксированная постоянная времени фильтра
np.random.seed(42)

t = np.linspace(t1 - 20, t2 + 20, 2000)
g = np.where((t >= t1) & (t <= t2), a, 0)

# --- Сетка параметров для исследования ---
param_combinations = [
    {'b': 0.5, 'c': 0,   'd': 0},      # Только слабый белый шум
    {'b': 2.0, 'c': 0,   'd': 0},      # Только сильный белый шум
]

# --- Функция для генерации и сохранения ---
def create_and_save_plot(filename, data):
    """Создает и сохраняет один график."""
    t, g, u, y_filt, title = data
    
    plt.figure(figsize=(12, 6))
    
    plt.plot(t, u, label='$u(t)$ (зашумлённый)', linewidth=1, alpha=0.5, color='orange')
    plt.plot(t, g, label='$g(t)$ (исходный)', linewidth=2.5, color='blue', linestyle='--')
    plt.plot(t, y_filt, label='$y(t)$ (фильтрованный)', linewidth=2.5, color='green')

    plt.title(title, fontsize=16)
    plt.xlabel('$t$', fontsize=14) # Изменено
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=14, loc='lower center')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()

    save_dir = r"images\1_4\1_4_3"
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=300)
    plt.close()
    print(f"График сохранен: {filepath}")

# --- Основной цикл ---
W1 = signal.TransferFunction([1], [T, 1])
t_sim = t - t[0]

for i, params in enumerate(param_combinations):
    b, c, d = params['b'], params['c'], params['d']
    
    xi = 2 * np.random.rand(len(t)) - 1
    noise = b * xi + c * np.sin(d * t)
    u = g + noise
    
    _, y_filt, _ = signal.lsim(W1, u, t_sim)
    
    # --- Умное формирование заголовка и имени файла ---
    if c == 0:
        title = f'Фильтрация сигнала при $b={b}, c=0$'
        filename = f'task_1_4_3_case_{i+1}_b{b}_c0.png'
    else:
        title = f'Фильтрация сигнала при $b={b}, c={c}, d={d}$'
        filename = f'task_1_4_3_case_{i+1}_b{b}_c{c}_d{d}.png'
    plot_data = (t, g, u, y_filt, title)
    create_and_save_plot(filename, plot_data)

print("\nВсе графики для пункта 1.4.3 сгенерированы.")