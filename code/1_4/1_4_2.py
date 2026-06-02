import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import os

# --- Общие параметры ---
t1, t2 = -17, 27
b = 0.8  # Зададим шум побольше для наглядности
np.random.seed(42)
t = np.linspace(t1 - 20, t2 + 20, 2000)
xi = 2 * np.random.rand(len(t)) - 1

# --- Обновленная функция для сохранения одного графика ---
def plot_and_save_single(filename, plot_data):
    """Функция для создания и сохранения одного графика."""
    fig, ax = plt.subplots(figsize=(12, 5)) # Оптимальный размер для одного графика

    t, g, u, y_filt, title = plot_data
    
    # Используем ax.set_title для заголовка самого графика
    ax.set_title(title, fontsize=16)

    ax.plot(t, g, label='Исходный сигнал $g(t)$', color='blue', lw=2)
    ax.plot(t, u, label='Зашумлённый сигнал $u(t)$', color='orange', alpha=0.6)
    # Добавил y(t) в легенду для полноты
    ax.plot(t, y_filt, label='Фильтрованный сигнал $y(t)$', color='green', lw=2.5)
    ax.grid(True, linestyle='--', alpha=0.7)
    # Увеличил шрифт легенды
    ax.legend(fontsize=14) 
    # Увеличил шрифт подписей осей
    ax.set_xlabel('$t$', fontsize=14)

    plt.tight_layout()

    # Сохранение
    save_dir = r"images\1_4_2"
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=300)
    # print(f"График сохранен в: {filepath}") # Можно закомментировать, чтобы не засорять вывод
    plt.close(fig)


# --- 1. Исследование влияния T (a=4) ---
a_fixed = 4
T_cases = [0.2, 2.0, 15.0]

g_fixed = np.where((t >= t1) & (t <= t2), a_fixed, 0)
u_fixed = g_fixed + b * xi

for i, T in enumerate(T_cases):
    system = signal.TransferFunction([1], [T, 1])
    _, y_filt, _ = signal.lsim(system, u_fixed, t - t[0])
    
    if T == 0.2:
        case = "Неудачный случай: "
    elif T == 2.0:
        case = "Удачный случай: "
    else:
        case = "Неудачный случай: "
        
    title = f'{case} $T={T}$, $a={a_fixed}$'
    
    # Генерируем уникальное имя файла для каждого графика
    filename = f"filtration_T_case_{i+1}_T_{T}.png"
    plot_data = (t, g_fixed, u_fixed, y_filt, title)
    
    plot_and_save_single(filename, plot_data)


# --- 2. Исследование влияния a (T=2.0) ---
T_fixed = 2.0
a_cases = [0.5, 4.0]

system_fixed = signal.TransferFunction([1], [T_fixed, 1])

for i, a in enumerate(a_cases):
    g = np.where((t >= t1) & (t <= t2), a, 0)
    u = g + b * xi
    _, y_filt, _ = signal.lsim(system_fixed, u, t - t[0])

    if a == 0.5:
        case = "Неудачный случай: "
    else:
        case = "Удачный случай: "

    title = f'{case} $a={a}$, $T={T_fixed}$'
    
    # Генерируем уникальное имя файла
    filename = f"filtration_a_case_{i+1}_a_{a}.png"
    plot_data = (t, g, u, y_filt, title)
    
    plot_and_save_single(filename, plot_data)

# plt.show() # Можно убрать, так как мы только сохраняем файлы