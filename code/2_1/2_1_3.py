import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import os

# --- Общие параметры ---
a, t1, t2 = 4, -17, 27

# --- Временная ось ---
N = 4000
t = np.linspace(t1 - 20, t2 + 20, N, endpoint=False)
g = np.where((t >= t1) & (t <= t2), a, 0)
t_sim = t - t[0]

# --- Функция для генерации и сохранения ---
def create_and_save_plot(filename, data):
    """Создает и сохраняет один сравнительный график сигналов."""
    t_axis, g_signal, u_signal, y_signal, title = data
    
    plt.figure(figsize=(12, 7))
    plt.plot(t_axis, u_signal, label=r'$u(t)$ (зашумлённый)', color='orange', alpha=0.8)
    plt.plot(t_axis, y_signal, label=r'$y(t)$ (фильтрованный)', color='blue', linewidth=2.5)
    plt.plot(t_axis, g_signal, label=r'$g(t)$ (исходный)', color='black', linestyle='--', linewidth=2)
    
    plt.title(title, fontsize=16)
    plt.xlabel('t', fontsize=14)
    plt.ylabel('Амплитуда', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlim(t_axis[0], t_axis[-1])
    plt.tight_layout()

    save_dir = r"C:\Users\fmusa\ITMOStudies\freq_methods\freq_methods_lab4\images\2_1_3"
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=300)
    plt.close()
    print(f"График сохранен: {filepath}")


# --- Набор исследуемых комбинаций параметров ---
param_combinations = [
    # --- 1. Исследование влияния частоты помехи d (при c = 1.5, b1 = 0.5) ---
    {'case': 'd_influence',  'c': 1.5, 'd': 5.3, 'omega_0': 5.0, 'b1': 0.5,  'title': 'Околорезонансная помеха'},
    {'case': 'd_influence',  'c': 1.5, 'd': 8.0, 'omega_0': 5.0, 'b1': 0.5,  'title': 'Далёкая помеха'},

    # --- 2. Исследование влияния амплитуды помехи c при неидеальной настройке фильтра (d = 5.3) ---
    {'case': 'c_influence',  'c': 1.5, 'd': 5.3, 'omega_0': 5.0, 'b1': 0.5,  'title': 'Малое "с"'},
    {'case': 'c_influence',  'c': 5.0, 'd': 5.3, 'omega_0': 5.0, 'b1': 0.5,  'title': 'Большое "с"'},
]


# --- Основной цикл ---
for i, params in enumerate(param_combinations):
    case, c, d, omega_0, b1, title = params.values()
    
    # 1. Генерируем зашумленный сигнал u(t) = g(t) + c*sin(d*t)
    noise = c * np.sin(d * t)
    u = g + noise
    
    # 2. Создаем передаточную функцию режекторного фильтра
    # W2(p) = (p^2 + w0^2) / (p^2 + b1*p + w0^2)
    num = [1, 0, omega_0**2]
    den = [1, b1, omega_0**2]
    W2_system = signal.TransferFunction(num, den)
    
    # 3. Фильтрация во временной области для получения y(t)
    _, y_lsim, _ = signal.lsim(W2_system, u, t_sim)
    
    # --- Формирование динамического заголовка с параметрами ---
    full_title = f"{title}\n" + rf"$c = {c}$, $b_1 = {b1}$, $d = {d}$"
    
    # --- Формирование имени файла ---
    filename = f'task_2_1_3_{case}_{i+1}_c{c}_d{d}_w0{omega_0}_b1{b1}.png'
        
    plot_data = (t, g, u, y_lsim, full_title)
    create_and_save_plot(filename, plot_data)

print("\nВсе графики для пункта 2.1.3 сгенерированы.")