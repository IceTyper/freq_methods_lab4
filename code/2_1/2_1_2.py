import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import os

# --- Общие параметры ---
a, t1, t2 = 4, -17, 27
c_noise = 1.5 # Амплитуда синусоидальной помехи

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

    save_dir = r"C:\Users\fmusa\ITMOStudies\freq_methods\freq_methods_lab4\images\2_1_2"
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=300)
    plt.close()
    print(f"График сохранен: {filepath}")

# --- Сетка параметров для исследования ---
# Частота помехи d, частота настройки фильтра omega_0, параметр b1
# b=0 для этого задания
param_combinations = [
    # --- 1. Исследование влияния b1 (d = 5, omega_0 = 5) ---
    {'case_name': 'b1_influence', 'd': 5.0, 'omega_0': 5.0, 'b1': 0.5, 'title': 'Удачная фильтрация'},
    {'case_name': 'b1_influence', 'd': 5.0, 'omega_0': 5.0, 'b1': 10.0, 'title': 'Неудачная фильтрация'},
    
    # --- 2. Исследование влияния d (b1 = 0.5) ---
    {'case_name': 'd_influence', 'd': 5.0, 'omega_0': 5.0, 'b1': 0.5, 'title': 'Удачная фильтрация'},
    {'case_name': 'd_influence', 'd': 7.0, 'omega_0': 5.0, 'b1': 0.5, 'title': 'Неудачная фильтрация'},
]

# --- Основной цикл ---
for i, params in enumerate(param_combinations):
    case_name, d, omega_0, b1, title = params.values()
    
    # 1. Генерируем зашумленный сигнал u(t) = g(t) + c*sin(d*t)
    noise = c_noise * np.sin(d * t)
    u = g + noise
    
    # 2. Создаем передаточную функцию режекторного фильтра
    # W2(p) = (p^2 + w0^2) / (p^2 + b1*p + w0^2)
    num = [1, 0, omega_0**2]
    den = [1, b1, omega_0**2]
    W2_system = signal.TransferFunction(num, den)
    
    # 3. Фильтрация во временной области для получения y(t)
    _, y_lsim, _ = signal.lsim(W2_system, u, t_sim)
    
    # --- Формирование имени файла ---
    filename = f'task_2_1_2_{case_name}_{i+1}_d{d}_w0{omega_0}_b1{b1}.png'
        
    plot_data = (t, g, u, y_lsim, title)
    create_and_save_plot(filename, plot_data)

print("\nВсе графики для пункта 2.1.2 сгенерированы.")