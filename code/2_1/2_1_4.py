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
dt = t[1] - t[0]  # Шаг дискретизации для спектрального анализа

# --- Функция для генерации и сохранения ---
def create_and_save_plot(filename, data):
    """Создает и сохраняет график сравнения временной и частотной фильтрации."""
    t_axis, g_signal, u_signal, y_time, y_freq, title = data
    
    plt.figure(figsize=(12, 7))
    # Зашумленный сигнал делаем чуть бледнее, чтобы он не перекрывал результаты фильтрации
    plt.plot(t_axis, u_signal, label=r'$u(t)$ (зашумлённый)', color='orange', alpha=0.4)
    plt.plot(t_axis, g_signal, label=r'$g(t)$ (исходный)', color='black', linestyle='--', linewidth=2)
    
    # Отрисовка двух методов фильтрации с обновленными подписями из скриншота
    plt.plot(t_axis, y_time, label=r'$y(t) = W_2(p)u(t)$', color='blue', linewidth=3)
    plt.plot(t_axis, y_freq, label=r'$y(t) = \mathcal{F}^{-1}\{W_2(i\omega)\hat{u}(\omega)\}$', color='red', linestyle=':', linewidth=2)
    
    plt.title(title, fontsize=16)
    plt.xlabel('t', fontsize=14)
    plt.ylabel('Амплитуда', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=14, loc='lower center')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlim(t_axis[0], t_axis[-1])
    plt.tight_layout()

    save_dir = r"images\2_1_4"
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=300)
    plt.close()
    print(f"График сохранен: {filepath}")


# --- Набор комбинаций параметров для демонстрации методов ---
param_combinations = [
    {'case': 'method_demo', 'c': 1.5, 'd': 5.0, 'omega_0': 5.0, 'b1': 0.5, 'title': 'Узкая полоса'},
    {'case': 'method_demo', 'c': 1.5, 'd': 5.0, 'omega_0': 5.0, 'b1': 5.0, 'title': 'Широкая полоса'},
]

# --- Основной цикл ---
for i, params in enumerate(param_combinations):
    case, c, d, omega_0, b1, title = params.values()
    
    # 1. Формируем входной сигнал u(t)
    noise = c * np.sin(d * t)
    u = g + noise
    
    # 2. Метод 1: Фильтрация во временной области (через lsim)
    num = [1, 0, omega_0**2]
    den = [1, b1, omega_0**2]
    W2_system = signal.TransferFunction(num, den)
    _, y_time, _ = signal.lsim(W2_system, u, t_sim)
    
    # 3. Метод 2: Фильтрация в частотной области (через FFT)
    u_fft = np.fft.fft(u)
    freqs = np.fft.fftfreq(N, d=dt)      # Частоты в Гц
    omega = 2 * np.pi * freqs            # Перевод в радианы в секунду (угловая частота)
    
    # Вычисляем частотную характеристику фильтра W2(i*omega)
    s = 1j * omega
    H = (s**2 + omega_0**2) / (s**2 + b1 * s + omega_0**2)
    
    # Фильтруем в спектральной области и возвращаем в t-область
    y_fft = u_fft * H
    y_freq = np.fft.ifft(y_fft).real
    
    # --- Формирование динамического заголовка с параметрами ---
    full_title = f"{title}\n" + rf"$c = {c}$, $b_1 = {b1}$, $d = {d}$"
    
    # --- Формирование имени файла ---
    filename = f'task_2_1_4_{case}_{i+1}_c{c}_d{d}_w0{omega_0}_b1{b1}.png'
        
    plot_data = (t, g, u, y_time, y_freq, full_title)
    create_and_save_plot(filename, plot_data)

print("\nВсе графики для пункта 2.1.4 успешно сгенерированы.")