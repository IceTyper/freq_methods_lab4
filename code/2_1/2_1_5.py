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

# --- Функция для генерации и сохранения графиков спектров ---
def create_and_save_plot(filename, data):
    """Создает и сохраняет один сравнительный график модулей Фурье-образов."""
    omega_axis, mag_g, mag_u, mag_y, title = data
    
    plt.figure(figsize=(12, 7))
    
    # Зашумленный сигнал (оранжевый, сплошной, на заднем плане за счет alpha)
    plt.plot(omega_axis, mag_u, label=r'$|\hat{u}(\omega)|$ (зашумлённый)', color='orange', alpha=0.7, linewidth=1.8)
    # Фильтрованный сигнал (синий, яркий, выделяющийся)
    plt.plot(omega_axis, mag_y, label=r'$|\hat{y}(\omega)|$ (фильтрованный)', color='blue', linewidth=2.5)
    # Исходный сигнал (черный, пунктирный, хорошо различимый под синим)
    plt.plot(omega_axis, mag_g, label=r'$|\hat{g}(\omega)|$ (исходный)', color='black', linestyle='--', linewidth=2)
    
    # Оформление по ГОСТ / требованиям читаемости
    plt.title(title, fontsize=16)
    plt.xlabel(r'$\omega$', fontsize=14)
    plt.ylabel(r'Модуль Фурье-образа', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=14, loc='upper right')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    
    # Ограничение оси частот для вывода наиболее значимой части спектра (w0=5, d=5 или 7)
    plt.xlim(0, 15)
    plt.tight_layout()

    save_dir = r"images\2_1_5"
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=300)
    plt.close()
    print(f"График сохранен: {filepath}")

# --- Сетка параметров для исследования (соответствует пунктам 2.1.3 и 2.1.4) ---
param_combinations = [
    # 1. Влияние b1 (ширина фильтра) при d = omega_0, c = 1.5
    {'case': 'b1_influence', 'c': 1.5, 'd': 5.0, 'omega_0': 5.0, 'b1': 0.5,  'title': 'Фурье-образы: Удачная фильтрация (узкий фильтр)'},
    {'case': 'b1_influence', 'c': 1.5, 'd': 5.0, 'omega_0': 5.0, 'b1': 10.0, 'title': 'Фурье-образы: Неудачная фильтрация (широкий фильтр)'},
    
    # 2. Влияние d (частота помехи) при b1 = 0.5, c = 1.5
    {'case': 'd_influence',  'c': 1.5, 'd': 5.0, 'omega_0': 5.0, 'b1': 0.5,  'title': 'Фурье-образы: Удачная фильтрация ($d = \omega_0$)'},
    {'case': 'd_influence',  'c': 1.5, 'd': 7.0, 'omega_0': 5.0, 'b1': 0.5,  'title': 'Фурье-образы: Неудачная фильтрация ($d \\neq \omega_0$)'},

    # 3. Влияние c (амплитуда помехи) при d = omega_0, b1 = 0.5
    {'case': 'c_influence',  'c': 1.5, 'd': 5.0, 'omega_0': 5.0, 'b1': 0.5,  'title': 'Фурье-образы: Удачная фильтрация (малая помеха)'},
    {'case': 'c_influence',  'c': 4.0, 'd': 5.0, 'omega_0': 5.0, 'b1': 0.5,  'title': 'Фурье-образы: Неудачная фильтрация (сильная помеха)'},
]

# --- Основной цикл ---
for i, params in enumerate(param_combinations):
    case, c, d, omega_0, b1, title = params.values()
    
    # 1. Формируем входной сигнал u(t) = g(t) + c*sin(dt)
    noise = c * np.sin(d * t)
    u = g + noise
    
    # 2. Моделируем фильтрацию во временной области через lsim
    num = [1, 0, omega_0**2]
    den = [1, b1, omega_0**2]
    W2_system = signal.TransferFunction(num, den)
    _, y_lsim, _ = signal.lsim(W2_system, u, t_sim)
    
    # 3. Переход в частотную область (FFT) с унитарным масштабированием
    # Множитель унитарного преобразования к угловой частоте: dt / sqrt(2 * pi)
    scaling = dt / np.sqrt(2 * np.pi)
    
    freqs = np.fft.fftfreq(N, d=dt)
    omega = 2 * np.pi * freqs
    
    # Используем маску только для положительных частот
    pos_mask = omega >= 0
    omega_pos = omega[pos_mask]
    
    # Расчет спектров
    mag_g = np.abs(np.fft.fft(g)[pos_mask]) * scaling
    mag_u = np.abs(np.fft.fft(u)[pos_mask]) * scaling
    mag_y = np.abs(np.fft.fft(y_lsim)[pos_mask]) * scaling
    
    # --- Формирование динамического заголовка с параметрами ---
    full_title = f"{title}\n" + rf"$c = {c}$, $b_1 = {b1}$, $d = {d}$"
    
    # --- Формирование имени файла ---
    filename = f'task_2_1_5_{case}_{i+1}_c{c}_d{d}_w0{omega_0}_b1{b1}.png'
        
    plot_data = (omega_pos, mag_g, mag_u, mag_y, full_title)
    create_and_save_plot(filename, plot_data)

print("\nВсе графики спектров для пункта 2.1.5 успешно сгенерированы.")