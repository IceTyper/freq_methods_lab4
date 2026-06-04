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

# --- Функция для генерации и сохранения графиков ---
def create_and_save_plot(filename, data):
    """Создает и сохраняет один сравнительный график модулей Фурье-образов."""
    omega_axis, mag_y, mag_W_u, title = data
    
    plt.figure(figsize=(12, 7))
    
    # Чтобы графики, которые идеально совпадают, были удобочитаемыми:
    # 1. Задаем нижний график широкой сплошной линией (синий)
    plt.plot(omega_axis, mag_y, label=r'$|\hat{y}(\omega)|$', 
             color='blue', linewidth=3.5, alpha=0.9)
    
    # 2. Накладываем верхний график более тонкой штриховой линией контрастного цвета (красный)
    plt.plot(omega_axis, mag_W_u, label=r'$|W_{2}(i\omega) \cdot \hat{u}(\omega)|$', 
             color='crimson', linestyle='--', linewidth=2.0)
    
    # Оформление по требованиям
    plt.title(title, fontsize=16)
    plt.xlabel(r'$\omega$', fontsize=14)
    plt.ylabel(r'Модуль Фурье-образа', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=14, loc='upper right')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    
    # Ограничение оси частот для отображения наиболее значимой части Фурье-образа
    plt.xlim(0, 15)
    plt.tight_layout()

    save_dir = r"images\2_1_6"
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=300)
    plt.close()
    print(f"График сохранен: {filepath}")

# --- Сетка параметров для исследования (соответствует предыдущим пунктам) ---
param_combinations = [
    # 1. Влияние b1 (ширина фильтра) при d = omega_0, c = 1.5
    {'case': 'b1_influence', 'c': 1.5, 'd': 5.0, 'omega_0': 5.0, 'b1': 0.5,  'title': 'Узкий фильтр'},
    {'case': 'b1_influence', 'c': 1.5, 'd': 5.0, 'omega_0': 5.0, 'b1': 10.0, 'title': 'Широкий фильтр'},
    
    # 2. Влияние d (частота помехи) при b1 = 0.5, c = 1.5
    {'case': 'd_influence',  'c': 1.5, 'd': 5.0, 'omega_0': 5.0, 'b1': 0.5,  'title': 'При $d = \omega_0$'},
    {'case': 'd_influence',  'c': 1.5, 'd': 7.0, 'omega_0': 5.0, 'b1': 0.5,  'title': 'При $d \\neq \omega_0$'},

    # 3. Влияние c (амплитуда помехи) при d = omega_0, b1 = 0.5
    {'case': 'c_influence',  'c': 1.5, 'd': 5.0, 'omega_0': 5.0, 'b1': 0.5,  'title': 'Малая помеха'},
    {'case': 'c_influence',  'c': 4.0, 'd': 5.0, 'omega_0': 5.0, 'b1': 0.5,  'title': 'Сильная помеха'},
]

# --- Основной цикл ---
for i, params in enumerate(param_combinations):
    case, c, d, omega_0, b1, title = params.values()
    
    # 1. Формируем входной зашумленный сигнал
    noise = c * np.sin(d * t)
    u = g + noise
    
    # 2. Временная фильтрация (получение y(t) через lsim)
    num = [1, 0, omega_0**2]
    den = [1, b1, omega_0**2]
    W2_system = signal.TransferFunction(num, den)
    _, y_lsim, _ = signal.lsim(W2_system, u, t_sim)
    
    # 3. Частотный анализ (FFT) с унитарным масштабированием к угловой частоте
    scaling = dt / np.sqrt(2 * np.pi)
    
    freqs = np.fft.fftfreq(N, d=dt)
    omega = 2 * np.pi * freqs
    
    # Используем маску только для положительных частот
    pos_mask = omega >= 0
    omega_pos = omega[pos_mask]
    
    # Расчет Фурье-образа отфильтрованного сигнала y(t)
    mag_y = np.abs(np.fft.fft(y_lsim)[pos_mask]) * scaling
    
    # Расчет Фурье-образа зашумленного сигнала u(t)
    u_fft = np.fft.fft(u)
    
    # Вычисление частотной характеристики фильтра W2(i*omega) аналитически
    s = 1j * omega
    H = (s**2 + omega_0**2) / (s**2 + b1 * s + omega_0**2)
    
    # Расчет теоретического Фурье-образа как произведения W2(i*omega) * u_hat(omega)
    mag_W_u = np.abs((H * u_fft)[pos_mask]) * scaling
    
    # --- Формирование динамического заголовка с параметрами ---
    full_title = f"{title}\n" + rf"$c = {c}$, $b_1 = {b1}$, $d = {d}$"
    
    # --- Формирование имени файла ---
    filename = f'task_2_1_6_{case}_{i+1}_c{c}_d{d}_w0{omega_0}_b1{b1}.png'
        
    plot_data = (omega_pos, mag_y, mag_W_u, full_title)
    create_and_save_plot(filename, plot_data)

print("\nВсе графики для пункта 2.1.6 успешно сгенерированы.")