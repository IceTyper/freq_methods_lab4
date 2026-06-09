import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import os

def plot_notch_filter_experiment():
    """
    Генерация двухпанельных графиков (временная и частотная области) 
    для исследования режекторного фильтра при репрезентативном наборе 
    комбинаций параметров b1 и d.
    """
    # --- Базовые параметры сигнала и фильтра ---
    t1, t2 = -17, 27
    a = 4.0          # Амплитуда полезного прямоугольного импульса
    c = 2.0          # Амплитуда гармонической помехи
    omega_0 = 5.0    # Центральная частота режекции фильтра
    
    # Временная сетка (расширена для корректного затухания переходных процессов)
    N = 4000
    t = np.linspace(t1 - 20, t2 + 20, N)
    dt = t[1] - t[0]
    
    # --- Репрезентативный набор экспериментальных случаев ---
    # Блок 1: Влияние ширины режекции b1 при точном совпадении d = omega_0
    # Блок 2: Влияние расстройки частоты помехи d при фиксированном b1
    cases = [
        # Блок 1: Исследование b1 (d = 5.0)
        {"b1": 0.1,  "d": 5.0,  "desc": "b1_0.1_d_5.0"},
        {"b1": 0.5,  "d": 5.0,  "desc": "b1_0.5_d_5.0"},
        {"b1": 1.0,  "d": 5.0,  "desc": "b1_1.0_d_5.0"},
        {"b1": 2.0,  "d": 5.0,  "desc": "b1_2.0_d_5.0"},
        {"b1": 5.0,  "d": 5.0,  "desc": "b1_5.0_d_5.0"},
        {"b1": 10.0, "d": 5.0,  "desc": "b1_10.0_d_5.0"},
        # Блок 2: Исследование d (b1 = 1.0)
        {"b1": 1.0, "d": 2.0,  "desc": "b1_1.0_d_2.0"},
        {"b1": 1.0, "d": 4.0,  "desc": "b1_1.0_d_4.0"},
        {"b1": 1.0, "d": 4.9,  "desc": "b1_1.0_d_4.9"},
        {"b1": 1.0, "d": 5.1,  "desc": "b1_1.0_d_5.1"},
        {"b1": 1.0, "d": 7.0,  "desc": "b1_1.0_d_7.0"},
    ]
    
    # Частотная сетка для БПФ (строго линейная)
    omega = 2 * np.pi * np.fft.fftfreq(N, dt)
    mask_freq = (omega >= 0) & (omega <= 25) # Маска для отображения значимого диапазона частот
    w_plot = omega[mask_freq]

    # --- Цикл по экспериментальным случаям ---
    for case in cases:
        b1 = case["b1"]
        d = case["d"]
        desc = case["desc"]
        
        # 1. Формирование сигналов
        g = np.where((t >= t1) & (t <= t2), a, 0.0)
        u = g + c * np.sin(d * t)
        
        # 2. Частотный анализ и расчет эталонного отклика (для корректных начальных условий)
        U_hat = np.fft.fft(u) * dt
        G_hat = np.fft.fft(g) * dt
        
        # Частотная передаточная функция W2(iw)
        numerator = omega_0**2 - omega**2
        denominator = (omega_0**2 - omega**2) + 1j * b1 * omega
        W2_freq = numerator / denominator
        
        Y_hat = U_hat * W2_freq
        y_fft = np.fft.ifft(Y_hat).real
        
        # 3. Моделирование во временной области с корректными начальными условиями
        # Используем State-Space представление для надежного задания X0
        sys_tf = signal.TransferFunction([1, 0, omega_0**2], [1, b1, omega_0**2])
        sys_ss = sys_tf.to_ss()
        
        # Аппроксимация начальных условий состояния на основе решения в частотной области
        x0_1 = y_fft[0]
        x0_2 = (y_fft[1] - y_fft[0]) / dt
        X0_correct = [x0_1, x0_2]
        
        _, y_lsim, _ = signal.lsim(sys_ss, u, t - t[0], X0=X0_correct)
        Y_hat_lsim = np.fft.fft(y_lsim) * dt
        
        # 4. Построение двухпанельного графика
        fig, axs = plt.subplots(1, 2, figsize=(16, 6))
        
        # --- Левый субплот: Временная область (п. 2.1.3) ---
        axs[0].plot(t, g, label=r'Исходный $g(t)$', color='blue', lw=2, zorder=2)
        axs[0].plot(t, u, label=r'Зашумленный $u(t)$', color='orange', alpha=0.5, lw=1.5, zorder=1)
        axs[0].plot(t, y_lsim, label=r'Фильтрованный $y(t)$', color='red', lw=2.5, alpha=0.7, zorder=3)
        
        axs[0].set_xlabel(r'$t$', fontsize=14)
        axs[0].set_ylabel(r'Амплитуда', fontsize=14)
        axs[0].legend(fontsize=14, loc='best')
        axs[0].grid(True, linestyle='--', alpha=0.6)
        axs[0].tick_params(labelsize=14)
        
        # --- Правый субплот: Частотная область (п. 2.1.5) ---
        # Спектр зашумленного сигнала (усилен для наглядности)
        axs[1].plot(w_plot, np.abs(U_hat[mask_freq]), label=r'$|\hat{u}(\omega)|$', 
                    color='orange', alpha=0.5, lw=1.5, zorder=1)
        # Спектр полезного сигнала (ориентир)
        axs[1].plot(w_plot, np.abs(G_hat[mask_freq]), label=r'$|\hat{g}(\omega)|$', 
                    color='blue', lw=3, linestyle='-', alpha=0.7, zorder=2)
        # Спектр фильтрованного сигнала (результат)
        axs[1].plot(w_plot, np.abs(Y_hat_lsim[mask_freq]), label=r'$|\hat{y}(\omega)|$', 
                    color='red', lw=2, linestyle=':', alpha=1, zorder=3)
        
        axs[1].set_xlabel(r'$\omega$', fontsize=14)
        axs[1].set_ylabel(r'Модуль Фурье-образа', fontsize=14)
        axs[1].legend(fontsize=14, loc='best')
        axs[1].grid(True, linestyle='--', alpha=0.6)
        axs[1].tick_params(labelsize=14)
        axs[1].set_xlim(0, 25) # Ограничение для наглядности
        
        plt.tight_layout()
        
        # --- Сохранение файла ---
        save_dir = os.path.join("images", "2_1", "2_1_(2, 3, 5)")
        os.makedirs(save_dir, exist_ok=True)
        filepath = os.path.join(save_dir, f'notch_experiment_{desc}.png')
        plt.savefig(filepath, dpi=300)
        plt.close(fig)
        
        print(f"График сохранен: {filepath}")

if __name__ == "__main__":
    plot_notch_filter_experiment()