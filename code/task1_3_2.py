# графики модулей Фурье-образов графиков
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import os

# Параметры
a, t1, t2 = 4, -17, 27
b, c, d = 0.5, 0, 0
T = 2.0
np.random.seed(42)

# Временная сетка
dt = 0.01
t = np.arange(t1 - 20, t2 + 20, dt)

# Сигналы
g = np.where((t >= t1) & (t <= t2), a, 0)
xi = 2 * np.random.rand(len(t)) - 1
u = g + b * xi

# Сдвиг времени для фильтрации
t_sim = t - t[0]

# Фильтрация
W1 = signal.TransferFunction([1], [T, 1])
_, y_filt, _ = signal.lsim(W1, u, t_sim)

# Унитарное преобразование Фурье по угловой частоте ω
# F{f}(ω) ≈ (dt / √(2π)) * FFT(f)
scale = dt / np.sqrt(2 * np.pi)
g_hat = np.fft.fftshift(np.fft.fft(g)) * scale
u_hat = np.fft.fftshift(np.fft.fft(u)) * scale
y_hat = np.fft.fftshift(np.fft.fft(y_filt)) * scale

# Ось угловых частот
w = np.fft.fftshift(np.fft.fftfreq(len(t), d=dt)) * 2 * np.pi

# Построение графика спектров
plt.figure(figsize=(10, 6))
plt.plot(w, np.abs(g_hat), label='$|\hat{g}(\omega)|$ (исходный)', linewidth=2, color='blue')
plt.plot(w, np.abs(u_hat), label='$|\hat{u}(\omega)|$ (зашумлённый)', linewidth=1, alpha=0.7, color='orange')
plt.plot(w, np.abs(y_hat), label='$|\hat{y}(\omega)|$ (фильтрованный)', linewidth=2, color='green')

plt.xlabel('Угловая частота $\omega$', fontsize=16)
plt.ylabel('Модуль Фурье-образа', fontsize=16)
plt.title('Сравнение модулей Фурье-образов сигналов', fontsize=18)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlim(-0.5, 0.5)
plt.tight_layout()

# Сохранение
save_dir = r"C:\Users\fmusa\ITMOStudies\freq_methods\freq_methods_lab4\images"
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, "comparison_of_Fourier_modules.png"), dpi=300)
plt.close()
