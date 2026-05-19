import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

# 1. Параметры
a = 1.0          # высота импульса
t1, t2 = -1, 1   # границы импульса
T = 0.5          # постоянная времени фильтра
b = 0.3          # уровень шума
dt = 0.01        # шаг по времени
t = np.arange(0, 10, dt)  # время  # время

# 2. Сигнал
g = np.where((t >= t1) & (t <= t2), a, 0)  # прямоугольный импульс
xi = 2 * np.random.rand(len(t)) - 1         # шум ~ U[-1,1]
u = g + b * xi                              # зашумлённый сигнал

# 3. Фильтр (временной метод)
sys = signal.TransferFunction([1], [T, 1])  # W(p) = 1/(T*p + 1)
y_time = signal.lsim(sys, u, t)[1]          # решение дифф. уравнения

# 4. Фильтр (частотный метод)
U = np.fft.fft(u)                           # спектр входа
omega = 2 * np.pi * np.fft.fftfreq(len(t), d=dt)  # угловые частоты
W = 1 / (1 + 1j * omega * T)                # частотная характеристика
y_freq = np.real(np.fft.ifft(U * W))        # обратное преобразование

# 5. Графики (пример)
plt.plot(t, g, label='исходный')
plt.plot(t, u, label='с шумом', alpha=0.5)
plt.plot(t, y_time, label='отфильтрованный')
plt.legend()
plt.show()