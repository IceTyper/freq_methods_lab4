# сравнительные графики исходного и фильтрованного сигналов
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import os

a, t1, t2 = 4, -17, 27
b, c, d = 0.5, 0, 0
T = 2.0
np.random.seed(42)

t = np.linspace(t1 - 20, t2 + 20, 2000)

g = np.where((t >= t1) & (t <= t2), a, 0)
xi = 2 * np.random.rand(len(t)) - 1
u = g + b * xi + c * np.sin(d * t)

t_sim = t - t[0]

W1 = signal.TransferFunction([1], [T, 1])
_, y_filt, _ = signal.lsim(W1, u, t_sim)


plt.figure(figsize=(10, 6))
plt.plot(t, u, label='$u(t)$ (зашумлённый)', linewidth=1, alpha=0.4, color='gray')
plt.plot(t, g, label='$g(t)$ (исходный)', linewidth=2.5, color='black', linestyle='--')
plt.plot(t, y_filt, label='$y(t)$ (фильтрованный)', linewidth=2.5, color='crimson')

plt.xlabel('Время $t$', fontsize=16)
plt.ylabel('Амплитуда', fontsize=16)
plt.title('Сравнение исходного и фильтрованного сигналов', fontsize=18)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()

save_dir = r"C:\Users\fmusa\ITMOStudies\freq_methods\freq_methods_lab4\images"
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, "first_comparison_of_signals.png"), dpi=300)
plt.close()