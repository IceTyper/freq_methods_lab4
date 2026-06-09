import numpy as np
import matplotlib.pyplot as plt
import os

# Расширенный репрезентативный набор значений Т для полноценного исследования
T_values = [0.1, 0.3, 0.5, 1.0, 2.0, 5.0]

# Строго линейная сетка частот
omega = np.linspace(0, 30, 1000)

plt.figure(figsize=(10, 6))

# Различные стили линий для однозначной визуальной различимости при печати на А4
line_styles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 5))]

for i, T in enumerate(T_values):
    A = 1 / np.sqrt(1 + (omega * T)**2)
    plt.plot(omega, A, label=r'$T = {}$'.format(T), linewidth=2.5, linestyle=line_styles[i % len(line_styles)])

# Возврат уровня частоты среза по ГОСТу (1/sqrt(2) ≈ 0.707)
plt.axhline(y=1/np.sqrt(2), color='gray', linestyle='--', linewidth=1.5, label=r'Уровень частоты среза')

# Оформление (без внутренних заголовков, ордината строго по методичке)
plt.xlabel(r'$\omega$', fontsize=14)
plt.ylabel(r'$|W_1(i\omega)|$', fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=14, loc='best')
plt.tight_layout()

# Сохранение по относительному пути начиная с папки images
save_dir = os.path.join("images", "1_4")
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, "1_4_1.png"), dpi=300)
plt.close()