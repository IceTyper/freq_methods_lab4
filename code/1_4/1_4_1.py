import os
import numpy as np
import matplotlib.pyplot as plt

# Директория для сохранения графиков
output_dir = r"images"

# Создание директории, если она ещё не существует
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Выбранные значения постоянной времени T
T_values = [0.1, 0.5, 1.0, 2.0]

# Расчет диапазона угловых частот \omega (логарифмическая шкала)
omega = np.logspace(-2, 2, 1000)

plt.figure(figsize=(10, 6))

# Построение АЧХ для каждого значения T
for T in T_values:
    # Расчет АЧХ по формуле: |W_1(i\omega)| = 1 / sqrt(1 + (T * \omega)^2)
    ach = 1 / np.sqrt(1 + (T * omega)**2)
    
    # Расчет частоты среза: \omega_c = 1 / T
    omega_c = 1 / T
    
    # Построение кривой
    plt.plot(omega, ach, label=f'T = {T}', linewidth=2)
    
    # Добавление точки частоты среза на уровень 1 / sqrt(2)
    plt.plot(omega_c, 1 / np.sqrt(2), 'ro', markersize=5)

# Линия уровня частоты среза (ослабление -3 дБ)
plt.axhline(y=1/np.sqrt(2), color='gray', linestyle='--', alpha=0.7, label='Уровень частоты среза')

# Оформление графика
plt.xscale('log')
plt.title('АЧХ фильтра для разных T', fontsize=14, pad=15)
plt.xlabel('$\omega$', fontsize=12)
plt.ylabel('$|W_1(i\omega)|$', fontsize=12)
plt.grid(True, which="both", linestyle='--', alpha=0.6)
plt.legend(fontsize=11, loc='lower left')
plt.xlim(0.01, 100)
plt.ylim(0, 1.1)

plt.tight_layout()

# Сохранение графика в соответствии с требованиями
plt.savefig(os.path.join(output_dir, '1_4_1.png'), dpi=300, bbox_inches='tight')
plt.close()