import numpy as np
import matplotlib.pyplot as plt
import os

def plot_notch_filter_afc():
    """
    Строит и сохраняет график АЧХ режекторного фильтра для репрезентативного
    набора значений параметра b1 на строго линейной сетке частот.
    """
    # --- Параметры для построения АЧХ ---
    omega_0 = 5.0
    
    # Репрезентативный набор значений b1 для полноценного исследования влияния на ширину режекции
    b1_values = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    
    # Строго линейная сетка частот. Диапазон расширен до 50 рад/с, 
    # чтобы наглядно продемонстрировать асимптотическое стремление АЧХ к единице на верхних частотах
    omega = np.linspace(0, 50, 5000)

    plt.figure(figsize=(12, 7))

    # Различные стили линий для однозначной визуальной различимости при печати на А4
    line_styles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 5))]

    # --- Цикл для расчета и построения АЧХ для каждого b1 ---
    for i, b1 in enumerate(b1_values):
        # Числитель и знаменатель частотной передаточной функции W2(iw)
        numerator = omega_0**2 - omega**2
        denominator = (omega_0**2 - omega**2) + 1j * b1 * omega
        
        # Рассчитываем АЧХ |W2(iw)|
        afc = np.abs(numerator / denominator)
        
        # Строим график с уникальным стилем для каждой линии
        plt.plot(omega, afc, label=f'$b_1 = {b1}$', linewidth=2.5, 
                 linestyle=line_styles[i % len(line_styles)])


    # --- Оформление графика ---
    # plt.title полностью запрещен согласно требованиям (подписи будут в LaTeX через \caption)
    plt.xlabel(r'$\omega$', fontsize=14)
    plt.ylabel(r'$|W_2(i\omega)|$', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=14, loc='best')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    
    # Ограничение осей для корректного отображения формы графика
    plt.xlim(0, 50)
    plt.ylim(-0.05, 1.2)
    plt.tight_layout()

    # --- Сохранение файла ---
    # Путь сохранения оставлен без изменений
    save_dir = r"images\2_1\2_1_1"
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, 'task_2_1_1_afc_comparison.png')
    plt.savefig(filepath, dpi=300)
    plt.close()

    print(f"График АЧХ режекторного фильтра сохранен: {filepath}")

# --- Запуск функции ---
if __name__ == "__main__":
    plot_notch_filter_afc()