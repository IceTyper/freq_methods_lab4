import numpy as np
import matplotlib.pyplot as plt
import os

def plot_notch_filter_afc():
    """
    Строит и сохраняет график АЧХ режекторного фильтра для различных
    значений параметра b1.
    """
    # --- Параметры для построения АЧХ ---
    # Выберем центральную частоту режекции для демонстрации
    omega_0 = 5.0
    # Выберем значения b1, чтобы показать его влияние:
    # узкий фильтр, средний и широкий
    b1_values = [0.5, 2.0, 10.0]
    
    # Создаем диапазон частот для графика
    omega = np.linspace(0, 15, 2000)
    
    plt.figure(figsize=(12, 7))
    
    # --- Цикл для расчета и построения АЧХ для каждого b1 ---
    for b1 in b1_values:
        # Числитель и знаменатель частотной передаточной функции W2(iw)
        numerator = omega_0**2 - omega**2
        denominator = (omega_0**2 - omega**2) + 1j * b1 * omega
        
        # Рассчитываем АЧХ |W2(iw)|
        afc = np.abs(numerator / denominator)
        
        # Строим график
        plt.plot(omega, afc, label=f'$b_1 = {b1}$', linewidth=2.5)

    # --- Оформление графика ---
    plt.title(f'АЧХ режекторного фильтра при $\omega_0=5$', fontsize=16)
    plt.xlabel(r'$\omega$', fontsize=14)
    plt.ylabel(r'$|W_2(i\omega)|$', fontsize=14)
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.legend(fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlim(0, 15)
    plt.ylim(-0.05, 1.05)
    plt.tight_layout()

    # --- Сохранение файла ---
    save_dir = r"images\2_1_1"
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, 'task_2_1_1_afc_comparison.png')
    plt.savefig(filepath, dpi=300)
    plt.close()
    
    print(f"График АЧХ режекторного фильтра сохранен: {filepath}")

# --- Запуск функции ---
if __name__ == "__main__":
    plot_notch_filter_afc()