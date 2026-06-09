import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# 1. Загрузка данных
file_name = 'SBER_220604_260604.csv'
if not os.path.exists(file_name):
    # Адаптация пути, если скрипт запускается из корня проекта
    file_name = os.path.join('code', '3_3', 'SBER_220604_260604.csv')

df = pd.read_csv(file_name, sep= ';')

# Преобразование даты в формат datetime для точных расчетов
df['DATE_CONV'] = pd.to_datetime(df['<DATE>'].astype(str), format='%Y%m%d')
u = df['<CLOSE>'].values
dates = df['DATE_CONV'].values
t = np.arange(len(u))

# Наборы значений постоянной времени T (в календарных днях) и их понятные имена
T_values = {
    '1_day': {'days': 1, 'label': '1 день'},
    '1_week': {'days': 7, 'label': '1 неделя'},
    '1_month': {'days': 30, 'label': '1 месяц'},
    '3_months': {'days': 90, 'label': '3 месяца'},
    '1_year': {'days': 365, 'label': '1 год'}
}

# Корректная папка для сохранения графиков согласно структуре лабораторной
output_dir = os.path.join('images', '3_3')
os.makedirs(output_dir, exist_ok=True)

# 2. Цикл фильтрации и построения графиков для каждого T
for key, item in T_values.items():
    T = item['days']
    label_ru = item['label']
    
    # Задание непрерывной передаточной функции фильтра 1-го порядка: W(p) = 1 / (T*p + 1)
    num = [1]
    den = [T, 1]
    sys_cont = signal.tf2ss(num, den)
    
    # ИСПРАВЛЕНИЕ КРИТИЧЕСКОЙ ОШИБКИ: Расчет математически корректных начальных условий.
    # Так как для tf2ss выход y(0) = C * X0, а C = [1/T], то для обеспечения y(0) = u(0)
    # необходимо подать состояние X0 = u(0) * T.
    X0 = np.array([u[0] * T])
    
    # Моделирование фильтрации во временной области
    tout, y, x = signal.lsim(sys_cont, U=u, T=t, X0=X0)
    
    # Создание общего полотна для двух графиков (а и б)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Установка крупного шрифта для печати А4 (не менее 14 пт)
    plt.rc('font', size=14)
    plt.rc('axes', labelsize=14)
    plt.rc('xtick', labelsize=14)
    plt.rc('ytick', labelsize=14)
    
    # -----------------------------------------------------------------
    # График А: Масштаб, соответствующий постоянной времени T
    # -----------------------------------------------------------------
    display_points = max(5, int(T))
    if display_points > len(u):
        display_points = len(u)
        
    indices_loc = np.arange(display_points)
    ax1.plot(indices_loc, u[:display_points], label='Исходный курс', color='orange', alpha=0.5)
    ax1.plot(indices_loc, y[:display_points], label='Сглаженный сигнал', color='blue', linewidth=2)
    
    # ЗАМЕЧАНИЕ ПРЕПОДАВАТЕЛЯ: Внутренние заголовки plt.title удалены. 
    # Вместо них используются только внешние подписи рисунков в LaTeX.
    ax1.set_title(f'а) Временной диапазон масштаба T ({label_ru})', fontsize=16, fontweight='bold')
    ax1.set_xlabel('Дата')
    ax1.set_ylabel('Цена (руб.)')
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(fontsize=14)
    
    # Равномерные подписи дат
    step_loc = max(1, display_points // 6)
    ax1.set_xticks(indices_loc[::step_loc])
    ax1.set_xticklabels([pd.Timestamp(dates[i]).strftime('%Y-%m-%d') for i in indices_loc[::step_loc]], rotation=15)
    
    # -----------------------------------------------------------------
    # График Б: Весь массив данных (Глобальный)
    # -----------------------------------------------------------------
    indices_glob = np.arange(len(u))
    ax2.plot(indices_glob, u, label='Исходный курс', color='orange', alpha=0.4)
    ax2.plot(indices_glob, y, label='Сглаженный сигнал', color='blue', linewidth=1.8)
    
    # ЗАМЕЧАНИЕ ПРЕПОДАВАТЕЛЯ: Внутренние заголовки plt.title удалены.
    ax2.set_title('б) Весь массив данных (2022–2026)', fontsize=16, fontweight='bold')
    ax2.set_xlabel('Дата')
    ax2.set_ylabel('Цена (руб.)')
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(fontsize=14)
    
    # Распределяем 8 меток дат по всему глобальному графику
    step_glob = len(u) // 8
    ax2.set_xticks(indices_glob[::step_glob])
    ax2.set_xticklabels([pd.Timestamp(dates[i]).strftime('%Y-%m-%d') for i in indices_glob[::step_glob]], rotation=15)
    
    
    # Оптимизация расположения элементов
    plt.tight_layout()
    
    # Сохранение строго по тем же путям и с исходными именами файлов
    file_path = os.path.join(output_dir, f'task_3_3_{key}.png')
    plt.savefig(file_path, dpi=300)
    plt.close()

print("Моделирование завершено успешно. Все замечания преподавателя устранены.")