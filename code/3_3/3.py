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

df = pd.read_csv(file_name, sep=';')

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

# 2. Создание целевой директории для сохранения картинок
save_dir = os.path.join('images', '3_3')
os.makedirs(save_dir, exist_ok=True)

# 3. Цикл фильтрации и построения графиков
for key, config in T_values.items():
    days_limit = config['days']
    text_label = config['label']
    
    # Пересчитываем календарные дни в торговые (в среднем ~252 торговых дня в году)
    T_steps = days_limit * (252 / 365)
    if T_steps < 1:
        T_steps = 1  # Защита от нулевой постоянной времени
        
    num = [1]
    den = [T_steps, 1]
    sys = signal.lti(num, den)
    
    # Фильтрация с учетом начальных условий u[0]
    t_sim, y, x_state = signal.lsim(sys, U=u, T=t, X0=[u[0]])
    
    # Создаем холст: слева - локальный масштаб "а", справа - глобальный "б"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(f'Линейная фильтрация биржевых данных (T = {text_label})', fontsize=14, fontweight='bold')
    
    # -----------------------------------------------------------------
    # График А: Временной диапазон масштаба T (Локальный)
    # -----------------------------------------------------------------
    # Вычисляем, сколько строк попадает в календарный период T
    start_date = dates[0]
    end_date = start_date + pd.Timedelta(days=days_limit)
    display_points = np.sum(dates <= end_date)
    
    # Защита: если точек слишком мало (для 1 дня), берем минимум 5 точек для наглядности линии
    if display_points < 5:
        display_points = 5
        
    # Строим строго по индексам для равномерного шага
    indices_loc = np.arange(display_points)
    ax1.plot(indices_loc, u[:display_points], label='Исходный курс', color='orange', marker='o', markersize=3, alpha=0.7)
    ax1.plot(indices_loc, y[:display_points], label='Сглаженный сигнал', color='blue', linewidth=2, marker='s', markersize=3)
    
    ax1.set_title(f'а) Временной диапазон масштаба T ({text_label})')
    ax1.set_xlabel('Дата')
    ax1.set_ylabel('Цена (руб.)')
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend()
    
    # Красивые и равномерные подписи дат
    step_loc = max(1, display_points // 6)
    ax1.set_xticks(indices_loc[::step_loc])
    ax1.set_xticklabels([pd.Timestamp(dates[i]).strftime('%Y-%m-%d') for i in indices_loc[::step_loc]], rotation=15)
    
    # -----------------------------------------------------------------
    # График Б: Весь массив данных (Глобальный)
    # -----------------------------------------------------------------
    indices_glob = np.arange(len(u))
    ax2.plot(indices_glob, u, label='Исходный курс', color='orange', alpha=0.4)
    ax2.plot(indices_glob, y, label='Сглаженный сигнал', color='blue', linewidth=1.8)
    
    ax2.set_title('б) Весь массив данных (2022–2026)')
    ax2.set_xlabel('Дата')
    ax2.set_ylabel('Цена (руб.)')
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend()
    
    # Распределяем 8 меток дат по всему глобальному графику
    step_glob = len(u) // 8
    ax2.set_xticks(indices_glob[::step_glob])
    ax2.set_xticklabels([pd.Timestamp(dates[i]).strftime('%Y-%m') for i in indices_glob[::step_glob]], rotation=15)
    
    # Оптимизация полей и сохранение
    plt.tight_layout()
    save_path = os.path.join(save_dir, f'task_3_3_{key}.png')
    plt.savefig(save_path, dpi=150)
    plt.close()

print("Все графики успешно перегенерированы и сохранены в папку images/3_3/")