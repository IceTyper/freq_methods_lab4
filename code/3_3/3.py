import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# 1. Загрузка данных (укажите правильное имя вашего файла)
file_name = 'code\\3_3\\SBER_220604_260604.csv'
df = pd.read_csv(file_name, sep=';')

# Преобразование даты в удобный формат и извлечение цены закрытия
df['DATE_CONV'] = pd.to_datetime(df['<DATE>'].astype(str), format='%Y%m%d')
u = df['<CLOSE>'].values
dates = df['DATE_CONV'].values
t = np.arange(len(u))

# Наборы значений постоянной времени T (в днях)
T_values = {
    '1_day': 1,
    '1_week': 7,
    '1_month': 30,
    '3_months': 90,
    '1_year': 365
}

# 2. Создание целевой директории images/3_3/
save_dir = os.path.join('images', '3_3')
os.makedirs(save_dir, exist_ok=True)

# 3. Цикл фильтрации и построения графиков
for label, T in T_values.items():
    # Задаем фильтр 1-го порядка в пространстве состояний: W(p) = 1 / (T*p + 1)
    # dx/dt = (-1/T)*x + (1/T)*u
    # y = 1*x + 0*u
    A = [[-1.0 / T]]
    B = [[1.0 / T]]
    C = [[1.0]]
    D = [[0.0]]
    sys = signal.StateSpace(A, B, C, D)
    
    # Коррекция начального условия, чтобы фильтр не стартовал с нуля
    X0 = [u[0]]
    
    # Моделирование линейного фильтра
    _, y, _ = signal.lsim(sys, U=u, T=t, X0=X0)
    
    # Создаем два подобъекта на одном рисунке (а и б)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(f'Линейная фильтрация биржевых данных (T = {label})', fontsize=14, fontweight='bold')
    
    # б) Весь массив данных
    ax1.plot(dates, u, label='Исходный курс (CLOSE)', color='orange', alpha=0.5)
    ax1.plot(dates, y, label='Сглаженный сигнал', color='blue', linewidth=1.5)
    ax1.set_title('б) Весь массив данных (2022–2026)')
    ax1.set_xlabel('Дата')
    ax1.set_ylabel('Цена (руб.)')
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend()
    
    # а) Временной диапазон масштаба T
    # Выбираем количество точек для отображения (минимум 15 для наглядности T=1)
    display_points = int(max(T, 15))
    display_points = min(display_points, len(u)) # защита от выхода за границы
    
    ax2.plot(dates[:display_points], u[:display_points], label='Исходный курс', color='orange', marker='o', markersize=3, alpha=0.7)
    ax2.plot(dates[:display_points], y[:display_points], label='Сглаженный сигнал', color='blue', linewidth=2, marker='s', markersize=3)
    ax2.set_title(f'а) Временной диапазон масштаба T ({label})')
    ax2.set_xlabel('Дата')
    ax2.set_ylabel('Цена (руб.)')
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend()
    
    plt.tight_layout()
    
    # 4. Сохранение графика в папку images/3_3/
    filename = f'task_3_3_{label}.png'
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=300)
    plt.close()
    
    print(f'График сохранен: {filepath}')