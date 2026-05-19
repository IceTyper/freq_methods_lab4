import numpy as np                 # Для математических операций и работы с массивами
import scipy.signal as signal      # Для определения системы и моделирования её реакции
import matplotlib.pyplot as plt  # Для построения графиков


## --- 1. Определение констант и параметров системы ---
## Параметры исходного сигнала (прямоугольный импульс)
a = 4.0          # Амплитуда импульса 'a'
t1 = -17.0      # Начало импульса 't1'
t2 = 27.0         # Конец импульса 't2'

## Параметры шума
b = 1.5    # Коэффициент усиления шума 'b'
NOISE_DISTRIBUTION_MIN = -1.0  # Минимальное значение для равномерного распределения
NOISE_DISTRIBUTION_MAX = 1.0   # Максимальное значение для равномерного распределения

## Параметры моделирования во времени
SAMPLES_COUNT = 8192       # Количество отсчетов для дискретизации времени
TIME_DOMAIN_END = 40.0     # Конец временной оси для моделирования

## Параметры фильтра (значения T для исследования)
FILTER_TIME_CONSTANTS_T = [0.1, 0.5, 1.0, 2.0]

## --- 2. Генерация временной оси и исходного сигнала ---
## Создание массива времени 'timeVector'
timeVector = np.linspace(0, TIME_DOMAIN_END, SAMPLES_COUNT, endpoint=False)

## Генерация исходного сигнала 'originalSignal'
## Прямоугольный импульс: g(t) = AMPLITUDE_A при t в [START_TIME_T1, END_TIME_T2], иначе 0
g = np.where((timeVector >= t1) & (timeVector <= t2), a, 0.0)

## --- 3. Генерация зашумленного сигнала 'noisySignal' ---
## Генерация белого шума 'noiseVector' с равномерным распределением U[-1, 1]
np.random.seed(42)  # Для воспроизводимости результатов
ksi = np.random.uniform(NOISE_DISTRIBUTION_MIN, NOISE_DISTRIBUTION_MAX, SAMPLES_COUNT)
## Формирование зашумленного сигнала u(t) = g(t) + b * xi(t)
u = g + b * ksi


## --- 4. Моделирование фильтрации для каждого значения T ---
## Словарь для хранения отфильтрованных сигналов
filteredSignalsDict = {}

## Проходим по всем значениям постоянной времени T
for timeConstantT in FILTER_TIME_CONSTANTS_T:
    ## Определение передаточной функции фильтра W(s) = 1 / (T*s + 1)
    ## Коэффициенты числителя и знаменателя полинома
    numeratorCoeffs = [1.0]
    denominatorCoeffs = [timeConstantT, 1.0]
    
    ## Создание объекта системы (динамической системы) из передаточной функции
    systemTransferFunction = signal.TransferFunction(numeratorCoeffs, denominatorCoeffs)
    
    ## Моделирование реакции системы на входной сигнал 'noisySignal'
    ## lsim: линейная системная модель (linear system simulation)
    _, outputSignal, _ = signal.lsim(systemTransferFunction, u, timeVector)
    
    ## Сохраняем отфильтрованный сигнал в словарь с ключом, соответствующему T
    filteredSignalsDict[timeConstantT] = outputSignal


## --- 5. Расчет и построение графиков ---
## --- График 1: Сравнение временных сигналов ---
plt.figure(figsize=(12, 8))

## Построение исходного сигнала
plt.plot(timeVector, g, label=f'Исходный сигнал $g(t), a={a}$', linewidth=2, color='tab:green')

## Построение зашумленного сигнала
plt.plot(timeVector, u, label=f'Зашумленный сигнал $u(t)$', alpha=0.7, color='tab:red')

## Построение отфильтрованных сигналов для разных T
colorsForFiltered = ['tab:blue', 'tab:purple', 'tab:pink', 'k']
for idx, (timeConstantT, filteredSignal) in enumerate(filteredSignalsDict.items()):
    plt.plot(timeVector, filteredSignal, 
             label=f'Отфильтрованный $y(t), T={timeConstantT}$', 
             color=colorsForFiltered[idx], linestyle='--')

## Настройка графика 1
plt.title('Сравнение исходного, зашумленного и отфильтрованных сигналов', fontsize=14, fontweight='bold')
plt.xlabel('Время $t$, с', fontsize=12)
plt.ylabel('Амплитуда', fontsize=12)
plt.legend(loc='upper right', fontsize=10)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.xlim(0, 30)  # Ограничение по оси X для лучшей наглядности

## Сохранение и отображение графика 1
plt.tight_layout()
plt.savefig('C:\\Users\\fmusa\\ITMOStudies\\freg_methods\\freg_methods_lab4\\images\\time_signals.png', dpi=300, bbox_inches='tight')
plt.close()

## --- График 2: Сравнение модулей Фурье-образов ---
## Расчет спектров сигналов
originalSpectrum = np.fft.fft(g)
noisySpectrum = np.fft.fft(u)

## Для сохранения отфильтрованных спектров будем использовать первый полученный сигнал
firstFilteredSignal = filteredSignalsDict[FILTER_TIME_CONSTANTS_T[0]]
filteredSpectrumsList = [np.fft.fft(firstFilteredSignal)]  # Будем добавлять остальные

## Добавляем остальные отфильтрованные сигналы в список для расчета спектров
for timeConstantT in FILTER_TIME_CONSTANTS_T[1:]:
    filteredSpectrumsList.append(np.fft.fft(filteredSignalsDict[timeConstantT]))

## Расчет частотной оси
frequencyAxis = np.fft.fftfreq(SAMPLES_COUNT, d=timeVector[1] - timeVector[0])

## Выбор только первой половины спектров (из-за симметрии для вещественных сигналов)
halfSize = SAMPLES_COUNT // 2
frequencyAxisPositive = frequencyAxis[:halfSize]
originalSpectrumPositive = originalSpectrum[:halfSize]
noisySpectrumPositive = noisySpectrum[:halfSize]
filteredSpectrumsPositive = [spectrum[:halfSize] for spectrum in filteredSpectrumsList]

## Расчет модулей спектров
originalSpectrumMagnitude = np.abs(originalSpectrumPositive)
noisySpectrumMagnitude = np.abs(noisySpectrumPositive)
filteredSpectrumsMagnitude = [np.abs(spectrum) for spectrum in filteredSpectrumsPositive]

## Расчет АЧХ фильтра для T=1.0 для сравнения
achFilterOne = 1.0 / np.sqrt((frequencyAxisPositive * 1.0)**2 + 1.0)

## Построение графика 2
plt.figure(figsize=(12, 8))

## Построение спектра исходного сигнала
plt.semilogy(frequencyAxisPositive, originalSpectrumMagnitude, 
             label=f'Спектр $\\hat{{g}}(\\omega)$', linewidth=2, color='tab:green')

## Построение спектра зашумленного сигнала
plt.semilogy(frequencyAxisPositive, noisySpectrumMagnitude, 
             label=f'Спектр $\\hat{{u}}(\\omega)$', alpha=0.7, color='tab:red')

## Построение спектров отфильтрованных сигналов
for idx, (timeConstantT, filteredSpectrumMagnitude) in enumerate(zip(FILTER_TIME_CONSTANTS_T, filteredSpectrumsMagnitude)):
    plt.semilogy(frequencyAxisPositive, filteredSpectrumMagnitude, 
                 label=f'Спектр $\\hat{{y}}(\\omega), T={timeConstantT}$', 
                 linestyle='--')

## Добавление АЧХ фильтра для T=1.0 для наглядности
plt.semilogy(frequencyAxisPositive, achFilterOne, 
             label='АЧХ $A(\\omega), T=1$', 
             color='black', linewidth=1.5, linestyle=':')

## Настройка графика 2
plt.title('Сравнение модулей Фурье-образов сигналов', fontsize=14, fontweight='bold')
plt.xlabel('Частота $\\omega$, рад/с', fontsize=12)
plt.ylabel('Модуль спектра $|\\hat{x}(\\omega)|$', fontsize=12)
plt.legend(loc='upper right', fontsize=10)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

## Сохранение и отображение графика 2
plt.tight_layout()
plt.savefig('C:\\Users\\fmusa\\ITMOStudies\\freg_methods\\freg_methods_lab4\\images\\fft_modulus.png', dpi=300, bbox_inches='tight')
plt.close()


## --- 6. Расчет и построение графика АЧХ для разных T ---
plt.figure(figsize=(10, 6))

## Диапазон частот для построения АЧХ
achFrequencyAxis = np.logspace(-1, 2, 500)  # Логарифмическая сетка для лучшего вида

## Цвета для кривых АЧХ
colorsForAch = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

## Построение кривых АЧХ для разных T
for idx, timeConstantT in enumerate(FILTER_TIME_CONSTANTS_T):
    achValues = 1.0 / np.sqrt((achFrequencyAxis * timeConstantT)**2 + 1.0)
    plt.semilogx(achFrequencyAxis, achValues, 
                 label=f'$T={timeConstantT}$', color=colorsForAch[idx], linewidth=2.5)

## Настройка графика АЧХ
plt.axvline(x=1.0, color='gray', linestyle='-', alpha=0.5, linewidth=1)  # Вертикальная линия для w=1
plt.text(1.1, 0.5, '$\\omega_c=1/T$', fontsize=12, rotation=-90, verticalalignment='center')
plt.title('Амплитудно-частотная характеристика (АЧХ) фильтра первого порядка', fontsize=14, fontweight='bold')
plt.xlabel('Частота $\\omega$, рад/с', fontsize=12)
plt.ylabel('АЧХ $A(\\omega)$', fontsize=12)
plt.legend(loc='best', fontsize=10)
plt.grid(True, which="both", ls="--", linewidth=0.5)
plt.xlim(achFrequencyAxis[0], achFrequencyAxis[-1])
plt.ylim(0, 1.1)

## Сохранение и отображение графика АЧХ
plt.tight_layout()
plt.savefig('C:\\Users\\fmusa\\ITMOStudies\\freg_methods\\freg_methods_lab4\\images\\bode_T.png', dpi=300, bbox_inches='tight')
plt.close()

## --- Сообщение о завершении программы ---
print("Моделирование завершено. Все графики успешно сохранены в директорию 'src/images/'.")