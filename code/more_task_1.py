## Имя файла: src/code/dynamic_filtering_additional_plots.py

import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import os

## Создание директории для изображений, если её нет
os.makedirs('src/images', exist_ok=True)

## --- 1. Константы ---
amplitudeA = 4.0
startTimeT1 = -17.0
endTimeT2 = 27.0
noiseIntensityB = 1.5
samplesCount = 8192
timeDomainEnd = 40.0
filterTimeConstantsT = [0.1, 0.5, 1.0, 2.0]

## --- 2. Генерация сигналов ---
timeVector = np.linspace(0, timeDomainEnd, samplesCount, endpoint=False)
originalSignal = np.where((timeVector >= startTimeT1) & (timeVector <= endTimeT2), amplitudeA, 0.0)

np.random.seed(42)
noiseVector = np.random.uniform(-1.0, 1.0, samplesCount)
noisySignal = originalSignal + noiseIntensityB * noiseVector

## --- 3. Функция для получения отфильтрованного сигнала (lsim) ---
def getFilteredSignalTime(timeConstantT, inputSignal, timeVec):
    numeratorCoeffs = [1.0]
    denominatorCoeffs = [timeConstantT, 1.0]
    systemTransferFunction = signal.TransferFunction(numeratorCoeffs, denominatorCoeffs)
    _, outputSignal, _ = signal.lsim(systemTransferFunction, inputSignal, timeVec)
    return outputSignal

## --- 4. Функция для фильтрации в частотной области ---
def getFilteredSignalFreq(timeConstantT, inputSignal, timeVec):
    ## Расчет спектра входа
    uSpectrum = np.fft.fft(inputSignal)
    freqAxis = np.fft.fftfreq(len(timeVec), d=timeVec[1] - timeVec[0])
    
    ## Расчет частотной передаточной функции W(iw)
    ## W(iw) = 1 / (T * i * w + 1)
    wAxis = 2 * np.pi * freqAxis
    wTransferFunction = 1.0 / (timeConstantT * 1j * wAxis + 1.0)
    
    ## Умножение в частотной области
    ySpectrum = uSpectrum * wTransferFunction
    
    ## Обратное преобразование Фурье
    outputSignal = np.real(np.fft.ifft(ySpectrum))
    return outputSignal, freqAxis, wTransferFunction, uSpectrum, ySpectrum

## --- ГРАФИК 4: Сравнение методов фильтрации (lsim vs ifft) ---
## Берем один показательный случай, например T = 1.0
timeConstantT_example = 1.0
signalLsim = getFilteredSignalTime(timeConstantT_example, noisySignal, timeVector)
signalFreq, freqAxis, wFunc, uSpec, ySpec = getFilteredSignalFreq(timeConstantT_example, noisySignal, timeVector)

plt.figure(figsize=(12, 6))
plt.plot(timeVector, signalLsim, label='Фильтрация во временной области (lsim)', linewidth=2, color='tab:blue')
plt.plot(timeVector, signalFreq, label='Фильтрация через обратное Фурье (ifft)', linestyle='--', linewidth=1.5, color='tab:red')
plt.title(f'Сравнение методов фильтрации при T={timeConstantT_example}', fontsize=14, fontweight='bold')
plt.xlabel('Время $t$, с', fontsize=12)
plt.ylabel('Амплитуда', fontsize=12)
plt.legend(loc='upper right', fontsize=10)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.xlim(0, 30)
plt.tight_layout()
plt.savefig('C:\\Users\\fmusa\\ITMOStudies\\freg_methods\\freg_methods_lab4\\images\\filter_methods_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

## --- ГРАФИК 6: Сравнение модулей спектров (Y_spec vs W*U_spec) ---
## Нужно сравнить |Y(w)| (полученный из FFT фильтра) и |W(iw)*U(w)|
## signalFreq мы уже имеем, посчитаем его спектр для проверки
ySpecDirect = np.fft.fft(signalFreq)
ySpecDirectMag = np.abs(ySpecDirect[:samplesCount // 2])
productSpecMag = np.abs(uSpec[:samplesCount // 2] * wFunc[:samplesCount // 2])
freqAxisPositive = freqAxis[:samplesCount // 2]

plt.figure(figsize=(12, 6))
plt.semilogy(freqAxisPositive, ySpecDirectMag, label='Модуль спектра $|\hat{y}(\omega)|$ (из фильтра)', linewidth=2, color='tab:blue')
plt.semilogy(freqAxisPositive, productSpecMag, label='Модуль произведения $|W_1(i\omega) \cdot \\hat{u}(\omega)|$', linestyle='--', linewidth=1.5, color='tab:red')
plt.title(f'Сравнение спектров при T={timeConstantT_example}', fontsize=14, fontweight='bold')
plt.xlabel('Частота $\\omega$, рад/с', fontsize=12)
plt.ylabel('Модуль спектра', fontsize=12)
plt.legend(loc='upper right', fontsize=10)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.xlim(0, 10) ## Ограничим частоты для наглядности
plt.tight_layout()
plt.savefig('C:\\Users\\fmusa\\ITMOStudies\\freg_methods\\freg_methods_lab4\\images\\spectrum_product_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

## --- ГРАФИКИ ДЛЯ ИССЛЕДОВАНИЯ ПАРАМЕТРОВ (Пункт 2 методички) ---

## А) Влияние T при фиксированном a (уже частично есть, но добавим сравнение всех на одном для наглядности неудачных/удачных)
plt.figure(figsize=(12, 8))
plt.plot(timeVector, originalSignal, label='Исходный $g(t)$', linewidth=2, color='tab:green')
plt.plot(timeVector, noisySignal, label='Зашумленный $u(t)$', alpha=0.5, color='gray')
colorsT = ['tab:blue', 'tab:orange', 'tab:purple', 'tab:red']
for idx, tVal in enumerate(filterTimeConstantsT):
    sig = getFilteredSignalTime(tVal, noisySignal, timeVector)
    plt.plot(timeVector, sig, label=f'Фильтр $T={tVal}$', color=colorsT[idx], linestyle='-')
plt.title('Влияние постоянной времени T на форму сигнала', fontsize=14, fontweight='bold')
plt.xlabel('Время $t$, с', fontsize=12)
plt.ylabel('Амплитуда', fontsize=12)
plt.legend(loc='upper right', fontsize=10)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.xlim(0, 30)
plt.tight_layout()
plt.savefig('C:\\Users\\fmusa\\ITMOStudies\\freg_methods\\freg_methods_lab4\\images\\influence_of_T.png', dpi=300, bbox_inches='tight')
plt.close()

## Б) Влияние a при фиксированном T
fixedT = 1.0
amplitudesA = [0.5, 2.0, 4.0, 8.0]
plt.figure(figsize=(12, 8))
colorsA = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
for idx, aVal in enumerate(amplitudesA):
    ## Генерируем сигнал для каждой амплитуды
    origSigA = np.where((timeVector >= startTimeT1) & (timeVector <= endTimeT2), aVal, 0.0)
    noisySigA = origSigA + noiseIntensityB * noiseVector ## Шум тот же, меняется SNR
    filtSigA = getFilteredSignalTime(fixedT, noisySigA, timeVector)
    plt.plot(timeVector, filtSigA, label=f'Фильтрация при $a={aVal}$', color=colorsA[idx], linestyle='-')

plt.title(f'Влияние амплитуды a на результат фильтрации (фиксировано T={fixedT})', fontsize=14, fontweight='bold')
plt.xlabel('Время $t$, с', fontsize=12)
plt.ylabel('Амплитуда', fontsize=12)
plt.legend(loc='upper right', fontsize=10)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.xlim(0, 30)
plt.tight_layout()
plt.savefig('C:\\Users\\fmusa\\ITMOStudies\\freg_methods\\freg_methods_lab4\\images\\influence_of_a.png', dpi=300, bbox_inches='tight')
plt.close()

print("Дополнительные графики успешно сохранены.")