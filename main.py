## Program to creat ECG ##


import matplotlib.pyplot as plt
import openpyxl
from openpyxl import load_workbook
import numpy as np
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import pandas as pd
import plotly.graph_objs as go

from plotly.subplots import make_subplots
from scipy.stats import iqr
from scipy.signal import butter, filtfilt
from scipy.interpolate import PchipInterpolator

import kivy
from kivy.app import App
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen



#Primeiro filtro para Remover linha de base

samplerate = 250
window_length = 1
overlap = 0.5

def isoline_correction(filtered_signal):
    # Esta função deve corrigir qualquer offset constante no sinal filtrado
    # Implementação fictícia para propósito ilustrativo
    offset = np.mean(filtered_signal)
    corrected_signal = filtered_signal - offset
    return corrected_signal, offset



def ecg_baseline_removal(signal, samplerate, window_length, overlap):
    # Propriedades do sinal


    L = signal.shape[0]  # comprimento do sinal

    if signal.ndim == 1:
        signal = signal.reshape(-1, 1)  # garante que o sinal seja 2D
    NCH = signal.shape[1]
    #NCH = 1

    baseline = np.zeros_like(signal)  # matriz para armazenar o baseline 0. 0.
    filtered_signal = np.zeros_like(signal)  # matriz para o sinal filtrado


    window_length = int(round(window_length * samplerate))  # comprimento da janela em amostras
    window_length = window_length + 1 - window_length % 2  # garante que o comprimento seja ímpar
    window_half_length = (window_length - 1) // 2  # metade do comprimento da janela


    if 0 <= overlap < 1:
        N = int(np.floor((L - window_length * overlap) / (window_length * (1 - overlap))))  # número de janelas
        center = np.round(window_length * (1 - overlap) * np.arange(N)) + window_half_length
        center = center.astype(int)
    elif overlap == 1:
        center = np.arange(1, L + 1)  # cada amostra é um centro de janela
        N = len(center)  # número de janelas
    else:
        raise ValueError('overlap must be a number between 0 and 1')


    for j in range(NCH):
        baseline_points = np.zeros(center.shape)  # aloca memória para os pontos do baseline
        for i in range(N):
            leftInt = max(center[i] - window_half_length, 0)
            rightInt = min(center[i] + window_half_length, L)
            baseline_points[i] = np.median(signal[leftInt:rightInt, j])  # mediana local

        interpolator = PchipInterpolator(center, baseline_points)
        baseline[:, j] = interpolator(np.arange(L))  # interpolação do baseline
        filtered_signal[:, j] = signal[:, j] - baseline[:, j]  # subtrai o baseline do sinal

        # Correção do offset constante
        filtered_signal[:, j], offset = isoline_correction(filtered_signal[:, j])
        baseline[:, j] += offset

        '''dados = filtered_signal[1:2000]
        print(dados)
        plt.plot(dados)
        plt.title('Baseline Removal')
        plt.xlabel('Time in ms')
        plt.ylabel('Voltage in mV')
        plt.show()'''

    return filtered_signal, baseline

#Segundo filtro para Filtrar sinal



import numpy as np
from scipy.signal import butter, filtfilt, sosfiltfilt


def isoline_correction(signal):
    offset = np.mean(signal, axis=0)
    return signal - offset


def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    sos = butter(order, normal_cutoff, btype='high', analog=False, output='sos')
    return sos


def highpass_filter(signal, samplerate, highpass_frequency, filter_type='Butterworth'):
    # Verificação e transposição do sinal
    if signal.ndim == 1:
        signal = signal.reshape(-1, 1)
    transposeflag = signal.shape[0] < signal.shape[1]
    if transposeflag:
        signal = signal.T

    if filter_type.lower() in ['smooth', 's']:
        case_var = 1
    elif filter_type.lower() in ['gauss', 'g']:
        case_var = 2
    elif filter_type.lower() in ['butterworth', 'b']:
        case_var = 3
    else:
        raise ValueError('Filter type not recognized')

    if not isinstance(signal, np.double):
        signal = signal.astype(np.double)

    NCH = signal.shape[1]

    l = int(round(samplerate * 10))
    filteredsignal = np.pad(signal, ((l, l), (0, 0)), 'constant')

    if highpass_frequency > samplerate / 2:
        print('Warning: Lowpass frequency above Nyquist frequency. Nyquist frequency is chosen instead.')
        highpass_frequency = np.floor(samplerate / 2 - 1)

    if case_var in [1, 3]:
        order = 3
        sos = butter_highpass(highpass_frequency, samplerate, order=order)
        for j in range(NCH):
            filteredsignal[:, j] = sosfiltfilt(sos, filteredsignal[:, j])
    else:
        sigmaf = highpass_frequency
        sigma = samplerate / (2 * np.pi * sigmaf)
        length_gauss = 2 * round(4 * sigma) + 1
        x = np.linspace(-length_gauss // 2, length_gauss // 2, length_gauss)
        h = -np.exp(-(x ** 2) / (2 * sigma ** 2))
        h[length_gauss // 2] = 1 + h[length_gauss // 2]
        for i in range(NCH):
            filteredsignal[:, i] = np.convolve(filteredsignal[:, i], h, mode='same')

    filteredsignal = filteredsignal[l:-l, :]
    filteredsignal = isoline_correction(filteredsignal)

    if transposeflag:
        filteredsignal = filteredsignal.T

    return filteredsignal.squeeze()


# Exemplo de uso
samplerate = 500
highpass_frequency = 0.5

# Sinal sintético de exemplo
t = np.linspace(0, 10, samplerate * 10)
signal = np.sin(2 * np.pi * 1 * t)

# Chamando a função
filtered_signal = highpass_filter(signal, samplerate, highpass_frequency, 'Butterworth')

# Mostrando os resultados
print("Filtered Signal:", filtered_signal)

# passa baixa

import numpy as np
from scipy.signal import butter, filtfilt, sosfiltfilt


def isoline_correction(signal):
    offset = np.mean(signal, axis=0)
    return signal - offset


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    sos = butter(order, normal_cutoff, btype='low', analog=False, output='sos')
    return sos


def lowpass_filter(signal, samplerate, lowpass_frequency, filter_type='Butterworth'):
    # Verificação e transposição do sinal
    if signal.ndim == 1:
        signal = signal.reshape(-1, 1)
    transposeflag = signal.shape[0] < signal.shape[1]
    if transposeflag:
        signal = signal.T

    if filter_type.lower() in ['smooth', 's']:
        case_var = 1
    elif filter_type.lower() in ['gauss', 'g']:
        case_var = 2
    elif filter_type.lower() in ['butterworth', 'b']:
        case_var = 3
    else:
        raise ValueError('Filter type not recognized')

    if not isinstance(signal, np.double):
        signal = signal.astype(np.double)

    NCH = signal.shape[1]

    l = int(round(samplerate * 10))
    filteredsignal = np.pad(signal, ((l, l), (0, 0)), 'constant')

    if lowpass_frequency > samplerate / 2:
        print('Warning: Lowpass frequency above Nyquist frequency. Nyquist frequency is chosen instead.')
        lowpass_frequency = np.floor(samplerate / 2 - 1)

    if case_var == 1:
        nw = round(samplerate / lowpass_frequency)
        for i in range(NCH):
            filteredsignal[:, i] = np.convolve(filteredsignal[:, i], np.ones((nw,)) / nw, mode='same')
    elif case_var == 2:
        sigmaf = lowpass_frequency
        sigma = samplerate / (2 * np.pi * sigmaf)
        length_gauss = 2 * round(4 * sigma) + 1
        x = np.linspace(-length_gauss // 2, length_gauss // 2, length_gauss)
        h = np.exp(-(x ** 2) / (2 * sigma ** 2))
        h = h / np.sum(h)
        for i in range(NCH):
            filteredsignal[:, i] = np.convolve(filteredsignal[:, i], h, mode='same')
    elif case_var == 3:
        order = 3
        sos = butter_lowpass(lowpass_frequency, samplerate, order=order)
        for j in range(NCH):
            filteredsignal[:, j] = sosfiltfilt(sos, filteredsignal[:, j])

    filteredsignal = filteredsignal[l:-l, :]
    filteredsignal = isoline_correction(filteredsignal)

    if transposeflag:
        filteredsignal = filteredsignal.T

    return filteredsignal.squeeze()


# Exemplo de uso
samplerate = 500
lowpass_frequency = 40

# Sinal sintético de exemplo
t = np

#chama os dois
import numpy as np
from scipy.signal import butter, filtfilt


def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y


def lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y


def ecg_high_low_filter(signal, samplerate, highpass_frequency, lowpass_frequency, filter_type='Butterworth'):
    if filter_type == 'Butterworth':
        # Apply highpass filter
        filtered_signal = highpass_filter(signal, highpass_frequency, samplerate)
        # Apply lowpass filter
        filtered_signal = lowpass_filter(filtered_signal, lowpass_frequency, samplerate)
    else:
        raise ValueError(f"Filter type {filter_type} not supported. Only 'Butterworth' is implemented.")

    return filtered_signal


# Exemplo de uso
samplerate = 500  # exemplo de taxa de amostragem em Hz
highpass_frequency = 0.5  # exemplo de frequência de corte passa-alta em Hz
lowpass_frequency = 40.0  # exemplo de frequência de corte passa-baixa em Hz

# Gerando um sinal ECG sintético para exemplo
t = np.linspace(0, 10, samplerate * 10)
signal = np.sin(2 * np.pi * 1 * t)  # sinal sintético de exemplo, substitua pelo seu sinal real

# Chamando a função
filtered_signal = ecg_high_low_filter(signal, samplerate, highpass_frequency, lowpass_frequency)

# Mostrando os resultados
print("Filtered Signal:", filtered_signal)


# Função para abrir a caixa de diálogo e selecionar o arquivo Excel
def selecionar_arquivo():
    Tk().withdraw()  # Ocultar a janela principal do Tkinter
    arquivo_selecionado = askopenfilename(
        title="Selecione o arquivo Excel",
        filetypes=[("Arquivo Excel", "*.xlsx *.xls")]
    )
    return arquivo_selecionado

caminho_do_arquivo = selecionar_arquivo()

# Verificar se um arquivo foi selecionado
if caminho_do_arquivo:
    # Carregar o arquivo Excel, especificando que queremos apenas a primeira coluna


    workbook = load_workbook(filename=caminho_do_arquivo)
    # Selecionar a primeira planilha ativa
    sheet = workbook.active
    # Iterar sobre as células da primeira coluna (A) e criar uma lista
    matriz_uma_coluna = [cell.value for cell in sheet['A']]
    # Exibir a lista como uma matriz de uma linha
    signal = np.array(matriz_uma_coluna)

    # PARTE DO GRAFICO Q DA ERRO NO MAC
    '''dados = signal[1:2000]
    plt.plot(dados)
    plt.title('Unfiltered ECG Signal Lead I')
    plt.xlabel('Time in ms')
    plt.ylabel('Voltage in mV')
    plt.show()'''




    filtered_signal, baseline = ecg_baseline_removal(signal, samplerate,window_length, overlap)

else:
    print("Nenhum arquivo foi selecionado.")










# Generate for rats

"""class MainWindow(Screen):


    def Calcular_ECG(self): #COMO VAI CHAMAR UMA VARIAVEL DE OUTRA FUNCAO

        # Generate simulated ECG data
        fs = 140  # Distance RR
        duration = 5  # Seconds
        t, simulated_ecg = generate_simulated_ecg(self, fs, duration)

        # simulated_ecg = nk.ecg_simulate(duration:=60, sampling_rate:=500, heart_rate:=70)

        # Find R-peaks
        peaks, _ = find_peaks(simulated_ecg, distance=100, height=0.5)

        # Plot the ECG signal with detected R-peaks
        plt.figure(figsize=(12, 4))
        plt.plot(simulated_ecg)
        plt.plot(peaks, simulated_ecg[peaks], "x", color='red', markersize=10)
        plt.title('Simulated ECG Signal with R-peaks Detected')
        plt.xlabel('Sample milisec')
        plt.ylabel('Amplitude')
        plt.grid(True)
        plt.show()

        # Print the indices of detected R-peaks
        print("Indices of R-peaks:", peaks)



        # Save ECG data and R-peak indices to an Excel file
        workbook = openpyxl.Workbook()
        sheet = workbook.active

        # Add headers
        sheet["A1"] = "Time (s)"
        sheet["B1"] = "ECG Signal"
        sheet["C1"] = "R-peak Indices"

        # Add ECG signal and time
        for i, (time, ecg_value) in enumerate(zip(t, simulated_ecg), start=2):
            sheet.cell(row=i, column=1, value=time)
            sheet.cell(row=i, column=2, value=ecg_value)

        # Add R-peak indices
        for i, peak in enumerate(peaks, start=2):
            sheet.cell(row=i, column=3, value=peak)

        workbook.save("test_ECG_One.xlsx")
        print("Excel file created :)")

class WindowManager(ScreenManager): #Cria a janela
    pass

kv = Builder.load_file("ecgscreenalem.kv") #Seleciona o arquivo em kv e cria o app

class ECG(App):  #Determina o nome do aplicativo
    def build(self):
        return kv

if __name__ == "__main__":
    ECG().run()"""




#
#
#
#Artigos para ver depois
#https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5822908/
#https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5351223/
#COMPUTER METHODS AND PROGRAMS IN BIOMEDICINE (PRINT) 0169-2607