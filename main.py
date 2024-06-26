## Program to creat ECG ##


import matplotlib.pyplot as plt
import openpyxl
from openpyxl import load_workbook
import numpy as np
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import plotly.graph_objs as go

from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import PchipInterpolator
from plotly.subplots import make_subplots
from scipy.stats import iqr
from scipy.interpolate import PchipInterpolator
from scipy.signal import butter, filtfilt, sosfiltfilt

import kivy
from kivy.app import App
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen


#Segundo filtro para Filtrar sinal ELE ESTA INICIALIZANDO DIRETO VER ISSO

def ecg_filter(signal, samplerate, filter_type, lowpass_frequency=None, highpass_frequency=None, filter_method='Butterworth'):
    # Ensure the signal is a 2D array (n_samples, n_channels)
    if signal.ndim == 1:
        signal = signal[:, np.newaxis]
        print('Passando por aqui')

    if signal.shape[1] > signal.shape[0]:
        signal = signal.T
        transpose_flag = True
    else:
        transpose_flag = False

    if filter_method.lower() in ['smooth', 's']:
        case_var = 1
    elif filter_method.lower() in ['gauss', 'g']:
        case_var = 2
    elif filter_method.lower() in ['butterworth', 'b']:
        case_var = 3
    else:
        raise ValueError('Filter type not recognized')

    # Convert signal to double if it's not already
    if not np.issubdtype(signal.dtype, np.float64):
        signal = signal.astype(np.float64)

    # Number of channels
    n_samples, n_channels = signal.shape
    print('Passando por aqui 2')

    # Extend signal to avoid bordering artifacts
    l = int(round(samplerate * 10))
    filteredsignal = np.pad(signal, ((l, l), (0, 0)), mode='constant')

    if lowpass_frequency and lowpass_frequency > samplerate / 2:
        print('Warning: Lowpass frequency above Nyquist frequency. Nyquist frequency is chosen instead.')
        lowpass_frequency = samplerate / 2 - 1

    if highpass_frequency and highpass_frequency > samplerate / 2:
        print('Warning: Highpass frequency above Nyquist frequency. Nyquist frequency is chosen instead.')
        highpass_frequency = samplerate / 2 - 1

    if filter_type == 'low':
        filteredsignal = apply_lowpass_filter(filteredsignal, samplerate, lowpass_frequency, case_var, n_channels)
    elif filter_type == 'high':
        filteredsignal = apply_highpass_filter(filteredsignal, samplerate, highpass_frequency, case_var, n_channels)
    elif filter_type == 'band':
        if lowpass_frequency is None or highpass_frequency is None:
            raise ValueError('Both lowpass_frequency and highpass_frequency must be specified for bandpass filter.')
        filteredsignal = apply_bandpass_filter(filteredsignal, samplerate, lowpass_frequency, highpass_frequency,
                                               case_var, n_channels)
    else:
        raise ValueError('Filter type not recognized')

    # Remove extension of signal
    filteredsignal = filteredsignal[l:-l, :]

    # Constant offset removal
    filteredsignal = isoline_correction(filteredsignal)

    if transpose_flag:
        filteredsignal = filteredsignal.T

    return filteredsignal


def apply_lowpass_filter(signal, samplerate, lowpass_frequency, case_var, n_channels):
    if case_var == 1:  # Smoothing filter
        nw = int(round(samplerate / lowpass_frequency))
        for i in range(n_channels):
            signal[:, i] = smooth(signal[:, i], nw)
    elif case_var == 2:  # Gaussian filter
        sigmaf = lowpass_frequency
        sigma = samplerate / (2 * np.pi * sigmaf)
        signal = gaussian_filter1d(signal, sigma, axis=0)
    elif case_var == 3:  # Butterworth filter
        order = 3
        sos = butter(order, 2 * lowpass_frequency / samplerate, btype='low', output='sos')
        for i in range(n_channels):
            signal[:, i] = sosfiltfilt(sos, signal[:, i])
    return signal


def apply_highpass_filter(signal, samplerate, highpass_frequency, case_var, n_channels):
    if case_var == 3:  # Butterworth filter
        order = 3
        sos = butter(order, 2 * highpass_frequency / samplerate, btype='high', output='sos')
        for i in range(n_channels):
            signal[:, i] = sosfiltfilt(sos, signal[:, i])
    else:
        raise NotImplementedError("High-pass filter is only implemented for Butterworth filter.")
    return signal


def apply_bandpass_filter(signal, samplerate, lowpass_frequency, highpass_frequency, case_var, n_channels):
    signal = apply_lowpass_filter(signal, samplerate, lowpass_frequency, case_var, n_channels)
    signal = apply_highpass_filter(signal, samplerate, highpass_frequency, case_var, n_channels)
    return signal


def isoline_correction(signal):
    return signal - np.mean(signal, axis=0)


def smooth(signal, window_len):
    s = np.r_[signal[window_len - 1:0:-1], signal, signal[-2:-window_len - 1:-1]]
    w = np.hanning(window_len)
    y = np.convolve(w / w.sum(), s, mode='valid')
    return y[int(window_len / 2 - 1):-int(window_len / 2)]


# Example usage
if __name__ == '__main__':
    samplerate = 500  # Example sample rate
    signal = np.sin(2 * np.pi * 1 * np.arange(0, 10, 1 / samplerate))  # Example signal
    lowpass_frequency = 40
    highpass_frequency = 1
    filtered_signal = ecg_filter(signal, samplerate, 'band', lowpass_frequency, highpass_frequency, 'Butterworth')



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


        filtered_signal = ecg_filter(signal, samplerate, 'band', lowpass_frequency, highpass_frequency, 'Butterworth')

    return filtered_signal, baseline



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