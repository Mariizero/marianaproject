## Program to creat ECG ##


import matplotlib.pyplot as plt
import openpyxl
import numpy as np
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import pandas as pd
import plotly.graph_objs as go

from plotly.subplots import make_subplots
from scipy.stats import iqr
from scipy.signal import find_peaks
from scipy.interpolate import PchipInterpolator

import kivy
from kivy.app import App
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen



#Primeiro filtro para Remover linha de base

def isoline_correction(filtered_signal):
    # Esta função deve corrigir qualquer offset constante no sinal filtrado
    # Implementação fictícia para propósito ilustrativo
    offset = np.mean(filtered_signal)
    corrected_signal = filtered_signal - offset
    return corrected_signal, offset

def ecg_baseline_removal(signal, samplerate, window_length, overlap):
    # Propriedades do sinal


    L, NCH = signal.shape  # comprimento do sinal e número de canais
    baseline = np.zeros_like(signal)  # matriz para armazenar o baseline
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

    return filtered_signal, baseline

# Exemplo de uso
# signal: matriz (numpy array) com o sinal ECG
# samplerate: taxa de amostragem (Hz)
# window_length: comprimento da janela (segundos)
# overlap: sobreposição (valor entre 0 e 1)

# signal = np.array(...)  # seu sinal ECG aqui
# samplerate = 500  # exemplo de taxa de amostragem
# window_length = 0.2  # exemplo de comprimento da janela em segundos
# overlap = 0.5  # exemplo de sobreposição

# filtered_signal, baseline = ecg_baseline_removal(signal, samplerate, window_length, overlap)


def ECG_Baseline_Removal():

    #ele levanta os dados para cima de forma q os segmentos fiquem o mais
    #proximos de 0
    print(df)


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
    df = pd.read_excel(caminho_do_arquivo, usecols=[0])
    signal = df
    dados = df[1:2000]

    print(dados)

    ecg_baseline_removal()

    '''plt.plot(dados)
    plt.title('Unfiltered ECG Signal Lead I')
    plt.xlabel('Time in ms')
    plt.ylabel('Voltage in mV')
    plt.show()'''

    #print(df.to_string(index=False))
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