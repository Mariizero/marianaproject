## Program to creat ECG ##

import neurokit2 as nk  # Load the package
import matplotlib.pyplot as plt
import openpyxl
import numpy as np

import pandas as pd
import plotly.graph_objs as go

from plotly.subplots import make_subplots
from scipy.stats import iqr
from scipy.signal import find_peaks

import kivy
from kivy.app import App
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen

# Generate 60 seconds of ECG signal (recorded at 500 samples/second) for human
# simulated_ecg = nk.ecg_simulate(duration:=60, sampling_rate:=500, heart_rate:=70) #method="daubechies"


def generate_wave(t, start, end, amplitude, frequency):
    wave = np.zeros_like(t)
    mask = (t >= start) & (t <= end)
    wave[mask] = amplitude * np.sin(2 * np.pi * frequency * (t[mask] - start) / (end - start))
    return wave


def generate_single_cycle(fs=100):
    duration = 1
    t = np.linspace(0, duration, fs * duration)

    p_start, p_end = 0, 0.05  # 0, 0.05
    p_wave = generate_wave(t, p_start, p_end, 0.1, 0.5)

    q_start, q_end = 0.20, 0.21  # 0.09, 0.10
    q_wave = generate_wave(t, q_start, q_end, -0.03, 0.5)

    r_start, r_end = 0.21, 0.23  # 0.10, 0.12
    r_wave = generate_wave(t, r_start, r_end, 0.6, 0.5)

    s_start, s_end = 0.23, 0.25  # 0.12, 0.14
    s_wave = generate_wave(t, s_start, s_end, -0.2, 0.5)

    t_start, t_end = 0.25, 0.32  # 0.14, 0.22
    t_wave = generate_wave(t, t_start, t_end, 0.2, 0.5)

    noise = 0.01 * np.random.randn(len(t))

    ecg = p_wave + q_wave + r_wave + s_wave + t_wave + noise

    return t, ecg


def generate_simulated_ecg(self, fs, duration):
    single_cycle_t, single_cycle_ecg = generate_single_cycle(fs)

    # Calculate the number of cycles needed to cover the duration
    num_cycles = duration

    # Repeat the single cycle to cover the entire duration
    ecg = np.tile(single_cycle_ecg, num_cycles)
    t = np.linspace(0, duration, len(ecg))

    # Add noise
    noise = 0.0 * np.random.randn(len(t))
    ecg += noise

    return t, ecg


def Calcular_ECG(self):  # COMO VAI CHAMAR UMA VARIAVEL DE OUTRA FUNCAO

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





















# Generate for rats

class MainWindow(Screen):

    def generate_wave(t, start, end, amplitude, frequency):
        wave = np.zeros_like(t)
        mask = (t >= start) & (t <= end)
        wave[mask] = amplitude * np.sin(2 * np.pi * frequency * (t[mask] - start) / (end - start))
        return wave


    def generate_single_cycle(fs=100):
        duration = 1
        t = np.linspace(0, duration, fs * duration)

        p_start, p_end = 0, 0.05  # 0, 0.05
        p_wave = generate_wave(t, p_start, p_end, 0.1, 0.5)

        q_start, q_end = 0.20, 0.21  # 0.09, 0.10
        q_wave = generate_wave(t, q_start, q_end, -0.03, 0.5)

        r_start, r_end = 0.21, 0.23  # 0.10, 0.12
        r_wave = generate_wave(t, r_start, r_end, 0.6, 0.5)

        s_start, s_end = 0.23, 0.25  # 0.12, 0.14
        s_wave = generate_wave(t, s_start, s_end, -0.2, 0.5)

        t_start, t_end = 0.25, 0.32  # 0.14, 0.22
        t_wave = generate_wave(t, t_start, t_end, 0.2, 0.5)

        noise = 0.01 * np.random.randn(len(t))

        ecg = p_wave + q_wave + r_wave + s_wave + t_wave + noise

        return t, ecg


    def generate_simulated_ecg(self, fs, duration):
        single_cycle_t, single_cycle_ecg = generate_single_cycle(fs)

        # Calculate the number of cycles needed to cover the duration
        num_cycles = duration

        # Repeat the single cycle to cover the entire duration
        ecg = np.tile(single_cycle_ecg, num_cycles)
        t = np.linspace(0, duration, len(ecg))

        # Add noise
        noise = 0.0 * np.random.randn(len(t))
        ecg += noise

        return t, ecg

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
    ECG().run()




#
#
#
#Artigos para ver depois
#https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5822908/
#https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5351223/
#COMPUTER METHODS AND PROGRAMS IN BIOMEDICINE (PRINT) 0169-2607