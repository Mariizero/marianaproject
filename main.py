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




def Calcular_ECG(self):

    # Generate TEM Q ABRIR um arquivo em EXCEL e pegar apenas a primeira coluna.






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