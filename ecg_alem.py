import kivy
from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.gridlayout import GridLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.uix.widget import Widget
from kivy.uix.spinner import Spinner
from kivy.properties import ObjectProperty
from kivy.lang import Builder
from kivy.garden.matplotlib.backend_kivyagg import FigureCanvasKivyAgg
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import pandas as pd
import openpyxl
import numpy as np
from scipy.signal import find_peaks
import math
from kivy.uix.screenmanager import ScreenManager, Screen
import serial.tools.list_ports
import serial
import time
from kivy.clock import Clock
from datetime import date
from io import BytesIO
from svglib.svglib import svg2rlg
from reportlab.pdfgen import canvas
from reportlab.graphics import renderPDF
import os
from datetime import datetime
import subprocess
import sqlite3
import wfdb



class MainWindow(Screen):

    Screen.Trava=0
    Screen.ard2=0
    Screen.nomep=0

    ### CONECTAR COM ARDUINO ###
    def ConectarArd(self):

        self.ids.IsCon.text = ". . ."

        global start
        global baud
        global n
        global ports

        n = -1

        baud = 115200 #TEM Q MUDAR ISSO TALVEZ NAO CONSIGA MAIS PLOTAR EM TEMPO REAL
        ports = list(serial.tools.list_ports.comports())

        Clock.schedule_interval(self.ConArd, 1 / 30.)

    def ConArd(self, dt):




        directory = "./your/directory/"
        ECGs = []
        for ecgfilename in sorted(os.listdir(directory )):
            if ecgfilename.endswith(".dat"):
                ecg = wfdb.rdsamp(directory  + ecgfilename.split(".")[0])
                ECGs.append(ecg)
        ECGs = np.asarray(ECGs)


### INICIAR COLETA ECG ###
    def ECGCol(self):

        global DadosECG
        global BG

        Screen.ard2.write(str.encode("c"))

        DadosECG = []


        BG = Clock.schedule_interval(self.LopECG, 1 / 1000)

    def LopECG(self, dt):

        valueECG = Screen.ard2.readline().decode('utf-8')

        if valueECG:

            if "E" in valueECG:

                ECGverdade = valueECG.split("E")[1].strip()

                valueECGTotal = float(ECGverdade)

                DadosECG.append(valueECGTotal)

                Dadosgraf = DadosECG[-500:]

                #print(DadosECG)

                plt.cla()

                plt.figure(num=2)

                plt.plot(Dadosgraf)



                ArdBox = self.ids.ArdBox
                ArdBox.clear_widgets()
                ArdBox.add_widget(FigureCanvasKivyAgg(plt.figure(num=2)), index=2)  # cgf = get current figure


### FINALIZAR COLETA ECG ###
    def ECGFim(self):

        global BG
        BG.cancel()
        self.SalvarECG()

        Screen.ard2.write(str.encode("p"))
        print("PAROU")


### SALVAR DADOS ECG ###
    def SalvarECG(self):

        NomePasta = Screen.nomep
    
        diretorio_pai = os.getcwd() #Pega o diretório onde o programa está
        diretorio_filho = 'TesteECG\\' + NomePasta #Gerar a subpasta

        caminho_arquivo = os.path.join(diretorio_pai, diretorio_filho)

        if not os.path.exists(caminho_arquivo):# Cria o diretório filho se não existir
            os.makedirs(caminho_arquivo)

        curr_datetime = datetime.now().strftime('%Y-%m-%d %H-%M-%S') + ".xlsx"

        caminho_final = os.path.join(caminho_arquivo, curr_datetime)

        print(caminho_final)

        dfw = pd.DataFrame(DadosECG)

        dfw.to_excel(caminho_final, engine='openpyxl')




    ####### NÃO MEXER #########
class WindowManager(ScreenManager): #Cria a janela
    pass

kv = Builder.load_file("ecgscreenalem.kv") #Seleciona o arquivo em kv e cria o app

class ECG(App):  #Determina o nome do aplicativo
    def build(self):
        return kv

if __name__ == "__main__":
    ECG().run()