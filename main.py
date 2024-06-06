## Program to creat ECG ##

import neurokit2 as nk  # Load the package
import matplotlib.pyplot as plt
import openpyxl
import numpy as np
import wfdb
import os
import csv
from tabulate import tabulate

import pandas as pd
import plotly.graph_objs as go

from plotly.subplots import make_subplots
from scipy.stats import iqr


# Generate 60 seconds of ECG signal (recorded at 500 samples/second) for human
#simulated_ecg = nk.ecg_simulate(duration:=60, sampling_rate:=500, heart_rate:=70) #method="daubechies"

#Generate for rats

simulated_ecg = []
simulated_ecg = nk.ecg_simulate(duration:=60, sampling_rate:=500, frequency:=50)
simulated_ecg = np.array(simulated_ecg, dtype:="object") ## Converter Array ##




b = simulated_ecg.tolist() #Uma linha em lista
print(b)


## Monta grafico ##

plt.plot(simulated_ecg)
plt.show()



## Create a Excel ##

workbook = openpyxl.Workbook()
sheet = workbook.active
data = [b] #TROCAR AQUI DEPOIS
for row in data:
    sheet.append(row)

workbook.save("test_ECG_One.xlsx")
print("Excel file created :)")
