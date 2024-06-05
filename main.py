## Program to creat ECG ##

import neurokit2 as nk  # Load the package
import matplotlib.pyplot as plt
import openpyxl
import pandas as pd
import numpy as np

simulated_ecg = nk.ecg_simulate(duration:=8, sampling_rate:=200, method="daubechies")

## Monta grafico ##
plt.plot(simulated_ecg)
plt.show()

## Converter Array ##

a = np.array(simulated_ecg)
b = a.tolist() #Uma linha em lista


## Create a Excel ##

workbook = openpyxl.Workbook()
sheet = workbook.active
data = [b]
for row in data:
    sheet.append(row)

workbook.save("test_ECG_One.xlsx")
print("Excel file created")