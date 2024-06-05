## Program to creat ECG ##

import neurokit2 as nk  # Load the package
import matplotlib.pyplot as plt
import openpyxl
import numpy as np


# Generate 60 seconds of ECG signal (recorded at 500 samples/second)
simulated_ecg = nk.ecg_simulate(duration:=60, sampling_rate:=500, heart_rate:=70) #method="daubechies"

## Monta grafico ##

plt.plot(simulated_ecg )
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
print("Excel file created :)")
