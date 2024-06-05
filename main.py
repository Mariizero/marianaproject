## Program to creat ECG ##

import neurokit2 as nk  # Load the package
import matplotlib.pyplot as plt
import openpyxl
import pandas as pd

#simulated_ecg = nk.ecg_simulate(duration:=8, sampling_rate:=200, method="daubechies")
simulated_ecg = 1

plt.plot(simulated_ecg)
plt.show()

## List ##

main_list = []
sub_list= simulated_ecg
main_list.append(sub_list)
print(main_list)
#print(main_list[0])

## Create a Excel ##

workbook = openpyxl.Workbook()
sheet = workbook.active
data = [main_list]
for row in data:
    sheet.append(row)

workbook.save("test_ECG_One.xlsx")
print("Excel file created")