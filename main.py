## Program to creat ECG ##

import neurokit2 as nk  # Load the package
import matplotlib.pyplot as plt

simulated_ecg = nk.ecg_simulate(duration:=8, sampling_rate:=200, method="daubechies")

plt.plot(simulated_ecg)
plt.show()
