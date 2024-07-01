## Program to creat ECG ##


import matplotlib.pyplot as plt
from openpyxl import load_workbook
import numpy as np
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import PchipInterpolator
from scipy.signal import butter, sosfiltfilt


# Define constants for filtering
lowpass_frequency = 40
highpass_frequency = 1
samplerate = 1000
window_length = 15
overlap = 0.5

def ecg_filter(signal, samplerate, filter_type, lowpass_frequency=None, highpass_frequency=None, filter_method='Butterworth'):
    if signal.ndim == 1:
        signal = signal[:, np.newaxis]

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

    if not np.issubdtype(signal.dtype, np.float64):
        signal = signal.astype(np.float64)

    n_samples, n_channels = signal.shape

    l = int(round(samplerate * 10))
    filteredsignal = np.pad(signal, ((l, l), (0, 0)), mode='constant')

    if lowpass_frequency and lowpass_frequency > samplerate / 2:
        lowpass_frequency = samplerate / 2 - 1

    if highpass_frequency and highpass_frequency > samplerate / 2:
        highpass_frequency = samplerate / 2 - 1

    if filter_type == 'low':
        filteredsignal = apply_lowpass_filter(filteredsignal, samplerate, lowpass_frequency, case_var, n_channels)
    elif filter_type == 'high':
        filteredsignal = apply_highpass_filter(filteredsignal, samplerate, highpass_frequency, case_var, n_channels)
    elif filter_type == 'band':
        if lowpass_frequency is None or highpass_frequency is None:
            raise ValueError('Both lowpass_frequency and highpass_frequency must be specified for bandpass filter.')
        filteredsignal = apply_bandpass_filter(filteredsignal, samplerate, lowpass_frequency, highpass_frequency, case_var, n_channels)
    else:
        raise ValueError('Filter type not recognized')

    filteredsignal = filteredsignal[l:-l, :]

    filteredsignal, _ = isoline_correction(filteredsignal)

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

def isoline_correction(signal, number_bins=None):
    if signal.ndim == 1:
        signal = signal[:, np.newaxis]

    # Alocate filtered signal
    filteredsignal = np.zeros_like(signal)
    # Number of channels in ECG
    number_channels = signal.shape[1]

    # Check for optional input
    if number_bins is None:
        number_bins = min(2**10, signal.shape[0])  # default number of bins for histogram

    # Alocate matrix for histogram frequencies
    frequency_matrix = np.zeros((number_bins, number_channels))
    # Alocate matrix for bin centers
    bins_matrix = np.zeros_like(frequency_matrix)
    offset = np.zeros(number_channels)

    # Constant offset removal
    for i in range(number_channels):
        frequency_matrix[:, i], bin_edges = np.histogram(signal[:, i], bins=number_bins)
        pos = np.argmax(frequency_matrix[:, i])  # find maximum of histogram
        offset[i] = (bin_edges[pos] + bin_edges[pos + 1]) / 2  # find most frequent amplitude in the ECG signal
        filteredsignal[:, i] = signal[:, i] - offset[i]  # remove offset

    return filteredsignal, offset

def smooth(signal, window_len):
    s = np.r_[signal[window_len - 1:0:-1], signal, signal[-2:-window_len - 1:-1]]
    w = np.hanning(window_len)
    y = np.convolve(w / w.sum(), s, mode='valid')
    return y[int(window_len / 2 - 1):-int(window_len / 2)]

def ecg_baseline_removal(signal, samplerate, window_length, overlap):
    L = signal.shape[0]
    if signal.ndim == 1:
        signal = signal.reshape(-1, 1)
    NCH = signal.shape[1]

    baseline = np.zeros_like(signal)
    filtered_signal = np.zeros_like(signal)

    window_length = int(round(window_length * samplerate))
    window_length = window_length + 1 - window_length % 2
    window_half_length = (window_length - 1) // 2

    if 0 <= overlap < 1:
        N = int(np.floor((L - window_length * overlap) / (window_length * (1 - overlap))))
        center = np.round(window_length * (1 - overlap) * np.arange(N)) + window_half_length
        center = center.astype(int)
    elif overlap == 1:
        center = np.arange(1, L + 1)
        N = len(center)
    else:
        raise ValueError('overlap must be a number between 0 and 1')

    for j in range(NCH):
        baseline_points = np.zeros(center.shape)
        for i in range(N):
            leftInt = max(center[i] - window_half_length, 0)
            rightInt = min(center[i] + window_half_length, L)
            baseline_points[i] = np.median(signal[leftInt:rightInt, j])

        interpolator = PchipInterpolator(center, baseline_points)
        baseline[:, j] = interpolator(np.arange(L))
        filtered_signal[:, j] = signal[:, j] - baseline[:, j]

        corrected_signal, offset = isoline_correction(filtered_signal[:, j][:, np.newaxis])
        filtered_signal[:, j] = corrected_signal.flatten()
        baseline[:, j] += offset
        filtered_signal[:, j] += 0.1

    '''plt.plot(filtered_signal[1:2000])
    plt.title('Baseline Removal')
    plt.xlabel('Time in ms')
    plt.ylabel('Voltage in mV')
    plt.show()'''

    return filtered_signal, baseline

def selecionar_arquivo():
    Tk().withdraw()
    arquivo_selecionado = askopenfilename(
        title="Selecione o arquivo Excel",
        filetypes=[("Arquivo Excel", "*.xlsx *.xls")]
    )
    return arquivo_selecionado

caminho_do_arquivo = selecionar_arquivo()

if caminho_do_arquivo:
    workbook = load_workbook(filename=caminho_do_arquivo)
    sheet = workbook.active
    matriz_uma_coluna = [cell.value for cell in sheet['A']]
    signal = np.array(matriz_uma_coluna)

    '''plt.plot(signal[1:2000])
    plt.title('Unfiltered ECG Signal Lead I')
    plt.xlabel('Time in ms')
    plt.ylabel('Voltage in mV')
    plt.show()'''

    # First, perform baseline removal
    filtered_signal, baseline = ecg_baseline_removal(signal, samplerate, window_length, overlap)

    # Then, pass the baseline-corrected signal through the bandpass filter
    final_filtered_signal = ecg_filter(filtered_signal, samplerate, 'band', lowpass_frequency, highpass_frequency, 'Butterworth')
    final_filtered_signal2 = final_filtered_signal + 0.05

    # Plot the final filtered signal
    '''plt.plot(final_filtered_signal2[1:2000])
    plt.title('Filtered ECG Signal')
    plt.xlabel('Time in ms')
    plt.ylabel('Voltage in mV') 
    plt.show()'''

    # Apply isoline correction to final_filtered_signal2
    corrected_final_filtered_signal2, _ = isoline_correction(final_filtered_signal2)

    # Plot the isoline-corrected final_filtered_signal2
    '''plt.plot(corrected_final_filtered_signal2[1:2000])
    plt.title('Isoline-Corrected Filtered ECG Signal')
    plt.xlabel('Time in ms')
    plt.ylabel('Voltage in mV')
    plt.show()'''


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
