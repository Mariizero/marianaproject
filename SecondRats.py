## Program to creat ECG RATS##

import matplotlib.pyplot as plt
from openpyxl import load_workbook
import numpy as np
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import PchipInterpolator
from scipy.signal import butter, sosfiltfilt


# Define constants for filtering
lowpass_frequency = 100
highpass_frequency = 0.5
notch_frequency = 50
samplerate = 1000
window_length = 1
overlap = 0.5
width = 1

def ecg_filter(signal, samplerate, filter_types, lowpass_frequency=None, highpass_frequency=None, notch_frequency=None, filter_method='Butterworth'):
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
        raise ValueError('Filter method not recognized')

    if not np.issubdtype(signal.dtype, np.float64):
        signal = signal.astype(np.float64)

    n_samples, n_channels = signal.shape
    l = int(round(samplerate * 10))
    filteredsignal = np.pad(signal, ((l, l), (0, 0)), mode='constant')

    if lowpass_frequency and lowpass_frequency > samplerate / 2:
        lowpass_frequency = samplerate / 2 - 1
    if highpass_frequency and highpass_frequency > samplerate / 2:
        highpass_frequency = samplerate / 2 - 1

    for filter_type in filter_types:
        if filter_type == 'low':
            filteredsignal = apply_lowpass_filter(filteredsignal, samplerate, lowpass_frequency, case_var, n_channels)
        elif filter_type == 'high':
            filteredsignal = apply_highpass_filter(filteredsignal, samplerate, highpass_frequency, case_var, n_channels)
        elif filter_type == 'notch':
            filteredsignal = apply_notch_filter(filteredsignal, samplerate, notch_frequency, width)
        elif filter_type == 'band':
            if lowpass_frequency is None or highpass_frequency is None:
                raise ValueError('Both lowpass_frequency and highpass_frequency must be specified for bandpass filter.')
            filteredsignal = apply_bandpass_filter(filteredsignal, samplerate, lowpass_frequency, highpass_frequency, case_var, n_channels)
        else:
            raise ValueError('Filter type not recognized')

    filteredsignal = filteredsignal[l:-l, :]
    filteredsignal, offset = isoline_correction(filteredsignal)

    
    '''plt.figure(figsize=(12, 6))
    plt.plot(filteredsignal[1:2000], label='Filtered Signal', color='blue', alpha=0.75)
    plt.axhline(y=offset, color='green', linestyle='--', label='Estimated Isoline')
    plt.legend()
    plt.title('High/Low-pass filtered signal')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.show()'''

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
            #print("Filtro 1 foi")
    return signal

def apply_highpass_filter(signal, samplerate, highpass_frequency, case_var, n_channels):
    if case_var == 3:  # Butterworth filter
        order = 3
        sos = butter(order, 2 * highpass_frequency / samplerate, btype='high', output='sos')
        for i in range(n_channels):
            signal[:, i] = sosfiltfilt(sos, signal[:, i])
            #print("Filtro 2 foi")
    else:
        raise NotImplementedError("High-pass filter is only implemented for Butterworth filter.")
    return signal

def apply_notch_filter(signal, samplerate, notch_frequency, width):
    # The spectrum will have peaks at k*f0Hz. K gives the greatest number n 
    # that can be chosen for a harmonic oscillation without going beyond the 
    # Nyquist frequency 
    K = int(np.floor(samplerate / 2 / notch_frequency))

    # Extend signal to avoid boundary effects
    extpoints = int(round(0.5 * np.ceil(samplerate / width)))
    signal_extended = np.pad(signal, ((extpoints, extpoints), (0, 0)), 'symmetric')

    L = signal_extended.shape[0]  # Length of the signal
    f = np.fft.fftfreq(L, d=1/samplerate)  # Frequency vector

    sigmaf = width  # Standard deviation of Gaussian bell used to select frequency
    sigma = int(np.ceil(L * sigmaf / samplerate))  # Sigma discrete
    lg = 2 * round(4 * sigma) + 1  # Size of Gaussian bell
    lb = (lg - 1) // 2  # Position of center of Gaussian bell
    
    # Gaussian bell creation
    g = gaussian_filter1d(np.eye(1, lg).flatten(), sigma)
    g = 1 / (np.max(g) - np.min(g)) * (np.max(g) - g)  # Scale Gaussian bell to be in interval [0;1]

    H = np.ones(L)  # Filter

    # Implementation of periodical Gaussian bells at k*f0Hz
    for k in range(1, K+1):
        b = np.argmin(np.abs(f - k * notch_frequency))  # Discrete position at which f = k*f0Hz
        H[b-lb:b+lb+1] = g  # Gaussian bell placed around k*f0Hz
        H[L-b-lb:L-b+lb+1] = g  # Gaussian bell placed symmetrically around samplerate - k*f0Hz

    H = np.tile(H, (signal_extended.shape[1], 1)).T  # Reproduce the filter for all channels
    X = np.fft.fft(signal_extended, axis=0)  # FFT of signal
    Y = H * X  # Filtering process in the Fourier Domain
    signal = np.real(np.fft.ifft(Y, axis=0))  # Reconstruction of filtered signal
    signal = signal[extpoints:-extpoints, :]  # Remove extended portions
    #print("Filtro 3 foi")
    
    return signal


def apply_bandpass_filter(signal, samplerate, lowpass_frequency, highpass_frequency, case_var, n_channels):
    signal = apply_lowpass_filter(signal, samplerate, lowpass_frequency, case_var, n_channels)
    signal = apply_highpass_filter(signal, samplerate, highpass_frequency, case_var, n_channels)
    #print("Filtro 4 foi")
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
        filtered_signal[:, j] += 0.05

    '''plt.figure(figsize=(14, 7))
    plt.plot(signal[1:2000], label='Signal', color='blue')
    plt.plot(baseline[1:2000], label='Estimated baseline', color='red')
    plt.legend()
    plt.title('Input signal')
    plt.xlabel('Time in ms')
    plt.ylabel('Voltage in mV')
    plt.show()'''

    '''plt.figure(figsize=(14, 7))
    plt.plot(filtered_signal[1:450])
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

    plt.figure(figsize=(12, 6))
    plt.plot(signal[1:450])
    plt.title('Unfiltered ECG Signal Lead I')
    plt.xlabel('Time in ms')
    plt.ylabel('Voltage in mV')
    plt.show()

    # First, perform baseline removal
    filtered_signal, baseline = ecg_baseline_removal(signal, samplerate, window_length, overlap)
    
    # Define the sequence of filters to be applied
    
    filter_types = ['low', 'high', 'notch', 'band']
    # Then, pass the baseline-corrected signal through the bandpass filter
    #final_filtered_signal = ecg_filter(filtered_signal, samplerate, 'band', lowpass_frequency, highpass_frequency, 'Butterworth')
    final_filtered_signal = ecg_filter(filtered_signal, samplerate, filter_types, lowpass_frequency, highpass_frequency, notch_frequency, 'Butterworth')
    final_filtered_signal2 = final_filtered_signal #+ 0.05

    

    # Plot the final filtered signal
    '''plt.figure(figsize=(12, 6))
    plt.plot(final_filtered_signal2[1:2000])
    plt.title('Filtered ECG Signal')
    plt.xlabel('Time in ms')
    plt.ylabel('Voltage in mV') 
    plt.show()'''

    # Apply isoline correction to final_filtered_signal2
    corrected_final_filtered_signal2, offset = isoline_correction(final_filtered_signal2)

    # Plot the isoline-corrected final_filtered_signal2
    plt.figure(figsize=(12, 6))
    plt.plot(corrected_final_filtered_signal2[1:450])
    plt.title('Isoline-Corrected Filtered ECG Signal')
    plt.xlabel('Time in ms')
    plt.ylabel('Voltage in mV')
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(signal[1:450], label='Original Signal', color='blue', alpha=0.5)
    plt.plot(corrected_final_filtered_signal2[1:450], label='Filtered Signal', color='red', alpha=0.75)
    #plt.axhline(y=offset, color='green', linestyle='--', label='Offset')
    plt.legend()
    plt.title('Signal and Filtered Signal')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.show()

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


