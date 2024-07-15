## Program to creat ECG ##


import matplotlib.pyplot as plt
from openpyxl import load_workbook
import numpy as np
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import PchipInterpolator
from scipy.signal import butter, sosfiltfilt
import scipy.signal as signal
from scipy.signal import decimate
from scipy.io import loadmat
from scipy.signal import find_peaks
from scipy.stats import scoreatpercentile
import pywt

# Define constants for filtering
lowpass_frequency = 150
highpass_frequency = 1
notch_frequency = 50
samplerate = 1000
window_length = 1
overlap = 0.5
width = 1


def ecg_filter(signal, samplerate, filter_types, lowpass_frequency=None, highpass_frequency=None, notch_frequency=None,
               filter_method='Butterworth'):
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
            filteredsignal = apply_bandpass_filter(filteredsignal, samplerate, lowpass_frequency, highpass_frequency,
                                                   case_var, n_channels)
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
            # print("Filtro 1 foi")
    return signal


def apply_highpass_filter(signal, samplerate, highpass_frequency, case_var, n_channels):
    if case_var == 3:  # Butterworth filter
        order = 3
        sos = butter(order, 2 * highpass_frequency / samplerate, btype='high', output='sos')
        for i in range(n_channels):
            signal[:, i] = sosfiltfilt(sos, signal[:, i])
            # print("Filtro 2 foi")
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
    f = np.fft.fftfreq(L, d=1 / samplerate)  # Frequency vector

    sigmaf = width  # Standard deviation of Gaussian bell used to select frequency
    sigma = int(np.ceil(L * sigmaf / samplerate))  # Sigma discrete
    lg = 2 * round(4 * sigma) + 1  # Size of Gaussian bell
    lb = (lg - 1) // 2  # Position of center of Gaussian bell

    # Gaussian bell creation
    g = gaussian_filter1d(np.eye(1, lg).flatten(), sigma)
    g = 1 / (np.max(g) - np.min(g)) * (np.max(g) - g)  # Scale Gaussian bell to be in interval [0;1]

    H = np.ones(L)  # Filter

    # Implementation of periodical Gaussian bells at k*f0Hz
    for k in range(1, K + 1):
        b = np.argmin(np.abs(f - k * notch_frequency))  # Discrete position at which f = k*f0Hz
        H[b - lb:b + lb + 1] = g  # Gaussian bell placed around k*f0Hz
        H[L - b - lb:L - b + lb + 1] = g  # Gaussian bell placed symmetrically around samplerate - k*f0Hz

    H = np.tile(H, (signal_extended.shape[1], 1)).T  # Reproduce the filter for all channels
    X = np.fft.fft(signal_extended, axis=0)  # FFT of signal
    Y = H * X  # Filtering process in the Fourier Domain
    signal = np.real(np.fft.ifft(Y, axis=0))  # Reconstruction of filtered signal
    signal = signal[extpoints:-extpoints, :]  # Remove extended portions
    # print("Filtro 3 foi")

    return signal


def apply_bandpass_filter(signal, samplerate, lowpass_frequency, highpass_frequency, case_var, n_channels):
    signal = apply_lowpass_filter(signal, samplerate, lowpass_frequency, case_var, n_channels)
    signal = apply_highpass_filter(signal, samplerate, highpass_frequency, case_var, n_channels)
    # print("Filtro 4 foi")
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
        number_bins = min(2 ** 10, signal.shape[0])  # default number of bins for histogram

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
    plt.plot(signal[1:2000], label='Filtered Signal', color='blue')
    plt.plot(baseline[1:2000], label='Baseline', color='red')
    plt.title('Estimated baseline')
    plt.xlabel('Time in ms')
    plt.ylabel('Voltage in mV')
    plt.show()


    plt.figure(figsize=(14, 7))
    plt.plot(filtered_signal[1:2000])
    plt.title('Baseline Removal')
    plt.xlabel('Time in ms')
    plt.ylabel('Voltage in mV')
    plt.show()'''

    return filtered_signal, baseline


# ENCONTRAR O PICO R AQUI
def butter_highpass_filter(signal, samplerate, highpass_frequency):
    order = 3
    n_channels = 1
    sos = butter(order, 2 * highpass_frequency / samplerate, btype='high', output='sos')

    if n_channels == 1:
        # Single-channel case
        filtered_signal3 = sosfiltfilt(sos, signal)
    else:
        # Multi-channel case
        filtered_signal3 = np.zeros_like(signal)
        for i in range(n_channels):
            filtered_signal3[:, i] = sosfiltfilt(sos, signal[:, i])

    return filtered_signal3


def butter_lowpass_filter(signal, samplerate, lowpass_frequency):
    order = 3
    n_channels = 1
    sos = butter(order, 2 * lowpass_frequency / samplerate, btype='low', output='sos')

    if n_channels == 1:
        # Single-channel case
        filtered_signal3 = sosfiltfilt(sos, signal)
    else:
        # Multi-channel case
        filtered_signal3 = np.zeros_like(signal)
        for i in range(n_channels):
            filtered_signal3[:, i] = sosfiltfilt(sos, signal[:, i])

    return filtered_signal3


def QRS_Detection(signal, samplerate, peaksQRS=False, mute=False):
    # Initialization
    flag_posR = peaksQRS
    if not mute:
        print('Detecting R Peaks...')

    # Ensure signal is numpy array of type float64
    signal = np.asarray(corrected_final_filtered_signal2, dtype=np.float64).flatten()

    # Check if signal is a vector
    if signal.ndim != 1:
        raise ValueError('The input ECG signal must be a vector!')

    # Check for small signal values
    if np.all(np.abs(signal) < np.finfo(float).eps):
        if not mute:
            print('The signal values are too small to process. Returning empty FPT table')
        return None

    # Denoise ECG: Highpass and Lowpass filtering
    highpass_frequency = 0.5
    lowpass_frequency = 30
    filtered_signal3 = butter_highpass_filter(signal, samplerate, highpass_frequency)
    filtered_signal3 = butter_lowpass_filter(filtered_signal3, samplerate, lowpass_frequency)

    # Downsampling if necessary
    fdownsample = 400
    if samplerate > fdownsample:
        r = int(np.floor(samplerate / fdownsample))
        signal = decimate(filtered_signal3, r)
        samplerate = samplerate / r

    # Perform wavelet transform using the 'db4' wavelet
    wavelet = 'db4'  #db4  haar
    waveletTes = 'haar'

    coeffs = pywt.wavedec(signal, wavelet, level=6)
    cA6, cD6, cD5, cD4, cD3, cD2, cD1 = coeffs  # cD3-start P // cD2-end S // cD1-peak S // cD4-start P

    # Reconstruct the signal from wavelet coefficients
    reconstructed_signal = pywt.upcoef('d', cD1, wavelet, level=1)  # Wave S  cd1 1
    reconstructed_signalSend = pywt.upcoef('d', cD2, wavelet, level=2)  # End of wave S cd2 2
    reconstructed_signalTest = pywt.upcoef('d', cD1, waveletTes, level=1)  # test


    # Absolute value to emphasize the peaks
    reconstructed_signal = np.abs(reconstructed_signal)
    reconstructed_signalSend = np.abs(reconstructed_signalSend)
    reconstructed_signalTest = np.abs(reconstructed_signalTest) #TEST

    # Find S peaks using distance
    # distance = int(samplerate * 0.6)  # Assuming heart rate is not more than 100 bpm (i.e., 60/100 * sampling_rate)
    # print(distance) 300

    # PARA O PICO S
    amplitude_mean = np.mean(reconstructed_signal)
    amplitude_mean2 = amplitude_mean * 10000
    amplitude_std = np.std(reconstructed_signal)
    amplitude_std2 = amplitude_std * 10000

    distance = int(
        amplitude_mean2 + 5 * amplitude_std2)  # Adjust as needed for your data // int(amplitude_mean2 + 2 * amplitude_std2)
    # print(distance) 143

    peaksS, _ = find_peaks(reconstructed_signal, distance=distance, height=np.mean(reconstructed_signal))
    peaksR = peaksS - 8  # Para estimar o R

    # PARA O OFFSET S
    amplitude_meanSend = np.mean(reconstructed_signalSend)
    amplitude_mean2Send = amplitude_meanSend * 10000
    amplitude_stdSend = np.std(reconstructed_signalSend)
    amplitude_std2Send = amplitude_stdSend * 10000
    distanceSend = int(amplitude_mean2Send + 5 * amplitude_std2Send)
    peaksSend, _ = find_peaks(reconstructed_signalSend, distance=distanceSend, height=np.mean(reconstructed_signalSend))

    #PARA O TESTE HAAR ####################
    amplitude_meanTest = np.mean(reconstructed_signalTest)
    amplitude_mean2Test = amplitude_meanTest * 100000

    amplitude_stdTest = np.std(reconstructed_signalTest)
    amplitude_std2Test = amplitude_stdTest * 100000

    distance = int(amplitude_mean2Test + 5 * amplitude_std2Test)  # Adjust as needed for your data // int(amplitude_mean2 + 2 * amplitude_std2)

    peaksTes, _ = find_peaks(reconstructed_signalTest, distance=distance, height=np.mean(reconstructed_signalTest))



    plt.figure(figsize=(12, 6))

    plt.subplot(4, 1, 1)
    plt.plot(signal[1:2000], label='Filtered ECG Signal', color='orange')
    plt.plot(peaksTes[:7], signal[peaksTes[:7]], 'ro', label='R Peaks')
    #plt.plot(peaksTest[:7], signal[peaksTest[:7]], 'go', label='P Peaks')
    plt.legend()

    plt.subplot(4, 1, 2)
    plt.plot(reconstructed_signalTest[1:2000], label='Approximation Coefficients', color='green')
    plt.plot(peaksTes[:7], reconstructed_signalTest[peaksTes[:7]], 'ro', label='P Peaks')  # Plot first 5 peaks for demonstration
    plt.legend()

    plt.subplot(4, 1, 3)
    plt.plot(reconstructed_signal[1:2000], label='Approximation Coefficients', color='green')
    plt.plot(peaksS[:8], reconstructed_signal[peaksS[:8]], 'ro',
             label='S Peaks')  # Plot first 5 S peaks for demonstration
    plt.plot(peaksR[:8], reconstructed_signal[peaksR[:8]], 'go',
             label='R Peaks')  # Plot first 5 R peaks for demonstration
    plt.plot(peaksSend[:3], reconstructed_signal[peaksSend[:3]], 'bo',
             label='OffS Peaks')  # Plot first 5 peaks for demonstration
    plt.legend()

    plt.subplot(4, 1, 4)
    plt.plot(signal[1:2000], label='Approximation Coefficients', color='green')
    plt.plot(peaksS[:8], signal[peaksS[:8]], 'ro', label='S Peaks')  # Plot first 5 S peaks for demonstration
    plt.plot(peaksR[:8], signal[peaksR[:8]], 'go', label='R Peaks')  # Plot first 5 peaks for demonstration
    plt.plot(peaksSend[:3], signal[peaksSend[:3]], 'bo', label='OffS Peaks')  # Plot first 5 peaks for demonstration
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Placeholder for FPT and further processing
    value_S = {'S Peaks': peaksS}  # Todos os valores dos picos aqui
    value_R = {'R Peaks': peaksR}  # Todos os valores dos picos aqui

    print(value_R)

    FPT_S = len(value_S['S Peaks'])
    FPT_R = len(value_R['R Peaks'])

    if not mute:
        print('Done')

    return FPT_S, FPT_R


##########

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

    # Define the sequence of filters to be applied

    filter_types = ['low', 'high', 'notch', 'band']
    # Then, pass the baseline-corrected signal through the bandpass filter
    # final_filtered_signal = ecg_filter(filtered_signal, samplerate, 'band', lowpass_frequency, highpass_frequency, 'Butterworth')
    final_filtered_signal = ecg_filter(filtered_signal, samplerate, filter_types, lowpass_frequency, highpass_frequency,
                                       notch_frequency, 'Butterworth')
    final_filtered_signal2 = final_filtered_signal  # + 0.05

    # Plot the final filtered signal
    '''plt.plot(final_filtered_signal2[1:2000])
    plt.title('Filtered ECG Signal')
    plt.xlabel('Time in ms')
    plt.ylabel('Voltage in mV') 
    plt.show()'''

    # Apply isoline correction to final_filtered_signal2
    corrected_final_filtered_signal2, offset = isoline_correction(final_filtered_signal2)

    # Plot the isoline-corrected final_filtered_signal2
    '''plt.plot(corrected_final_filtered_signal2[1:2000])
    plt.title('Isoline-Corrected Filtered ECG Signal')
    plt.xlabel('Time in ms')
    plt.ylabel('Voltage in mV')
    plt.show()'''

    '''plt.figure(figsize=(12, 6))
    plt.plot(signal[1:2000], label='Original Signal', color='blue', alpha=0.5)
    plt.plot(corrected_final_filtered_signal2[1:2000], label='Filtered Signal', color='red', alpha=0.75)
    #plt.axhline(y=offset, color='green', linestyle='--', label='Offset')
    plt.legend()
    plt.title('Signal, Filtered Signal, and Offset')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.show()'''

    # Perform QRS detection
    FPT_S, FPT_R = QRS_Detection(signal, samplerate, peaksQRS=True, mute=True)

    print(f"Total number of S peaks: {FPT_S}")
    print(f"Total number of R peaks: {FPT_R}")

else:
    print("Nenhum arquivo foi selecionado.")


