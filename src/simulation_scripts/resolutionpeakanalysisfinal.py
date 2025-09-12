import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

# def detect_peaks(T2_values, signal, threshold=0.01, resolution_criteria=0.9):
#     """
#     Detect peaks in T2 distributions and check their resolution.

#     Parameters:
#     - T2_values: 1D array, T2 relaxation times.
#     - signal: 1D array, signal corresponding to each T2 value.
#     - threshold: float, minimum relative amplitude for a peak to be considered.
#     - resolution_criteria: float, criteria for judging two peaks to be resolved.

#     Returns:
#     - resolved_peaks: list of tuples (T2_value, amplitude, width) for resolved peaks.
#     - spurious_peaks: list of tuples for spurious peaks that were filtered out.
#     - is_resolved: binary value indicating whether peaks are resolved (1 for resolved, 0 for not).
#     """

#     # Step 1: Find peaks in the signal
#     peaksidx, _ = find_peaks(signal)
#     resolved_peaks = []
#     spurious_peaks = []
#     resolution_count = 0

#     # for every peak (i) and peak index (peak), calculate the amplitude and then the corresponding T2 value.
#     for position, peakidx in enumerate(peaksidx):
#         amplitude = signal[peakidx]
#         T2_peak = T2_values[peakidx]
        
#         # Step 2: Check if peak is spurious (based on T2 value less than 40)
#     if T2_peak < 40:
#         spurious_peaks.append((T2_peak, amplitude))
#         continue

#         # Step 3: Calculate width (second moment around peak)
#         left_base = T2_values[peaksidx[max(0, position-1)]] if position > 0 else T2_peak
#         right_base = T2_values[peaksidx[min(position+1, len(peaksidx)-1)]] if position < len(peaksidx)-1 else T2_peak
#         width = right_base - left_base
        
#         # Step 4: Check for resolution
#         if position > 0:  # Ensure there is a previous peak to compare with
#             previous_peak = peaksidx[position-1]
#             min_between = np.min(signal[previous_peak:peakidx])  # Find minimum between peaks
#             smaller_amplitude = min(signal[previous_peak], amplitude)
            
#             # Check if the minimum between peaks is less than 90% of the smaller peak's amplitude
#             if min_between < resolution_criteria * smaller_amplitude:
#                 resolved_peaks.append((T2_peak, amplitude, width))
#                 resolution_count += 1

#     # Determine if any peaks are resolved
#     is_resolved = 1 if resolution_count > 0 else 0

#     return resolved_peaks, spurious_peaks, is_resolved

# def detect_peaks(T2_values, signal, threshold=0.01, resolution_criteria=0.9):
#     """
#     Detect peaks in T2 distributions and check their resolution.

#     Parameters:
#     - T2_values: 1D array, T2 relaxation times.
#     - signal: 1D array, signal corresponding to each T2 value.
#     - threshold: float, minimum relative amplitude for a peak to be considered.
#     - resolution_criteria: float, criteria for judging two peaks to be resolved.

#     Returns:
#     - resolved_peaks: list of tuples (T2_value, amplitude, width) for resolved peaks.
#     - spurious_peaks: list of tuples for spurious peaks that were filtered out.
#     - is_resolved: binary value indicating whether peaks are resolved (1 for resolved, 0 for not).
#     """

#     # Step 1: Find peaks in the signal
#     peaksidx, _ = find_peaks(signal)
#     resolved_peaks = []
#     spurious_peaks = []
#     resolution_count = 0
    
#     spuriousT2val = T2_values[peaksidx][np.where(T2_values[peaksidx] < 40)]
#     spurioussignal = signal[peaksidx][np.where(T2_values[peaksidx] < 40)]
#     for i, T2val in enumerate(spuriousT2val):
#         amplitude = spurioussignal[i]
#         spurious_peaks.append((T2val, amplitude))
#     # signal[peaksidx][np.where(T2_values[peaksidx] < 40)] = 0
#     if np.size(signal) < 150:
#         print(signal)
#     signal[np.where(T2_values < 40)] = 0
#     peaksidx, _ = find_peaks(signal)
#     # for every peak (i) and peak index (peak), calculate the amplitude and then the corresponding T2 value.
#     for position, peakidx in enumerate(peaksidx):
#         amplitude = signal[peakidx]
#         T2_peak = T2_values[peakidx]
        

#         # Step 3: Calculate width (second moment around peak)
#         left_base = T2_values[peaksidx[max(0, position-1)]] if position > 0 else T2_peak
#         right_base = T2_values[peaksidx[min(position+1, len(peaksidx)-1)]] if position < len(peaksidx)-1 else T2_peak
#         width = right_base - left_base
        
#         # Step 4: Check for resolution
#         if position > 0:  # Ensure there is a previous peak to compare with
#             previous_peak = peaksidx[position-1]
#             min_between = np.min(signal[previous_peak:peakidx])  # Find minimum between peaks
#             smaller_amplitude = min(signal[previous_peak], amplitude)
            
#             # Check if the minimum between peaks is less than 90% of the smaller peak's amplitude
#             if min_between < resolution_criteria * smaller_amplitude:
#                 resolved_peaks.append((T2_peak, amplitude))
#                 resolved_peaks.append((T2_values[previous_peak], signal[previous_peak]))
#                 resolution_count += 1
#             else:
#                 resolved_peaks.append(T2_values[previous_peak], signal[previous_peak])
#     # Determine if any peaks are resolved
#     is_resolved = 1 if resolution_count > 0 else 0

#     return resolved_peaks, spurious_peaks, is_resolved


def detect_peaks(T2_values, signal, threshold=0.01, resolution_criteria=0.9):
    """
    Detect peaks in T2 distributions and check their resolution.

    Parameters:
    - T2_values: 1D array, T2 relaxation times.
    - signal: 1D array, signal corresponding to each T2 value.
    - threshold: float, minimum relative amplitude for a peak to be considered.
    - resolution_criteria: float, criteria for judging two peaks to be resolved.

    Returns:
    - resolved_peaks: list of tuples (T2_value, amplitude, width) for resolved peaks.
    - spurious_peaks: list of tuples for spurious peaks that were filtered out.
    - is_resolved: binary value indicating whether peaks are resolved (1 for resolved, 0 for not).
    """

    # Step 1: Find peaks in the signal
    peaksidx, _ = find_peaks(signal)
    resolved_peaks = []
    spurious_peaks = []
    resolution_count = 0
    
    spuriousT2val = T2_values[peaksidx][np.where(T2_values[peaksidx] < 40)]
    spurioussignal = signal[peaksidx][np.where(T2_values[peaksidx] < 40)]
    for i, T2val in enumerate(spuriousT2val):
        amplitude = spurioussignal[i]
        spurious_peaks.append((T2val, amplitude))
    # signal[peaksidx][np.where(T2_values[peaksidx] < 40)] = 0
    if np.size(signal) < 150:
        print(signal)
    signal[np.where(T2_values < 40)] = 0
    peaksidx, _ = find_peaks(signal)
    # for every peak (i) and peak index (peak), calculate the amplitude and then the corresponding T2 value.
    for position, peakidx in enumerate(peaksidx):
        amplitude = signal[peakidx]
        T2_peak = T2_values[peakidx]

        # Step 3: Calculate width (second moment around peak)
        left_base = T2_values[peaksidx[max(0, position-1)]] if position > 0 else T2_peak
        right_base = T2_values[peaksidx[min(position+1, len(peaksidx)-1)]] if position < len(peaksidx)-1 else T2_peak
        width = right_base - left_base
        
        # Step 4: Check for resolution
        if position > 0:  # Ensure there is a previous peak to compare with
            previous_peak = peaksidx[position-1]
            min_between = np.min(signal[previous_peak:peakidx])  # Find minimum between peaks
            smaller_amplitude = min(signal[previous_peak], amplitude)
            
            # Check if the minimum between peaks is less than 90% of the smaller peak's amplitude
            if min_between < resolution_criteria * smaller_amplitude:
                resolved_peaks.append((T2_peak, amplitude, min_between))
                resolved_peaks.append((T2_values[previous_peak], signal[previous_peak], min_between))
                resolution_count += 1
            # else:
            #     resolved_peaks.append((T2_values[previous_peak], signal[previous_peak]))
    # Determine if any peaks are resolved
    is_resolved = 1 if resolution_count > 0 else 0

    return resolved_peaks, spurious_peaks, is_resolved

# Example use case with overlapping signals
T2_values = np.linspace(0, 10, 1000)

# Generate multiple signals for testing
signals = [
    np.exp(-((T2_values - 4)**2) / (2 * 0.4**2)) + 0.8 * np.exp(-((T2_values - 5.5)**2) / (2 * 0.4**2)),
    np.exp(-((T2_values - 3)**2) / (2 * 0.3**2)) + 0.6 * np.exp(-((T2_values - 5)**2) / (2 * 0.5**2)),
    np.exp(-((T2_values - 2)**2) / (2 * 0.5**2)) + 0.5 * np.exp(-((T2_values - 4)**2) / (2 * 0.6**2)),
]

# Initialize resolution stats
resolution_stats = []

# # Loop through each signal to detect peaks and record resolution
# for idx, signal in enumerate(signals):
#     resolved_peaks, spurious_peaks, is_resolved = detect_peaks(T2_values, signal)
    
#     print(f"Signal {idx+1}:")
#     print("Detected Peaks Indices:", resolved_peaks)
#     print("Spurious Peaks:", spurious_peaks)
#     print(f"Peaks are {'resolved' if is_resolved else 'not resolved'}.\n")
    
#     resolution_stats.append(is_resolved)

# # Calculate and print the percentage of resolved peaks
# percentage_resolved = (sum(resolution_stats) / len(resolution_stats)) * 100
# print(f"Percentage of Resolved Peaks: {percentage_resolved:.2f}%")

# Example use case with sample data

# Plot the overlapping peaks for visualization
# plt.plot(T2_values, signal)
# plt.title('Overlapping Peaks in T2 Distribution')
# plt.xlabel('T2 (ms)')
# plt.ylabel('Signal Amplitude')
# plt.title('Resolvable Peaks (70% rule)')
# plt.savefig("test.png")

# resolved_peaks, spurious_peaks = detect_peaks(T2_values, signal)

# print("Resolved Peaks:", resolved_peaks)
# print("Spurious Peaks:", spurious_peaks)

# Plot the signal to visualize the resolvable overlapping peaks



# signal = np.exp(-((x - 3)**2) / (2 * 0.2**2)) + 0.5 * np.exp(-((x - 6)**2) / (2 * 0.2**2))

# # Check resolution of the peaks and remove spurious components
# resolved_peaks_info, spurious_peaks_info = check_resolution(signal, x)

# # Plot the signal and the resolved peaks
# plt.plot(x, signal, label='MRR Signal')
# for peak in resolved_peaks_info['resolved_peaks']:
#     plt.plot(peak['T2_value'], peak['height'], "x", label=f"Resolved Peak at T2={peak['T2_value']:.2f}")
# plt.legend()
# plt.savefig("test.png")
