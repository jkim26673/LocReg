import sys
import os
print("Setting system path")
sys.path.append(".")  # Replace this path with the actual path to the parent directory of Utilities_functions

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def find_min_between_peaks(data, ref_ind, sel_ind):
    """
    this function takes the vector of y_peaks, and finds the reference y peak and selected y peak and finds the difference
    """
    peaks, _ = find_peaks(data)
    # if len(peaks) == 1:
    #     raise AssertionError("Need to have more than 1 peak for resolution analysis.")
    #     ret
    peak_heights = data[peaks]
    sorted_peak_indices = np.argsort(peak_heights)[::-1]  # Indices of peaks sorted by height
    
    # Select top two highest peaks
    if len(peaks) > 2:
        # Ignore the first peak and select the next two highest peaks
        top_two_peak_indices = sorted_peak_indices[1:3]  # Ignore the first peak, take the next two
    else:
        # If there are exactly 2 peaks, select them both
        top_two_peak_indices = sorted_peak_indices[:2]
    
    # Get the actual indices of the top two peaks in the data
    top_two_peaks = peaks[top_two_peak_indices]
    # print("top_two_peaks", top_two_peaks)
    peaks = top_two_peaks
    min_between_peaks = np.abs(data[peaks[ref_ind]] - data[peaks[sel_ind]])
    return data[peaks[ref_ind]] , data[peaks[sel_ind]], min_between_peaks

def check_resolution(peak1_amplitude, peak2_amplitude, min_between_peaks):
    """
    Checks if two overlapping components are adequately resolved.
    
    Parameters:
    peak1_amplitude (float): Amplitude of the first peak.
    peak2_amplitude (float): Amplitude of the second peak.
    min_between_peaks (float): The minimum amplitude between the two peaks.

    Returns:
    bool: True if the components are resolved, False otherwise.
    """
    smaller_amplitude = min(peak1_amplitude, peak2_amplitude)
    # print("peak1_amplitude", peak1_amplitude)
    # print("peak2_amplitude", peak2_amplitude)

    # Check if the minimum between components is < 80% of the smaller component's amplitude
    if min_between_peaks < 0.7 * smaller_amplitude:
        return 1
    else:
        return 0



# # Sample data
# data = np.array([0, 1, 0, 2, 1, 0, 3, 2, 1, 0, 1, 2, 1, 0])

# # Find local peaks
# peaks, _ = find_peaks(data)

# # Print peaks
# print("Local peaks at indices:", peaks)
# print("Local peak values:", data[peaks])

# # Optional: Plotting
# plt.plot(data)
# plt.plot(peaks, data[peaks], "x")
# plt.title("Local Peaks")
# plt.xlabel("Index")
# plt.ylabel("Value")
# plt.savefig("test.png")  # Save the plot as a PNG file
# plt.show()