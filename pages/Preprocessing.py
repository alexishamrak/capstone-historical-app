import scipy.signal
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from agcounts.extract import get_counts
from scipy.signal import butter,filtfilt
np.random.seed(1)

left_hand = pd.read_csv('left_hand_lm.csv')
right_hand = pd.read_csv('right_hand_hm.csv')
# left_leg = pd.read_csv('left_leg_lm.csv')
# right_leg = pd.read_csv('right_leg_hm.csv')

# values for collecting activity counts
freq = 50
epoch = 10

# sorting left hand data into more specific arrays
lh_time = left_hand["time"]
lh_time = np.array(lh_time)
lh_X = left_hand["x-acceleration"]
lh_X = np.array(lh_X)
lh_Y = left_hand["y-acceleration"]
lh_Y = np.array(lh_Y)
lh_Z = left_hand["z-acceleration"]
lh_Z = np.array(lh_Z)
lh_raw = left_hand[["x-acceleration", "y-acceleration", "z-acceleration"]]
lh_raw = np.array(lh_raw)

# collecting count activity from left hand
lh_counts = get_counts(lh_raw, freq=freq, epoch=epoch)
lh_counts = pd.DataFrame(lh_counts, columns=["Axis1", "Axis2", "Axis3"])
lh_count_mag = np.sqrt(lh_counts["Axis1"]**2+lh_counts["Axis2"]**2+lh_counts["Axis3"]**2)
print(lh_counts)

# plotting counts as histogram
# plt.hist(lh_counts, bins=10)
# plt.show()

# plotting raw XYZ data from left hand
fig, axs = plt.subplots(3)
fig.suptitle('Raw XYZ Data from Left Hand')
axs[0].plot(lh_time, lh_X)
axs[0].set_title('Raw X Data', fontsize=8)
axs[1].plot(lh_time, lh_Y)
axs[1].set_title('Raw Y Data', fontsize=8)
axs[2].plot(lh_time, lh_Z)
axs[2].set_title('Raw Z Data', fontsize=8)
plt.show()

# plotting filtered XYZ data from left hand
lh_X_hat = scipy.signal.savgol_filter(lh_X, 51, 3)
plt.plot(lh_time, lh_X)
plt.plot(lh_time,lh_X_hat, color='red')
plt.title("Filtered Left Hand X Data")
plt.legend(['Raw', 'Filtered'])
plt.show()
lh_Y_hat = scipy.signal.savgol_filter(lh_Y, 51, 3)
plt.plot(lh_time, lh_Y)
plt.plot(lh_time,lh_Y_hat, color='red')
plt.title("Filtered Left Hand Y Data")
plt.legend(['Raw', 'Filtered'])
plt.show()
lh_Z_hat = scipy.signal.savgol_filter(lh_Z, 51, 3)
plt.plot(lh_time, lh_Z)
plt.plot(lh_time,lh_Z_hat, color='red')
plt.title("Filtered Left Hand Z Data")
plt.legend(['Raw', 'Filtered'])
plt.show()

# calculating magnitude of left hand use
lh_mag = np.sqrt(lh_X**2+lh_Y**2+lh_Z**2)
# print(lh_mag)
plt.plot(lh_mag)
plt.title("Left Hand Magnitude")
plt.show()

# sorting right hand data into more specific arrays
rh_time = right_hand["time"]
rh_time = np.array(rh_time)
rh_X = right_hand["x-acceleration"]
rh_X = np.array(rh_X)
rh_Y = right_hand["y-acceleration"]
rh_Y = np.array(rh_Y)
rh_Z = right_hand["z-acceleration"]
rh_Z = np.array(rh_Z)
rh_raw = left_hand[["x-acceleration", "y-acceleration", "z-acceleration"]]
rh_raw = np.array(rh_raw)

# collecting count activity from right hand
rh_counts = get_counts(rh_raw, freq=freq, epoch=epoch)
rh_counts = pd.DataFrame(lh_counts, columns=["Axis1", "Axis2", "Axis3"])
rh_count_mag = np.sqrt(rh_counts["Axis1"]**2+rh_counts["Axis2"]**2+rh_counts["Axis3"]**2)
print(rh_counts)

# plotting raw XYZ data from right hand
fig, axs = plt.subplots(3)
fig.suptitle('Raw XYZ Data from Right Hand')
axs[0].plot(rh_time, rh_X)
axs[0].set_title('Raw X Data', fontsize=8)
axs[1].plot(rh_time, rh_Y)
axs[1].set_title('Raw Y Data', fontsize=8)
axs[2].plot(rh_time, rh_Z)
axs[2].set_title('Raw Z Data', fontsize=8)
plt.show()

# plotting filtered XYZ data from right hand
rh_X_hat = scipy.signal.savgol_filter(rh_X, 51, 3)
plt.plot(rh_time, rh_X)
plt.plot(rh_time, rh_X_hat, color='red')
plt.title("Filtered Right Hand X Data")
plt.legend(['Raw', 'Filtered'])
plt.show()
rh_Y_hat = scipy.signal.savgol_filter(rh_Y, 51, 3)
plt.plot(rh_time, rh_Y)
plt.plot(rh_time, rh_Y_hat, color='red')
plt.title("Filtered Right Hand Y Data")
plt.legend(['Raw', 'Filtered'])
plt.show()
rh_Z_hat = scipy.signal.savgol_filter(rh_Z, 51, 3)
plt.plot(rh_time, rh_Z)
plt.plot(rh_time, rh_Z_hat, color='red')
plt.title("Filtered Right Hand Z Data")
plt.legend(['Raw', 'Filtered'])
plt.show()

# calculating magnitude of right hand use
rh_mag = np.sqrt(rh_X**2+rh_Y**2+rh_Z**2)
# print(rh_mag)
plt.plot(rh_mag)
plt.title("Right Hand Magnitude")
plt.show()

# calculating magnitude use ratio between left and right hand
mag_ratio = np.sum(lh_mag)/np.sum(rh_mag)
hand_magnitude_ratio = np.log(mag_ratio)
print(f"Magnitude ratio between hands is: {hand_magnitude_ratio}")

# calculating paretic/non-paretic use ratio
# I have no idea if I'm doing this right..........
use_ratio = np.sum(lh_count_mag)/np.sum(rh_count_mag)
hand_use_ratio = np.log(use_ratio)
print(f"Use ratio between hands is: {hand_use_ratio}")

# future implementation of code for data collected from right and left legs

# lh_time = left_hand["time"]
# ll_X = left_leg["x-acceleration"]
# ll_Y = left_leg["y-acceleration"]
# ll_Z = left_leg["z-acceleration"]
# ll_raw = left_hand[["x-acceleration", "y-acceleration", "z-acceleration"]]
# ll_raw = np.array(ll_raw)
# ll_counts = get_counts(ll_raw, freq=freq, epoch=epoch)
# ll_counts = pd.DataFrame(ll_counts, columns=["Axis1", "Axis2", "Axis3"])

# ll_time = left_hand["time"]
# rl_X = right_leg["x-acceleration"]
# rl_Y = right_leg["y-acceleration"]
# rl_Z = right_leg["z-acceleration"]"
# rl_raw = left_hand[["x-acceleration", "y-acceleration", "z-acceleration"]]
# rl_raw = np.array(rl_raw)
# rl_counts = get_counts(rl_raw, freq=freq, epoch=epoch)
# rl_counts = pd.DataFrame(rl_counts, columns=["Axis1", "Axis2", "Axis3"])

