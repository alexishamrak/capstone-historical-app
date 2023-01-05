import scipy.signal
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from agcounts.extract import get_counts

# lh: left hand
# rh: right hand
# ll: left leg
# rl: right leg

def main():

    left_hand = pd.read_csv('left_hand_lm.csv')
    right_hand = pd.read_csv('right_hand_hm.csv')
    # left_leg = pd.read_csv('left_leg_lm.csv')
    # right_leg = pd.read_csv('right_leg_hm.csv')

    # values for collecting activity counts
    freq = 50
    epoch = 10

    # function for sorting data into separate arrays for time, X, Y, and Z data
    def sorting_data(dataset):
        time = dataset["time"]
        time = np.array(time)
        X = dataset["x-acceleration"]
        X = np.array(X)
        Y = dataset["y-acceleration"]
        Y = np.array(Y)
        Z = dataset["z-acceleration"]
        Z = np.array(Z)
        raw = dataset[["x-acceleration", "y-acceleration", "z-acceleration"]]
        raw = np.array(raw)
        return time, X, Y, Z, raw

    # function for calculating the number of activity counts
    def collecting_counts(raw_data):
        # frequency is the sampling rate (50 Hz), epochs was arbitrarily set to 10
        # get_counts() is calculating the activity count from the accelerometer data
        raw_counts = get_counts(raw_data, freq=freq, epoch=epoch)
        raw_counts = pd.DataFrame(raw_counts, columns=["Axis1", "Axis2", "Axis3"])
        raw_count_mag = np.sqrt(raw_counts["Axis1"] ** 2 + raw_counts["Axis2"] ** 2 + raw_counts["Axis3"] ** 2)
        return raw_counts, raw_count_mag

    # function for filtering raw data
    def filter_data(X_data, Y_data, Z_data):
        X_hat = scipy.signal.savgol_filter(X_data, 51, 3)
        Y_hat = scipy.signal.savgol_filter(Y_data, 51, 3)
        Z_hat = scipy.signal.savgol_filter(Z_data, 51, 3)
        return X_hat, Y_hat, Z_hat

    # function for calculating magnitude and converting the count magnitude array to binary
    def converting_mag2binary(x_filt, y_filt, z_filt, count_mag):
        mag = np.sqrt(x_filt ** 2 + y_filt ** 2 + z_filt ** 2)
        mag_len = len(count_mag)
        mag_binary = []

        for i in range(mag_len):
            if count_mag[i] >= 2:
                mag_binary.append(1)
            else:
                mag_binary.append(0)

        return mag_binary, mag

    # function for calculating the magnitude ratio
    def magnitude_ratio(paretic_mag, non_paretic_mag):
        mag_ratio = np.sum(paretic_mag) / np.sum(non_paretic_mag)
        return mag_ratio

    # function for converting the count magnitude array to binary
    def converting_countmag2binary(count_mag):
        mag_len = len(count_mag)
        mag_count_binary = []

        for i in range(mag_len):
            if count_mag[i] >= 2:
                mag_count_binary.append(1)
            else:
                mag_count_binary.append(0)

        mag_count_final = (np.sum(mag_count_binary)) * epoch

        return mag_count_final

    # function for calculating the paretic/non-paretic limb use ratio
    def use_ratio(paretic_count_mag, non_paretic_count_mag):
        use_ratio = paretic_count_mag / non_paretic_count_mag
        return use_ratio

    lh_time, lh_X, lh_Y, lh_Z, lh_raw = sorting_data(left_hand)
    rh_time, rh_X, rh_Y, rh_Z, rh_raw = sorting_data(right_hand)
    # ll_time, ll_X, ll_Y, ll_Z, ll_raw = sorting_data(left_leg)
    # rl_time, rl_X, rl_Y, rl_Z, rl_raw = sorting_data(right_leg)

    lh_counts, lh_count_mag = collecting_counts(lh_raw)
    rh_counts, rh_count_mag = collecting_counts(rh_raw)
    # ll_counts, ll_count_mag = collecting_counts(ll_raw)
    # rl_counts, rl_count_mag = collecting_counts(rl_raw)

    lh_X_hat, lh_Y_hat, lh_Z_hat = filter_data(lh_X, lh_Y, lh_Z)
    rh_X_hat, rh_Y_hat, rh_Z_hat = filter_data(rh_X, rh_Y, rh_Z)
    # ll_X_hat, ll_Y_hat, ll_Z_hat = filter_data(ll_X, ll_Y, ll_Z)
    # rl_X_hat, rl_Y_hat, rl_Z_hat = filter_data(rl_X, rl_Y, rl_Z)

    lh_mag_bin, lh_mag = converting_mag2binary(lh_X_hat, lh_Y_hat, lh_Z_hat, lh_count_mag)
    rh_mag_bin, rh_mag = converting_mag2binary(rh_X_hat, rh_Y_hat, rh_Z_hat, rh_count_mag)
    # ll_mag_bin, ll_mag = converting_mag2binary(ll_X_hat, ll_Y_hat, ll_Z_hat, ll_count_mag)
    # rl_mag_bin, rl_mag = converting_mag2binary(rl_X_hat, rl_Y_hat, rl_Z_hat, rl_count_mag)

    hand_mag_ratio = magnitude_ratio(lh_mag_bin, rh_mag_bin)
    print(f"Magnitude ratio between hands is: {hand_mag_ratio}")
    # leg_mag_ratio = magnitude_ratio(ll_mag_bin, rl_mag_bin)
    # print(f"Magnitude ratio between legs is: {leg_mag_ratio}")

    lh_count = converting_countmag2binary(lh_count_mag)
    rh_count = converting_countmag2binary(rh_count_mag)
    # ll_count = converting_countmag2binary(ll_count_mag)
    # rl_count = converting_countmag2binary(rl_count_mag)

    hand_use_ratio = use_ratio(lh_count, rh_count)
    print(f"Use ratio between hands is: {hand_use_ratio}")
    # leg_use_ratio = use_ratio(ll_count, rl_count)
    # print(f"Use ratio between legs is: {leg_use_ratio}")

if __name__ == "__main__":
    main()
