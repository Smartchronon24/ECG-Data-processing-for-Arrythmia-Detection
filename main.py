import wfdb
import os
import numpy as np
import pandas as pd
from scipy import signal as sg

root='mit-bih-arrhythmia-database-1.0.0/'
results = {}

for filename in os.listdir(root):
    if filename.endswith('.dat'):
        record_name = filename[:-4]

        record = wfdb.rdrecord(os.path.join(root, record_name), sampfrom=0, sampto=4000)
        annotation = wfdb.rdann(os.path.join(root, record_name), 'atr', sampfrom=0, sampto=4000,
                                shift_samps=True)

        class preprocess():
            def bp_filter(self, signal):
                result = None
                sig = signal.copy()

                sampling_rate = 360
                cutoff_freq = 0.5
                window_size = int(sampling_rate / cutoff_freq)

                baseline = np.convolve(sig, np.ones(window_size) / window_size, mode='same')

                sig = sig - baseline

                for index in range(len(signal)):
                    sig[index] = signal[index]

                    if (index >= 1):
                        sig[index] += 2 * sig[index - 1]

                    if (index >= 2):
                        sig[index] -= sig[index - 2]

                    if (index >= 6):
                        sig[index] -= 2 * signal[index - 6]

                    if (index >= 12):
                        sig[index] += signal[index - 12]

                result = sig.copy()

                for index in range(len(signal)):
                    result[index] = -1 * sig[index]

                    if (index >= 1):
                        result[index] -= result[index - 1]

                    if (index >= 16):
                        result[index] += 32 * sig[index - 16]

                    if (index >= 32):
                        result[index] += sig[index - 32]

                max_val = max(max(result), -min(result))
                result = result / max_val
                return result

            def derivative(self, signal):
                result = signal.copy()

                for index in range(len(signal)):
                    result[index] = 0

                    if (index >= 1):
                        result[index] -= 2 * signal[index - 1]

                    if (index >= 2):
                        result[index] -= signal[index - 2]

                    if (index >= 2 and index <= len(signal) - 2):
                        result[index] += 2 * signal[index + 1]

                    if (index >= 2 and index <= len(signal) - 3):
                        result[index] += signal[index + 2]

                    result[index] = (result[index] * annotation.fs) / 8
                return result

            def squaring(self, signal):
                result = signal.copy()

                for index in range(len(signal)):
                    result[index] = signal[index] ** 2
                return result

            def moving_window(self, signal):

                result = signal.copy()
                win_size = round(0.150 * annotation.fs)
                sum = 0

                for j in range(win_size):
                    sum += signal[j] / win_size
                    result[j] = sum

                for index in range(win_size, len(signal)):
                    sum += signal[index] / win_size
                    sum -= signal[index - win_size] / win_size
                    result[index] = sum
                return result

            def solve(self, signal):
                input_signal = signal.iloc[:, 1].to_numpy()

                global bpass
                bpass = self.bp_filter(input_signal.copy())

                global der
                der = self.derivative(bpass.copy())

                global sqr
                sqr = self.squaring(der.copy())

                global mwin
                mwin = self.moving_window(sqr.copy())

                return mwin

        QRS = preprocess()
        ecg = pd.DataFrame(np.array([list(range(len(record.adc()))), record.adc()[:, 0]]).T,
                           columns=['TimeStamp', 'ecg'])
        output = QRS.solve(ecg)

        class heart_rate():

            def __init__(self, signal, samp_freq):

                self.RR1, self.RR2, self.probable_peaks, self.r_locs, self.peaks, self.result = ([] for i in range(6))
                self.SPKI, self.NPKI, self.Threshold_I1, self.Threshold_I2, self.SPKF, self.NPKF, self.Threshold_F1, self.Threshold_F2 = (
                    0 for i in range(8))

                self.T_wave = False
                self.m_win = mwin
                self.b_pass = bpass
                self.samp_freq = samp_freq
                self.signal = signal
                self.win_150ms = round(0.15 * self.samp_freq)

                self.RR_Low_Limit = 0
                self.RR_High_Limit = 0
                self.RR_Missed_Limit = 0
                self.RR_Average1 = 0

            def approx_peak(self):

                slopes = sg.fftconvolve(self.m_win, np.full((25,), 1) / 25, mode='same')

                for i in range(round(0.5 * self.samp_freq) + 1, len(slopes) - 1):
                    if (slopes[i] > slopes[i - 1]) and (slopes[i + 1] < slopes[i]):
                        self.peaks.append(i)

            def adjust_rr_interval(self, ind):

                self.RR1 = np.diff(self.peaks[max(0, ind - 8): ind + 1]) / self.samp_freq

                self.RR_Average1 = np.mean(self.RR1)
                RR_Average2 = self.RR_Average1

                if (ind >= 8):
                    for i in range(0, 8):
                        if (self.RR_Low_Limit < self.RR1[i] < self.RR_High_Limit):
                            self.RR2.append(self.RR1[i])

                            if (len(self.RR2) > 8):
                                self.RR2.remove(self.RR2[0])
                                RR_Average2 = np.mean(self.RR2)

                if (len(self.RR2) > 7 or ind < 8):
                    self.RR_Low_Limit = 0.92 * RR_Average2
                    self.RR_High_Limit = 1.16 * RR_Average2
                    self.RR_Missed_Limit = 1.66 * RR_Average2

            def searchback(self, peak_val, RRn, sb_win):

                if (RRn > self.RR_Missed_Limit):
                    win_rr = self.m_win[peak_val - sb_win + 1: peak_val + 1]

                    coord = np.asarray(win_rr > self.Threshold_I1).nonzero()[0]

                    if (len(coord) > 0):
                        for pos in coord:
                            if (win_rr[pos] == max(win_rr[coord])):
                                x_max = pos
                                break
                    else:
                        x_max = None

                    if (x_max is not None):
                        self.SPKI = 0.25 * self.m_win[x_max] + 0.75 * self.SPKI
                        self.Threshold_I1 = self.NPKI + 0.25 * (self.SPKI - self.NPKI)
                        self.Threshold_I2 = 0.5 * self.Threshold_I1

                        win_rr = self.b_pass[x_max - self.win_150ms: min(len(self.b_pass) - 1, x_max)]

                        coord = np.asarray(win_rr > self.Threshold_F1).nonzero()[0]

                        if (len(coord) > 0):
                            for pos in coord:
                                if (win_rr[pos] == max(win_rr[coord])):
                                    r_max = pos
                                    break
                        else:
                            r_max = None

                        if (r_max is not None):
                            if self.b_pass[r_max] > self.Threshold_F2:
                                self.SPKF = 0.25 * self.b_pass[r_max] + 0.75 * self.SPKF
                                self.Threshold_F1 = self.NPKF + 0.25 * (self.SPKF - self.NPKF)
                                self.Threshold_F2 = 0.5 * self.Threshold_F1

                                self.r_locs.append(r_max)

            def find_t_wave(self, peak_val, RRn, ind, prev_ind):

                if (self.m_win[peak_val] >= self.Threshold_I1):
                    if (ind > 0 and 0.20 < RRn < 0.36):
                        curr_slope = max(np.diff(self.m_win[peak_val - round(self.win_150ms / 2): peak_val + 1]))
                        last_slope = max(
                            np.diff(
                                self.m_win[self.peaks[prev_ind] - round(self.win_150ms / 2): self.peaks[prev_ind] + 1]))

                        if (curr_slope < 0.5 * last_slope):
                            self.T_wave = True
                            self.NPKI = 0.125 * self.m_win[peak_val] + 0.875 * self.NPKI

                    if (not self.T_wave):
                        if (self.probable_peaks[ind] > self.Threshold_F1):
                            self.SPKI = 0.125 * self.m_win[peak_val] + 0.875 * self.SPKI
                            self.SPKF = 0.125 * self.b_pass[ind] + 0.875 * self.SPKF

                            self.r_locs.append(self.probable_peaks[ind])

                        else:
                            self.SPKI = 0.125 * self.m_win[peak_val] + 0.875 * self.SPKI
                            self.NPKF = 0.125 * self.b_pass[ind] + 0.875 * self.NPKF

                elif (self.m_win[peak_val] < self.Threshold_I1) or (
                        self.Threshold_I1 < self.m_win[peak_val] < self.Threshold_I2):
                    self.NPKI = 0.125 * self.m_win[peak_val] + 0.875 * self.NPKI
                    self.NPKF = 0.125 * self.b_pass[ind] + 0.875 * self.NPKF

            def adjust_thresholds(self, peak_val, ind):

                if (self.m_win[peak_val] >= self.Threshold_I1):

                    self.SPKI = 0.125 * self.m_win[peak_val] + 0.875 * self.SPKI

                    if (self.probable_peaks[ind] > self.Threshold_F1):
                        self.SPKF = 0.125 * self.b_pass[ind] + 0.875 * self.SPKF

                        self.r_locs.append(self.probable_peaks[ind])

                    else:
                        self.NPKF = 0.125 * self.b_pass[ind] + 0.875 * self.NPKF

                elif (self.m_win[peak_val] < self.Threshold_I2) or (
                        self.Threshold_I2 < self.m_win[peak_val] < self.Threshold_I1):
                    self.NPKI = 0.125 * self.m_win[peak_val] + 0.875 * self.NPKI
                    self.NPKF = 0.125 * self.b_pass[ind] + 0.875 * self.NPKF

            def update_thresholds(self):

                self.Threshold_I1 = self.NPKI + 0.25 * (self.SPKI - self.NPKI)
                self.Threshold_F1 = self.NPKF + 0.25 * (self.SPKF - self.NPKF)
                self.Threshold_I2 = 0.5 * self.Threshold_I1
                self.Threshold_F2 = 0.5 * self.Threshold_F1
                self.T_wave = False

            def ecg_searchback(self):

                self.r_locs = np.unique(np.array(self.r_locs).astype(int))

                win_200ms = round(0.2 * self.samp_freq)

                for r_val in self.r_locs:
                    coord = np.arange(r_val - win_200ms, min(len(self.signal), r_val + win_200ms + 1), 1)

                    if (len(coord) > 0):
                        for pos in coord:
                            if (self.signal[pos] == max(self.signal[coord])):
                                x_max = pos
                                break
                    else:
                        x_max = None

                    if (x_max is not None):
                        self.result.append(x_max)

            def find_r_peaks(self):

                self.approx_peak()

                for ind in range(len(self.peaks)):

                    peak_val = self.peaks[ind]
                    win_300ms = np.arange(max(0, self.peaks[ind] - self.win_150ms),
                                          min(self.peaks[ind] + self.win_150ms, len(self.b_pass) - 1), 1)
                    max_val = max(self.b_pass[win_300ms], default=0)

                    if (max_val != 0):
                        x_coord = np.asarray(self.b_pass == max_val).nonzero()
                        self.probable_peaks.append(x_coord[0][0])

                    if (ind < len(self.probable_peaks) and ind != 0):

                        self.adjust_rr_interval(ind)

                        if (self.RR_Average1 < self.RR_Low_Limit or self.RR_Average1 > self.RR_Missed_Limit):
                            self.Threshold_I1 /= 2
                            self.Threshold_F1 /= 2

                        RRn = self.RR1[-1]

                        self.searchback(peak_val, RRn, round(RRn * self.samp_freq))

                        self.find_t_wave(peak_val, RRn, ind, ind - 1)

                    else:
                        self.adjust_thresholds(peak_val, ind)

                    self.update_thresholds()
                self.ecg_searchback()
                return self.result

        signal = ecg.iloc[:, 1].to_numpy()

        hr = heart_rate(signal, annotation.fs)
        result = hr.find_r_peaks()
        result = np.array(result)

        result = result[result > 0]

        heartRate = (60 * annotation.fs) / np.average(np.diff(result[1:]))

        results[record_name] = {
            'Heart Rate (BPM)': heartRate,
            'R-Peaks': result
        }

results_df = pd.DataFrame.from_dict(results, orient='index')

results_df.to_csv('ecg_results.csv')