import os
import time
import scipy
import numpy as np
import socket
import pandas as pd
import pytrigno
import matplotlib.pyplot as plt


def record(host, n, samples_per_read, t, root_path, exp_name):
    """
    Communication with and data acquisition from a Delsys Trigno wireless EMG system.
    Delsys Trigno Control Utility needs to be installed and running on, and the device
    needs to be plugged in. Records can be run with a device connected to a remote machine.

    Args:
        host: host of a remote machine
        n: number of emg channels
        samples_per_read： number of samples per read
        t: number of batches (running time * 2000 / samples_per_read)

    Returns: None
    """
    dev = pytrigno.TrignoEMG(channel_range=(0, 0), samples_per_read=samples_per_read,
                             host=host)
    dev.set_channel_range((0, n - 1))
    dev.start()
    data_sensor = []
    try:
        data_w_time = np.zeros((n + 1, samples_per_read))
        for i in range(int(t)):
            # while True:
            data = dev.read() * 1e6
            system_time = time.time()
            data_w_time[0, :] = system_time
            data_w_time[1:, :] = data
            print(data_w_time)
            assert data_w_time.shape == (n + 1, samples_per_read)
            temp = data_w_time.copy()
            data_sensor.append(temp)  # change to 'data' if system time is not needed
        print(n, '-channel achieved')
        dev.stop()

        data_sensor = np.reshape(np.transpose(np.array(data_sensor), (0, 2, 1)),
                                 (-1, n + 1))  # change to 'n' if system time is not needed
        np.save(os.path.join(root_path, exp_name), data_sensor)
        # np.save(exp_name, data_sensor)
    except Exception:
        data_sensor = np.reshape(np.transpose(np.array(data_sensor), (0, 2, 1)),
                                 (-1, n + 1))  # change to 'n' if system time is not needed
        np.save(os.path.join(root_path, exp_name), data_sensor)
        raise IOError("Error! But saved.")


def emg_rectification(x, Fs):
    # Fs 采样频率，在EMG信号中是1000Hz
    # wp 通带截止频率    ws 阻带截止频率
    x_mean = np.mean(x)
    raw = x - x_mean * np.ones_like(x)
    t = np.arange(0, raw.size / Fs, 1 / Fs)
    EMGFWR = abs(raw)

    # 线性包络 Linear Envelope
    NUMPASSES = 3
    LOWPASSRATE = 6  # 低通滤波4—10Hz得到包络线

    Wn = LOWPASSRATE / (Fs / 2)
    [b, a] = scipy.signal.butter(NUMPASSES, Wn, 'low')
    EMGLE = scipy.signal.filtfilt(b, a, EMGFWR, padtype='odd', padlen=3 * (max(len(b), len(a)) - 1))  # 低通滤波

    plt.figure()
    plt.subplot(2, 1, 1)
    # plt.plot(t, raw)
    plt.plot(t, EMGLE)
    plt.xlabel('Time(s)')
    plt.ylabel('Filtered EMG Voltage(\muV)')
    # plt.show()

    # normalized_EMG = normalize(EMGLE, 'range');
    ref = max(EMGLE)
    normalized_EMG = EMGLE / ref

    plt.subplot(2, 1, 2)
    plt.plot(t, normalized_EMG)
    plt.xlabel('Time(s)')
    plt.ylabel('Normalized EMG')
    # plt.ylim(0, 1)
    # plt.title(code)

    # print(EMGfig, '-dpng', [code '.png'])
    y = normalized_EMG
    # y = EMGLE
    return [y, t]


def export(emg, fs, channel_num):
    for i in range(channel_num):
        [emg_BRD, t] = emg_rectification(emg[:, i + 1], fs)
    plt.show()


if __name__ == "__main__":
    fs = 2000
    emg_channel_num = 6
    emg_samples_per_read = 2000
    exp_name = time.time()
    exp_name = str(exp_name)
    host = '192.168.10.30'
    # host = '192.168.1.117'
    root_dir = 'emg'
    total_time = 60
    # emg = np.load('emg/1701252631.8887734.npy')
    # export(emg, fs, emg_channel_num)
    record(host, emg_channel_num, emg_samples_per_read, total_time, root_dir, exp_name)
