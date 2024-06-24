import os
import time
import scipy
import numpy as np
import pytrigno
import matplotlib.pyplot as plt
import pandas as pd


def emg_rectification(x, Fs, idx):
    # Fs 采样频率，在EMG信号中是1000Hz
    # wp 通带截止频率    ws 阻带截止频率
    x_mean = np.mean(x)
    raw = x - x_mean * np.ones_like(x)
    t = np.arange(0, raw.size / Fs, 1 / Fs)
    EMGFWR = abs(raw)

    # 线性包络 Linear Envelope
    NUMPASSES = 3
    LOWPASSRATE = 5  # 低通滤波4—10Hz得到包络线

    Wn = LOWPASSRATE / (Fs / 2)
    [b, a] = scipy.signal.butter(NUMPASSES, Wn, 'low')
    EMGLE = scipy.signal.filtfilt(b, a, EMGFWR, padtype='odd', padlen=3 * (max(len(b), len(a)) - 1))  # 低通滤波

    # plt.figure()
    # plt.subplot(2, 1, 1)
    # # plt.plot(t, raw)
    # plt.plot(t, EMGLE)
    # plt.xlabel('Time(s)')
    # plt.ylabel('Filtered EMG Voltage(\muV)')
    # plt.show()

    ref = 407

    # if idx == 0:
    #     ref = 407.11
    # elif idx == 1:
    #     ref = 467.59
    # elif idx == 2:
    #     ref = 176.04
    # elif idx == 3:
    #     ref = 176.04
    # elif idx == 4:
    #     ref = 87.11
    # elif idx == 5:
    #     ref = 339.21
    # else:
    #     ref = max(EMGLE)

    # if idx == 0:
    #     ref = 417.90
    # elif idx == 1:
    #     ref = 126.73
    # elif idx == 2:
    #     ref = 408.73
    # elif idx == 3:
    #     ref = 250.28
    # else:
    #     ref = max(EMGLE)

    # if idx == 0:
    #     ref = 407.11
    # elif idx == 1:
    #     ref = 250.37
    # elif idx == 2:
    #     ref = 468.31
    # elif idx == 3:
    #     ref = 467.59
    # else:
    #     ref = max(EMGLE)

    # if idx == 0:
    #     ref = 703.47
    # elif idx == 1:
    #     ref = 250.37
    # elif idx == 2:
    #     ref = 468.31
    # elif idx == 3:
    #     ref = 467.59
    # else:
    #     ref = max(EMGLE)
    # ref = 1
    # ref = max(EMGLE)
    normalized_EMG = EMGLE / ref

    # plt.subplot(2, 1, 2)
    # plt.plot(t, normalized_EMG)
    # plt.xlabel('Time(s)')
    # plt.ylabel('Normalized EMG')
    # plt.ylim(0, 1)
    # plt.title(code)

    # print(EMGfig, '-dpng', [code '.png'])
    y = normalized_EMG
    # y = EMGLE
    return [EMGLE, y, t]


def first_high_value(arr_copy, value=0.5):
    arr = arr_copy.copy()
    for i in range(arr.shape[0]):
        if arr[i] > value:
            return i
    return -1


def export(data, fs, channel_num, value=0.5):
    raw = [([]) for _ in range(len(data))]
    emg = [([]) for _ in range(len(data))]
    time = [([]) for _ in range(len(data))]
    start_max = fs * 0
    start = fs * 0
    # end = fs * 205
    end = -1
    # plt.subplot(211)
    # plt.plot(data[0][:, 1])
    # plt.subplot(212)
    # plt.plot(data[0][:, 2])
    for j in range(len(data)):
        for i in range(channel_num):
            [m, y, t] = emg_rectification(data[j][:, i + 1], fs, i)
            raw[j].append(m)
            emg[j].append(y)
            time[j].append(t)
        print('-' * 25, 'max', '-' * 25)
        for i in range(channel_num):
            print(time[j][i][np.argmax(emg[j][i][start_max:end]) + start_max])
        print('-' * 25, '1st', '-' * 25)
        for i in range(channel_num):
            print(time[j][i][first_high_value(emg[j][i][start_max:], value) + start_max])

        # plt.figure(figsize=(6, 8))
        # plt.subplot(311)
        # plt.plot(time[j][0][start:end], emg[j][0][start:end])
        # plt.subplot(312)
        # plt.plot(time[j][1][start:end], emg[j][1][start:end])
        # plt.subplot(313)
        # plt.plot(time[j][2][start:end], emg[j][2][start:end])

        # if j == 0 or j == 1 or j == 2:
        #     plt.figure(figsize=(6, 2.5))
        #     plt.subplots_adjust(bottom=0.180)
        #     plt.plot(time[j][0][start:end], emg[j][0][start:end], label='Anterior')
        #     plt.plot(time[j][2][start:end], emg[j][2][start:end], label='Posterior')
        #     plt.plot(time[j][1][start:end], emg[j][1][start:end], label='Medius')
        #     plt.legend()
        #     plt.xlabel('Time(s)')
        # if j == 0 or j == 1 or j == 2:
        #     plt.figure(figsize=(6, 2.5))
        #     plt.subplots_adjust(bottom=0.180)
        #     plt.plot(time[j][3][start:end], emg[j][3][start:end], label='Sternal')
        #     plt.plot(time[j][4][start:end], emg[j][4][start:end], label='Clavicular')
        #     plt.plot(time[j][5][start:end], emg[j][5][start:end], label='Costal')
        #     plt.legend()
        #     plt.xlabel('Time(s)')
        # if j == 5 or j == 4:
        #     plt.figure(figsize=(6, 2.5))
        #     plt.subplots_adjust(bottom=0.180)
        #     plt.plot(time[j][4][start:end], emg[j][4][start:end], label='Lateral')
        #     plt.plot(time[j][5][start:end], emg[j][5][start:end], label='long')
        #     plt.legend()
        #     plt.xlabel('Time(s)')

        musc_label = ['Brachiorad', 'Brachialis', 'Biceps_long', 'Biceps_short', 'Triceps', 'Deltoid']
        plt.figure(figsize=(6, 8))
        plt.subplots_adjust(top=0.930, hspace=0.330)
        for i in range(channel_num):
            plt.subplot(channel_num, 1, i + 1)
            plt.plot(time[j][i][start:end], emg[j][i][start:end])
            plt.ylabel(musc_label[i], weight='bold')
        #     if j == 5 and i == 0:
        #         plt.ylim(0, 0.015)
        #     if j == 5 and i == 2:
        #         plt.ylim(0, 0.25)

        # plt.figure(figsize=(6, 6))
        # if subject == 'zhuo':
        #     plt.subplot(211)
        #     plt.plot(time[j][0][start:end], emg[j][0][start:end])
        #     plt.subplot(212)
        #     plt.plot(time[j][1][start:end], emg[j][1][start:end])
        # elif subject == 'chenzui':
        #     plt.subplot(211)
        #     plt.plot(time[j][2][start:end], emg[j][2][start:end])
        #     plt.subplot(212)
        #     plt.plot(time[j][3][start:end], emg[j][3][start:end])

        # plt.figure(figsize=(6, 8))
        # plt.subplot(411)
        # plt.plot(time[j][0][start:end], emg[j][0][start:end])
        # # plt.ylim(0, 500)
        # plt.subplot(412)
        # plt.plot(time[j][1][start:end], emg[j][1][start:end])
        # # plt.ylim(0, 200)
        # plt.subplot(413)
        # plt.plot(time[j][2][start:end], emg[j][2][start:end])
        # plt.subplot(414)
        # plt.plot(time[j][3][start:end], emg[j][3][start:end])
        # plt.subplot(411)
        # plt.plot(time[j][0][start:end], emg[j][0][start:end])
        # # plt.ylim(0, 500)
        # plt.subplot(412)
        # plt.plot(time[j][2][start:end], emg[j][2][start:end])
        # # plt.ylim(0, 200)
        # plt.subplot(413)
        # plt.plot(time[j][3][start:end], emg[j][3][start:end])
        # plt.subplot(414)
        # plt.plot(time[j][1][start:end], emg[j][1][start:end])
        plt.xlabel('Time(s)')
        # plt.ylabel('Filtered EMG Voltage(muV)')

    print('-' * 25, 'average emg of max 200', '-' * 25)
    num = 100
    print(max(np.partition(raw[i][0][:end], -num)[-num:].mean() for i in range(len(data))))
    print(max(np.partition(raw[i][1][:end], -num)[-num:].mean() for i in range(len(data))))
    print(max(np.partition(raw[i][2][:end], -num)[-num:].mean() for i in range(len(data))))
    print(max(np.partition(raw[i][3][:end], -num)[-num:].mean() for i in range(len(data))))
    print(max(np.partition(raw[i][4][:end], -num)[-num:].mean() for i in range(len(data))))
    print(max(np.partition(raw[i][5][:end], -num)[-num:].mean() for i in range(len(data))))

    num = min(raw[j][0].shape[0] for j in range(len(data)))
    rawd = []
    emgd = []
    for j in range(len(data)):
        rawd.append(np.asarray(raw[j])[:, :num])
        emgd.append(np.asarray(emg[j])[:, :num])
    raw = np.asarray(rawd)[:, :, start:end]
    emg = np.asarray(emgd)[:, :, start:end]
    m = np.ones([len(data), channel_num])
    for j in range(len(data)):
        for i in range(channel_num):
            m[j, i] = np.max(raw[j, i, :])
    # plt.subplot(411)
    # plt.plot(m[:, 0])
    # plt.subplot(412)
    # plt.plot(m[:, 1])
    # plt.subplot(413)
    # plt.plot(m[:, 2])
    # plt.subplot(414)
    # plt.plot(m[:, 3])

    # 取前 num 个最大数的平均值
    print('-'*25, 'average emg of max 200', '-' * 25)
    num = 100
    print(max(np.partition(raw[:, 0, :end], -num)[i, -num:].mean() for i in range(raw.shape[0])))
    print(max(np.partition(raw[:, 1, :end], -num)[i, -num:].mean() for i in range(raw.shape[0])))
    print(max(np.partition(raw[:, 2, :end], -num)[i, -num:].mean() for i in range(raw.shape[0])))
    print(max(np.partition(raw[:, 3, :end], -num)[i, -num:].mean() for i in range(raw.shape[0])))
    print(max(np.partition(raw[:, 4, :end], -num)[i, -num:].mean() for i in range(raw.shape[0])))
    print(max(np.partition(raw[:, 5, :end], -num)[i, -num:].mean() for i in range(raw.shape[0])))
    # print(np.partition(raw[:, 0, :end], -num)[:, -num:].mean())
    # print(np.partition(raw[:, 1, :end], -num)[:, -num:].mean())
    # print(np.partition(raw[:, 2, :end], -num)[:, -num:].mean())
    # print(np.partition(raw[:, 3, :end], -num)[:, -num:].mean())
    # print(np.partition(raw[:, 4, :end], -num)[:, -num:].mean())
    # print(np.partition(raw[:, 5, :end], -num)[:, -num:].mean())

    print('-'*25, 'max_emg_raw', '-' * 25)
    print(np.max(raw[:, 0, :]))
    print(np.max(raw[:, 1, :]))
    print(np.max(raw[:, 2, :]))
    print(np.max(raw[:, 3, :]))
    print(np.max(raw[:, 4, :]))
    print(np.max(raw[:, 5, :]))
    print('-'*25, 'max_emg', '-'*25)
    print(np.max(emg[:, 0, :]))
    print(np.max(emg[:, 1, :]))
    print(np.max(emg[:, 2, :]))
    print(np.max(emg[:, 3, :]))
    print(np.max(emg[:, 4, :]))
    print(np.max(emg[:, 5, :]))
    plt.show()


if __name__ == "__main__":
    # subject = 'chenzui'
    # subject = 'zhuo'
    fs = 1000
    emg_channel_num = 6
    emg = [np.asarray(pd.read_excel('../../HKSI digital twins/Muscle modeling/files/bench press/yuetian/0408/MVC EMG/yt 2024_04_08 13_42_49.xlsx')),
           np.asarray(pd.read_excel('../../HKSI digital twins/Muscle modeling/files/bench press/yuetian/0408/MVC EMG/yt 2024_04_08 13_58_46.xlsx')),
           np.asarray(pd.read_excel('../../HKSI digital twins/Muscle modeling/files/bench press/yuetian/0408/MVC EMG/yt 2024_04_08 14_10_32.xlsx')),
           np.asarray(pd.read_excel('../../HKSI digital twins/Muscle modeling/files/bench press/yuetian/0408/Test EMG/bptest 2024_04_08 19_25_06.xlsx')),
           np.asarray(pd.read_excel('../../HKSI digital twins/Muscle modeling/files/bench press/yuetian/0408/Test EMG/bptest 2024_04_08 19_29_21.xlsx')),
           np.asarray(pd.read_excel('../../HKSI digital twins/Muscle modeling/files/bench press/yuetian/0408/Test EMG/bptest 2024_04_08 19_33_53.xlsx')),
           np.asarray(pd.read_excel('../../HKSI digital twins/Muscle modeling/files/bench press/yuetian/0408/Test EMG/bptest 2024_04_08 19_35_47.xlsx')),
           np.asarray(pd.read_excel('../../HKSI digital twins/Muscle modeling/files/bench press/yuetian/0408/Test EMG/bptest 2024_04_08 19_38_03.xlsx')),
           # np.asarray(pd.read_excel('../../HKSI digital twins/Muscle modeling/files/bench press/yuetian/0408/Test EMG/ytRobot 2024_04_08 21_12_06.xlsx')),
           np.asarray(pd.read_excel('../../HKSI digital twins/Muscle modeling/files/bench press/yuetian/0408/Test EMG/ytRobot 2024_04_08 21_13_45.xlsx')),
           np.asarray(pd.read_excel('../../HKSI digital twins/Muscle modeling/files/bench press/yuetian/0408/Test EMG/ytRobot 2024_04_08 21_15_37.xlsx')),
           np.asarray(pd.read_excel('../../HKSI digital twins/Muscle modeling/files/bench press/yuetian/0408/Test EMG/ytRobot 2024_04_08 21_18_16.xlsx')),
           np.asarray(pd.read_excel('../../HKSI digital twins/Muscle modeling/files/bench press/yuetian/0408/Test EMG/ytRobot 2024_04_08 21_19_53.xlsx')),
           np.asarray(pd.read_excel('../../HKSI digital twins/Muscle modeling/files/bench press/yuetian/0408/Test EMG/ytRobot 2024_04_08 21_22_39.xlsx')),
           ]
    # emg1 = [np.load('emg/5.npy')]
    # emg = [np.load('emg_240313/9.5.npy')]

    # emg = [np.load('emg/2.npy'),
    #        # np.load('emg/3.npy'),
    #        np.load('emg/3.npy'),
    #        # np.load('emg/5.npy'),
    #        # np.load('emg/6.npy'),
    #        np.load('emg/4.npy')]
    # emg = [np.load('emg/chenzui-right-shoulder-2.npy')]
#     # emg = [np.load('emg/chenzui-right-shoulder.npy'),
    #        np.load('emg/chenzui-right-shoulder-2.npy')]
    # emg[0] = emg[0][:2000*10, :]
    # emg = [
    #     # np.load('emg/1701251776.753853.npy'),
    #     # np.load('emg/1701251933.1794827.npy'),
    #     # np.load('emg/1701252030.3090453.npy'),
    #     # np.load('emg/1701252135.141656.npy'),
    #     # np.load('emg/1701252242.8745039.npy'),
    #     # np.load('emg/1701252339.6082103.npy'),
    #     # np.load('emg/1701252423.2515116.npy'),
    #     np.load('emg/1701252530.5062706.npy'),
    #     np.load('emg/1701252631.8887734.npy'),
    #     np.load('emg/1701252721.6080053.npy'),
    #     np.load('emg/1701252885.7887957.npy'),  # highlight
    #     # np.load('emg/1701252959.9981768.npy'),
    #     np.load('emg/1701253050.469565.npy'),  # highlight
    #     np.load('emg/1701253127.6713908.npy'),
    #     np.load('emg/1701253221.6221592.npy'),
    #     np.load('emg/1701253323.446435.npy'),
    #     np.load('emg/1701253404.2738903.npy'),
    #     np.load('emg/1701253479.5198448.npy'),
    #     # np.load('emg/1701253556.7260988.npy'),
    #     # np.load('emg/1701253652.689899.npy'),
    # ]
    export(emg, fs, emg_channel_num, value=0.1)
