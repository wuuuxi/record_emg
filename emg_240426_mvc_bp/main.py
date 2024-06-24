import os
import time
import scipy
import numpy as np
import socket
import pandas as pd
import pytrigno
import matplotlib.pyplot as plt


def emg_rectification(x, Fs, idx, channel_num):
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
    mvc = [0.0883, 0.0617, 0.1057, 0.0478, 0.1561, 0.1552, 0.1293, 0.1499, 0.3021, 0.0988, 0.9063, 0.3818]
    ref = 0
    for i in range(channel_num):
        if idx == i:
            ref = mvc[i]
    if ref == 0:
        print('No MVC data in channel', idx)
        ref = 1
    # ref = max(EMGLE)
    normalized_EMG = EMGLE / ref

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
    end = -10
    musc_label = ['PM1', 'PM2', 'LD', 'PM3', 'TMaj', 'TMin', 'Inf', 'Sup', 'Del1', 'Cor', 'Del2', 'Del3']
    # plt.subplot(211)
    # plt.plot(data[0][:, 1])
    # plt.subplot(212)
    # plt.plot(data[0][:, 2])
    for j in range(len(data)):
        for i in range(channel_num):
            [m, y, t] = emg_rectification(data[j][:, i], fs, i, channel_num)
            raw[j].append(m)
            emg[j].append(y)
            time[j].append(t)
        # print('-' * 25, 'max', '-' * 25)
        # for i in range(channel_num):
        #     print(time[j][i][np.argmax(emg[j][i][start_max:end]) + start_max])
        # print('-' * 25, '1st', '-' * 25)
        # for i in range(channel_num):
        #     print(time[j][i][first_high_value(emg[j][i][start_max:], value) + start_max])

        plt.figure(figsize=(10, 7))
        plt.subplots_adjust(top=0.930, wspace=0.280, hspace=0.330, left=0.085, right=0.985)
        for i in range(channel_num):
            plt.subplot(int(channel_num/2), 2, i + 1)
            plt.plot(time[j][i][start:end], raw[j][i][start:end])
            plt.ylabel(musc_label[i], weight='bold')
            if i == 10:
                plt.xlabel('Time(s)', weight='bold')
        plt.xlabel('Time(s)', weight='bold')
        # plt.ylabel('Filtered EMG Voltage(muV)')

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
    num = 200
    mvc_table = np.zeros([channel_num, len(data)])
    for j in range(len(data)):
        print('-' * 25, 'average emg of max', num, ',', 'No.', j + 1, '-' * 25)
        for k in range(channel_num):
            mvc = np.partition(raw[j][k][:end], -num)[-num:].mean()
            mvc_table[k, j] = mvc
            print(musc_label[k], ':', mvc)
    pd.DataFrame(mvc_table).round(4).to_excel('output.xlsx', index=False)

    # 取前 num 个最大数的平均值
    # average of the max num numbers.
    num = 200
    print('-' * 25, 'average emg of max', num, '-' * 25)
    for j in range(channel_num):
        print(musc_label[j], ':', '\t', max(np.partition(raw[:, j, :end], -num)[i, -num:].mean() for i in range(raw.shape[0])))

    print('-' * 25, 'max_emg_raw', '-' * 25)
    for j in range(channel_num):
        print(musc_label[j], ':', '\t', np.max(raw[:, j, :]))
    print('-' * 25, 'max_emg', '-' * 25)
    for j in range(channel_num):
        print(musc_label[j], ':', '\t', np.max(emg[:, j, :]))
    plt.show()


def from_csv_to_xlsx(csv_name):
    csv_file = csv_name + '.csv'
    xlsx_file = csv_name + '.xlsx'
    data_frame = pd.read_csv(csv_file)
    data_frame.to_excel(xlsx_file, index=False)


def from_excel_to_emg(csv_name):
    states = pd.read_excel(csv_name + '.xlsx')
    emg = np.asarray([states['AI 0'], states['AI 1'], states['AI 2'],
                      states['AI 3'], states['AI 4'], states['AI 5'],
                      states['AI 12'], states['AI 13'], states['AI 14'],
                      states['AI 15'], states['AI 16'], states['AI 17']]).T
    return emg


if __name__ == "__main__":
    fs = 1000
    emg_channel_num = 12
    # label = ['20kg', '30kg', '40kg', '50kg']

    # emg_label = ['testmvc 2024_04_26 16_51_55',
    #              'testmvc 2024_04_26 16_53_40',  # 俯卧，背阔肌
    #              'testmvc 2024_04_26 16_59_39',  # 侧卧，冈上肌
    #              'testmvc 2024_04_26 17_08_52',  # 侧卧，冈下肌
    #              'testmvc 2024_04_26 17_23_11',
    #              'testmvc 2024_04_26 17_25_22',  # 胸大肌，坐姿夹胸
    #              'testmvc 2024_04_26 18_10_15',  # 三角肌后束，背向墙面
    #              'testmvc 2024_04_26 18_52_17',  # 三角肌前束，三角肌中束
    #              'testmvc 2024_04_26 19_03_52',  # 三角肌前束，三角肌中束
    #              'testmvc 2024_04_26 19_17_31',  # 背阔肌，坐姿划船
    #              'testmvc 2024_04_26 19_31_34',
    #              'testmvc 2024_04_26 19_41_53',  # 三角肌后束，肩部外展
    #              'testmvc 2024_04_26 19_47_39',  # 背阔肌, 引体向上
    #              ]

    emg_label = ['test 2024_04_26 16_04_48',
                 'test 2024_04_26 16_08_33',
                 'test 2024_04_26 16_12_06',
                 'test 2024_04_26 16_14_22']

    # state1 = from_excel_to_emg('test 2024_04_26 16_04_48')
    # state2 = from_excel_to_emg('test 2024_04_26 16_08_33')
    # state3 = from_excel_to_emg('test 2024_04_26 16_12_06')
    # state4 = from_excel_to_emg('test 2024_04_26 16_14_22')

    emg = []
    for label in emg_label:
        mvc = from_excel_to_emg(label)
        emg.append(mvc)

    # mvc1 = from_excel_to_emg('testmvc 2024_04_26 16_51_55')
    # mvc2 = from_excel_to_emg('testmvc 2024_04_26 16_53_40')  # 俯卧，背阔肌
    # mvc3 = from_excel_to_emg('testmvc 2024_04_26 16_59_39')  # 侧卧，冈上肌
    # mvc4 = from_excel_to_emg('testmvc 2024_04_26 17_08_52')  # 侧卧，冈下肌
    # mvc5 = from_excel_to_emg('testmvc 2024_04_26 17_23_11')
    # mvc6 = from_excel_to_emg('testmvc 2024_04_26 17_25_22')  # 胸大肌，坐姿夹胸
    # mvc7 = from_excel_to_emg('testmvc 2024_04_26 18_10_15')  # 三角肌后束，背向墙面
    # mvc8 = from_excel_to_emg('testmvc 2024_04_26 18_52_17')  # 三角肌前束，三角肌中束
    # mvc9 = from_excel_to_emg('testmvc 2024_04_26 19_03_52')  # 三角肌前束，三角肌中束
    # mvc10 = from_excel_to_emg('testmvc 2024_04_26 19_17_31')  # 背阔肌，坐姿划船
    # mvc11 = from_excel_to_emg('testmvc 2024_04_26 19_31_34')
    # mvc12 = from_excel_to_emg('testmvc 2024_04_26 19_41_53')  # 三角肌后束，肩部外展
    # mvc13 = from_excel_to_emg('testmvc 2024_04_26 19_47_39')  # 背阔肌，引体向上
    # print('finish reading')
    #
    # # emg = [state1, state2, state3, state4]
    # emg = [mvc1, mvc2, mvc3, mvc4, mvc5, mvc6, mvc7, mvc8, mvc9, mvc10, mvc11, mvc12, mvc13]
    # # emg = [mvc5, mvc6]

    export(emg, fs, emg_channel_num, value=0.1)
