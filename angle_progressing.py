import os
import time
import scipy
import numpy as np
import socket
import pandas as pd
import pytrigno
import matplotlib.pyplot as plt
import sys


def monotonically_increasing(arr, arrt):
    diff = arr[1:] - arr[:-1]
    diff = np.where(diff > 0, diff, np.zeros_like(diff))
    begin_flag = False
    begin_index = 0
    for i in range(diff.shape[0]):
        if begin_flag is False and diff[i] > 0:
            begin_flag = True
            begin_index = i
        if begin_flag is True and diff[i] <= 0:
            begin_flag = False
            if i - begin_index > 100:
                print(begin_index + 1, '-', i + 1, '\t', arrt[begin_index], 's', '-', arrt[i], 's')


def monotonically_decreasing(arr, arrt):
    diff = arr[1:] - arr[:-1]
    diff = np.where(diff < 0, diff, np.zeros_like(diff) * (-1))
    begin_flag = False
    begin_index = 0
    for i in range(diff.shape[0]):
        if begin_flag is False and diff[i] < 0:
            begin_flag = True
            begin_index = i
        if begin_flag is True and diff[i] >= 0:
            begin_flag = False
            if i - begin_index > 100:
                print(begin_index + 1, '-', i + 1, '\t', arrt[begin_index], 's', '-', arrt[i], 's')


def monotonically_increasing_with_threshold(arr, arrt, threshold1, threshold2):
    diff = arr[1:] - arr[:-1]
    # diff = np.where(diff > 0, diff, np.zeros_like(diff))
    begin_flag = False
    begin_index = 0
    for i in range(diff.shape[0]):
        if begin_flag is False and diff[i] > 0 and threshold1 < arr[i] < threshold2:
            begin_flag = True
            begin_index = i
        if begin_flag is True and diff[i] <= 0:
            begin_flag = False
            if i - begin_index > 10:
                # print(begin_index + 1, '-', i + 1, '\t',
                #       arrt[begin_index], 's', '-', arrt[i], 's', "({:.2f}s)".format(arrt[i] - arrt[begin_index]), '\t',
                #       arr[begin_index], '°', '-', arr[i], '°')
                print(begin_index + 1, '-', i + 1, '\t',
                      arrt[begin_index], ',', arrt[i], "({:.2f}s)".format(arrt[i] - arrt[begin_index]), '\t',
                      arr[begin_index], '°', '-', arr[i], '°')
        if begin_flag is True and arr[i] > threshold2:
            begin_flag = False
            if i - begin_index > 10:
                # print(begin_index + 1, '-', i + 1, '\t',
                #       arrt[begin_index], 's', '-', arrt[i], 's', "({:.2f}s)".format(arrt[i] - arrt[begin_index]), '\t',
                #       arr[begin_index], '°', '-', arr[i], '°')
                print(begin_index + 1, '-', i + 1, '\t',
                      arrt[begin_index], ',', arrt[i], "({:.2f}s)".format(arrt[i] - arrt[begin_index]), '\t',
                      arr[begin_index], '°', '-', arr[i], '°')


def monotonically_decreasing_with_threshold(arr, arrt, threshold1, threshold2):
    diff = arr[1:] - arr[:-1]
    diff = np.where(diff < 0, diff, np.zeros_like(diff) * (-1))
    begin_flag = False
    begin_index = 0
    for i in range(diff.shape[0]):
        if begin_flag is False and diff[i] < 0 and threshold1 < arr[i] < threshold2:
            begin_flag = True
            begin_index = i
        if begin_flag is True and diff[i] >= 0:
            begin_flag = False
            if i - begin_index > 10:
                # print(begin_index + 1, '-', i + 1, '\t',
                #       arrt[begin_index], 's', '-', arrt[i], 's', "({:.2f}s)".format(arrt[i] - arrt[begin_index]), '\t',
                #       arr[begin_index], '°', '-', arr[i], '°')
                print(begin_index + 1, '-', i + 1, '\t',
                      arrt[begin_index], ',', arrt[i], "({:.2f}s)".format(arrt[i] - arrt[begin_index]), '\t',
                      arr[begin_index], '°', '-', arr[i], '°')
        if begin_flag is True and threshold1 > arr[i]:
            begin_flag = False
            if i - begin_index > 10:
                # print(begin_index + 1, '-', i + 1, '\t',
                #       arrt[begin_index], 's', '-', arrt[i], 's', "({:.2f}s)".format(arrt[i] - arrt[begin_index]), '\t',
                #       arr[begin_index], '°', '-', arr[i], '°')
                print(begin_index + 1, '-', i + 1, '\t',
                      arrt[begin_index], ',', arrt[i], "({:.2f}s)".format(arrt[i] - arrt[begin_index]), '\t',
                      arr[begin_index], '°', '-', arr[i], '°')


def find_nearest_idx(arr, value):
    arr = np.asarray(arr)
    array = abs(np.asarray(arr) - value)
    idx = array.argmin()
    return idx


def first_high_value(start_time, time, arr_copy, bar):
    timeidx = find_nearest_idx(time, start_time)
    arr = arr_copy[timeidx:].copy()
    for i in range(arr.shape[0]):
        if arr[i] > bar:
            return i + timeidx
    return -1


def max_high_value(time, arr_copy):
    arr = arr_copy.copy()
    maxidx = np.argmax(arr)
    print('max:', time[maxidx])


def all_value(elbow, time, value, time1, time2):
    timeidx1 = find_nearest_idx(time, time1)
    timeidx2 = find_nearest_idx(time, time2)
    elbow = elbow[timeidx1:timeidx2]
    time = time[timeidx1:timeidx2]

    idx = find_nearest_idx(elbow, value)
    print(time[idx], 's', '\t', elbow[idx], '°')
    #
    # arrt[begin_index], 's', '-', arrt[i], 's', "({:.2f}s)".format(arrt[i] - arrt[begin_index]), '\t',
    # arr[begin_index], '°', '-', arr[i], '°')
    #
    # arr = arr_copy[timeidx1:].copy()
    # for i in range(arr.shape[0]):
    #     if arr[i] > bar:
    #         return i + timeidx
    # return -1


# def find_intersection(elbow, time, value):
#     # 创建插值函数
#     f_interp = scipy.interpolate.interp1d(time, elbow, kind='cubic')
#
#     # 定义一个求解器函数，计算两条曲线的差值
#     def equations(x):
#         return [f_interp(x) - g(x)]
#
#     # 构造一个迭代范围
#     x = np.linspace(0, int(time[-1]), int(time[-1])*2000)
#
#     # 计算曲线上的其他点
#     y = f_interp(x)
#
#     # 使用NumPy的argwhere函数找到函数值接近0的索引
#     indices = np.argwhere(np.isclose(y, value, atol=1e-6)).flatten()
#
#     # 提取交点的x坐标
#     intersection_points = x[indices]
#     print(x[indices])
#     print(y[indices])
#
#     return intersection_points


def find_intersection(elbow, time, value, threshold):
    print(value)
    # 计算每个点与目标数的差值
    differences = np.abs(elbow - value)

    # 找到接近目标数的点的索引
    close_indices = np.where(differences < threshold)[0]

    # 根据索引提取接近目标数的点
    elbow_points = elbow[close_indices]
    time_points = time[close_indices]

    # 输出接近目标数的点
    for i in range(len(close_indices)):
        print('Elbow points: ', elbow[close_indices[i]], 'Time points: ', time[close_indices[i]])
        # print(f"Time points: {time[i]}")
    return elbow_points, time_points


def from_mot_to_xlsx(time_label, csv_name):
    csv_file = time_label + csv_name + '.mot'
    xlsx_file = time_label + csv_name + '.xlsx'
    data_frame = pd.read_csv(csv_file)
    data_frame.to_excel(xlsx_file, index=False)


def read_force_mot(file_path):
    data_frame = pd.read_csv(file_path).values
    columns = data_frame[9][0].split('\t')
    df = pd.DataFrame(columns=columns)  # First line including the label of every column.

    for line in data_frame[10:]:
        data = line[0].split('\t')
        row = pd.Series(data, index=df.columns)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    return df


def find_target_angle(file_path, time_label='time', angle_label='arm_flex_r', angle_range=None, first_time_stop=50,
                      first_high_value_threshold=120):
    if angle_range is None:
        angle_range = [0, 125]
    angle = read_force_mot(file_path)
    # angle = pd.read_excel(file_path)

    # first value
    if first_time_stop > 0:
        time = np.asarray(angle[time_label]).astype(float)
        elbow = np.asarray(angle[angle_label]).astype(float)
        # arm_flex = np.asarray(angle['arm_flex_r'])
        idx = find_nearest_idx(time, first_time_stop)
        time = time[:idx]
        elbow = elbow[:idx]
        # arm_flex = arm_flex[:idx]
    else:
        time = np.asarray(angle[time_label]).astype(float)
        elbow = np.asarray(angle[angle_label]).astype(float)
        # arm_flex = np.asarray(angle['arm_flex_r'])
    start_time = 5
    idx = first_high_value(start_time, time, elbow, first_high_value_threshold)
    max_high_value(time, elbow)
    plt.figure(figsize=(10, 4.5))
    plt.plot(time, elbow, label='elbow')
    # plt.plot(time, arm_flex, label='arm_flex')
    plt.legend()
    # plt.xlim([0, 60])
    if idx >= 0:
        print('1st:', time[idx])

    start_angle = angle_range[0]
    end_angle = angle_range[1]
    monotonically_increasing_with_threshold(elbow, time, start_angle, end_angle)
    monotonically_decreasing_with_threshold(elbow, time, start_angle, end_angle)


if __name__ == "__main__":
    # folder_path = '../../HKSI digital twins/Muscle modeling/files/bench press/yuetian/0408/MVNX/'
    # file_name = 'MVN-006_Xsens_jointangle_q_Raja.xlsx'
    # find_target_angle(folder_path + file_name)
    folder_path = ('D:/OneDrive - The Chinese University of Hong Kong/Experiment/20240517_HKSI_deadlift/'
                   'Resorted/deadlift/MVN-007 45kg/')
    file_name = 'MVN-007_grf.mot'
    find_target_angle(folder_path + file_name, angle_label='barbell_left_py', angle_range=[0.6, 1.2])
    # barbell_left_py

    plt.show()

    # t_cz_1 = - 5.1 + 3.3
    # t_cz_2 = - 3.56 + 4.816
    # t_cz_3 = - 3.217 + 5.633
    # t_cz_4 = - 2.23 + 4.617
    # t_cz_5 = - 3.273 + 5.317

    # timestep_emg_1 = [[13.082, 15.082, 15.082, 15.649],
    #                   [16.182, 17.882, 17.882, 18.399],
    #                   [18.982, 20.615, 20.615, 21.215],
    #                   [21.732, 23.582, 23.582, 24.098],
    #                   [24.832, 26.298, 26.298, 27.115]]

    timestep_emg_1 = [[7.649, 8.949, 8.949, 10.532],
                      [11.149, 12.333, 12.333, 14.099],
                      [14.632, 15.665, 15.666, 17.215],
                      [17.799, 18.898, 18.898, 20.566],
                      [20.933, 21.933, 21.933, 23.366],
                      [23.933, 25.049, 25.049, 26.132]]

    timestep_emg_2 = [[8.783, 9.5, 9.5, 10.65],
                      [11.4, 11.983, 11.983, 13.15],
                      [13.933, 14.833, 14.833, 15.999],
                      [16.75, 17.75, 17.75, 18.633],
                      [19.416, 20.366, 20.366, 21.149],
                      [21.883, 22.533, 22.533, 23.566]]

    timestep_emg_3 = [[11.616, 13.866, 13.866, 14.849],
                      [15.599, 17.283, 17.283, 18.133],
                      [18.633, 20.266, 20.266, 21.182],
                      [21.632, 23.282, 23.282, 24.216],
                      [24.399, 26.116, 26.116, 26.999],
                      [27.199, 28.749, 28.749, 29.515],
                      [29.882, 31.249, 31.249, 32.049],
                      [32.049, 33.499, 33.499, 34.798]]

    timestep_emg_4 = [[7.266, 8.166, 8.166, 9.316],
                      [10.616, 11.466, 11.466, 12.732],
                      [13.782, 14.466, 14.466, 15.682],
                      [16.615, 17.399, 17.399, 18.349],
                      [19.282, 20.039, 20.039, 21.049]]

    timestep_emg_5 = [[6.566, 7.283, 7.283, 8.333],
                      [9.133, 9.789, 9.789, 10.766],
                      [11.483, 12.099, 12.099, 13.149],
                      [13.916, 14.536, 14.536, 15.565],
                      [16.199, 16.966, 16.966, 17.816],
                      [18.482, 19.032, 19.032, 20.065]]
