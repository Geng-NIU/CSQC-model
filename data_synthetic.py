import scipy.io as scio
import numpy as np
from scipy import interpolate, stats
import copy
import math
import pickle
from time import time
"""this module for generating benchmark data."""
t0 = time()


# field error (RMSE & REAA) calculate function
m = 3200


def REAA(inter):
    """
    Function to calculate the REAA error between interpolation field and the ground ture field.
    The mathmatical function and detail parameter explain can be found in the paper.
    inter represent the interplotion field.
    data3 is the original grund ture field.
    m is grid amount for the whole field.
    """
    def mean(inter):
        return sum(sum(i) for i in inter) / m
    return abs(mean(inter) - mean(data3)) / mean(data3)


def RMSE(inter1):
    """
    Function to calculate the REAA error between interpolation field and the ground ture field.
    """
    return (sum(sum(i) for i in (inter1 - data3) ** 2) / m) ** 0.5


'''
Loading the converted readable radar data
 (the radar data can free download from the NOAA website https://www.ncdc.noaa.gov/data-access/radar-data/nexrad).
'''
# Initial parameters setting as well as the various scenarios variables setting for the main program.
n = 100
n1 = 35
RMSE_new1, RMSE_old1, RMSE_removed1, RMSE_remove1 = [], [], [], []
REAA_new1, REAA_old1, REAA_removed1, REAA_remove1 = [], [], [], []
observation_data_stats = []
point_coord0_stats = []
inter_new_stats = []
point_values = []
data_X, data_y = [], []
fixed_X = np.random.randint(0, 10, size=int(n * .3))
fixed_Y = np.random.randint(0, 80, size=int(n * .3))
fixed_coord = np.vstack((fixed_X, fixed_Y)).T

for c in range(data1.shape[1]):  # 5
    data2 = data1[0, c].T
    observation_data_stats_temp = []
    point_coord0_stats_temp = []
    inter_new_stats_temp = []
    for t in range(1, data2.shape[0]):  # 11 time steps
        print("t", t)
        data3 = data2[t, :, :].T
        data3[data3 < 0] = 0
        coord_move_each_step = coord[c, t - 1, :, :]  # coord generated from dynamic model
        combine_coord = np.vstack((coord_move_each_step, fixed_coord))
        Y = combine_coord[:, 1]  # agent location at each time step
        X = combine_coord[:, 0]
        point_coord = [79 - Y, X]
        point_values0 = data3[79 - Y, X]  # ground true values of each time step
        point_coord0 = list(zip(79 - Y, X))  # coordinates index for later interpolation
        point_coord0 = np.array(point_coord0)
        # mark the point value which is 0 in ground true
        zero_No = []  # point value is 0
        for i in range(n):
            if data3[79 - Y[i], X[i]] == 0:
                zero_No.append(i)
        # basic regular error (normal distribution)
        mu0 = np.random.uniform(-0.15, 0.15, size=len(point_values0))  # -0.15 -- 0.15
        sigma0 = np.random.uniform(0.1, 0.2, size=len(point_values0))  #
        noise = np.random.normal(mu0 * point_values0, sigma0 * point_values0,
                                 size=len(point_values0))  # basic error
        observation_data = point_values0 + noise

        # basic regular error (wald distribution)
        # a0 = np.random.uniform(-0.15, 0.15, size=len(point_values0))  # mean
        # # b0 = np.random.uniform(0.1, 0.2, size=len(point_values0))  #
        # b0 = a0 * 2
        # noise = np.random.wald(a0 * (point_values0+0.0000001), b0 * (point_values0+0.0000001),
        #                        size=len(point_values0))
        # observation_data = point_values0 + noise

        choice_No = np.random.choice(n, size=n1, replace=False)
        # noise (normal distribution)
        mu1 = np.random.uniform(0.2, 0.6 + 0.2, size=len(list(choice_No)))  # 0.5-2
        sigma1 = np.random.uniform(0.6 + 0.5, 0.6 + 1, size=len(list(choice_No)))  # 0.5-3 # 0.13-0.3
        noise1 = np.random.normal(mu1 * observation_data[choice_No], sigma1 * observation_data[choice_No],
                                  size=len(list(choice_No)))  # noise

        # noise (wald distribution)
        # a1 = np.random.uniform(0.2, 0.6+0.2, size=len(choice_No))  # 0.6
        # # b1 = np.random.uniform(0.6+0.5, 0.6+1, size=len(choice_No))  # 0.6
        # b1 = a1 * 2
        # noise1 = np.random.wald(a1 * (observation_data[choice_No]), b1 * (observation_data[choice_No]),
        #                         size=len(choice_No))

        observation_data[choice_No] = observation_data[choice_No] + noise1  # chose points add noise1 values
        observation_data[zero_No] = 0
        observation_data[observation_data < 0] = 0
        for i in range(observation_data.shape[0]):
            if observation_data[i] < 0:
                observation_data[i] = 0

        # time step noise threshold define
        R = observation_data - point_values0
        noise_set = []
        non_noise_set = []
        for i in choice_No:
            if (abs(R[i]) > 0.5 * point_values0[i]) & (abs(R[i]) > 0.1):
                noise_set.append(i)
            else:
                non_noise_set.append(i)

        noise_values = []  # noise point values
        for i in noise_set:
            noise_values_temp = observation_data[i]
            noise_values.append(noise_values_temp)

        noise_coord = []  # noise point coordinates
        for i in noise_set:
            noise_coord_temp = point_coord0[i]
            noise_coord.append(noise_coord_temp)

        # delete noise observations which beyond threshold.
        after_removed_data = np.delete(observation_data, noise_set, axis=0)
        after_removed_coord = np.delete(point_coord0, noise_set, axis=0)

        observation_data_stats_temp.append(observation_data)
        point_coord0_stats_temp.append(point_coord0)
        # spatial interpolation
        noise_values = np.array(noise_values)
        newfunc = interpolate.NearestNDInterpolator(point_coord0,
                                                    observation_data)  # with noise and basic error, all CS data
        oldfunc = interpolate.NearestNDInterpolator(point_coord0,
                                                    point_values0)  # original without noise
        afterremovedfunc = interpolate.NearestNDInterpolator(after_removed_coord,
                                                             after_removed_data)  # remove known noise, not predict
        grid_x, grid_y = np.mgrid[0:80:80j, 0:40:40j]
        inter_new = newfunc(grid_x, grid_y)
        inter_old = oldfunc(grid_x, grid_y)
        inter_afterremoved = afterremovedfunc(grid_x, grid_y)
        inter_new_stats_temp.append(inter_new)

        # last time step
        data3_last = data2[t - 1, :, :].T
        data3_last[data3_last < 0] = 0
        row_last = np.random.choice(80, n)
        col_last = np.random.choice(40, n)
        row_last = list(row_last)
        col_last = list(col_last)
        point_coord_last = [row_last, col_last]
        point_values0_last = data3_last[tuple(point_coord_last)]
        point_coord0_last = list(zip(row_last, col_last))
        point_coord0_last = np.array(point_coord0_last)

        # mark the point value which is 0 in ground true
        zero_No_last = []
        for i in range(n):
            if data3_last[point_coord_last[0][i], point_coord_last[1][i]] == 0:
                zero_No.append(i)

        # last time step basic regular error (normal distribution)
        mu0_last = np.random.uniform(0.15, 0.15, size=len(point_values0_last))
        sigma0_last = np.random.uniform(0.1, 0.2, size=len(point_values0_last))
        noise0_last = np.random.normal(mu0_last * point_values0_last, mu0_last * point_values0_last,
                                       size=len(point_values0_last))  # basic error
        observation_data_last = point_values0_last + noise0_last  # only with basic error

        # # last time step basic regular error (wald distribution)
        # a0_last = np.random.uniform(0.15, 0.15, size=len(point_values0_last))
        # # b0_last = np.random.uniform(0.05, 0.1, size=len(point_values0_last))
        # b0_last = a0_last * 2
        # noise0_last = np.random.wald(a0_last * (point_values0_last),
        #                              b0_last * (point_values0_last),
        #                              size=len(point_values0_last))
        # observation_data_last = point_values0_last + noise0_last

        choice_No_last = np.random.choice(n, size=n1, replace=False)

        # last time step noise (normal distribution)
        mu1_last = np.random.uniform(0.2, 0.6 + 0.2, size=len(list(choice_No_last)))  # 0.5-2
        sigma1_last = np.random.uniform(0.6 + 0.5, 0.6 + 1, size=len(list(choice_No_last)))  # 0.5-3
        noise1_last = np.random.normal(mu1_last * point_values0_last[choice_No_last],
                                       sigma1_last * point_values0_last[choice_No_last],
                                       size=len(list(choice_No_last)))  # noise

        # # # last time step noise (wald distribution)
        # a1_last = np.random.uniform(0.2, 0.6+0.2, size=len(choice_No_last))  # 0.6  # mean
        # # b1_last = np.random.uniform(0.6+0.5, 0.6+1, size=len(choice_No_last))  # 0.6 # scale
        # b1_last = a1_last * 2
        # noise1_last = np.random.wald(a1_last * (point_values0_last[choice_No_last]),
        #                              b1_last * (point_values0_last[choice_No_last]),
        #                              size=len(choice_No_last))

        # chose points add noise1 values
        observation_data_last[choice_No_last] = observation_data_last[choice_No_last] + noise1_last
        # 索引之前 ground true 为 0 的情况，相应的观测值也为0
        observation_data_last[zero_No_last] = 0
        observation_data_last[observation_data_last < 0] = 0
        # if observation values less than 0, then assignment it as 0
        for i in range(observation_data_last.shape[0]):
            if observation_data_last[i] < 0:
                observation_data_last[i] = 0

        # last time step: noise threshold define & remove & spatial interpolation
        R_last = observation_data_last - point_values0_last
        noise_set_last = []
        non_noise_set_last = []
        # for i in range(len(R_last)):
        for i in choice_No_last:
            if (abs(R_last[i]) > 0.5 * point_values0_last[i]) & (abs(R_last[i]) > 0.1):
                noise_set_last.append(i)
            else:
                non_noise_set_last.append(i)

        # last time step: noise point values & coordinates
        noise_values_last = []  # noise point values
        for i in noise_set_last:
            noise_values_last_temp = observation_data_last[i]
            noise_values_last.append(noise_values_last_temp)

        noise_coord_last = []  # noise point coordinates
        for i in noise_set_last:
            noise_coord_last_temp = point_coord0_last[i]
            noise_coord_last.append(noise_coord_last_temp)

        point_values0_last = np.array(point_values0_last)
        # noise_set_last = np.array(noise_set_last)
        # last time step: remove
        after_removed_data_last = np.delete(observation_data_last, noise_set_last, axis=0)
        after_removed_coord_last = np.delete(point_coord0_last, noise_set_last, axis=0)

        # last time step: spatial interpolation
        newfunc_last = interpolate.NearestNDInterpolator(point_coord0_last,
                                                         observation_data_last)  # without basic error and noise
        oldfunc_last = interpolate.NearestNDInterpolator(point_coord0_last,
                                                         point_values0_last)  # without noise but with basic error
        afterremovedfunc_last = interpolate.NearestNDInterpolator(after_removed_coord_last,
                                                                  after_removed_data_last)  # with basic error & noise
        inter_new_last = newfunc_last(grid_x, grid_y)

        # REAA
        REAA_new = REAA(inter_new)
        REAA_old = REAA(inter_old)
        REAA_removed = REAA(inter_afterremoved)
        # REAA_remove = REAA(inter_remove)

        # RMSE
        RMSE_new = RMSE(inter_new)
        RMSE_old = RMSE(inter_old)
        RMSE_removed = RMSE(inter_afterremoved)
        # RMSE_remove = RMSE(inter_remove)

        RMSE_new1.append(RMSE_new), RMSE_removed1.append(RMSE_removed)
        REAA_new1.append(REAA_new), REAA_removed1.append(REAA_removed)
        # REAA_remove1.append(REAA_removed), RMSE_remove1.append(RMSE_removed)
        # RMSE_old_sta.append(RMSE_old), REAA_old_sta.append(REAA_old)
        # window based features
        data_X_temp = []
        for ws in range(1, 6):  # test!!
            # ws = 4  # window size, ws = 1 corresponding window is 3 * 3 if haven't reach boundary
            rd, ad = [], []
            win_added_var, win_added_mean, win_added_std = [], [], []
            win_added_max, win_added_min, win_added_range = [], [], []
            window_new = []  # all observation points(CS data)
            for i in range(point_coord0.shape[0]):
                if point_coord[0][i] - ws < 0:
                    start_row = 0
                    if point_coord[1][i] - ws < 0:
                        start_col = 0
                        window0 = inter_new[start_row:point_coord[0][i] + ws + 1,
                                  start_col:point_coord[1][i] + ws + 1]
                    elif point_coord[1][i] - ws >= 0:
                        window0 = inter_new[start_row:point_coord[0][i] + ws + 1,
                                  point_coord[1][i] - ws:point_coord[1][i] + ws + 1]
                elif point_coord[0][i] - ws >= 0:
                    if point_coord[1][i] - ws < 0:
                        start_col = 0
                        window0 = inter_new[point_coord[0][i] - ws:point_coord[0][i] + ws + 1,
                                  start_col:point_coord[1][i] + ws + 1]
                    elif point_coord[1][i] - ws >= 0:
                        window0 = inter_new[point_coord[0][i] - ws:point_coord[0][i] + ws + 1,
                                  point_coord[1][i] - ws:point_coord[1][i] + ws + 1]
                mean0 = np.mean(window0)
                var0 = np.var(window0)
                std0 = np.std(window0, ddof=1)  # unbiased
                max0 = np.max(window0)
                min0 = np.min(window0)
                range0 = np.max(window0) - np.min(window0)
                central_value = inter_new[point_coord0[i, 0], point_coord0[i, 1]]  # window central

                ad0 = central_value - mean0
                # rd0 = (ad0/mean0)*100
                win_added_mean.append(mean0), win_added_var.append(var0), win_added_std.append(std0)
                win_added_max.append(max0), win_added_min.append(min0), win_added_range.append(range0)
                ad.append(ad0)
                # rd.append(rd0)
                window0 = np.ravel(window0)
                window_new.append(window0)

            # last time step: window
            window_last = []
            for i in range(point_coord0_last.shape[0]):
                if point_coord[0][i] - ws < 0:
                    start_row = 0
                    if point_coord[1][i] - ws < 0:
                        start_col = 0
                        window1 = inter_new_last[start_row:point_coord[0][i] + ws + 1,
                                  start_col:point_coord[1][i] + ws + 1]
                    elif point_coord[1][i] - ws >= 0:
                        window1 = inter_new_last[start_row:point_coord[0][i] + ws + 1,
                                  point_coord[1][i] - ws:point_coord[1][i] + ws + 1]
                elif point_coord[0][i] - ws >= 0:
                    if point_coord[1][i] - ws < 0:
                        start_col = 0
                        window1 = inter_new_last[point_coord[0][i] - ws:point_coord[0][i] + ws + 1,
                                  start_col:point_coord[1][i] + ws + 1]
                    elif point_coord[1][i] - ws >= 0:
                        window1 = inter_new_last[point_coord[0][i] - ws:point_coord[0][i] + ws + 1,
                                  point_coord[1][i] - ws:point_coord[1][i] + ws + 1]
                window1 = np.ravel(window1)
                window_last.append(window1)

            # correlation
            corr = []
            for i in range(len(window_new)):
                corr0 = np.corrcoef(window_new[i], window_last[i])
                corr0 = corr0[0][1]
                corr.append(corr0)
            # judge NaN
            for i in range(len(corr)):
                if math.isnan(corr[i]):
                    corr[i] = np.nan_to_num(corr[i])

            # difference with surround points
            data4 = copy.deepcopy(data3)
            for i in range(point_coord0.shape[0]):
                data4[point_coord0[i][0], point_coord0[i][1]] = observation_data[i]

            d0 = 6  # d0 means distance range from one central point to surround others. can be variable.
            difference_set = []
            difference_set2 = []
            range_set = []
            for i in range(point_coord0.shape[0]):
                row = point_coord0[i][0]
                col = point_coord0[i][1]
                values0 = data4[row, col]
                set0 = []
                for j in range(point_coord0.shape[0]):
                    d = ((point_coord0[i][0] - point_coord0[j][0]) ** 2 + (
                            point_coord0[i][1] - point_coord0[j][1]) ** 2) ** 0.5
                    if d <= d0:
                        row1 = point_coord0[j][0]
                        col1 = point_coord0[j][1]
                        values1 = data4[row1, col1]
                        set0.append(values1)
                sets = np.array(set0)
                mean_value = np.mean(sets)
                range0 = np.max(sets) - np.min(sets)
                difference = values0 - mean_value
                difference2 = abs(range0 - values0)
                difference_set.append(difference)
                difference_set2.append(difference2)
                range_set.append(range0)
            difference_set = np.array(difference_set)
            difference_set2 = np.array(difference_set2)
            range_set = np.array(range_set)

            data_X0 = np.vstack((win_added_mean, win_added_std, win_added_max, win_added_min,
                                 win_added_range, ad, corr,
                                 # win_added_var,
                                 difference_set, difference_set2, range_set)).T
            data_X_temp.append(data_X0)
        data_X_temp = np.array(data_X_temp)
        data_X_temp = np.hstack((data_X_temp[0], data_X_temp[1], data_X_temp[2], data_X_temp[3], data_X_temp[4]))
        data_X.append(data_X_temp)
        data_y0 = np.zeros(n)
        for i in noise_set:
            data_y0[i] = 1
        data_y.append(data_y0)

data_X = np.array(data_X)
data_y = np.array(data_y)

duration = time() - t0
print("duration: %f" % duration)
