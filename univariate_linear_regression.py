import pandas as pd
import numpy as np
import matplotlib.pyplot


def cost_function(in_arr, out_arr, theta1):
    m = in_arr.shape[0]
    diff_sqr = 0
    for i2 in range(m):
        xi = in_arr[i2].reshape(1, 2)
        yi_pre = np.dot(xi, theta1.T)
        yi_org = out_arr[i2]
        diff_sqr += ((yi_pre - yi_org) * (yi_pre - yi_org))
    j = (1 / (2.0 * m)) * diff_sqr
    return j[(0, 0)]


def compute_j_theta_per_iteration(in_arr, out_arr, alpha, theta):
    m = in_arr.shape[0]
    j = cost_function(in_arr, out_arr, theta)
    temp = np.array([0.0, 0.0]).reshape(theta.shape)
#     print temp
    for i1 in range(m):
        xi = arr_in[i1].reshape(1, 2)
        yi_pre = np.dot(xi, theta.T)
        yi_org = arr_out[i1].reshape(1, 1)
        diff_sqr = (yi_pre - yi_org)
        temp += (diff_sqr * xi).reshape(theta.shape)
    theta_new = theta.T - ((alpha / m) * temp).reshape(theta.T.shape)
    # print theta_new.reshape(1, 2).tolist()
    return j, theta_new.reshape(1, 2).tolist()


if __name__ == '__main__':
    data_set = pd.read_csv("ex1.txt", sep=",", header=None)
    data_set.columns = ["Population_in_10000s", "Profit_in_$10000"]
    # print data_set.head()
    x_axis = data_set.Population_in_10000s.values
    y_axis = data_set["Profit_in_$10000"].values
    axes = matplotlib.pyplot.gca()
    axes.set_xlim([min(x_axis) - 1, max(x_axis) + 2])
    axes.set_ylim([min(y_axis) - 1, max(y_axis) + 2])
    matplotlib.pyplot.plot(x_axis, y_axis, 'rx')
    print "Plotting Input data PLOT...."
    matplotlib.pyplot.show()
    x_list = []
    y_list = []
    for x in range(len(x_axis)):
        x_list.append([1.0, x_axis[x]])
        y_list.append([y_axis[x]])

    arr_in = np.array(x_list)
    arr_out = np.array(y_axis).reshape(len(y_axis), 1)

    print "Feature size :- {}".format(arr_in.shape)
    print "Output size :- {}".format(arr_out.shape)

    theta_test = np.array([0, 0]).reshape(1, 2)
    print "Testing Cost Function with theta = {}...".format(theta_test)
    j_test = cost_function(arr_in, arr_out, theta_test)
    print "Cost Function :- {}".format(j_test)

    num_iteration = 1500
    out_list = []
    alpha = 0.01
    theta = np.array([0, 0]).reshape(1, 2)
    j_per_iteration = []
    x_ticks = []
    for i in range(num_iteration):
        out_mod = compute_j_theta_per_iteration(arr_in, arr_out, alpha, theta)
        if i != 0:
            j_per_iteration.append(out_mod[0])
            x_ticks.append(i + 1)
        out_list.append(out_mod)
        theta = np.array(out_mod[1])
        # print theta
    final_values = out_list[-1]
    print "Minimised Cost function :- %s" % final_values[0]
    J_final = final_values[0]
    theta_final = final_values[1][0]
    print "Theta values :- %s" % theta_final

    print "x_ticks :{}".format(x_ticks[:5])
    print "J value per ticks :{}".format(j_per_iteration[:5])
    matplotlib.pyplot.plot(x_ticks, j_per_iteration)
    print "Plotting cost function plot...."
    matplotlib.pyplot.show()

    axes = matplotlib.pyplot.gca()
    axes.set_xlim([min(x_axis) - 1, max(x_axis) + 2])
    axes.set_ylim([min(y_axis) - 1, max(y_axis) + 2])
    y_line = []
    for i in x_axis:
        y_line.append(theta_final[0] + theta_final[1] * i)
    matplotlib.pyplot.plot(x_axis, y_axis, 'rx')
    matplotlib.pyplot.plot(x_axis, y_line, 'k-')
    matplotlib.pyplot.show()
