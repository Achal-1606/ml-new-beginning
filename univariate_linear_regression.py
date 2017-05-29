import pandas as pd
import numpy as np
import matplotlib.pyplot
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


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


def cost_function_3d_plot(in_list):
    x = []
    y = []
    z = []
    for s in in_list:
        z.append(s[0])
        x.append(s[1][0][0])
        y.append(s[1][0][1])
    # print x_axis
    # print y_axis
    # print z_axis
    print "Plotting 3d plot for Cost Function vs theta..."
    fig = matplotlib.pyplot.figure()
    ax = fig.gca(projection='3d')
    x_i, y_i = np.meshgrid(x, y)
    surf = ax.plot_surface(x_i, y_i, z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.zaxis.set_major_locator(LinearLocator(10))
    fig.colorbar(surf, shrink=0.5, aspect=5)
    matplotlib.pyplot.show()

if __name__ == '__main__':
    data_set = pd.read_csv("E:\\ml_repository\ex1.txt", sep=",", header=None)
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

    num_iteration = 1000
    out_list = []
    alpha = 0.01
    theta = np.array([0, 0]).reshape(1, 2)
    for i in range(num_iteration):
        out_mod = compute_j_theta_per_iteration(arr_in, arr_out, alpha, theta)
        out_list.append(out_mod)
        theta = np.array(out_mod[1])
        # print theta
    final_values = out_list[-1]
    print "Minimised Cost function :- %s" % final_values[0]
    J_final = final_values[0]
    theta_final = final_values[1][0]
    print "Theta values :- %s" % theta_final

    cost_function_3d_plot(out_list)

    axes = matplotlib.pyplot.gca()
    axes.set_xlim([min(x_axis) - 1, max(x_axis) + 2])
    axes.set_ylim([min(y_axis) - 1, max(y_axis) + 2])
    y_line = []
    for i in x_axis:
        y_line.append(theta_final[0] + theta_final[1] * i)
    matplotlib.pyplot.plot(x_axis, y_axis, 'rx')
    matplotlib.pyplot.plot(x_axis, y_line, 'k-')
    matplotlib.pyplot.show()
