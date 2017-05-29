import numpy as np


def main():
    new_arr = np.mat([[3, 4], [2, 16]])
    new_arr_inv = np.linalg.inv(new_arr)
    print "Multiplicative inverse of \n {} \n is :- \n {}".format(new_arr, new_arr_inv)
    print "Array transpose is:- \n {}".format(new_arr.T)
    print "(A * A^-1) :- \n {}".format(np.dot(new_arr, new_arr_inv))


if __name__ == "__main__":
    main()
