import math
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (10,7)

def plt_signal(x,y, xT, yT, dfn):
    plt.stem(x, y, 'rx', use_line_collection=True)
    plt.xticks(x, x)
    plt.xlabel('n')
    plt.ylabel(dfn)
    plt.show()

    plt.stem(xT, yT, 'rx')
    plt.xlabel('n')
    plt.ylabel(dfn)
    plt.show()


# init variant data
option = 5
N = 30 + option % 5
T = 0.0005 * (1 + option % 3)
a = pow(-1, option) * (0.8 + 0.005 * option)
C = 1 + option % 5
w_0 = math.pi / (6 + option % 5)
m = 5 + option % 5
U = option
n_0 = option % 5 + 3
n_imp = option % 5 + 5
B_1 = 1.5 + option % 5
B_2 = 5.7 - option % 5
B_3 = 2.2 + option % 5
w_1 = math.pi / (4 + option % 5)
w_2 = math.pi / (8 + option % 5)
w_3 = math.pi / (16 + option % 5)
a_1 = 1.5 - option % 5
a_2 = 0.7 + option % 5
a_3 = 1.4 + option % 5


def singleton(m = 0):
    y = np.zeros((N-1,))
    y[m] = 1
    x = [x for x in range(0, N-1)]
    xT = np.multiply(range(0, N-1), T)
    plt_signal(x, y, xT, y, 'δ(n)')


def singleton_discrete(m = 0):
    y = np.concatenate((np.zeros(m,), np.ones(N-1 - m,)))
    x = [x for x in range(0, N-1)]
    xT = np.multiply(range(0, N-1), T)
    plt_signal(x, y, xT, y, 'σ(n)')


def exp_seq(m = 0):
    x = [x for x in range(0, N-1)]
    xT = np.multiply(range(0, N-1), T)
    y = [0 for _ in range(m)] + [pow(a, p) for p in x[0:N-1-m]]
    yT = [0 for _ in range(m)] + [(a ** (1 / T)) ** p * (-(p / T % 2) * 2 + 1) for p in xT[0:N-1-m]]
    plt_signal(x, y, xT, yT, 's_1(n)')


def harm_discrete():
    x = [x for x in range(0, N-1)]
    # y = [complex(C, 0) * cmath.exp(1j * complex(w_0, 0) * complex(p, 0)) for p in x]
    y = [C * math.cos(w_0*p) for p in x]
    yi = [C * math.sin(w_0*p) for p in x]
    plt_signal(x, y, x, yi, 's_2(n)')


def signals_delayed():
    singleton(m)
    singleton_discrete(m)
    exp_seq(m)


def singleton_rect_discrete():
    x = range(0, N-1)
    y = np.concatenate((np.zeros(n_0,), np.multiply(np.ones(n_imp,), U), (np.zeros(N - 1 - n_0 - n_imp,))))

    plt.stem(x, y, 'rx', label='s_3', use_line_collection=True)
    plt.xticks(x, x)
    plt.xlabel('n')
    plt.ylabel('s_3(n)')
    plt.legend()
    plt.show()


def harm_discrete_combined():
    x = range(0, 5*N-1)
    y1 = [B_1 * math.sin(w_1*p) for p in x]
    y2 = [B_2 * math.sin(w_2*p) for p in x]
    y3 = [B_3 * math.sin(w_3*p) for p in x]
    y = np.sum([np.multiply(y1, a_1), np.multiply(y2, a_2), np.multiply(y3, a_3)], axis = 0)

    plt.figure(figsize=(13, 4))
    plt.plot(x, y, label='s_4')
    plt.plot(x, y1, label='x_1')
    plt.plot(x, y2, label='x_2')
    plt.plot(x, y3, label='x_3')
    plt.xlabel('n')
    plt.legend()
    plt.show()


def sinus_discrete():
    x = range(0, N-1)
    y = [(a ** p) * math.cos(w_0 * p) for p in x]
    plt.stem(x, y, label='s_5', use_line_collection=True)
    plt.xticks(x, x)
    plt.legend()
    plt.xlabel('n')
    plt.ylabel('s_5(n)')
    plt.show()


def rect_seq_repeated():
    len = n_0 + n_imp*5 * 2
    x = range(0, len)
    y = np.zeros(len)
    for n in range(n_0, len):
        if ((n-n_0)//n_imp) % 2 == 0:
            y[n] = U

    plt.stem(x, y, 'rx', label='s_6', use_line_collection=True)
    plt.xlabel('n')
    plt.ylabel('s_6(n)')
    plt.legend()
    plt.show()


# build plots

singleton()

singleton_discrete()

exp_seq()

harm_discrete()

singleton(m)
singleton_discrete(m)
exp_seq(m)

singleton_rect_discrete()

harm_discrete_combined()

sinus_discrete()

rect_seq_repeated()