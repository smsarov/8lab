import numpy as np
import matplotlib.pyplot as plt

def fs(t, v, kounter):
  A = np.array([[-0.4, 0.02, 0], [0, 0.8, -0.1], [0.003, 0, 1]])
  kounter[0] += 1
  return np.dot(A, v)

initial_conditions = [1, 1, 2]

# https://cyclowiki.org/wiki/%D0%9C%D0%B5%D1%82%D0%BE%D0%B4_%D0%A0%D1%83%D0%BD%D0%B3%D0%B5-%D0%9A%D1%83%D1%82%D1%82%D1%8B
def step(t, v, h):
    k1 = fs(t, v, kounter)
    k2 = fs(t + h / 2, v + h * k1 / 2, kounter)
    k3 = fs(t + h, v - h * k1 + 2 * h * k2, kounter)
    return v + (h / 6) * (k1 + 4 * k2 + k3)


def try_step(t, v, h):

    v2 = step(t, v, h)
    t2 = t + h

    v2_1 = step(t, v, h / 2)
    t2_1 = t + h / 2
    v2_2 = step(t2_1, v2_1, h / 2)

    R = np.linalg.norm(v2 - v2_2) / 7

    return R, t2, v2_2


def solve(t0, T, h0, eps):
    t = t0
    h = h0
    v = np.array(initial_conditions)
    kounter = [0]

    def step(t, v, h):
        k1 = fs(t, v, kounter)
        k2 = fs(t + h / 2, v + h * k1 / 2, kounter)
        k3 = fs(t + h, v - h * k1 + 2 * h * k2, kounter)
        return v + (h / 6) * (k1 + 4 * k2 + k3)

    def try_step(t, v, h):

        v2 = step(t, v, h)
        t2 = t + h

        v2_1 = step(t, v, h / 2)
        t2_1 = t + h / 2
        v2_2 = step(t2_1, v2_1, h / 2)

        R = np.linalg.norm(v2 - v2_2) / 7

        return R, t2, v2_2

    _t = []
    _h = []
    _R = []
    _steps = []
    _v = []

    _t.append(t0)
    _h.append(h0)
    _R.append(0)
    _steps.append(0)
    _v.append(initial_conditions)


    while t < T:
        R, t2, v2 = try_step(t, v, h)

        while R > eps:
            h /= 2
            R, t2, v2 = try_step(t, v, h)

        while R < eps / 64:
            h *= 2
            R, t2, v2 = try_step(t, v, h)

        t = t2
        v = v2

        _t.append(t)
        _h.append(h)
        _R.append(R)
        _steps.append(kounter[0])
        _v.append(v)

    return {
        't': _t,
        'h': _h,
        'R': _R,
        'steps': _steps,
        'v': _v
    }


t_0 = 1.5
T = 2.5
h_0 = 0.1
N = 10000
eps = 0.001

solve(t_0, T, h_0, eps)


# h(eps)
# min(h) (eps)
# steps(eps)
# result (eps)

epses = [1e-3, 1e-4, 1e-5, 1e-6]
results = [solve(t_0, T, h_0, eps) for eps in epses]

fig, axs = plt.subplots(2, 2)
fig.suptitle('h(t)')
for i, res in enumerate(results):
    eps = epses[i]

    x = res['t']
    y = res['h']

    p = i // 2
    q = i % 2

    axs[p, q].plot(x, y)
    axs[p, q].set_title(f"{eps=}")
plt.show()




epses = np.linspace(1e-8, 1e-3, 10000)
results = [solve(t_0, T, h_0, eps) for eps in epses]


plt.semilogx()
plt.title('зависимость min(h) от eps')
y = [min(res['h']) for res in results]
plt.plot(epses, y)
plt.show()


plt.semilogx()
plt.title('зависимость числа шагов steps oт eps')

y = [res['steps'][-1] for res in results]
plt.plot(epses, y)
plt.show()


plt.semilogx()
plt.title('зависимость координат решения oт eps')
y = [res['v'][-1] for res in results]
z1 = list(map(lambda v: v[0], y))

for i in range(len(y[0])):
    z_i = list(map(lambda v: v[i], y))
    plt.plot(epses, z_i)
plt.show()
