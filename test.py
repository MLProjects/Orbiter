import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sys
import select
import tty
import termios

def isData():
    return select.select([sys.stdin], [], [], 0.1) == ([sys.stdin], [], [])

old_settings = termios.tcgetattr(sys.stdin)


def calc_relative_positions(position_array):
    relative_positions = np.zeros((position_array.size, position_array.size))
    for i in range(0, position_array.size):
        for j in range(0, position_array.size):
            relative_positions[i, j] = position_array[j] - position_array[i]
    return relative_positions
old_settings


def norm_xy_2D(matrix_x, matrix_y):
    module = np.power(np.power(matrix_x, 2) + np.power(matrix_y, 2), 0.5)
    module_x = np.zeros(module.shape)
    module_y = np.zeros(module.shape)
    for i in range(0, module.shape[0]):
        for j in range(0, module.shape[1]):
            if(module[i][j] != 0.0):
                module_x[i][j] = matrix_x[i][j] / module[i][j]
                module_y[i][j] = matrix_y[i][j] / module[i][j]
    return module_x, module_y


def norm_xy_1D(matrix_x, matrix_y):
    module = np.power(np.power(matrix_x, 2) + np.power(matrix_y, 2), 0.5)
    module_x = np.zeros(module.shape)
    module_y = np.zeros(module.shape)
    for i in range(0, module.shape[0]):
        if(module[i] != 0.0):
            module_x[i] = matrix_x[i] / module[i]
            module_y[i] = matrix_y[i] / module[i]
    return module_x, module_y


def cursor(thrust, count):
    try:
        tty.setcbreak(sys.stdin.fileno())

        sys.stdout.write("\r%d" % thrust)
        sys.stdout.flush()

        # print isData()

        if isData():
            print('Press ', count)
            count = 0
            c = sys.stdin.read(3)
            # print str(c)
            if c == '\x1b[A':
                thrust = 1.0
            elif c == '\x1b[B':
                thrust = -1.0
        else:
            # print 'No press ', count
            thrust = 0.0

        return thrust, count

    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)


dt = 0.01  # Time step
G = 1  # Gravitational constant
fix = np.array([0, 1])  # If zero then planet doesn't move
thrust = np.array([0.0, 0.0])
mass = np.array([100, 1])
speed_x = np.array([0, 0])
speed_y = np.array([0, 7])
position_x = np.array([0, -2])
position_y = np.array([0, 0])
a_x = np.zeros(position_x.shape)
a_y = np.zeros(position_y.shape)
count = 0
thrust_loc = 0


print(position_x)
print(position_x.size)

position_x_matrix = calc_relative_positions(position_x)
position_y_matrix = calc_relative_positions(position_y)

print(position_x_matrix)
print(position_x_matrix.shape[0])

#plt.plot(position_x, position_y, 'ro')
#plt.axis([-6, 6, -6, 6])
# plt.show()

t = 0
endt = 10
# while(t<endt):

#lt.plot(position_x, position_y, 'ro')
#plt.axis([-6, 6, -6, 6])
# plt.show()

fig, ax = plt.subplots()
xdata, ydata = [], []
planets, = plt.plot([], [], 'bo', ms=6)
trails, = plt.plot([], [], 'ro', ms=1)


def init():
    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)
    return planets, trails,


def update(i):
    global position_x_matrix, position_y_matrix, position_x, position_y
    global a_x, a_y, speed_x, speed_y, dt, t, ax, thrust, thrust_loc, count
    thrust_loc, count = cursor(thrust_loc, count)
    thrust[1] = 0.1 * thrust_loc
    for iterator in range(0, 5):
        position_x_matrix = calc_relative_positions(position_x)
        position_y_matrix = calc_relative_positions(position_y)
        distance_2 = np.power(position_x_matrix, 2) + \
            np.power(position_y_matrix, 2)
        position_x_matrix, position_y_matrix = norm_xy_2D(
            position_x_matrix, position_y_matrix)
        a_x = np.zeros(position_x.shape)
        a_y = np.zeros(position_y.shape)
        speed_xnorm, speed_ynorm = norm_xy_1D(speed_x, speed_y)
        for i in range(0, a_x.size):
            a_x[i] = a_x[i] + speed_xnorm[i] * thrust[i] / dt
            a_y[i] = a_y[i] + speed_ynorm[i] * thrust[i] / dt
            for j in range(0, a_x.size):
                if(distance_2[i][j] != 0.0):
                    a_x[i] = a_x[i] + G * mass[j] * \
                        position_x_matrix[i][j] / distance_2[i][j]
                    a_y[i] = a_y[i] + G * mass[j] * \
                        position_y_matrix[i][j] / distance_2[i][j]

        # Update positions
        speed_x = speed_x + 0.5 * a_x * dt * fix
        speed_y = speed_y + 0.5 * a_y * dt * fix
        position_x = position_x + fix * speed_x * dt + 0.5 * a_x * dt * dt * fix
        position_y = position_y + fix * speed_y * dt + 0.5 * a_y * dt * dt * fix
        speed_x = speed_x + 0.5 * a_x * dt * fix
        speed_y = speed_y + 0.5 * a_y * dt * fix

        t = t + dt
    xdata.append(position_x)
    ydata.append(position_y)
    trails.set_data(xdata, ydata)
    planets.set_data(position_x, position_y)
    return planets, trails

ani = FuncAnimation(fig, update, frames=100, init_func=init, blit=True)
plt.show()


print(position_x_matrix)
print(position_y_matrix)
