import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

x = np.arange(0, 2*np.pi, 0.1)
y = np.sin(x)

fig, axes = plt.subplots(nrows=6)


def plot(ax):
    return ax.plot(x, y, animated=True)[0]


lines = [plot(ax) for ax in axes]
print(type(lines[1]))

def animate(i):
    for j, line in enumerate(lines, start=0):
        line.set_ydata(np.sin(j*x + i/10.0))
    return lines

dog = 1
while True:
    dog += 1
    # We'd normally specify a reasonable "interval" here...
    ani = animation.FuncAnimation(fig, animate, range(1, 200), 
                                interval=0, blit=True)
    print(dog)

    plt.pause(0.05)