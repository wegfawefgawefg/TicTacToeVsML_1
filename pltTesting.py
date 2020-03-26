import matplotlib.pyplot as plt

x = [1,2,3,4,5]
y = [1,2,3,4,5]

xNum = 0
yNum = 0
while(True):
    xNum += 1
    yNum += 2* xNum
    x.append(xNum)
    y.append(yNum)
    plt.plot(x, y)
    plt.pause(0.05)