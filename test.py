def extractWindow(data, maxWindowSize):
    top = len(data)
    bottom = 0
    intervalSize = (top - bottom) / maxWindowSize

    newData = []
    sampleIndex = 0
    for i in range( maxWindowSize ):
        newData.append(data[int(sampleIndex)])
        sampleIndex += intervalSize

    assert (len(newData) == maxWindowSize)
    return newData

a = [0, 1, 2]
b = [0, 1, 2, 3, 4, 5]
c = list(range(0, 100))

wA = extractWindow(a, 20)
wB = extractWindow(b, 20)
wC = extractWindow(c, 20)

print(len(wA))
print(len(wB))
print(len(wC))