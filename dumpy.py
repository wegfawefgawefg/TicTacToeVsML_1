'''
DEPRECATED
USE IF U R RETARD
'''

from random import random
incNum = -1

################
#   mm

#   dot

#    outer product

#   transpose
def T(matrix):
    shape = getShape(matrix)
    numDims = len(shape)
    data = matrix['data']
    if numDims == 3: #  axis = 1
        newData = []
        for b in range(0, shape[0]):
            miniMat = {
                'shape':[shape[1], shape[2]],
                'data':data[b]
            }
            newData.append(T(miniMat))
        matrix['data'] = newData
        return matrix
    if numDims == 2:
        rows = []
        for x in range(0, shape[1]):
            row = []
            for y in range(0, shape[0]):
                row.append(data[y][x])
            rows.append(row)
        matrix['data'] = rows
        return matrix

def getShape(matrix):
    return matrix['shape']

#   print matrix
def printmat(matrix):
    shape = getShape(matrix)
    numDims = len(shape)
    data = matrix['data']
    if numDims == 3:
        print('[', end='')
        for z in range(0, shape[0]):
            batch = data[z]
            if z == 0:
                print('[', end='')
            else:
                print(' [', end='')
            for y in range(0, shape[1]):
                row = batch[y]
                if y == 0:
                    print('[', end='')
                else:
                    print('  [', end='')
                for x in range(0, shape[2]):
                    num = row[x]
                    print(num, end='')
                    print(',', end='')
                print(']')
            print('],')
        print(']')
    elif numDims == 2:
        print('[', end='')
        for y in range(0, shape[0]):
            row = data[y]
            if y == 0:
                print('[', end='')
            else:
                print(' [', end='')            
            for x in range(0, shape[1]):
                num = row[x]
                print(num, end='')
                print(',', end='')
            print('],')
        print(']')
    elif numDims == 1:
        print('[', end='')
        for i in range(0, shape[0]):
            num = data[i]
            print(num, end='')
            print(',', end='')#   pow elementwise
        print(']')

#   slicing
def index(matrix, slices):
    shape = getShape(matrix)
    numDims = len(shape)
    if numDims == 1:


###############
#   scal mult

#   scal div

#   scal add

#   scal minus (use add)



def one():
    return 1

def zero():
    return 0

def inc(reset=False):
    global incNum
    if reset:
        incNum = -1
    incNum += 1
    return incNum

def randMat(dims):
    return genMatrix(dims, random)

def zeros(dims):
    return genMatrix(dims, zero)

def ones(dims):
    return genMatrix(dims, one)

def step(dims):
    inc(reset=True)
    return genMatrix(dims, inc)

def genMatrix(dims, genFunc):
    numDims = len(dims)
    mat = {
        'shape':dims,
        'data':None
    }
    if numDims == 3:
        batch = []
        for z in range(0, dims[0]):
            matrix = []
            for y in range(0, dims[1]):
                row = []
                for x in range(0, dims[2]):
                    row.append(genFunc())
                matrix.append(row)
            batch.append(matrix)
        mat['data'] = batch
    elif numDims == 2:
        matrix = []
        for y in range(0, dims[0]):
            row = []
            for x in range(0, dims[1]):
                row.append(genFunc())
            matrix.append(row)
        mat['data'] = matrix
    elif numDims == 1:
        vec = []
        for i in range(0, dims[0]):
            vec.append(genFunc())
        mat['data'] = vec
    return mat

a = randMat((6,6, 6))
printmat(a)


#   relu
#   derrelu

