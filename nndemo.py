import numpy as np
import random 
import tictactoe as tt
from tqdm import trange
import matplotlib.pyplot as plt

np.random.seed(1)


#   ----    activation functions    ----    #
def relu(x):
    return (x > 0) * x
def derRelu(x):
    return x > 0

#   ----------------------------------  #
def pickBestMove(nextMoves, player, computersPlayer):
    oneHots = [oneHotTicTacToe(nextMove, tt.togglePlayer(player), computersPlayer) for nextMove in nextMoves]
    trainingSessions = [forward(network, oneHot, dropout=True) for oneHot in oneHots]

    goodMoves = []
    okayMoves = []
    badMoves = []
    for i in range(len(trainingSessions)):
        trainingSession = trainingSessions[i]
        score = trainingSession['outputs'][-1]
        loss = score[0]
        tie = score[1]
        win = score[2]

        moveScore = {"move":nextMoves[i],
                    "score":score,
                    "trainingSession":trainingSession}

        if win > tie and win > loss:
            goodMoves.append(moveScore)
        elif tie > loss:
            okayMoves.append(moveScore)
        else:
            badMoves.append(moveScore)

    if goodMoves:
        bestMove = None
        bestWin = -100000
        for move in goodMoves:
            win = move["score"][2]
            if win > bestWin:
                bestMove = move
                bestWin = win
    elif okayMoves:
        bestMove = None
        bestTie = -100000
        for move in okayMoves:
            tie = move["score"][1]
            if tie > bestTie:
                bestMove = move
                bestTie = tie
    else: # only bad moves :(
        bestMove = None
        bestLoss = 100000
        for move in badMoves:
            loss = move["score"][0]
            if loss < bestLoss:
                bestMove = move
                bestLoss = loss
    return bestMove

def oneHotTicTacToe(board, player, computersPlayer):
    xs = np.zeros((3,3))
    for y in range(0, 3):
        for x in range(0, 3):
            piece = board[y][x]
            if piece == 2:
                xs[y][x] = 1.0
    os = np.zeros((3,3))
    for y in range(0, 3):
        for x in range(0, 3):
            piece = board[y][x]
            if piece == 1:
                os[y][x] = 1.0

    xs = xs.flatten()
    os = os.flatten()
    playerOne = 1 if player == 1 else 0
    playerTwo = 1 if player == 2 else 0
    computerIsO = 1 if computersPlayer == 1 else 0
    computerIsX = 1 if computersPlayer == 2 else 0
    boardOneHot = np.append(xs, os)
    boardOneHot = np.append(boardOneHot, playerOne)
    boardOneHot = np.append(boardOneHot, playerTwo)
    boardOneHot = np.append(boardOneHot, computerIsO)
    boardOneHot = np.append(boardOneHot, computerIsX)
    return boardOneHot

def forward(network, inputData, dropout=False):
    #   index of layer lines up with index of dropout mask
    #   #   there is one less mask than layers
    #   index of layer output lines up with index of layer
    layers = network["layers"]
    outputs = []
    masks = []

    layerOutput = relu(inputData.dot(layers[0]))
    if dropout:
        dropoutMask = np.random.randint(2, size=layerOutput.shape)
        layerOutput *= dropoutMask
        layerOutput *= 2
        masks.append(dropoutMask)
    outputs.append(layerOutput)

    for l in range(1, len(layers) - 1):
        layerOutput = relu(outputs[-1].dot(layers[l]))
        if dropout:
            dropoutMask = np.random.randint(2, size=layerOutput.shape)
            layerOutput *= dropoutMask
            layerOutput *= 2
            masks.append(dropoutMask)        
        outputs.append(layerOutput)

    layerOutput = relu(outputs[-1].dot(layers[-1]))
    outputs.append(layerOutput)
    
    trainingSession = {
        "input":inputData,
        "outputs":outputs,
        "dropoutMasks":masks,
    }
    return trainingSession

def genNetwork(shape):
    layers = []
    for l in range(len(shape)-1):
        inputSize = shape[l]
        outputSize = shape[l+1]
        newLayer = weightScaler * (2.0 * np.random.random((inputSize, outputSize)) - 1.0)
        layers.append(newLayer)

    network = {
        "shape":shape,
        "layers":layers
    }
    return network

def backward(network, trainingSessions, truth):
    #   index of layer lines up with index of dropout mask
    #   #   there is one less mask than layers
    #   index of layer output lines up with index of layer
    layers = network['layers']

    error = 0
    for trainingSession in trainingSessions:
        inputData = trainingSession['input']
        outputs = trainingSession['outputs']
        dropoutMasks = trainingSession['dropoutMasks']

        #   compute deltas
        deltas = []
        lastLayerDelta = truth - outputs[-1]
        error += (lastLayerDelta ** 2).mean()
        deltas.append(lastLayerDelta)

        for l in range(len(layers) - 2, -1, -1):
            layerDelta = deltas[-1].dot( layers[l+1].T ) * derRelu( outputs[l] )
            layerDelta *= dropoutMasks[l]
            deltas.append(layerDelta)

        #   update weights
        deltas = list(reversed(deltas))
        #   #   index of layer delta lines up with layer index (after reverse)

        # layers[0] += alpha * inputData.T.dot( deltas[0] )
        layers[0] += alpha * np.outer(inputData.T, deltas[0])
        for l in range(1, len(layers)):
            # layers[l] += alpha * outputs[l-1].T.dot( deltas[l] )
            layers[l] += alpha * np.outer( outputs[l-1].T, deltas[l] )
    
    error = error / len(trainingSessions)
    return error

#   train with random moves
#   train with intentional moves

alpha = 0.01
weightScaler = 0.1
train = True

numInputs = 22
numOutputs = 3
# network = genNetwork((22, 128, 128, 128, 3))
network = genNetwork((22, 32, 32, 3))
numTrains = 100000

makeRandomMoves = False

#   graph data
gamesX = []

oWins = []
oWinsCount = 0

xWins = []
xWinsCount = 0

ties = []
tiesCount = 0

errors = []

maxWindowSize = 10

dummyData = [0] * maxWindowSize
print(dummyData)

fig, axes = plt.subplots(nrows=4)
lines = []
lines.append(axes[0].plot(dummyData, dummyData)[-1])
lines.append(axes[1].plot(dummyData, dummyData)[-1])
lines.append(axes[2].plot(dummyData, dummyData)[-1])
lines.append(axes[3].plot(dummyData, dummyData)[-1])

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


verbose = True
logInterval = 100
plotInterval = 20
if train:
    for i in trange(numTrains):
        # if i > 1000:
        #     makeRandomMoves = False

        board = tt.genBoard()
        movesLeft = True
        winner = False
        player = 2
        computersPlayer = random.randint(1,2)

        playerOneTrainingSessions = []
        playerTwoTrainingSessions = []

        if verbose and i % logInterval == 0:
            print("NEW GAME")
        while(movesLeft and not winner):
            if verbose and i % logInterval == 0:
                if player == 2:
                    print("X's Turn")
                else: # player == 1
                    print("O's Turn")
                tt.printBoard(board)

            nextMoves = tt.listNextBoards(board, player)
            if makeRandomMoves:
                randomMove = random.randint(0, len(nextMoves)-1)
                randomMove = nextMoves[randomMove]
                oneHot = oneHotTicTacToe(randomMove, tt.togglePlayer(player), computersPlayer)
                trainingSession = forward(network, oneHot, dropout=True)
                bestMove = randomMove
            else:
                bestMoveDict = pickBestMove(nextMoves, player, computersPlayer)
                bestMove = bestMoveDict['move']
                score = bestMoveDict['score']
                trainingSession = bestMoveDict['trainingSession']

            if player == 1:
                playerOneTrainingSessions.append(trainingSession)
            else: # player == 2
                playerTwoTrainingSessions.append(trainingSession)

            board = bestMove
            player = tt.togglePlayer(player)
                        
            winner = tt.getWinner(board)
            movesLeft = not tt.noMoreMoves(board)

        #   do backprop here
        playerOneTruth = None
        playerTwoTruth = None
        if winner == False:
            truth = np.array([0.0, 1.0, 0.0])
            playerOneTruth = truth
            playerTwoTruth = truth
        elif winner == 1:
            playerOneTruth = np.array([0.0, 0.0, 1.0])
            playerTwoTruth = np.array([1.0, 0.0, 0.0])
        else: # winner == 2
            playerTwoTruth = np.array([0.0, 0.0, 1.0])
            playerOneTruth = np.array([1.0, 0.0, 0.0])


        if winner:
            if winner == 2:
                xWinsCount += 1
            else: # winner == 1
                oWinsCount += 1
        else:
            tiesCount += 1

        if verbose and i % logInterval == 0:
            tt.printBoard(board)

            if winner:
                if winner == 2:
                    print("WINNER: X")
                else: # winner == 1
                    print("WINNER: O")
            else:
                print("TIE")

        errorOne = backward(network, playerOneTrainingSessions, playerOneTruth)
        errorTwo = backward(network, playerTwoTrainingSessions, playerTwoTruth)
        error = errorOne + errorTwo / 2

        if i % plotInterval == 0:
            ties.append( tiesCount / (float(i+1)))
            xWins.append(xWinsCount / (float(i+1)))
            oWins.append(oWinsCount / (float(i+1)))
            errors.append(error)        
            gamesX.append(i)

            wGamesX = extractWindow(gamesX, maxWindowSize)
            wOWins =  extractWindow(oWins,  maxWindowSize)
            wXWins =  extractWindow(xWins,  maxWindowSize)
            wTies =   extractWindow(ties,   maxWindowSize)
            wErrors = extractWindow(errors, maxWindowSize)

            for i in range(0, len(lines)):
                lines[i].set_xdata(wGamesX)
            lines[0].set_ydata(wOWins)
            lines[1].set_ydata(wXWins)
            lines[2].set_ydata(wTies)
            lines[3].set_ydata(wErrors)

            plt.pause(0.0001)

            # plt.plot(gamesX, oWins)
            # plt.ylabel('oWins')
            # plt.xlabel('num games')

            # plt.plot(gamesX, xWins)
        
        #   plotting
        if i % logInterval == 0:

    
            #   output
            trainsNumOut = "itter: " + str(i)
            mseOut = "mse: " + str(error)
            alphaOut = "alpha: " + str(alpha)
            outList = [ trainsNumOut,
                        mseOut,
                        alphaOut]
            out = " ".join(outList)

            print(out)
        alpha *= 0.99999

#   +   try random training
#   make graphs to determine if it is actually learning
#   #   heatmap specific first moves
#   #   count number of lethals missed

#   try different architectures

#   verify loss function is correct

#   continuous training until minimum error
#   save the network

#   integrate pytorch

#   add some endboards for ground truth
groundTruthBoards = [
    [[2, 0, 2],
     [0, 1, 0],
     [2, 0, 2]],

    [[2, 1, 2],
     [2, 1, 0],
     [0, 0, 0]],

    [[1, 2, 1],
     [2, 1, 2],
     [0, 1, 2]],

    [[2, 1, 2],
     [1, 1, 2],
     [2, 0, 1]],
]

groundTruthTurns = [2, 1, 2, 1]
groundTruthTruths = [
    [0, 0, 1],
    [0, 0, 1],
    [0, 1, 0],
    [0, 1, 0]
]