import numpy as np
import copy

class Tic():
    def __init__(self,size):
        self.size = size
        self.build(size)

    def build(self,size):
        arr = []
        for i in range(size):
            arr.append(np.zeros(size))

        self.board = np.array(arr)

    def state(self,actor):
        return actor * self.board.reshape((1,self.size**2))

    def valid(self,loc):
        if self.board[loc[0]][loc[1]] == 0:
            return True
        return False

    def play(self,a,loc):
        if self.board[loc[0]][loc[1]] != 0:
            self.print()
            print(loc)
            print(a)
            print("INVALID PLAY")
            exit(1)
        else:
            self.board[loc[0]][loc[1]] = a

    def done(self):
        amDone = True
        isWinner = False
        winner = 0

        rDFound = False
        cDFound = False

        rD = self.board[0][0]
        cD = np.rot90(self.board)[0][0]

        for i in range(len(self.board)): #For each row
            r = self.board[i] #Row
            c = np.rot90(self.board)[i] #Column

            rFound = False
            rLastE = r[0]
            cFound = False
            cLastE = c[0]

            if r[i] != rD or r[i] == 0: #Check Diagonals
                rDFound = True
            if c[i] != cD or c[i] == 0:
                cDFound = True

            for j in range(len(r)): #For each item in the row/column
                rE = r[j]
                cE = c[j]

                if rE != rLastE or rE == 0:
                    rFound = True
                if rE == 0.0:
                    amDone = False

                if cE != cLastE or cE == 0:
                    cFound = True

            if not rFound:
                isWinner = True
                winner = rLastE
            if not cFound:
                isWinner = True
                winner = cLastE

        if not rDFound:
            isWinner = True
            winner = rD
        if not cDFound:
            isWinner = True
            winner = cD

        return [isWinner or amDone,winner]

    def print(self):
        self.setPrintBuffer()
        for line in self.myPrint:
            print(line)

    def setPrintBuffer(self):
        self.myPrint = []
        for i in range(len(self.board)):
            r = copy.deepcopy(self.board[i])
            line1 = ""
            line2 = ""
            for j in range(len(r)):
                e = r[j]
                symbol = ' '
                if e == 1:
                    symbol = 'x'
                if e == -1:
                    symbol = 'o'

                line1 = line1 + (" " + symbol + " ")
                line2 = line2 + ("---")

                if j < len(r) - 1:
                    line1 += "|"
                    line2 += "+"
            self.myPrint.append(line1)
            if i < len(self.board) - 1:
                self.myPrint.append(line2)