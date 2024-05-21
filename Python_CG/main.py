import numpy as np

def printMtx(mtx):
    # .shape[0] refers to the number of row
    # .shape[1] refers to the number of column
    for rWkr in range (mtx.shape[0]):
        for cWkr in range (mtx.shape[1]):
            print(mtx[rWkr, cWkr], end = ' ')
        print()
def main():
    mtxA = np.array([[1.5004, 1.3293, 0.8439],[1.3293, 1.2436, 0.6936],[0.8439, 0.6936, 1.2935]])
    printMtx(mtxA)


if __name__ == "__main__":
    main()
