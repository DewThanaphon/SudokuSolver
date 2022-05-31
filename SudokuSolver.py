#author: Thanaphon Rianthong
#github: https://github.com/DewThanaphon/SudokuSolver.git

import math
import numpy as np
import cv2

import tensorflow as tf

#algorithm to solve a sudoku problem
def checkPossibility(problem):
    poss = {}
    minis = int(math.sqrt(len(problem)))

    for r in range(len(problem)):
        poss[r] = {}
        for c in range(len(problem)):
            mini = []
            if problem[r,c]==0:
                mini = list(range(1,len(problem)+1))
                #check row
                for minir in range(len(problem)):
                    try:
                        if problem[minir, c]!=0: mini.remove(problem[minir, c])
                    except: pass

                #check column
                for minic in range(len(problem)):
                    try:
                        if problem[r, minic]!=0: mini.remove(problem[r, minic])
                    except: pass

                #check small square
                for minir in range((int(math.ceil((r+1)/minis)-1)*minis), int(math.ceil((r+1)/minis))*minis):
                    for minic in range((int(math.ceil((c+1)/minis)-1)*minis), int(math.ceil((c+1)/minis))*minis):
                        try:
                            if problem[minir, minic]!=0: mini.remove(problem[minir, minic])
                        except: pass
                
            poss[r][c] = mini

    return poss

def clearPossibility(problem, poss, row, col, num):
    minis = int(math.sqrt(len(problem)))

    for r in poss:
        try: poss[r][col].remove(num)
        except: pass
    
    for c in poss[row]:
        try: poss[row][c].remove(num)
        except: pass

    for r in range((int(math.ceil((row+1)/minis)-1)*minis), int(math.ceil((row+1)/minis))*minis):
        for c in range((int(math.ceil((col+1)/minis)-1)*minis), int(math.ceil((col+1)/minis))*minis):
            try: poss[r][c].remove(num)
            except: pass

    return poss

def fillNumber(problem, poss):
    for row in poss:
        for col in poss[row]:
            if len(poss[row][col])==1:
                problem[row, col] = poss[row][col][0]
                poss = clearPossibility(problem, poss, row, col, poss[row][col][0])

    return problem, poss

def checkFixedCases(problem, poss):
    minis = int(math.sqrt(len(problem)))
    for row in poss:
        mini = np.zeros(len(problem))
        for col in poss[row]:
            for num in poss[row][col]:
                mini[num-1] += 1

        if len(np.where(mini==1)[0])>0:
            for i in np.where(mini==1)[0]:
                for col in poss[row]:
                    try:
                        if i+1 in poss[row][col]:
                            poss[row][col] = [i+1]
                    except: pass

    for col in range(len(problem)):
        mini = np.zeros(len(problem))
        for row in poss:
            for num in poss[row][col]:
                mini[num-1] += 1

        if len(np.where(mini==1)[0])>0:
            for i in np.where(mini==1)[0]:
                for row in poss:
                    try:
                        if i+1 in poss[row][col]:
                            poss[row][col] = [i+1]
                    except: pass

    for minir in range(minis):
        for minic in range(minis):
            mini = np.zeros(len(problem))
            for row in range(int(minir*minis), int((minir+1)*minis)):
                for col in range(int(minic*minis), int((minic+1)*minis)):
                    for num in poss[row][col]:
                        mini[num-1] += 1

            if len(np.where(mini==1)[0])>0:
                for i in np.where(mini==1)[0]:
                    for row in range(int(minir*minis), int((minir+1)*minis)):
                        for col in range(int(minic*minis), int((minic+1)*minis)):
                            try:
                                if i+1 in poss[row][col]:
                                    poss[row][col] = [i+1]
                            except: pass

    return poss

def generateCase(numCase, maxCase = 3):
    allCase = []
    for i in range(maxCase):
        for j in range(numCase):
            if i==0:
                allCase.append([j])

            else:
                for k in range(len(allCase)):
                    if not j in allCase[k]:
                        case = allCase[k].copy()
                        case.extend([j])
                        case.sort()
                        if not case in allCase and len(case)<=maxCase:
                            allCase.append(case)

    return allCase

def limitCases(problem, poss):
    minis = int(math.sqrt(len(problem)))
    for row in poss:
        blank = []
        for c in poss[row]:
            if len(poss[row][c])>1: blank.append(c)

        cases = generateCase(len(blank))
        for case in cases:
            if len(case)>1:
                mini = np.zeros(len(problem))
                blankCol = []
                for inx in case:
                    for j in poss[row][blank[inx]]:
                        mini[j-1] += 1
                        blankCol.append(blank[inx])
                
                if len(np.where(mini>0)[0])==len(case):
                    for i in np.where(mini>0)[0]:
                        for c in poss[row]:
                            if not c in blankCol:
                                try: poss[row][c].remove(i+1)
                                except: pass

    for col in poss[1]:
        blank = []
        for r in poss:
            if len(poss[r][col])>1: blank.append(r)

        cases = generateCase(len(blank))
        for case in cases:
            if len(case)>1:
                mini = np.zeros(len(problem))
                blankRow = []
                for inx in case:
                    for j in poss[blank[inx]][col]:
                        mini[j-1] += 1
                        blankRow.append(blank[inx])
                
                if len(np.where(mini>0)[0])==len(case):
                    for i in np.where(mini>0)[0]:
                        for r in poss:
                            if not r in blankRow:
                                try: poss[r][col].remove(i+1)
                                except: pass

    for minir in range(minis):
        for minic in range(minis):
            blank = []
            for r in range(int(minir*minis), int((minir+1)*minis)):
                for c in range(int(minic*minis), int((minic+1)*minis)):
                    if len(poss[r][c])>1: blank.append([r, c])

            cases = generateCase(len(blank))
            for case in cases:
                if len(case)>1:
                    mini = np.zeros(len(problem))
                    blankInx = []
                    for inx in case:
                        for j in poss[blank[inx][0]][blank[inx][1]]:
                            mini[j-1] += 1
                            blankInx.append(blank[inx])
                    
                    if len(np.where(mini>0)[0])==len(case):
                        for i in np.where(mini>0)[0]:
                            for r in range(int(minir*minis), int((minir+1)*minis)):
                                for c in range(int(minic*minis), int((minic+1)*minis)):
                                    if not [r, c] in blankInx:
                                        try: poss[r][c].remove(i+1)
                                        except: pass

    for row in poss:
        mini = np.zeros([minis, len(problem)])
        for c in poss[row]:
            if len(poss[row][c])>0:
                for num in poss[row][c]:
                    inx = int(math.floor(c/minis))
                    mini[inx, num-1] += 1
        
        for num in range(len(problem)):
            if sum(mini[:, num]>0)==1:
                inx = np.where(mini[:, num]>0)[0][0]
                for r in range((int(math.ceil((row+1)/minis)-1)*minis), int(math.ceil((row+1)/minis))*minis):
                    for c in range(int(inx*minis), int((inx+1)*minis)):
                        if r!=row:
                            try: poss[r][c].remove(num+1)
                            except: pass

    for col in poss:
        mini = np.zeros([minis, len(problem)])
        for r in poss:
            if len(poss[r][col])>0:
                for num in poss[r][col]:
                    inx = int(math.floor(r/minis))
                    mini[inx, num-1] += 1
        
        for num in range(len(problem)):
            if sum(mini[:, num]>0)==1:
                inx = np.where(mini[:, num]>0)[0][0]
                for r in range(int(inx*minis), int((inx+1)*minis)):
                    for c in range((int(math.ceil((col+1)/minis)-1)*minis), int(math.ceil((col+1)/minis))*minis):
                        if c!=col:
                            try: poss[r][c].remove(num+1)
                            except: pass

    return poss

def checkDone(poss):
    for row in poss:
        for col in poss[row]:
            if len(poss[row][col])!=0:
                return False
    
    return True

def sudokuSolverAlgorithm(problem):
    problem = np.array(problem)

    #first check possibility
    poss = checkPossibility(problem)

    while True:
        #check fixed cases
        poss = checkFixedCases(problem, poss)

        #limit cases
        poss = limitCases(problem, poss)

        #fill only fixed case
        problem, poss = fillNumber(problem, poss)
        if checkDone(poss): break
    
    return problem

#image processing
def imageCreate(imgOri):
    img = imgOri.copy()
    imggray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    (_, imgbw) = cv2.threshold(imggray, 200, 255, cv2.THRESH_BINARY)
    row, col = imgbw.shape
    hisRow = np.zeros(row)
    for r in range(row):
        hisRow[r] = sum(imgbw[r,:])/255

    hisCol = np.zeros(col)
    for c in range(col):
        hisCol[c] = sum(imgbw[:,c])/255

    diffRow = np.where(hisRow<=(3*min(hisRow)))[0][1:]-np.where(hisRow<=(3*min(hisRow)))[0][:-1]
    diffRow[-1] = 100
    linesRow = np.where(hisRow<=(3*min(hisRow)))[0][np.where(diffRow>1)[0]]
    imglines = img
    for lineRow in linesRow:
        imglines = cv2.line(imglines, (0, lineRow), (col, lineRow), (0, 255, 0), 3)

    diffCol = np.where(hisCol<=(3*min(hisCol)))[0][1:]-np.where(hisCol<=(3*min(hisCol)))[0][:-1]
    diffCol[-1] = 100
    linesCol = np.where(hisCol<=(3*min(hisCol)))[0][np.where(diffCol>1)[0]]
    for lineCol in linesCol:
        imglines = cv2.line(imglines, (lineCol, row), (lineCol, 0), (0, 255, 0), 3)

    imgs = []
    for ir in range(1, len(linesRow)):
        for ic in range(1, len(linesCol)):
            imgmini = imgbw[linesRow[ir-1]:linesRow[ir], linesCol[ic-1]:linesCol[ic]]
            imgmini = cv2.resize(imgmini, (32, 32), interpolation = cv2.INTER_AREA)
            imgs.append(imgmini)

    return imgs, linesRow, linesCol

#sudoku problem creation from an image
def digitClassification(model, imgs, bound=3):
    imgs = np.array(imgs)
    inx = []
    for i in range(len(imgs)):
        r, _ = np.where(imgs[i, bound:32-bound, bound:32-bound]>0)
        if len(r)<(32-(2*bound))**2:
            inx.append(i)

    imgs = np.reshape(imgs, (len(imgs), 32, 32, 1))
    problem = np.zeros([9,9])
    for i in inx:
        r = math.floor(i/9)
        c = i%9

        digit = np.argmax(model.predict(np.array([imgs[i]])))
        problem[r,c] = digit+1

    return problem, inx

def sudokuArrayCreate(imgOri, model):
    if isinstance(imgOri, str):
        img = cv2.imread(imgOri)

    imgs, _, _ = imageCreate(img)
    return digitClassification(model, imgs)

#put texts in the image to be an output image
def putTextSudoku(imgOri, problem, inx):
    if isinstance(imgOri, str):
        img = cv2.imread(imgOri)

    imgs, linesRow, linesCol = imageCreate(img)

    width = np.mean(np.diff(linesCol))
    height = np.mean(np.diff(linesRow))

    for i in range(len(imgs)):
        if not i in inx:
            r = math.floor(i/9)
            c = i%9

            org = (int(linesCol[c]+(width/3)), int(linesRow[r]+((2*height)/3)))
            img = cv2.putText(img, "%d"%(problem[r,c]), org, cv2.FONT_HERSHEY_SIMPLEX, 
                                width/60, (255, 0, 0), 2, cv2.LINE_AA)

    return img

#main function return 2 outputs: array of answer, and an image in BGR format
def sudokuSolver(imgOri, model='Default'):
    if model=='Default': model = tf.keras.models.load_model('digitModel.h5')
    else: model = tf.keras.models.load_model(model)

    problem, inx = sudokuArrayCreate(imgOri, model)
    problem = sudokuSolverAlgorithm(problem)
    img = putTextSudoku(imgOri, problem, inx)

    return problem, img

if __name__=="__main__":
    print('test')
    print(tf.__version__)