import numpy as np
from PIL import Image
from math import ceil, sqrt
from math import floor

import matplotlib.pyplot as plt



def checkByte(a):
    if a > 255:
        a = 255
    if a < 0:
        a = 0
    return a


def svertka(a, b):
    sum = 0
    for i in range(len(a)):
        for j in range(len(a[0])):
            sum += a[i][j] * b[i][j]
    return sum


def median(a):
    c = []
    for i in range(len(a)):
        for j in range(len(a[0])):
            c.append(a[i][j])
    c.sort()
    return c[ceil(len(c) / 2)]


def max(a):
    c = []
    for i in range(len(a)):
        for j in range(len(a[0])):
            c.append(a[i][j])
    c.sort()
    return c[len(c) - 1]


def min(a):
    c = []
    for i in range(len(a)):
        for j in range(len(a[0])):
            c.append(a[i][j])
    c.sort()
    return c[0]


im = Image.open('C:\\Users\\Артем\\PycharmProjects\\net_3\\venv\\new\\кошка 3.jpeg').resize((30, 30))
pixels = im.load()

plt.imshow(im)
plt.show()

imFinal = im.copy()
pixels2 = imFinal.load()
'''
filter = [
    [-1, -1, 0, 0, 0],
    [0, -1, -1, -1, 0],
    [0, -1, 9, -1, 0],
    [0, -1, -1, -1, 0],
    [0, 0, 0, 0, 0]
]'''

filter = [
    [-1, -1, -1],
    [-1, 9, -1],
    [-1, -1, -1]
]
'''
filter = [
    [0, 0, 0, 1, 0, 0, 0],
    [0, 0, 1, 1, 1, 0, 0],
    [0, 1, 1, 1, 1, 1, 0],
    [1, 1, 1, 1, 1, 1, 1],
    [0, 1, 1, 1, 1, 1, 0],
    [0, 0, 1, 1, 1, 0, 0],
    [0, 0, 0, 1, 0, 0, 0]
]
'''

'''
filter = [
    [-1, -1, -1],
    [-1, 9, -1],
    [-1, -1, -1]
]
'''

'''
filter = [
    [0.5, 1.5, 2, 1.5, 0.5],
    [1.5, 3.5, 5, 3.5, 1.5],
    [  2,   5, 10,  5,   2],
    [1.5, 3.5, 5, 3.5, 1.5],
    [0.5, 1.5, 2, 1.5, 0.5]
]'''


div = 0
for i in range(len(filter)):
    for j in range(len(filter[0])):
        div += filter[i][j]
if div == 0:
    div = 1
res_net = []
for i in range(floor(len(filter)/2), im.width - floor(len(filter)/2)):

    for j in range(floor(len(filter)/2), im.height - floor(len(filter)/2)):

        matrR = []
        matrG = []
        matrB = []

        for n in range(-floor(len(filter)/2), ceil(len(filter)/2)):


            rowR = []
            rowG = []
            rowB = []
            for m in range(-floor(len(filter)/2), ceil(len(filter)/2)):

                r, g, b = pixels[i + n, j + m]
                rowR.append(r)
                rowG.append(g)
                rowB.append(b)
            matrR.append(rowR)
            matrG.append(rowG)
            matrB.append(rowB)


        r = checkByte(round(svertka(matrR, filter) / div))
        g = checkByte(round(svertka(matrG, filter) / div))
        b = checkByte(round(svertka(matrB, filter) / div))


       # r = checkByte(min(matrR))
       # g = checkByte(min(matrG))
       # b = checkByte(min(matrB))
        '''
        if r < 512:
            pixels2[i, j] = (255, 255, 255)
        else:
            pixels2[i, j] = (0, 0, 0)'''
        pixels2[i, j] = (r, g, b)
        res_net .append((r+g+b)/ 765.0 * 0.99 + 0.01)

print(res_net)

'''res_net_2 = []
count = 1
for i in range(10):

    n.quary(res_net)
    for j in n.quary(res_net):
        if max(n.quary(res_net))==n.quary(res_net)[2]:
            count+=1
    print(f'еффективность нейросети равна {count}
'''


#plp.grid()
#plp.show()
plt.imshow(imFinal)
plt.show()