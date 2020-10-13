import numpy as np
from PIL import Image
import math


def HistogramEqualization(array):
    # array: 2-dimension
    depth = 256
    histogram = np.zeros((depth,), dtype=np.int)
    f = np.zeros((depth,), dtype=np.int)
    pixelCount = np.prod(array.shape)
    for i in array.flat:
        histogram[i] += 1

    sum = 0
    factor = depth / pixelCount

    for i in range(depth):
        sum += histogram[i]
        f[i] = factor * sum

    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            array[i, j] = f[array[i, j]]

    return array


def HistogramEqualizationFunction(array):
    depth = 256
    histogram = np.zeros((depth,), dtype=np.int)
    f = np.zeros((depth,), dtype=np.int)
    pixelCount = np.prod(array.shape)
    for i in array.flat:
        histogram[i] += 1

    sum = 0
    factor = depth / pixelCount

    for i in range(depth):
        sum += histogram[i]
        f[i] = factor * sum
    return f


def mirroring(array, i_in, j_in):
    i_out = i_in
    j_out = j_in
    # if i_in < 0:
    #     i_out = -i_in
    # if j_in < 0:
    #     j_out = -j_in
    if i_in >= array.shape[0]:
        i_out = array.shape[0] - (i_in + 1 - array.shape[0])
    if j_in >= array.shape[1]:
        j_out = array.shape[1] - (j_in + 1 - array.shape[1])
    return array[i_out,j_out]


def padding(array, windowSize):
    scale0 = math.ceil(array.shape[0] / windowSize)
    scale1 = math.ceil(array.shape[1] / windowSize)

    newArray = np.zeros((scale0 * windowSize, scale1 * windowSize,), dtype=np.int)

    for i in range(scale0 * windowSize):
        for j in range(scale1 * windowSize):
            newArray[i,j] = mirroring(array, i, j)
    return newArray


def linearInterpolation(f0, f0location, f1, f1location, targetLocation, array):

    x0=float(f0location[0])
    y0=float(f0location[1])
    x1=float(f1location[0])
    y1=float(f1location[1])
    x=float(targetLocation[0])
    y=float(targetLocation[1])
    xint=int(x)
    yint=int(y)

    if x0==x1:
        alpha=(y-y0)/(y1-y0)
        return int((1-alpha)*f0[array[xint,yint]]+alpha*f1[array[xint,yint]])
    else:
        alpha=(x-x0)/(x1-x0)
        return int((1-alpha)*f0[array[xint,yint]]+alpha*f1[array[xint,yint]])


def bilinearInterpolation(f00, f00location, f01, f01location, f10, f10location, f11, f11location, targetLocation,
                          array):
    x1=float(f00location[0])
    x2=float(f11location[0])

    y1=float(f00location[1])
    y2=float(f11location[1])

    x=float(targetLocation[0])
    y=float(targetLocation[1])

    val1=((x2-x)*(y2-y))/((x2-x1)*(y2-y1))*f00[array[targetLocation[0],targetLocation[1]]]
    val2=((x-x1)*(y2-y))/((x2-x1)*(y2-y1))*f10[array[targetLocation[0],targetLocation[1]]]
    val3=((x2-x)*(y-y1))/((x2-x1)*(y2-y1))*f01[array[targetLocation[0],targetLocation[1]]]
    val4=((x-x1)*(y-y1))/((x2-x1)*(y2-y1))*f11[array[targetLocation[0],targetLocation[1]]]

    return (int)(val1+val2+val3+val4)


def atCorner(i, j, shape, halfWindowSize):
    if i <= halfWindowSize and j <= halfWindowSize:
        return True
    if i >= shape[0] - halfWindowSize and j <= halfWindowSize:
        return True
    if i <= halfWindowSize and j >= shape[1] - halfWindowSize:
        return True
    if i >= shape[0] - halfWindowSize and j >= shape[1] - halfWindowSize:
        return True
    return False


def atXEdge(i, j, shape, halfWindowSize):
    if j <= halfWindowSize or j >= shape[1] - halfWindowSize:
        return True
    return False


def atYEdge(i, j, shape, halfWindowSize):
    if i <= halfWindowSize or i >= shape[0] - halfWindowSize:
        return True
    return False


def getLocation(f_i, f_j, windowSize):
    halfWindowSize = windowSize / 2
    return [int(f_i * windowSize + halfWindowSize), int(f_j * windowSize + halfWindowSize)]


# ADE
def adaptiveHistogramEqualization(array, windowSize):
    halfWindowSize = windowSize / 2
    originalShape = array.shape

    depth = 256

    # padding
    newArray = padding(array, windowSize)
    backupArray=newArray.copy()

    # get all f
    # 3-dimension. hard to understand.
    windowNum0 = round(newArray.shape[0] / windowSize)
    windowNum1 = round(newArray.shape[1] / windowSize)
    functionList = np.zeros((windowNum0, windowNum1, depth), dtype=np.int)
    for i in range(windowNum0):
        for j in range(windowNum1):
            functionList[i, j, :] = HistogramEqualizationFunction(
                newArray[i * windowSize:(i + 1) * windowSize, j * windowSize:(j + 1) * windowSize])

    # calculate corner case and normal case
    for i in range(newArray.shape[0]):
        for j in range(newArray.shape[1]):
            print(i,j)
            ibias = i - halfWindowSize
            jbias = j - halfWindowSize
            if atCorner(i, j, newArray.shape, halfWindowSize):
                newArray[i, j] = functionList[math.floor(i / windowSize), math.floor(j / windowSize), backupArray[i, j]]
            elif atXEdge(i, j, newArray.shape, halfWindowSize):
                f0 = functionList[math.floor(ibias / windowSize), math.floor(j / windowSize)]
                f0location = getLocation(math.floor(ibias / windowSize), math.floor(j / windowSize), windowSize)
                f1 = functionList[math.floor(ibias / windowSize) + 1, math.floor(j / windowSize)]
                f1location = getLocation(math.floor(ibias / windowSize) + 1, math.floor(j / windowSize), windowSize)
                newArray[i, j] = linearInterpolation(f0, f0location, f1, f1location, [i, j], backupArray)
                # print(newArray[i, j])

            elif atYEdge(i, j, newArray.shape, halfWindowSize):
                f0 = functionList[math.floor(i / windowSize), math.floor(jbias / windowSize)]
                f0location = getLocation(math.floor(i / windowSize), math.floor(jbias / windowSize), windowSize)
                f1 = functionList[math.floor(i / windowSize), math.floor(jbias / windowSize) + 1]
                f1location = getLocation(math.floor(i / windowSize), math.floor(jbias / windowSize) + 1, windowSize)
                newArray[i, j] = linearInterpolation(f0, f0location, f1, f1location, [i, j], backupArray)
                # print(newArray[i,j])

            else:
                f00 = functionList[math.floor(ibias / windowSize), math.floor(jbias / windowSize)]
                f00location = getLocation(math.floor(ibias / windowSize), math.floor(jbias / windowSize), windowSize)
                f01 = functionList[math.floor(ibias / windowSize), math.floor(jbias / windowSize) + 1]
                f01location = getLocation(math.floor(ibias / windowSize), math.floor(jbias / windowSize) + 1,
                                          windowSize)
                f10 = functionList[math.floor(ibias / windowSize) + 1, math.floor(jbias / windowSize)]
                f10location = getLocation(math.floor(ibias / windowSize) + 1, math.floor(jbias / windowSize),
                                          windowSize)
                f11 = functionList[math.floor(ibias / windowSize) + 1, math.floor(jbias / windowSize) + 1]
                f11location = getLocation(math.floor(ibias / windowSize) + 1, math.floor(jbias / windowSize) + 1,
                                          windowSize)
                newArray[i, j] = bilinearInterpolation(f00, f00location, f01, f01location, f10, f10location, f11,
                                                       f11location, [i, j], backupArray)
                # print(newArray[i, j])


    # cut it back
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            array[i, j] = newArray[i, j]
            if newArray[i,j]>=256:
                # maybe cuz interpolation and float
                array[i,j]=255


if __name__ == "__main__":
    fileAddr=input()
    # fileAddr = "../asset/image/building.jpg"
    fileName = fileAddr.split("/")[-1].split(".")[0]
    rawImage = Image.open(fileAddr)
    imageArray = np.array(rawImage, dtype=np.int)

    if len(imageArray.shape) == 3:
        # color
        for i in range(3):
            adaptiveHistogramEqualization(imageArray[:,:,i],256)
    else:
        # gray
        adaptiveHistogramEqualization(imageArray,16)



    newImage = Image.fromarray(np.uint8(imageArray))
    newImage.save("../asset/image/" + fileName + "_modified.jpg")
    print("save success")
