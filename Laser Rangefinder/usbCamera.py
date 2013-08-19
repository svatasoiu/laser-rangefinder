import cv2.cv as cv
import pyopencl as cl
import numpy
import Image
import statsFunctions as stats

TRIALRUNS = 5

print("Loading constants...")
constantFile = open('constants.txt','r')
c1 = float(constantFile.readline())
c2 = float(constantFile.readline())
constantFile.close()
print("Done!")

print("")

#constants for difference = 13.5cm
#c1 = 10567.
#c2 = -8.757

print("Loading kernel from findLaser.cl ...")
inputFile = open('findLaser.cl','r')
kernelArr = inputFile.readlines()
kernel = ""
for line in kernelArr:
    kernel += str(line)
inputFile.close()
print("Done!")

original = "Original"
HSVWindow = "HSV Image"
filtered = "Filtered Image"

def onMouseMove(event, x, y, flags, param):
    print("X Position = " + str(x) + " Y Position = " + str(y))
    imgHSV = param[1][y][x]
    print("hue: " + str(imgHSV[0]) + " saturation: " + str(imgHSV[1]) + " value: " + str(imgHSV[2]))

    img = param[2][y][x]
    print("blue: " + str(img[0]) + " green: " + str(img[1]) + " red: " + str(img[2]))

cv.NamedWindow(original, cv.CV_WINDOW_AUTOSIZE) #640 x 480
cv.NamedWindow(HSVWindow, cv.CV_WINDOW_AUTOSIZE)
#cv.NamedWindow(filtered, cv.CV_WINDOW_AUTOSIZE)
camera_index = 1
capture = cv.CaptureFromCAM(camera_index)

def GetThresholdedImage(imgHSV):        
    imgThresh=cv.CreateImage(cv.GetSize(imgHSV),cv.IPL_DEPTH_8U, 1)
    cv.InRangeS(imgHSV, cv.Scalar(170,160,60), cv.Scalar(180,256,256), imgThresh) 
    return imgThresh

for platform in cl.get_platforms():
    for device in platform.get_devices():
        print("===============================================================")
        print("Platform name:", platform.name)
        print("Platform profile:", platform.profile)
        print("Platform vendor:", platform.vendor)
        print("Platform version:", platform.version)
        print("---------------------------------------------------------------")
        print("Device name:", device.name)
        print("Device type:", cl.device_type.to_string(device.type))
        print("Device memory: ", device.global_mem_size//1024//1024, 'MB')
        print("Device max clock speed:", device.max_clock_frequency, 'MHz')
        print("Device compute units:", device.max_compute_units)
        print("Device max work group size:", device.max_work_group_size)

print("Creating some Context...")
ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)
print("Done!")

mf = cl.mem_flags

def parallelSumRed(imgRGBA, width, height):
    global c1
    global c2
    C = 0.
    F = 259.*(C + 255.)/(255.*(259. - C))
    #print(F)

    #Create buffers
    #host -> device
    width_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=numpy.int32(width))
    height_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=numpy.int32(height))
    dest_sum_buf = cl.Buffer(ctx, mf.WRITE_ONLY | mf.COPY_HOST_PTR, hostbuf=numpy.int32(0))
    dest_sumY_buf = cl.Buffer(ctx, mf.WRITE_ONLY | mf.COPY_HOST_PTR, hostbuf=numpy.int32(0))

    dest_N_buf = cl.Buffer(ctx, mf.WRITE_ONLY | mf.COPY_HOST_PTR, hostbuf=numpy.int32(0))
    F_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=numpy.float32(F))
    
    clImage = cl.Image(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, cl.ImageFormat(cl.channel_order.RGBA, cl.channel_type.UNORM_INT8),
                               (640, 480), None, imgRGBA.tostring() )
    clOutImage = cl.Image(ctx, mf.WRITE_ONLY | mf.COPY_HOST_PTR, cl.ImageFormat(cl.channel_order.RGBA, cl.channel_type.UNORM_INT8),
                               (640, 480), None, imgRGBA.tostring() )

    sampler = cl.Sampler(ctx,
                         False, #  Non-normalized coordinates
                         cl.addressing_mode.CLAMP_TO_EDGE,
                         cl.filter_mode.NEAREST)

    #compile openCL code
    prg = cl.Program(ctx, kernel).build()

    #define grid size
    gridSizeX = 640
    gridSizeY = 480
    
    globalWorkSize = (gridSizeX, gridSizeY)

    #run kernel
    prg.getLaserCoord(queue, globalWorkSize, 
            clImage, clOutImage, sampler, width_buf, height_buf, dest_sum_buf, dest_N_buf, dest_sumY_buf) #can't use Intel CPU for now, need to install NVidia drivers; use AMD for now

    #set up output buffers
    sumX = numpy.empty_like(0)
    sumY = numpy.empty_like(0)
    N = numpy.empty_like(0)
    buff = numpy.zeros(width * height * 4, numpy.uint8) #output is numpy array of (640, 480, 4); need to convert to RGBA -> RGB -> BGR and then display
    origin = (0,0,0)
    region = (width, height,1)

    #device -> host
    cl.enqueue_copy(queue, sumX, dest_sum_buf) #from 3rd arg on device to 2nd arg on host
    cl.enqueue_copy(queue, N, dest_N_buf)
    cl.enqueue_copy(queue, sumY, dest_sumY_buf)
    
    cl.enqueue_read_image(queue, clOutImage, origin, region, buff).wait()
    
    #print("N = " + str(N) + "; SumX = " + str(sumX) + "; SumY = " + str(sumY))
    
    #print(buff) #remember that every fourth value is alpha = 255
    offsetX = 0
    offsetY = 0
    
    if N!=0:
        print("LASER (x,y) = (" + str(sumX/N) + ", " + str(sumY/N) + ")")

    if N>5:
        offsetX = sumX/N-320.
        offsetY = sumY/N-240.

    return (buff, int(offsetX), int(offsetY))

def repeat():
    global capture #declare as globals since we are assigning to them now
    global camera_index
    global done
    
    frame = cv.QueryFrame(capture)
    cv.Smooth(frame, frame, cv.CV_GAUSSIAN,3,3)

    imgHsv = cv.CreateImage(cv.GetSize(frame),8, 3)
    cv.CvtColor(frame, imgHsv, cv.CV_BGR2HSV)
    #imgHsv2 = GetThresholdedImage(imgHsv)
    #print(numpy.asarray(cv.GetMat(imgHsv)))

    imgRGBA = cv.CreateImage(cv.GetSize(frame),8, 4)
    cv.CvtColor(frame, imgRGBA, cv.CV_BGR2RGBA)
    
    cv.Smooth(imgRGBA, imgRGBA, cv.CV_GAUSSIAN,3,3)
    (filteredImg,offsetX,offsetY) = parallelSumRed(imgRGBA, 640, 480) #3D array
    
    d = numpy.sqrt(offsetX*offsetX + offsetY*offsetY)

    if d!=0:    
        print("Distance = " + str(c1/d+c2) + "cm")
        print("OffsetX = " +str(offsetX) + "; OffsetY = " + str(offsetY))
        print("")
    
    imgRGB = cv.CreateImage(cv.GetSize(frame), 8, 3)
    #cv.CvtColor(Image.fromarray(filteredImg), imgRGB, cv.CV_RGBA2RGB)

    imgRGBA = cv.fromarray(numpy.reshape(filteredImg, (480, 640, 4)))
    if offsetX!=0 or offsetY!=0:
        cv.Rectangle(imgRGBA, (320+offsetX-6,240+offsetY-6), (320+offsetX+6, 240+offsetY+6), (255,0,255,255), 1,8)
        cv.Line(imgRGBA, (0,240+offsetY), (639,240+offsetY),  (255,0,255,255), 1,8)
        cv.Line(imgRGBA, (320+offsetX,0), (320+offsetX,479),  (255,0,255,255), 1,8)
    
    cv.ShowImage(HSVWindow, imgRGBA)
    cv.ShowImage(original, frame)
    
    cv.SetMouseCallback(original, onMouseMove, [cv.CV_EVENT_MOUSEMOVE, numpy.asarray(cv.GetMat(imgHsv)), numpy.asarray(cv.GetMat(frame))])
    #cv.SetMouseCallback(HSVWindow, onMouseMove, [cv.CV_EVENT_MOUSEMOVE, numpy.asarray(cv.GetMat(imgHsv)), numpy.asarray(cv.GetMat(frame))])

    #cv.ShowImage(filtered, imgHsv2)
    c = cv.WaitKey(10)
    
    if (str(c)=="27"): #if ESC is pressed
        print("Thank You!")
        done = True
    if (str(c)=="99"): #'c' for calibration
        calibration(int(input("How many data points: ")))

def calibration(N): #N = number of calibration points
    global c1
    global c2
    Darr = []
    offArr = []
    dataArr = []
    i = 0
    while (i < N):
        D = int(input("Distance in CM: "))
        (offset,d) = findAverageOffset(TRIALRUNS)
        dataArr.append(d)
        print("Offset at " + str(D) + "CM = " + str(offset) + " pixels")
        if raw_input("Keep (y/n)? ") == "y":
            if offset != 0:
                Darr.append(D)
                offArr.append(offset)
                i += 1
        print("")
    (c1, c2, R2) = stats.fitInverseFunction(numpy.array(Darr), numpy.array(offArr)) # d = c1/offset + c2
    writeDataToFile("constants.txt", c1, c2, R2, Darr, offArr, dataArr)

def writeDataToFile(filename, c1, c2, R2, Darr, offArr, data):
    dataFile = open(filename, 'w')
    dataFile.write(str(c1) + '\n' + str(c2) + '\n' + str(R2))
    dataFile.write('\n\n')

    for i in range(len(Darr)):
        dataFile.write("Data Run #" + str(i+1) + '\n')
        dataFile.write("_______________________\n")
        for j in range(TRIALRUNS - 1):
            dataFile.write("Distance: " + str(Darr[i]) + "cm | Offset: " + str(data[i][j]) + " pixels\n")
        dataFile.write("Distance: " + str(Darr[i]) + "cm | Average Offset: " + str(offArr[i]) + " pixels\n")
        dataFile.write("_______________________\n\n")
    dataFile.close()
        

def findAverageOffset(N):
    s = 0
    data = []
    for i in range(N):
        o = findOffsetInOneFrame()
        if i != 0:
            s += o
            data.append(o)
    return (s/(N-1),data)
 
def findOffsetInOneFrame():
    global capture #declare as globals since we are assigning to them now
    global camera_index
    
    frame = cv.QueryFrame(capture)
    cv.Smooth(frame, frame, cv.CV_GAUSSIAN,3,3)

    imgHsv = cv.CreateImage(cv.GetSize(frame),8, 3)
    cv.CvtColor(frame, imgHsv, cv.CV_BGR2HSV)
    #imgHsv2 = GetThresholdedImage(imgHsv)
    #print(numpy.asarray(cv.GetMat(imgHsv)))

    imgRGBA = cv.CreateImage(cv.GetSize(frame),8, 4)
    cv.CvtColor(frame, imgRGBA, cv.CV_BGR2RGBA)
    
    cv.Smooth(imgRGBA, imgRGBA, cv.CV_GAUSSIAN,3,3)
    (filteredImg,offsetX,offsetY) = parallelSumRed(imgRGBA, 640, 480) #3D array

    return numpy.sqrt(offsetX*offsetX + offsetY*offsetY) #d

#calibration(5)
done = False
while (done == False):
    repeat()
    
cv.DestroyAllWindows()

##frame = cv.QueryFrame(capture)
##cv.Smooth(frame, frame, cv.CV_GAUSSIAN,3,3)
##imgHsv = cv.CreateImage(cv.GetSize(frame),8, 3)
##cv.CvtColor(frame, imgHsv, cv.CV_BGR2HSV)
##
##mat = numpy.asarray(cv.GetMat(imgHsv))
##parallelSumRed(mat, 640, 480)
