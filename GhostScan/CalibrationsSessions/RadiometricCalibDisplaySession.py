from abc import ABC
from GhostScan.CalibrationSession import CalibrationSession
from GhostScan.Calibrations import RadiometricCalibration
import cv2
import os, sys, time, math
import numpy as np
import matplotlib.pyplot as plt
import os
DISPLAY_HEIGHT = 1080
DISPLAY_WIDTH = 1920

class RadiometricCalibDisplaySession(CalibrationSession, ABC):
    def __init__(self, camera, radio_calib, path='CalibrationImages/DisplayRadiometric', exposure=60000, numPhaseShift = 4):
        # Set camera, destination path
        self.camera = camera
        self.path = path
        # self.g = None
        self.radio_calib = radio_calib
        self.g = self.radio_calib.load_calibration_data()
        self.exposure = exposure
        self.camera.setExposure(exposure)
        self.NumPatterns = 256
        self.displayWidth = DISPLAY_WIDTH
        self.displayHeight = DISPLAY_HEIGHT
        self.setDefPattern()
        self.sequenceNum = 2 * numPhaseShift
        
        if not os.path.exists('CapturedImages/'):
            os.mkdir('CapturedImages/')
        if not os.path.exists('CapturedImages/sequenceImages/'):
            os.mkdir('CapturedImages/sequenceImages/')
        if not os.path.exists('CapturedImages/sequenceImages/undistortRadioCalib/'):
            os.mkdir('CapturedImages/sequenceImages/undistortRadioCalib/')
        if not os.path.exists('CapturedImages/sequenceImages/undistortRadioCalib/radioCalibResults/'):
            os.mkdir('CapturedImages/sequenceImages/undistortRadioCalib/radioCalibResults/')
        if not os.path.exists('CapturedImages/sequenceImages/undistortRadioCalib/undistortResults/'):
            os.mkdir('CapturedImages/sequenceImages/undistortRadioCalib/undistortResults/')
        if not os.path.exists('CalibrationImages/RadiometricTest/'):
            os.mkdir('CalibrationImages/RadiometricTest/')
        if not os.path.exists('CalibrationNumpyData/RadiometricTest/'):
            os.mkdir('CalibrationNumpyData/RadiometricTest/')
        if not os.path.exists('CalibrationImages/DisplayRadiometric'):
            os.mkdir('CalibrationImages/DisplayRadiometric')
            

        

    def calculateDisplayCalibration(self):
        
        pixelValues = self.avgImagePixel()

        DisplayToRadianceData = []

        for pixelValue in pixelValues:
            num = math.exp(self.g[round(pixelValue)] - math.log(self.exposure))
            DisplayToRadianceData.append(num)

        imgSave = 'CapturedImages/sequenceImages/undistortRadioCalib/radioCalibResults'
            
        fig = plt.figure()
        plt.title("Display to Relative Radiance Curve")
        plt.xlabel('Display pixel value')
        plt.ylabel('Relative Radiance')
        plt.plot(pixelValues, DisplayToRadianceData, 'k')
        # plt.show()
        fig.savefig(imgSave + '/Display to Relative Radiance Curve' + '.png')
        # fig.savefig('Display to Relative Radiance Curve.png')
        return DisplayToRadianceData, pixelValues

    def calUpValue(self, imgPath):
        img = cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE)
        img = np.array(img, dtype='uint8')
        row, col = img.shape[0], img.shape[1]
        arr = []
        for i in range(row):
            cols = []
            for j in range(col):
                val = math.exp(self.g[img[i][j]] - math.log(self.exposure))
                cols.append(val)
            arr.append(cols)
        return np.array(arr)


    def calculateReflectivity(self):
        imgPath = 'CapturedImages/sequenceImages/RadioDisplayTest/0.png'
        arr = self.calUpValue(imgPath)

        # img = cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE)
        # img = np.array(img, dtype='uint8')
        # row, col = img.shape[0], img.shape[1]

        DisplayToRadianceData, pixelValues = self.calculateDisplayCalibration()
        DisplayToRadianceValue = DisplayToRadianceData[255]
        
        # arr = []
        # for i in range(row):
        #     cols = []
        #     for j in range(col):
        #         val = math.exp(self.g[img[i][j]] - math.log(self.exposure)) / DisplayToRadianceValue
        #         cols.append(val)
        #     arr.append(cols)
        # arr = np.array(arr)
        # print(arr)
        
        arr = [[j/DisplayToRadianceValue for j in i] for i in arr]
        imgSave = 'CapturedImages/sequenceImages/undistortRadioCalib/undistortResults'
        fig = plt.figure()
        plt.imshow(arr, cmap='viridis')
        plt.colorbar()
        # plt.show()
        fig.savefig(imgSave + '/Reflectivity Map' + '.png')
        print('Reflectivity Calculation Successful! Reconstructed images viewable at CapturedImages/sequenceImages/undistortRadioCalib/undistortResults')
        # plt.savefig('Reflectivity Map.png')

        Reflectivity = np.array(arr)
        return Reflectivity, DisplayToRadianceData, pixelValues

    def radiometricCalibUndistortImages(self):
        Reflectivity = self.calculateReflectivity()
        imgSeqPath = 'CapturedImages/sequenceImages/RadioDisplayTest/'
        # sequenceNum = 8
        for i in range(self.sequenceNum):
            upValue = self.calUpValue(imgSeqPath + str(i) + '.png')
            # print(upValue)
            radiance = np.divide(upValue, Reflectivity)
            # print(radiance)
            # print(radiance.shape[0], radiance.shape[1])
            correctedImage = self.imageCorrection(radiance)
            # print(correctedImage)
            cv2.imwrite('CapturedImages/sequenceImages/undistortRadioCalib/undistortResults/' + str(i) + '.png', correctedImage)

    def imageCorrection(self, radiance):
        DisplayToRadianceData = self.calculateDisplayCalibration()
        row, col = radiance.shape[0], radiance.shape[1]
        imgCorr = []
        for i in range(row):
            cols = []
            for j in range(col):
                t = np.array(abs(DisplayToRadianceData - radiance[i][j]))
                t_min = np.min(t)
                tt = self.findPixelValue(t_min, t)
                cols.append(tt)
            imgCorr.append(cols)
        return np.array(imgCorr)

    def findPixelValue(self, value, arr):
        for i in range(len(arr)):
            if arr[i] == value:
                return i


    def setPatternScale(self):
        patternSize = min(self.displayHeight, self.displayWidth)
        boarderSize = math.floor((max(self.displayHeight, self.displayWidth) - patternSize)/2)

        self.patterns = np.zeros((self.displayHeight, self.displayWidth, self.NumPatterns), dtype=np.float32)
        
        return patternSize, boarderSize

    def setDefPattern(self):
        #patternSize, boarderSize = self.setPatternScale(2)
        patternSize, boarderSize = self.setPatternScale()

        patternSize = max(self.displayHeight, self.displayWidth)
        #patternSize = min(self.displayHeight, self.displayWidth)
        
        for i in range(self.NumPatterns):
            self.patterns[..., i] = i * np.ones((DISPLAY_HEIGHT,DISPLAY_WIDTH))


    def captureImages(self):
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        cv2.waitKey(1)
        
        window_name = 'projector'
        cv2.namedWindow(window_name)
        cv2.moveWindow(window_name, self.displayWidth, 0)  #self.displayWidth
            
        
        if not os.path.exists('CalibrationNumpyData/'):
            os.mkdir('CalibrationNumpyData/')
        if not os.path.exists('CalibrationNumpyData/DisplayRadiometric/'):
            os.mkdir('CalibrationNumpyData/DisplayRadiometric/')
       
        for i in range(self.NumPatterns):
            if i % 10 == 0:
                print('picture taken:' + str(i))

            cur_img = self.patterns[... ,i]
            cv2.imshow(window_name, self.patterns[..., i].astype(np.uint8))
            cv2.waitKey(40)
            self.camera.getImage(name = 'DisplayRadiometric/'+str(i), calibration=True)
            
        print('Capture Complete! Numpy data viewable at CalibrationImages/DisplayRadiometric/')    
        cv2.waitKey(1000)
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        cv2.waitKey(1)
        
    def captureTest(self):
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        cv2.waitKey(1)
        
        window_name = 'projector'
        cv2.namedWindow(window_name)
        cv2.moveWindow(window_name, self.displayWidth, 0)  #self.displayWidth
            
        
        if not os.path.exists('CalibrationNumpyData/'):
            os.mkdir('CalibrationNumpyData/')
        if not os.path.exists('CalibrationImages/RadiometricTest/'):
            os.mkdir('CalibrationImages/RadiometricTest/')
       
        pattern = np.ones((DISPLAY_HEIGHT,DISPLAY_WIDTH))
        for i in range(DISPLAY_WIDTH):
            pattern[:,i] = i / DISPLAY_WIDTH * 255
        
        cv2.imshow(window_name, pattern.astype(np.uint8))
        cv2.waitKey(40)
        self.camera.getImage(name = 'RadiometricTest/'+ 'testImage', calibration=True)
        
        pattern = np.ones((DISPLAY_HEIGHT,DISPLAY_WIDTH)) * 255
        cv2.imshow(window_name, pattern.astype(np.uint8))
        cv2.waitKey(40)
        self.camera.getImage(name = 'RadiometricTest/'+ 'testImageDim', calibration=True)
        
        pattern = np.ones((DISPLAY_HEIGHT,DISPLAY_WIDTH)) * 128
        cv2.imshow(window_name, pattern.astype(np.uint8))
        cv2.waitKey(40)
        self.camera.getImage(name = 'RadiometricTest/'+ 'vignettingcalibration', calibration=True)
        
        print('Capture Complete! Test image viewable at CalibrationImages/RadiometricTest/')
        cv2.waitKey(1000)
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        cv2.waitKey(1) 
        
        
        
    def avgImagePixel(self):
        pixelValues = []
        files = []
        # Load images
        
        for file in os.listdir(self.path):
            # Only use .raw files(Balser)
            #if file.endswith(".raw") or file.endswith(".Raw") or file.endswith(".RAW"):
            #    files.append(file)
            if file.endswith(".png") or file.endswith(".Png") or file.endswith(".PNG"):
                files.append(file)
        # Sort files depending on their exposure time from lowest to highest        
        files.sort(key=lambda x: int(x[:-4]))
        # We used exposure time as filenames

        for filename in files:
            #image = np.fromfile(self.path + '/' + filename, dtype=np.uint8)
            # image = cv2.imread(self.path + '/' + filename, cv2.IMREAD_UNCHANGED);
            image = cv2.imread(self.path + '/' + filename, 0)
            average = image.mean(axis=0).mean(axis=0) 
            pixelValues.append(average)

        x = list(range(256))
        imgSave = 'CapturedImages/sequenceImages/undistortRadioCalib/radioCalibResults'
        fig = plt.figure()
        plt.title("Display Value to Average Pixel Value Curve")
        plt.xlabel('Display pixel value')
        plt.ylabel('Average camera pixel value')
        plt.plot(x, pixelValues)
        # plt.show()
        fig.savefig(imgSave + '/Display Value to Average Pixel Value Curve' + '.png')
        # plt.savefig('Display Value to Average Pixel Value Curve.png')

        return pixelValues


    # def calibrate_HDR(self, smoothness=1000):
    #     # Call radiometric calibration
    #     # This computes and returns the camera response function and calculates a HDR image saved in path as PNG and .np
    #     self.g, le = self.radio_calib.get_camera_response(smoothness)
    #     self.radio_calib.plotCurve("Camera response")
    #     return self.g, le

    def load_calibration(self):
        self.radio_calib.load_calibration_data()

    def calibrate_image(self, exposure, path='CalibrationImages/Distorted'):
        # Calibrate radiometric calibrated images from a single exposure
        # Returns set of undistorted images and corresponding g function
        imgs, g = self.radio_calib.calibrate_image(exposure, path)
        return imgs, g


