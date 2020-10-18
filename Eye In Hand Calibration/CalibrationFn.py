# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 09:55:48 2018

@author: Joe
"""

from __future__ import print_function
import cv2
import numpy as np
import os
import glob

class CCalibration:

    def __init__(self):
        self.m_srcPoints = [] # 在图像平面的二维点
        self.m_dstPoints = [] # 在世界坐标系中的三维点
    
    def getChessboards(self, Ch_name):
        self.file_num = len([name for name in os.listdir(Ch_name) if os.path.isfile(os.path.join(Ch_name, name))])
        
    def setBorderSize(self, borderSize):
        #棋盘格模板规格
        self.w, self.h = borderSize
    
    def addChessboardPoints(self, Document):        
        # 世界坐标系中的棋盘格点,例如(0,0,0), (1,0,0), (2,0,0) ....,(8,5,0)，去掉Z坐标，记为二维矩阵
        dstCandidateCorners = np.zeros((self.w*self.h, 3), np.float32)
        dstCandidateCorners[:,:2] = np.mgrid[0:self.w, 0:self.h].T.reshape(-1,2)*10
        
        for i in range(self.file_num):
            name = Document + str(i) + ".bmp"
            img = cv2.imread(name)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # 找到棋盘格角点
            found, srcCandidateCorners = cv2.findChessboardCorners(gray, (self.w,self.h), None)
            # 如果找到足够点对，将其存储起来
            if(found):
                # 阈值
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 300, 0.001)
                srcCandidateCorners = cv2.cornerSubPix(gray, srcCandidateCorners, (5,5), (-1,-1), criteria)
            if(srcCandidateCorners.shape[0] == self.w*self.h):
                self.m_srcPoints.append(srcCandidateCorners)
                self.m_dstPoints.append(dstCandidateCorners)
                print(name)
            else:
                print(name + "is not found.")   
            # 将角点在图像上显示
            cv2.drawChessboardCorners(img, (self.w,self.h), srcCandidateCorners, found)
#            cv2.imshow('findCorners',gray)
            cv2.imshow('findCorners',img)
            cv2.waitKey(200)
        cv2.destroyAllWindows()
        
    def calibrate(self, gray, pos_of_Ex, pos_of_In, Ex_name, In_name, Dis_name):
        ret, self.cameraMatrix, self.distCoeffs, rvecs, tvecs = cv2.calibrateCamera(self.m_dstPoints, self.m_srcPoints, (gray.shape[1],gray.shape[0]), None, None)
        print("單眼校正物差:" + str(ret))
        #新建xml文档对象
        myExtrinsic = cv2.FileStorage(pos_of_Ex, cv2.FILE_STORAGE_WRITE)
        myIntrinsic = cv2.FileStorage(pos_of_In, cv2.FILE_STORAGE_WRITE)
        for i in range(len(rvecs)):
            rotation = cv2.Rodrigues(rvecs[i])[0]
            Extrinsic = cv2.hconcat((rotation, tvecs[i]))
            myExtrinsic.write(Ex_name + str(i), Extrinsic)
        myExtrinsic.release()  
        
        myIntrinsic.write(In_name, self.cameraMatrix)
        myIntrinsic.write(Dis_name, self.distCoeffs)
        myIntrinsic.release()
 
    def drawCbCorners(self, Document, pos_of_Ex, pos_of_In, Ex_name, In_name, outFile):
        myExtrinsic = cv2.FileStorage(pos_of_Ex, cv2.FILE_STORAGE_READ)
        myIntrinsic = cv2.FileStorage(pos_of_In, cv2.FILE_STORAGE_READ)
        
        Intrinsic = myIntrinsic.getNode(In_name).mat()
        for n in range(self.file_num):
            Extrinsic = myExtrinsic.getNode(Ex_name+str(n)).mat()
            name = Document + str(n) + ".bmp"
            image = cv2.imread(name) 
            
            # 畫角點
            for i in range(self.w):
                for j in range(self.h):
                    P = np.reshape(np.array([i*10, j*10, 0, 1]), (4,1))
                    camera_axit = np.dot(Extrinsic, P)
                    Z = camera_axit[2]
                    image_axit_P = (1/Z) * np.dot(Intrinsic,camera_axit)
                    cv2.circle(image, (image_axit_P[0],image_axit_P[1]), 3, (0,0,255), -1)
            
            # 畫軸
            world_axit_O = np.reshape(np.array([0, 0, 0, 1]), (4,1))
            world_axit_X = np.reshape(np.array([10, 0, 0, 1]), (4,1))
            world_axit_Y = np.reshape(np.array([0, 10, 0, 1]), (4,1))
            world_axit_Z = np.reshape(np.array([0, 0, 10, 1]), (4,1))
            
            camera_axit = np.dot(Extrinsic, world_axit_O)
            Z = camera_axit[2]
            image_axit_O = (1/Z) * np.dot(Intrinsic,camera_axit)         
            camera_axit = np.dot(Extrinsic, world_axit_X)
            Z = camera_axit[2]
            image_axit_X = (1/Z) * np.dot(Intrinsic,camera_axit)           
            camera_axit = np.dot(Extrinsic, world_axit_Y)
            Z = camera_axit[2]
            image_axit_Y = (1/Z) * np.dot(Intrinsic,camera_axit)  
            camera_axit = np.dot(Extrinsic, world_axit_Z)
            Z = camera_axit[2]
            image_axit_Z = (1/Z) * np.dot(Intrinsic,camera_axit)
            
            cv2.line(image, (image_axit_O[0], image_axit_O[1]), (image_axit_X[0], image_axit_X[1]), (0, 0, 255), 5)
            cv2.line(image, (image_axit_O[0], image_axit_O[1]), (image_axit_Y[0], image_axit_Y[1]), (0, 255, 0), 5)
            cv2.line(image, (image_axit_O[0], image_axit_O[1]), (image_axit_Z[0], image_axit_Z[1]), (255, 0, 0), 5)
            
            cv2.imwrite(outFile + str(n) + ".bmp", image)
            cv2.imshow('CheckCorners', image)
            cv2.waitKey(200)        
        cv2.destroyAllWindows()       
        
    def StereoCalibration(self, m_dstPoints_L, m_srcPoints_L, m_srcPoints_R, cameraMatrix_L, distCoeffs_L, cameraMatrix_R, distCoeffs_R):
        ref_L = cv2.imread("Chessboards/Chessboards_L/ChessboardL_0.bmp",0)
        ref_R = cv2.imread("Chessboards/Chessboards_R/ChessboardR_0.bmp",0)
        myStereoParameter = cv2.FileStorage("data/StereoParameter.xml", cv2.FILE_STORAGE_WRITE)
    
        retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = cv2.stereoCalibrate(m_dstPoints_L, m_srcPoints_L, m_srcPoints_R,
                                                                                                      cameraMatrix_L, distCoeffs_L, 
                                                                                                      cameraMatrix_R, distCoeffs_R,
                                                                                                      ref_L.shape, criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1000, 1e-6), flags = cv2.CALIB_FIX_INTRINSIC)
        print("雙眼校正物差:" + str(retval))
        Baseline = np.linalg.norm(T)
        
        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, ref_L.shape, R, T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=0, newImageSize=ref_L.shape)
    
        map1, map2 = cv2.initUndistortRectifyMap(cameraMatrix_L, distCoeffs_L, R1, P1, ref_L.shape, cv2.CV_32FC1)
        map3, map4 = cv2.initUndistortRectifyMap(cameraMatrix_R, distCoeffs_R, R2, P2, ref_R.shape, cv2.CV_32FC1)
        
        myStereoParameter.write("focal",P1[0,0])
        myStereoParameter.write("Baseline",Baseline)
        myStereoParameter.write("R",R)
        myStereoParameter.write("T",T)
        myStereoParameter.write("map1",map1)
        myStereoParameter.write("map2",map2)
        myStereoParameter.write("map3",map3)
        myStereoParameter.write("map4",map4)
        myStereoParameter.write("P1",P1)
        myStereoParameter.write("P2",P2)
    	
        print ("focal = "+str(P1[0,0])+"  Baseline = "+str(Baseline))
        print("stereoCalibration is finished.")
        myStereoParameter.release();
#        im1 = cv2.imread("Chessboards/Chessboards_L/ChessboardL_0.bmp")
#        im2 = cv2.imread("Chessboards/Chessboards_R/ChessboardR_0.bmp")
#        img1 = cv2.remap(im1, map1, map2, interpolation=cv2.INTER_LINEAR)
#        img2 = cv2.remap(im2, map3, map4, interpolation=cv2.INTER_LINEAR)
#        im1 = cv2.resize(im1, (416,416))
#        img1 = cv2.resize(img1, (416,416))
#        im2 = cv2.resize(im2, (416,416))
#        img2 = cv2.resize(img2, (416,416))
#        cv2.imshow("unrectify1", im1)
#        cv2.imshow("rectify1", img1)
#        cv2.imshow("unrectify2", im2)
#        cv2.imshow("rectify2", img2)
#        cv2.waitKey(0)
#        cv2.destroyAllWindows()
    
        