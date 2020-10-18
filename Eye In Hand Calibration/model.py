# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import tensorflow as tf
import cv2

class Eye_in_Hand_Calibration():
    def __init__(self):
        self.myExtrinsic = cv2.FileStorage("data/Extrinsic_L.xml", cv2.FILE_STORAGE_READ)
        self.myEEF = cv2.FileStorage("data/forward matrix.xml", cv2.FILE_STORAGE_READ)
        self.TA = []
        self.RA = []
        self.tA = []
        self.TB = []
        self.RB = []
        self.tB = []
        for i in range(999):
            if not self.myExtrinsic.getNode("ExtrinsicL_"+str(i)).empty():
                A = self.myExtrinsic.getNode("ExtrinsicL_"+str(i)).mat()
                self.TA.append(np.linalg.inv(np.concatenate([A, np.reshape([0,0,0,1], [1,4])], axis=0)))
                self.RA.append(self.TA[i][:3,:3].astype(np.float32))
                self.tA.append(self.TA[i][:3,3:].astype(np.float32))
            if not self.myEEF.getNode("EEF_"+str(i)).empty():
                self.TB.append(self.myEEF.getNode("EEF_"+str(i)).mat())
                self.RB.append(self.TB[i][:3,:3].astype(np.float32))
                self.tB.append(self.TB[i][:3,3:].astype(np.float32))
            
    def deg2rad(self, deg):
        return deg * np.pi / 180.0
    
    
    def rad2deg(self, rad):
        return rad * 180.0 / np.pi
    
    def RtXZ(self, q):
        RX = []
        tX = []
        RZ = []
        tZ = []
        
        RX.append(tf.math.cos(self.deg2rad(q[2])) * tf.math.cos(self.deg2rad(q[1])))
        RX.append(tf.math.cos(self.deg2rad(q[2])) * tf.math.sin(self.deg2rad(q[1])) * tf.math.sin(self.deg2rad(q[0])) - tf.math.sin(self.deg2rad(q[2])) * tf.math.cos(self.deg2rad(q[0])))
        RX.append(tf.math.cos(self.deg2rad(q[2])) * tf.math.sin(self.deg2rad(q[1])) * tf.math.cos(self.deg2rad(q[0])) + tf.math.sin(self.deg2rad(q[2])) * tf.math.sin(self.deg2rad(q[0])))
        RX.append(tf.math.sin(self.deg2rad(q[2])) * tf.math.cos(self.deg2rad(q[1])))
        RX.append(tf.math.sin(self.deg2rad(q[2])) * tf.math.sin(self.deg2rad(q[1])) * tf.math.sin(self.deg2rad(q[0])) + tf.math.cos(self.deg2rad(q[2])) * tf.math.cos(self.deg2rad(q[0])))
        RX.append(tf.math.sin(self.deg2rad(q[2])) * tf.math.sin(self.deg2rad(q[1])) * tf.math.cos(self.deg2rad(q[0])) - tf.math.cos(self.deg2rad(q[2])) * tf.math.sin(self.deg2rad(q[0])))
        RX.append(-tf.math.sin(self.deg2rad(q[1])))
        RX.append(tf.math.cos(self.deg2rad(q[1])) * tf.math.sin(self.deg2rad(q[0])))
        RX.append(tf.math.cos(self.deg2rad(q[1])) * tf.math.cos(self.deg2rad(q[0])))
        RX = tf.reshape(RX, [3,3])
        
        for i in range(3):
            tX.append(q[i+3])
        tX = tf.reshape(tX, [3,1])
        
        RZ.append(tf.math.cos(self.deg2rad(q[8])) * tf.math.cos(self.deg2rad(q[7])))
        RZ.append(tf.math.cos(self.deg2rad(q[8])) * tf.math.sin(self.deg2rad(q[7])) * tf.math.sin(self.deg2rad(q[6])) - tf.math.sin(self.deg2rad(q[8])) * tf.math.cos(self.deg2rad(q[6])))
        RZ.append(tf.math.cos(self.deg2rad(q[8])) * tf.math.sin(self.deg2rad(q[7])) * tf.math.cos(self.deg2rad(q[6])) + tf.math.sin(self.deg2rad(q[8])) * tf.math.sin(self.deg2rad(q[6])))
        RZ.append(tf.math.sin(self.deg2rad(q[8])) * tf.math.cos(self.deg2rad(q[7])))
        RZ.append(tf.math.sin(self.deg2rad(q[8])) * tf.math.sin(self.deg2rad(q[7])) * tf.math.sin(self.deg2rad(q[6])) + tf.math.cos(self.deg2rad(q[8])) * tf.math.cos(self.deg2rad(q[6])))
        RZ.append(tf.math.sin(self.deg2rad(q[8])) * tf.math.sin(self.deg2rad(q[7])) * tf.math.cos(self.deg2rad(q[6])) - tf.math.cos(self.deg2rad(q[8])) * tf.math.sin(self.deg2rad(q[6])))
        RZ.append(-tf.math.sin(self.deg2rad(q[7])))
        RZ.append(tf.math.cos(self.deg2rad(q[7])) * tf.math.sin(self.deg2rad(q[6])))
        RZ.append(tf.math.cos(self.deg2rad(q[7])) * tf.math.cos(self.deg2rad(q[6])))
        RZ = tf.reshape(RZ, [3,3])
        
        for i in range(3):
            tZ.append(q[i+9])
        tZ = tf.reshape(tZ, [3,1])
            
        return RX, tX, RZ, tZ
    
    def calibration_loss(self, q):
        RX,tX,RZ,tZ = self.RtXZ(q)
        errfunc = [0, 0]
        
        for i in range(len(self.RA)):
            errfunc[0] += tf.norm(tf.matmul(self.RA[i], RX) - tf.matmul(RZ, self.RB[i]), ord=2, axis=(0,1))
            errfunc[1] += tf.pow(tf.norm((tf.matmul(self.RA[i], tX) + self.tA[i] - tf.matmul(RZ, self.tB[i]) - tZ) / 100.0), 2)
            
        errfunc[0] = tf.sqrt(errfunc[0])
        errfunc[1] = tf.sqrt(errfunc[1])
        
        loss = errfunc[0] + errfunc[1]
        return loss, errfunc
    
    def q_body(self, q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,q11,q12):
        with tf.name_scope('Eular_Angles'):
            q = tf.Variable([q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,q11,q12], name='q')  # X的x,y,z,roll,pitch,yaw and Z的x,y,z,roll,pitch,yaw
        return q
    
    def drawCbCornersHE(self, Document, pos_of_In, pos_of_HE, In_name, outFile):
        myIntrinsic = cv2.FileStorage(pos_of_In, cv2.FILE_STORAGE_READ)
        myHandEye = cv2.FileStorage(pos_of_HE, cv2.FILE_STORAGE_READ)
        
        Intrinsic = myIntrinsic.getNode(In_name).mat()
        Ttc = myHandEye.getNode("Ttc").mat()
        Tbw = myHandEye.getNode("Tbw").mat()
        
        for n in range(len(self.TB)):        
            Extrinsic = np.dot(Ttc, np.dot(np.linalg.inv(self.TB[n]), np.linalg.inv(Tbw)))
            Extrinsic = Extrinsic[:3, :]
            name = Document + str(n) + ".bmp"
            image = cv2.imread(name) 
            
            # 畫角點
            for i in range(19):
                for j in range(12):
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
        
    
        
    
        