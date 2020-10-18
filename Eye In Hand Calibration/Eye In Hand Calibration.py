# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 21:27:13 2019

@author: ASUS
"""
import numpy as np
import tensorflow as tf
import cv2

import CameraCalibration, CalibrationFn
import model
tf.reset_default_graph()

def _main():
    lr = 1e-0
    q1 = 0.0
    q2 = 30.0
    q3 = -90.0
    q4 = 35.0
    q5 = 160.0
    q6 = 20.0
    q7 = 180.0
    q8 = 0.0
    q9 = 0.0
    q10 = -405.0
    q11 = 40.0
    q12 = 30.0

    # ======建model======
    EIHCal = model.Eye_in_Hand_Calibration()
    q = EIHCal.q_body(q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,q11,q12)
    loss, errfunc = EIHCal.calibration_loss(q)
    train_step = tf.train.AdamOptimizer(lr).minimize(loss)
    
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    
    # ======訓練======
    for i in range(1000):
        _, Loss = sess.run([train_step, errfunc])
        print(i,": ",Loss)
    
    # ======存檔======
    myLM_TbwTtc = cv2.FileStorage("data/LM_TbwTtc.xml", cv2.FILE_STORAGE_WRITE)
    expend = np.reshape([0,0,0,1], [1,4])
    Ttc = np.concatenate([sess.run(EIHCal.RtXZ(q))[0], sess.run(EIHCal.RtXZ(q))[1]], axis=1)
    Ttc = np.concatenate([Ttc, expend], axis=0)
    Tbw = np.concatenate([sess.run(EIHCal.RtXZ(q))[2], sess.run(EIHCal.RtXZ(q))[3]], axis=1)
    Tbw = np.concatenate([Tbw, expend], axis=0)
    myLM_TbwTtc.write("Ttc", Ttc)
    myLM_TbwTtc.write("Tbw", Tbw)
    myLM_TbwTtc.release()
    
    # ======畫圖驗證======
    EIHCal.drawCbCornersHE("Chessboards/Chessboards_L/ChessboardL_", "data/Intrinsic_L.xml", "data/LM_TbwTtc.xml", "IntrinsicL", "Chessboards_HE/Chessboards_HE_")
    
if __name__ == '__main__':
#    CameraCalibration._main()
    _main() 