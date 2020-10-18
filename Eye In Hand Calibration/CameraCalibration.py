# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 10:56:58 2018

@author: Joe
"""

from __future__ import print_function
import numpy as np
import cv2
import os
import CalibrationFn

def _main():
    # ==================左眼校正===============
    Chessboards_L = CalibrationFn.CCalibration()
    Chessboards_L.getChessboards("Chessboards/Chessboards_L")
    reference = cv2.imread("Chessboards/Chessboards_L/ChessboardL_0.bmp")
    Chessboards_L.setBorderSize((19,12));
    Chessboards_L.addChessboardPoints("Chessboards/Chessboards_L/ChessboardL_")
    Chessboards_L.calibrate(reference, "data/Extrinsic_L.xml", "data/Intrinsic_L.xml", "ExtrinsicL_" , "IntrinsicL", "Distortion_L")
    Chessboards_L.drawCbCorners("Chessboards/Chessboards_L/ChessboardL_", "data/Extrinsic_L.xml", "data/Intrinsic_L.xml", "ExtrinsicL_" , "IntrinsicL", "CheckChessboards/CheckChessboards_L/CheckChessboardsL_")

    # ==================左眼校正===============
    Chessboards_R = CalibrationFn.CCalibration()
    Chessboards_R.getChessboards("Chessboards/Chessboards_R")
    reference = cv2.imread("Chessboards/Chessboards_R/ChessboardR_0.bmp")
    Chessboards_R.setBorderSize((19,12));
    Chessboards_R.addChessboardPoints("Chessboards/Chessboards_R/ChessboardR_")
    Chessboards_R.calibrate(reference , "data/Extrinsic_R.xml" , "data/Intrinsic_R.xml" , "ExtrinsicR_" , "IntrinsicR", "Distortion_R")
    Chessboards_R.drawCbCorners("Chessboards/Chessboards_R/ChessboardR_", "data/Extrinsic_R.xml", "data/Intrinsic_R.xml", "ExtrinsicR_" , "IntrinsicR", "CheckChessboards/CheckChessboards_R/CheckChessboardsR_")
    
    # ==================雙眼校正===============
    Chessboards_L.StereoCalibration(Chessboards_L.m_dstPoints, Chessboards_L.m_srcPoints, Chessboards_R.m_srcPoints,
                      Chessboards_L.cameraMatrix, Chessboards_L.distCoeffs, 
                      Chessboards_R.cameraMatrix, Chessboards_R.distCoeffs,)
if __name__ == '__main__':
    _main() 
