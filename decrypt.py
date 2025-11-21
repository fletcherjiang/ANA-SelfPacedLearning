#!/usr/bin/python
#coding=utf-8
 
import cv2
import numpy as np
import os

# Simplified version for open source - no encryption
key = None

def get_key(username=None, userpasswd=None):
    """
    Placeholder function for compatibility.
    No encryption is used in the open source version.
    """
    return None
    
def cv_imread(file_path, key=None):
    """
    Read image file without decryption.
    This is the open source version that reads standard image files.
    """
    # Read image directly without decryption
    cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), -1)
    return cv_img
