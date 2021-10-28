# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 18:37:49 2021

@author: Yan Chang
"""
import os

if __name__ == '__main__':
    for i in [10, 30, 50, 70, 90, 110, 130]:
        command = "python main.py train --data-dir release-data --log-file cnn-logs-channels=" + str(i) + "csv --model-save cnn-channels=" + str(i) + ".torch --model simple-cnn --cnn-n1-channels " + str(i) 
        os.system(command)
        