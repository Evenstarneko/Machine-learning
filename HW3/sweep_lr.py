# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 18:37:49 2021

@author: Yan Chang
"""
import os

if __name__ == '__main__':
    for i in [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1]:
        command = "python main.py train.py train --data-dir relase-data --log-file ff-logs-lr=" + str(i) + "csv --model-save ff-lr=" + str(i) + ".torch --model simple-ff --learning-rate " + str(i) 
        os.system(command)
        