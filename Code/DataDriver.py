# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 2023

@author: youngj
"""
import csv
from numpy import *
import time
import os
from openpyxl import Workbook

wb = Workbook()
ws = wb.active
def get_file_path_by_name(file_dir):
    '''
    Gets the absolute path of all files in the specified path
    :param file_dir:
    :return: .csv file path
    '''
    x = []
    for root, dirs, files in os.walk(file_dir):  # get all file
        for file in files:  # traversal all file name
            if os.path.splitext(file)[1] == '.csv':   #.csv file
                x.append(os.path.join(root, file))
    print('total file：', len(x))
    return x


Load_index = ['0', '1', '2', '3', 'Cross']
savingPath="/Users/youngj/Local/Project/2022_Rolling_Bearing_Fault_Diagnosis/RollingBearFault_Data/"
for i in range(5):
    FilePath = get_file_path_by_name("/Users/youngj/Local/Project/2022_Rolling_Bearing_Fault_Diagnosis/RollingBearFault_Data/SEResNet/Load_"+Load_index[i])
    for filename in FilePath:
        with open(filename, 'r') as csvfile:
            csvreader = csv.reader(csvfile)
            header = next(csvreader)  # consume the header row
            top_1 = []
            mean_precision = []
            mean_recall = []
            mean_f1_score = []
            for row in csvreader:
                top_1.append(float(row[2]))  # convert to float if needed
                mean_precision.append(float(row[3]))
                mean_recall.append(float(row[4]))
                mean_f1_score.append(float(row[5]))
            # find the max value of each column
            max_top_1 = max(top_1)
            max_mean_precision = max(mean_precision)
            max_mean_recall = max(mean_recall)
            max_mean_f1_score = max(mean_f1_score)

            # add the filename and max values to the worksheet
            ws.append([filename])
            ws.append(['', 'max top-1', 'max mean precision', 'max mean recall', 'max mean F1 score'])
            ws.append(['', max_top_1, max_mean_precision, max_mean_recall, max_mean_f1_score])
            ws.append([])  # add a blank row for spacing

        # save the workbook to an Excel file
    wb.save(savingPath+Load_index[i]+'.xlsx')
    print("Load"+Load_index[i]+" is done")