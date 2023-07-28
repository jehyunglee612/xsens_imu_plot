import matplotlib.pyplot as plt
import csv
import numpy as np
import math
import serial
import csv
from datetime import datetime

class Serial_Reader:
    def __init__(self, arduino_port, baud, fileName):
        self.arduino_port = arduino_port
        self.baud = baud
        self.fileName = fileName
        
    def read_serial(self, samples):
        ser = serial.Serial(self.arduino_port, self.baud)
        file = open(self.fileName, "a")
        
        line = 0 #start at 0 because our header is 0 (not real data)
        self.sensor_data = [] #store data

        start = datetime.now()
        start_time = start.strftime("%H:%M:%S")
        print("Start Time =", start_time)
        # collect the samples
        while line < samples:
            getData=ser.readline()
            dataString = getData.decode('utf-8')
            data=dataString[0:][:-2]

            readings = data.split(",")

            self.sensor_data.append(readings)
            if(line%(samples/10) == 0):
                print("Progress: ", line/(samples/100), "%")
            line = line+1
        print("Progress: 100%")
        print("Data collection complete!")
        end = datetime.now()
        end_time = end.strftime("%H:%M:%S")
        print("Finish Time =", end_time)
        
        
        # create the CSV
        with open(self.fileName, 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(self.sensor_data)

        file.close()
            
    def print_data(self):
        print("Sensor data:")
        print(self.sensor_data)
        

length_of_arduino_data = 1000

sr = Serial_Reader("COM3", 9600, "serial_data_230721_3.csv")
sr.read_serial(length_of_arduino_data)
