import serial
import csv

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

        # collect the samples
        while line <= samples:
            getData=ser.readline()
            dataString = getData.decode('utf-8')
            data=dataString[0:][:-2]

            readings = data.split(",")

            self.sensor_data.append(readings)
            line = line+1
        
        # create the CSV
        with open(self.fileName, 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(self.sensor_data)

        print("Data collection complete!")
        file.close()
            
    def print_data(self):
        print("Sensor data:")
        print(self.sensor_data)
        
print("start collecting data from serial port")
sr = Serial_Reader("COM3", 9600, "analog-data.csv")
sr.read_serial(30)

      





