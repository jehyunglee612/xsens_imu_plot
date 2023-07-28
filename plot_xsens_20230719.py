import matplotlib.pyplot as plt
import csv
import numpy as np
import math
import serial
import csv
from datetime import datetime

class ZoomPan:
    def __init__(self):
        self.press = None
        self.cur_xlim = None
        self.cur_ylim = None
        self.x0 = None
        self.y0 = None
        self.x1 = None
        self.y1 = None
        self.xpress = None
        self.ypress = None


    def zoom_factory(self, ax, base_scale = 2.):
        def zoom(event):
            cur_xlim = ax.get_xlim()
            cur_ylim = ax.get_ylim()

            xdata = event.xdata # get event x location
            ydata = event.ydata # get event y location

            if event.button == 'down':
                # deal with zoom in
                scale_factor = 1 / base_scale
            elif event.button == 'up':
                # deal with zoom out
                scale_factor = base_scale
            else:
                # deal with something that should never happen
                scale_factor = 1
                print(event.button)

            new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
            new_height = (cur_ylim[1] - cur_ylim[0])

            relx = (cur_xlim[1] - xdata)/(cur_xlim[1] - cur_xlim[0])
            rely = (cur_ylim[1] - ydata)/(cur_ylim[1] - cur_ylim[0])

            ax.set_xlim([xdata - new_width * (1-relx), xdata + new_width * (relx)])
            ax.set_ylim([ydata - new_height * (1-rely), ydata + new_height * (rely)])
            ax.figure.canvas.draw()

        fig = ax.get_figure() # get the figure of interest
        fig.canvas.mpl_connect('scroll_event', zoom)

        return zoom

    def pan_factory(self, ax):
        def onPress(event):
            if event.inaxes != ax: return
            self.cur_xlim = ax.get_xlim()
            self.cur_ylim = ax.get_ylim()
            self.press = self.x0, self.y0, event.xdata, event.ydata
            self.x0, self.y0, self.xpress, self.ypress = self.press

        def onRelease(event):
            self.press = None
            ax.figure.canvas.draw()

        def onMotion(event):
            if self.press is None: return
            if event.inaxes != ax: return
            dx = event.xdata - self.xpress
            dy = event.ydata - self.ypress
            self.cur_xlim -= dx
            self.cur_ylim -= dy
            ax.set_xlim(self.cur_xlim)
            ax.set_ylim(self.cur_ylim)

            ax.figure.canvas.draw()

        fig = ax.get_figure() # get the figure of interest

        # attach the call back
        fig.canvas.mpl_connect('button_press_event',onPress)
        fig.canvas.mpl_connect('button_release_event',onRelease)
        fig.canvas.mpl_connect('motion_notify_event',onMotion)

        #return the function
        return onMotion

def roll_pitch_calculator(filename, xsens_file):
    index = []
    time = []
    acc_x = []
    acc_y = []
    acc_z = []
    gyro_x = []
    gyro_y = []
    gyro_z = []

    roll = []
    pitch = []
    
    # reading data from a xsens csv file 'Data.csv'
    with open(filename, newline='') as file:
        
        reader = csv.reader(file, delimiter = ',')
        
        # if xsens_file == True:
        #     # store the headers in a separate variable,
        #     # move the reader object to point on the next row
        #     headings = next(reader)
        #     headings = next(reader)
        #     headings = next(reader)
        #     headings = next(reader)
        #     headings = next(reader)
        #     headings = next(reader)
        #     headings = next(reader)
        #     headings = next(reader)
        
        for row in reader:
            index.append(float(row[0]))
            if xsens_file == True:
                time.append(int(row[1]))
                acc_x.append(float(row[5]))
                acc_y.append(float(row[6]))
                acc_z.append(float(row[7]))
                gyro_x.append(float(row[8]))
                gyro_y.append(float(row[9]))
                gyro_z.append(float(row[10]))
            else:
                acc_x.append(float(row[1]))
                acc_y.append(float(row[2]))
                acc_z.append(float(row[3]))
                gyro_x.append(float(row[4]))
                gyro_y.append(float(row[5]))
                gyro_z.append(float(row[6]))
                
    for i in range(len(acc_x)):
        roll.append(math.atan2(acc_y[i], math.sqrt(acc_x[i] ** 2.0 + acc_z[i] ** 2.0)))
        pitch.append(math.atan2(-acc_x[i], math.sqrt(acc_y[i] ** 2.0 + acc_z[i] ** 2.0)))

    return index, roll, pitch
def delete_first_n_rows(input_file, output_file, delete_lines):
    with open(input_file, 'r', newline='') as csvfile:
        csvreader = csv.reader(csvfile)
        data = list(csvreader)
    
    if len(data) <= delete_lines:
        print("The CSV file contains fewer than five rows. Nothing to delete.")
        return
    
    data = data[delete_lines:]  # Skip the first five rows
    
    with open(output_file, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(data)
        
def get_first_column(csv_file):
    first_column = []
    with open(csv_file, 'r', newline='') as csvfile:
        csvreader = csv.reader(csvfile, delimiter = ',')
        # print(csvreader)
        # first_data = list(csvreader)[0][0]
        # print("type of first data: ", type(first_data))
        for row in list(csvreader):
            if row:  # Check if the row is not empty
                first_column.append(int(row[0]))
    first_index = first_column[0]
    for i in range(len(first_column)):
        first_column[i] = first_column[i] - first_index
    return first_column

################## Parameters ##################
starting_index_arduino = 200
starting_index_xsens = 0


################## File Name ##################
xsens_raw_file = '15_20230720_151055_577.csv'
arduino_file = 'arduino_data.csv'
################################################


delete_first_n_rows('analog-data.csv', arduino_file, starting_index_arduino)
delete_first_n_rows(xsens_raw_file, 'xsens_data.csv', 8+starting_index_xsens)
# delete_first_n_rows(sen_files_data.csv', starting_index_xsens)
index_arduino, roll_arduino, pitch_arduino = roll_pitch_calculator(arduino_file, False)
index_xsens, roll_xsens, pitch_xsens = roll_pitch_calculator('xsens_data.csv', True)

# print("first column: ", get_first_column(arduino_file)[:10])
#plotting
print("start plotting")
fig = plt.figure()
xsens = fig.add_subplot(111)
xsens.set_title("Roll")
xsens.plot(get_first_column(arduino_file), roll_arduino, label = 'roll_arduino', linewidth=0.7, color = 'r')
xsens.plot(get_first_column('xsens_data.csv'), roll_xsens, label = 'roll_xsens', linewidth=0.7, color = 'g')
xsens.legend(['arduino', 'xsens'], loc='upper right')

scale = 1.2
zp = ZoomPan()
figZoom = zp.zoom_factory(xsens, base_scale = scale)
figPan = zp.pan_factory(xsens)

plt.show()