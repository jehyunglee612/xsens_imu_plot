import os, sys
import csv
from matplotlib.gridspec import GridSpec
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.widgets import TextBox
from scipy.signal import find_peaks
import pandas as pd

from models import * 

TAKES = [
    '20230427_140140_{}.csv',
    '20230427_141658_{}.csv',
    '20230427_142027_{}.csv',
    '20230427_142756_{}.csv',
    '20230427_143425_{}.csv',
    '20230427_144243_{}.csv',
    '20230427_144738_{}.csv',
    '20230427_151114_{}.csv',
    '20230628_145856_{}.csv',
    '20230628_150702_{}.csv',
    '20230628_151454_{}.csv',
    '20230628_151843_{}.csv',
    '20230628_152612_{}.csv',
    '20230628_153051_{}.csv',
    '20230703_170312_{}.csv',
    '20230706_174524_{}.csv',
    '20230706_175114_{}.csv',
    '20230706_175252_{}.csv',
    '20230706_180402_{}.csv',
    '20230706_181537_{}.csv',
    '20230706_183016_{}.csv',
    '20230706_183906_{}.csv',
]

t_idx = 2
contact_mode = 'fh_volley'
contact_color = 'xkcd:light red'
none_color = 'xkcd:light blue'
video_name = None

if len(sys.argv) > 1:
    t_idx = int(sys.argv[1])

if len(sys.argv) > 2:
    contact_mode = sys.argv[2]

if len(sys.argv) > 3:
    video_name = sys.argv[3]

class PlotSavedData():
    label_map = {
        0: {'name': 'remove', 'color': 'xkcd:light red', 's_color': 'xkcd:scarlet'},
        1: {'name': 'splitstep', 'color': 'xkcd:light green', 's_color': 'xkcd:green'},
        2: {'name': 'approach', 'color': 'xkcd:light blue', 's_color': 'xkcd:blue'},
        3: {'name': 'return', 'color': 'xkcd:blue green', 's_color': 'xkcd:dark teal'},
        4: {'name': 'takeback', 'color': 'xkcd:peach', 's_color': 'xkcd:orange'},
        5: {'name': 'swing', 'color': 'xkcd:taupe', 's_color': 'xkcd:puce'},
        6: {'name': 'contact', 'color': 'xkcd:lilac', 's_color': 'xkcd:violet'},
        7: {'name': 'followthrough', 'color': 'xkcd:azure', 's_color': 'xkcd:electric blue'},
    }

    sensors = ['lwrist', 'rwrist']
    plot_headers = ['Acc_X', 'Acc_Y', 'Acc_Z']
    
    data_root = 'data_aligned_wrists'
    data_file = TAKES[t_idx]
    print(data_file)

    label_path = os.path.join(data_root, f"{data_file[:-7]}_labels_full.csv")
    peak_path = os.path.join(data_root, f"{data_file[:-7]}_labels_peak_120.csv")

    single_click_threshold = 0.01
    select_dist_threshold = 0.005
    frame_diff_threshold = 0.03
    
    auto_window = 120
    auto_peak_mult = 0.6
    auto_peak_min = 40

    time_unit = 1000000 # Data time is in microseconds
    min_w = 0.5 # At least `min_w` seconds of data must be shown in the window
    max_w = 60 # At most `max_w` seconds of data can be shown in the window
    padding = 1 # `padding` seconds are added to the beginning and end of the x-axis

    left_delay = 0

    base_line_colors = ['r', 'g', 'b', 'y']

    def __init__(self):
        self.ax_dict = {}
        self.data_dict = {}
        self.plot_dict = {}
        self.labels = []
        self.label_text = {}

        self.contact_instances = []
        self.remove_instances = []

        self.is_labeled = os.path.exists(self.label_path)
        if self.is_labeled:
            self.sensors.append('racket')
        self.has_peaks = os.path.exists(self.peak_path)

        self.setup_plot()
        self.setup_labels()
        if not self.has_peaks:
            self.detect_windows()

        plt.show()

    def setup_plot(self):
        fig = plt.figure(figsize=(20, 10))
        plt.subplots_adjust(left=0.05, bottom=0.05, right=0.9, top=0.95)
        self.axes = []

        if self.is_labeled:
            gs = GridSpec(6, 3, figure=fig, height_ratios=[3,3,3,3,3,1])
        else:
            gs = GridSpec(5, 3, figure=fig, height_ratios=[3,3,3,3,1])

        self.axes.append(fig.add_subplot(gs[0, :]))
        self.axes.append(fig.add_subplot(gs[1, :]))
        self.axes.append(fig.add_subplot(gs[2, :]))
        self.axes.append(fig.add_subplot(gs[3, :]))
        if self.is_labeled:
            self.axes.append(fig.add_subplot(gs[4, :]))
        self.axes.append(fig.add_subplot(gs[-1, 0]))
        self.axes.append(fig.add_subplot(gs[-1, 1]))
        self.axes.append(fig.add_subplot(gs[-1, 2]))

        # Set listeners
        fig.canvas.mpl_connect('button_press_event', self.on_click)
        fig.canvas.mpl_connect('scroll_event', self.on_scroll)
        fig.canvas.mpl_connect('key_release_event', self.on_key_release)

        # Load data
        self.x_axis = None
        for sensor in self.sensors:
            df = pd.read_csv(os.path.join(self.data_root, self.data_file.format(sensor)))
            self.data_dict[sensor] = df[self.plot_headers].to_numpy()
            self.data_dict[sensor] = self.fill_nan(self.data_dict[sensor])
            if sensor is 'rwrist' and self.x_axis is None:
                self.x_axis = df['SampleTimeFine'].to_numpy() / self.time_unit
        self.x_axis -= self.x_axis[0]

        # Calculate absolute value for left wrist
        kernel_size = 30

        self.lwrist_abs = np.array([[abs(j) for j in i] for i in self.data_dict['rwrist']])
        self.lwrist_abs_conv = np.array([np.convolve(self.lwrist_abs[:,i], np.ones(kernel_size) / kernel_size, 'same') for i in range(self.lwrist_abs.shape[1])]).T
        self.lwrist_abs_conv_sum = np.sum(self.lwrist_abs_conv, axis=1)

        self.lwrist_grad_abs = np.array([[abs(j) for j in i] for i in np.gradient(self.data_dict['rwrist'], axis=0)])
        self.lwrist_grad_abs_conv = np.array([np.convolve(self.lwrist_grad_abs[:,i], np.ones(kernel_size) / kernel_size, 'same') for i in range(self.lwrist_grad_abs.shape[1])]).T
        self.lwrist_grad_abs_conv_sum = np.sum(self.lwrist_grad_abs_conv, axis=1)

        self.peak_data = self.lwrist_abs_conv_sum + 2 * self.lwrist_grad_abs_conv_sum
        self.peak_data = np.convolve(self.peak_data, np.ones(kernel_size * 2) / kernel_size * 2, 'same')

        self.auto_peaks = find_peaks(self.peak_data, height=self.auto_peak_min, distance=self.auto_window, prominence=self.auto_window//4)[0]

        # Calculate graph axis
        self.min_x = self.x_axis[0] - self.padding
        self.max_x = self.x_axis[-1] + self.padding
        self.cur_xlim = (self.min_x, self.max_x)
        self.max_w = self.max_x - self.min_x
        
        self.plot_data(0, np.stack((self.lwrist_abs_conv_sum, 4 * self.lwrist_grad_abs_conv_sum), axis=1), 'separate')
        self.plot_data(1, self.peak_data, 'combined')
        self.plot_data(2, self.data_dict['lwrist'], 'lwrist')
        self.plot_data(3, self.data_dict['rwrist'], 'rwrist')
        if self.is_labeled:
            self.plot_data(4, self.data_dict['racket'], 'racket')

        # Add label text
        for idx, label in self.label_map.items():
            self.label_text[idx] = plt.text(1.02, 1-(idx+1)*0.12, f"{idx}: {label['name']}", 
                                            transform=list(self.ax_dict.values())[0].transAxes,
                                            color=label['color'], fontsize=12, verticalalignment='top')
        
        # Auto detect inputs
        self.auto_window_input = TextBox(self.axes[-3], 'Window', initial=self.auto_window)
        self.auto_window_input.on_submit(self.on_window_submit)
        self.auto_peak_mult_input = TextBox(self.axes[-2], 'Peak threshold', initial=self.auto_peak_mult)
        self.auto_peak_mult_input.on_submit(self.on_peak_mult_submit)
        self.auto_peak_min_input = TextBox(self.axes[-1], 'Peak min', initial=self.auto_peak_min)
        self.auto_peak_min_input.on_submit(self.on_peak_min_submit)

    # Plot data
    def plot_data(self, idx, data, label):
        self.ax_dict[label] = self.axes[idx]
        ax = self.ax_dict[label]
        ax.clear()
        if len(data.shape) == 1:
            ax.plot(self.x_axis, data, color=self.base_line_colors[0], linestyle=':')
        else:
            for i in range(data.shape[1]):
                ax.plot(self.x_axis, data[:, i], color=self.base_line_colors[i], linestyle=':')

        
        for peak in self.labels:
            art = self.add_rect(x=peak.x_start, width=peak.x_end-peak.x_start, color=peak.art[0].get_facecolor(), to_list=[label])
            peak.art += art
                
        ax.set_title(label)
        ax.set_xlim(self.cur_xlim)

    def fill_nan(self, data):
        for c in range(data.shape[1]):
            mask = np.isnan(data[:,c])
            data[mask,c] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), data[~mask,c])
        return data

    ''' Auto label functions '''
    
    def setup_labels(self):
        if self.is_labeled:
            f = open(self.label_path, 'r')
            all_labels = list(csv.reader(f))[1:]
            f.close()

            type_to_mode = {label['name']: idx for idx, label in self.label_map.items()}

            for i, (label_type, label_name, x_start, x_end) in enumerate(all_labels):
                x_start = float(x_start)
                x_end = float(x_end)
                label_mode =  type_to_mode[label_name]
                label_color = self.label_map[label_mode]['color']
                if label_type == 'single':
                    art = self.add_line(x=x_start, color=label_color, linestyle='--', to_list=['racket'])
                    label = Label('single', label_name, label_mode, x_start, art=art)
                    self.labels.append(label)
                elif label_type == 'range':
                    art = self.add_rect(x=x_start, width=x_end-x_start, color=label_color, to_list=['racket'])
                    label = Label('range', label_name, label_mode, x_start, x_end=x_end, art=art)
                    self.labels.append(label)

                    if label_name == 'contact':
                        self.contact_instances.append(x_start)
                    if label_name == 'remove':
                        self.remove_instances.append((x_start, x_end))

        if self.has_peaks:
            print('loading peaks')
            with open(self.peak_path, 'r') as f:
                all_peaks = list(csv.reader(f))[1:]

            for (_, _name, _start, _end) in all_peaks:
                _start = float(_start)
                _end = float(_end)

                art = self.add_rect(x=_start, width=_end-_start, color=contact_color, to_list=['separate', 'combined', 'lwrist', 'rwrist'])
                label = Label('range', _name, 0, _start, x_end=_end, art=art)
                label.auto = True
                self.labels.append(label)

        plt.draw()

    def on_window_submit(self, text):
        self.auto_window = int(text)
        self.detect_windows()

    def on_peak_mult_submit(self, text):
        self.auto_peak_mult = float(text)
        self.detect_windows()

    def on_peak_min_submit(self, text):
        self.auto_peak_min = float(text)
        self.detect_windows()

    def delete_existing_auto_labels(self):
        i = 0
        while i < len(self.labels):
            if self.labels[i].auto:
                self.delete_label(self.labels[i])
                del self.labels[i]
            else:
                i += 1

    def detect_windows(self):
        self.delete_existing_auto_labels()

        for peak in self.auto_peaks:
            if peak - self.auto_window//2 < 0 or peak + self.auto_window//2 >= self.peak_data.shape[0]:
                continue

            x_start = self.x_axis[peak - self.auto_window//2]
            x_end = self.x_axis[peak + self.auto_window//2]

            is_contact = False
            for contact in self.contact_instances:
                if x_start <= contact <= x_end:
                    is_contact = True
                    break
            if not self.is_labeled:
                is_contact = True

            if is_contact:
                color = contact_color
                name = contact_mode
            else:
                color = none_color
                name = 'none'

            art = self.add_rect(x=x_start, width=x_end-x_start, color=color, to_list=['separate', 'combined', 'lwrist', 'rwrist'])
            label = Label('range', name, 0, x_start, x_end=x_end, art=art)
            label.auto = True
            self.labels.append(label)

        plt.draw()


    ''' Event handlers '''

    def on_scroll(self, event):
        if event.key == 'shift':
            self.zoom(event)
        else:
            self.scroll(event)
        plt.draw()

    def scroll(self, event):
        width = self.cur_xlim[1] - self.cur_xlim[0]
        if event.step > 0:
            scroll_length = width/10
        elif event.step < 0:
            scroll_length = -width/10

        new_min_x = self.cur_xlim[0] + scroll_length
        new_max_x = self.cur_xlim[1] + scroll_length

        if new_min_x < self.min_x:
            new_min_x = self.min_x
            new_max_x = min(self.max_x, new_min_x + width)
        if new_max_x > self.max_x:
            new_max_x = self.max_x
            new_min_x = max(self.min_x, new_max_x - width)

        self.cur_xlim = (new_min_x, new_max_x)

        for ax in self.ax_dict.values():
            ax.set_xlim(self.cur_xlim)

    def zoom(self, event):
        base_scale = 2.0
        xdata = event.xdata
        if event.step > 0:
            scale_factor = 1/base_scale
        elif event.step < 0:
            scale_factor = base_scale

        if not xdata:
            return
            
        cur_xrange = (self.cur_xlim[1] - self.cur_xlim[0])*.5
        new_min_x = max(self.min_x, xdata - cur_xrange*scale_factor)
        new_max_x = min(self.max_x, xdata + cur_xrange*scale_factor)

        width = new_max_x - new_min_x
        mid = (new_max_x + new_min_x) / 2
        if width < self.min_w:
            new_min_x = mid - self.min_w/2
            new_max_x = mid + self.min_w/2
        elif width > self.max_w:
            new_min_x = mid - self.max_w/2
            new_max_x = mid + self.max_w/2
        width = self.min_w

        if new_min_x < self.min_x:
            new_min_x = self.min_x
            new_max_x = min(self.max_x, new_min_x + width)
        if new_max_x > self.max_x:
            new_max_x = self.max_x
            new_min_x = max(self.min_x, new_max_x - width)

        self.cur_xlim = (new_min_x, new_max_x)

        for ax in self.ax_dict.values():
            ax.set_xlim(self.cur_xlim)
    
    def on_click(self, event):
        # Check if click is outside axes
        if not event.xdata or not event.ydata:
            return

        # Allow clicks only in the sensor axes
        clicked_axes_idx = 0
        for i, ax in enumerate(self.axes):
            if event.inaxes == ax:
                clicked_axes_idx = i
                break
        if not clicked_axes_idx < len(self.sensors):
            return

        if event.button == 3: # RIGHT
            label_to_delete = self.get_closest_label(event.xdata)
            if label_to_delete:
                self.labels.remove(label_to_delete)
                self.delete_label(label_to_delete)
        elif event.button == 1: # LEFT
            label_to_toggle = self.get_closest_label(event.xdata)
            if label_to_toggle:
                new_color = none_color if label_to_toggle.name == contact_mode else contact_color
                new_name = 'none' if label_to_toggle.name == contact_mode else contact_mode
                label_to_toggle.name = new_name
                self.delete_label(label_to_toggle)
                label_to_toggle.art = self.add_rect(
                    x=label_to_toggle.x_start, width=label_to_toggle.x_end-label_to_toggle.x_start, color=new_color)

        plt.draw()


    def on_key_release(self, event):
        key = event.key
        if key == 'r':
            self.cur_xlim = (self.min_x, self.max_x)
            for ax in self.ax_dict.values():
                ax.set_xlim(self.cur_xlim)
        elif key == 'left':
            self.move_lwrist(1)
        elif key == 'right':
            self.move_lwrist(-1)
        elif key == '+':
            self.save_labels()
        elif key == 'q':
            plt.close()
            return

        plt.draw()

    
    def get_closest_label(self, xdata):
        closest = None
        min_dist = np.inf
        for label in self.labels:
            dist = np.inf
            if label.type == 'single':
                dist = abs(label.x_start - xdata)
            elif label.type == 'range':
                if label.x_start <= xdata <= label.x_end:
                    dist = 0
                else:
                    dist = min(abs(label.x_start - xdata), abs(label.x_end - xdata))
            if dist <= min_dist:
                closest = label
                min_dist = dist

        ratio_dist = abs(min_dist / (self.cur_xlim[1] - self.cur_xlim[0]))
        if ratio_dist > self.select_dist_threshold:
            closest = None

        return closest

    def save_labels(self):
        with open(self.peak_path, 'w', newline='') as f:
            writer = csv.writer(f)
            header = ['label_type', 'label_name', 'label_start', 'label_end']
            writer.writerow(header)
            for label in sorted(self.labels, key=lambda l: l.x_start):
                if not label.auto:
                    continue
                writer.writerow(label.get_row())

        if self.left_delay != 0:
            lwrist_csv = os.path.join(self.data_root, self.data_file.format('lwrist'))
            df = pd.read_csv(lwrist_csv)
            original_time = df['SampleTimeFine'].copy()
            if self.left_delay > 0:
                df.iloc[:-self.left_delay] = df.iloc[self.left_delay:].values
            else:
                df.iloc[-self.left_delay:] = df.iloc[:self.left_delay].values
            df['SampleTimeFine'] = original_time
            # df = df.iloc[:, 1:]
            
            os.rename(lwrist_csv, lwrist_csv.replace('.csv', '_original.csv'))
            df.to_csv(lwrist_csv, index=False)

        print("Saved labels.")

    def delete_label(self, label):
        for art in label.art:
            art.remove()
        label.art = []

    def add_line(self, x, color, linestyle, to_list=[]):
        lines = []
        if len(to_list) == 0:
            to_list = self.sensors
        for plot_label in to_list:
            lines.append(self.ax_dict[plot_label].axvline(x=x, color=color, linestyle=linestyle))
        return lines

    def add_rect(self, x, width, color, to_list=[]):
        rects = []
        if len(to_list) == 0:
            to_list = ['separate', 'combined', 'lwrist', 'rwrist']
        for plot_label in to_list:
            min_y, max_y = self.ax_dict[plot_label].get_ylim()
            rect = Rectangle(
                (x, min_y), width, max_y-min_y,
                alpha=0.5, facecolor=color)
            rects.append(self.ax_dict[plot_label].add_patch(rect))
        return rects
    
    def move_lwrist(self, direction):
        self.left_delay += direction
        shifted_left = self.data_dict['lwrist'].copy()
        if self.left_delay > 0:
            shifted_left = np.concatenate([shifted_left[self.left_delay:], np.zeros((self.left_delay, shifted_left.shape[1]))])
        elif self.left_delay < 0:
            shifted_left = np.concatenate([np.zeros((-self.left_delay, shifted_left.shape[1])), shifted_left[:self.left_delay]])
        self.plot_data(2, shifted_left, 'lwrist')

if __name__ == '__main__':
    plot = PlotSavedData()