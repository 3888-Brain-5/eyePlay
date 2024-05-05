from serial import Serial
from serial.tools import list_ports
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal.windows import gaussian
from scipy.ndimage import convolve
from numpy.fft import fft, fftfreq
import csv
import pygame
import serial

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
from numpy.fft import fft, fftfreq, ifft, fftshift, ifftshift
import serial


def auto_select_serial_port():
    ports = list_ports.comports()
    for port in ports:
        if "USB Serial Port" in port.description:
            return port.device
    return None


def read_arduino(ser, inputBufferSize):
    data = ser.read(inputBufferSize)
    out = [(int(data[i])) for i in range(0, len(data))]
    return out


def read_and_process_data(ser, inputBufferSize):
    data = ser.read(inputBufferSize)
    out = [(int(data[i])) for i in range(0, len(data))]
    data_in = np.array(out)
    result = []
    i = 1
    while i < len(data_in) - 1:
        if data_in[i] > 127:
            # Found beginning of frame
            # Extract one sample from 2 bytes
            intout = (np.bitwise_and(data_in[i], 127)) * 128
            i = i + 1
            intout = intout + data_in[i]
            result = np.append(result, intout)
        i = i + 1
    return result


def process_gaussian_fft(t, data_t, sigma_gauss):
    nfft = len(data_t)
    dt = t[1] - t[0]
    maxf = 1/dt
    f_fft = np.arange(-maxf/2, maxf/2, 1/(nfft*dt))
    data_f = np.fft.fftshift(np.fft.fft(data_t))
    gauss_filter = np.exp(-(f_fft)**2/sigma_gauss**2)
    data_f_filtered = data_f * gauss_filter
    data_t_filtered = np.fft.ifft(np.fft.ifftshift(data_f_filtered))
    return f_fft, np.abs(data_t_filtered)


def update_plot(ax, line, x_data, y_data):
    line.set_data(x_data, y_data)
    ax.relim()
    ax.autoscale_view()
    ax.figure.canvas.draw_idle()


def main(baudrate, inputBufferSize, sigma_gauss, sampling_rate, max_time):
    selected_port = auto_select_serial_port()
    if selected_port:
        print(f"Automatically selected port: {selected_port}")
        filtered_data_collection = []
        recent_filtered_data = []
        max_samples = int(sampling_rate * max_time)
        with Serial(port=selected_port, baudrate=baudrate, timeout= inputBufferSize / 20000.0) as ser:
            plt.ion()
            fig, ax = plt.subplots()
            line, = ax.plot([], [], 'g-')
            ax.set_xlabel('Time (seconds)')
            ax.set_ylabel('Value')

            while True:
                data_temp = read_and_process_data(ser, inputBufferSize)
                num_samples = len(data_temp)
                T = np.linspace(0, num_samples / sampling_rate, num_samples)
                _, data_temp_filtered = process_gaussian_fft(T, data_temp, sigma_gauss)

                filtered_data_collection.append(data_temp_filtered)
                recent_filtered_data.append(data_temp_filtered)
                if len(recent_filtered_data) > max_samples:
                    recent_filtered_data.pop(0)

                update_plot(ax, line, T, data_temp_filtered)

                plt.pause(0.05)

                if plt.waitforbuttonpress(timeout=0):
                    break

            plt.ioff()
            plt.show()
            np.save('filtered_data.npy', np.array(filtered_data_collection))

    else:
        print("No suitable serial port found.")


if __name__ == "__main__":
    main(230400, 10000, 25, 10000, 10)
