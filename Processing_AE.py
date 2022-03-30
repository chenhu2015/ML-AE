# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 11:13:43 2021

@author: huxxx
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
#AE analysis package

#process data
#input: a file name
#output: number of AE events in that file, a list of record time, a list of AE events
def read_AE_file(filename):
    number_of_AE = 0
    record_time = []
    AE_list = []
    time_history = []
    with open(filename) as file:  
        #skip the header
        for _ in range(12):
            next(file)
        #read real values  
        for line in file:
            if "Record" in line: 
                time_history = []
                number_of_AE += 1
            elif "Timestamp" in line:
                record_time.append(float(line.split()[1]))
            elif line != '\n':
                time_history.append(float(line))  
                if  len(time_history) == 4000:
                    AE_list.append(time_history)
                
    return number_of_AE, record_time, AE_list


#plot AE events
#default sample interval = 0.05 us
#default voltage range [-16V,16V]
def plot_AE_event(time_history):
    fig, ax1 = plt.subplots()
        
    color = 'tab:red'
    ax1.set_xlabel('time [micro-sec]')
    ax1.set_ylabel('voltage [V]', color=color)
    ax1.plot(np.arange(len(time_history))*0.05,np.array(time_history), color=color) 
    ax1.tick_params(axis='y', labelcolor=color)
    #plt.ylim(-16,16)
       
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()

#plot original and FFT of a time history
#default sample interval = 0.05 us
#default plot use first 2000 samples (100 us) 
def plot_FFT(time_history):
    timestep = 0.05
    amplitude = time_history[0:2000]
    spectrum = np.fft.fft(amplitude)
    # Find the positive frequencies
    frequency = np.fft.fftfreq( spectrum.size, d = timestep)
    index = np.where(frequency >= 0.)
    # Scale the real part of the FFT and clip the data
    #clipped_spectrum = timestep*spectrum[index].real
    clipped_spectrum = timestep*np.abs(spectrum[index])
    clipped_frequency = frequency[index]
        
    # Create a figure
    fig = plt.figure()
    # Adjust white space between plots
    fig.subplots_adjust(hspace=1.0)
    # Create x-y plots of the amplitude and transform with labeled axes
    data1 = fig.add_subplot(2,1,1)
    plt.xlabel('Time [micro-sec]')
    plt.ylabel('Amplitude [V]')
    plt.title('Time Domain')
    real_index = 0.05 * np.arange(4000)
    data1.plot(real_index,time_history, color='red', label='Amplitude')
    plt.legend()
    plt.minorticks_on()
    #plt.xlim(0., 10.)
    data2 = fig.add_subplot(2,1,2)
    plt.xlabel('Frequency [MHz]')
    plt.ylabel('FFT Amplitude')
    plt.title('Frequency Domain')
    data2.plot(clipped_frequency,clipped_spectrum, color='blue', linestyle='solid', 
     marker='None', label='FFT', linewidth=1.5)
    plt.legend()
    plt.minorticks_on()
    plt.xlim(0., 1.)
    # Show the data
    plt.show()  
    
    
#calculate first arrival of a AE event
#input: time history
#output: first arrival time
def calculate_first_arrival(time_history):
    #use first 500 points to calculate noise level with mean and std
    event = np.array(time_history)
    noise = event[0:500]
    noise_mean = np.mean(noise)
    noise_std = np.std(noise)
    #set threshold
    upper_threshold = noise_mean + noise_std*50
    lower_threshold = noise_mean - noise_std*50
    #print(upper_threshold)
    #print(lower_threshold)
    
    #find local max and min index
    finding_region = event[500:2000]
    local_max_ind = argrelextrema(finding_region, np.greater)[0]
    local_min_ind = argrelextrema(finding_region, np.less)[0]
    local_ind = np.sort((np.concatenate((local_max_ind,local_min_ind))))
    #print(local_ind)
    
    #find first max or min
    max_min_index = 0
    for index in local_ind:
        if finding_region[index] > upper_threshold or finding_region[index] < lower_threshold:
            max_min_index = index+500
            break
    
    #print(max_min_index)
    #print(event[max_min_index])
    
    #elimate bad signals
    if max_min_index == 0:
        print("bad signal")
        return max_min_index
    elif np.max(event) > 16 or np.min(event) < -16:
        print("bad signal")
        return 0
    
    #find first_arrival
    else:
        for i in range(max_min_index-1,1,-1):            
            if event[i]*event[i-1] < 0:
                first_arrival = i
                break
            elif event[i] > event[i+1] and event[i] > event[i-1]:
                first_arrival = i
                break
            elif event[i] < event[i+1] and event[i] < event[i-1]:
                first_arrival = i
                break
    
    return first_arrival


#calculate RMS
#input:AE event, first_arrival, time_period to do RMS,default is 15 us(300 samples)
def calculate_RMS(time_history,first_arrival,data_period=300):
    data = time_history[first_arrival:first_arrival+data_period]
    data_square = np.square(data)
    RMS = np.sqrt(np.sum(data_square)/data_period)
    return RMS


#plot cumulative RMS vs time, and AE cumulative vs time
def plot_cum(record_time_list,RMS_cum_list):
    fig, ax1 = plt.subplots()
        
    color = 'tab:red'
    ax1.set_xlabel('time')
    ax1.set_ylabel('cumulative RMS [V]', color=color)
    ax1.plot(np.array(record_time_list),np.array(RMS_cum_list), color=color) 
    ax1.tick_params(axis='y', labelcolor=color)
    #plt.ylim(-4,4)
    
    
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        
    color = 'tab:blue'
    ax2.set_ylabel('cumulative AE count', color=color)  # we already handled the x-label with ax1
    ax2.plot(np.array(record_time_list)+200,np.arange(len(record_time_list)), color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    #plt.ylim(-2000,8000)
       
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()


#plot slope of cumulutive RMS vs time
def plot_AE_slope(record_time_list,RMS_cum_list):
    
    '''
    #first we need to smooth the curve, by average the curve with around n points
    smooth_interval = 501
    paddling the y-array
    #for i in range(int((smooth_interval-1)/2)):
        record_time_list = np.insert(record_time_list,[0],record_time_list[0])
        record_time_list = np.insert(record_time_list,[-1],record_time_list[-1])
    
    new_time_list = []
    for i in range(len(record_time_list)-(smooth_interval-1)):
        new_value = sum(record_time_list[i:i+smooth_interval])/smooth_interval
        new_time_list.append(new_value)
    
    record_time_list = new_time_list
    plot_cum(record_time_list,RMS_cum_list)
    '''

    gap = 1000
    time_list = []
    slope_list = []
    for i in range(len(record_time_list)-gap):
        slope = (RMS_cum_list[i+gap] - RMS_cum_list[i]) / (record_time_list[i+gap] - record_time_list[i])
        slope_list.append(slope)
        time_list.append(record_time_list[i+int(gap/2)])
        
        
    #find local maximum slope index
    local_slope_index = 0
    max_slope = 0
    for i in range(len(time_list)):
        if time_list[i] > 400 and time_list[i] < 600:
            if slope_list[i] > max_slope:
                local_slope_index = i
                max_slope = slope_list[i]
    
    print(local_slope_index,time_list[local_slope_index],slope_list[local_slope_index])
    local_slope_index =  local_slope_index + int(gap/2)
    print(local_slope_index)
    
    fig, ax1 = plt.subplots()
    
    color = 'tab:red'
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Cumulative AE Count', color=color)  # we already handled the x-label with ax1
    ax1.plot(np.array(record_time_list),np.arange(len(record_time_list)), color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    #plt.ylim(-4,4)    
    
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        
    color = 'tab:blue'
    ax2.set_ylabel('Cumulative AE count Slope [1/s]', color=color)
    ax2.plot(np.array(time_list),np.array(slope_list), color=color) 
    ax2.tick_params(axis='y', labelcolor=color)
    #plt.ylim(-2000,8000)
       
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()
    
    return local_slope_index


#plot slope of strain vs time
def plot_strain_slope(strain_list,stress_list):
    
    gap = 100
    new_strain_list = []
    slope_list = []
    for i in range(len(strain_list)-gap):
        slope = (stress_list[i+gap] - stress_list[i]) / (strain_list[i+gap] - strain_list[i])
        slope_list.append(slope)
        new_strain_list.append(strain_list[i+int(gap/2)])
        
        
    #find local minimum slope index
    local_slope_index = 0
    min_slope = 10000
    for i in range(len(new_strain_list)):
        if new_strain_list[i] > 0 and new_strain_list[i] < 1.2:
            if slope_list[i] < min_slope:
                local_slope_index = i
                min_slope = slope_list[i]
    
    print(local_slope_index,new_strain_list[local_slope_index],slope_list[local_slope_index])
    local_slope_index =  local_slope_index + int(gap/2)
    print(local_slope_index)
    
    fig, ax1 = plt.subplots()
    
    color = 'tab:red'
    ax1.set_xlabel('strain')
    ax1.set_ylabel('slope [Mpa]', color=color)
    ax1.plot(np.array(new_strain_list),np.array(slope_list), color=color) 
    ax1.tick_params(axis='y', labelcolor=color)
    #plt.ylim(-4,4)
    
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()
    
    return local_slope_index


#find stress of a given time
#input: a list of axial/hoop stress value, a list of stress time, time to find stress
#output: axial stress and hoop stress
def find_stress(axial_stress_list,hoop_stress_list,stress_time,time):
    time_index = 0
    for i in range(len(stress_time)-1):
        if stress_time[i] < time and stress_time[i+1] >= time:
            time_index = i
            break
    
    axial_stress = axial_stress_list[time_index]
    hoop_stress = hoop_stress_list[time_index]
              
    return axial_stress, hoop_stress


def main():
    RMS_list = []
    first_arrival_list = []
    real_record_time_list = []
    record_time_list = []
    data_file = ['1','2','3','4','5','6','7','8','9','10']
    time_last = 0
    for file in data_file:
        print("Read file"+file)
        number_of_AE,record_time,AE_list = read_AE_file("C:/Users/huxxx/OneDrive/Desktop/SiC/SiC real specimen testing/01_22_2022/" + file + ".txt")
        real_record_time_list = real_record_time_list + record_time
        record_time_adjustment = record_time[0]-time_last
        for i in range(number_of_AE):
            record_time[i] = record_time[i] - record_time_adjustment
        time_last = record_time[-1]
        record_time_list = record_time_list + record_time
        
        #get first arrival and RMS 
        for i in range(number_of_AE):
            first_arrival = calculate_first_arrival(AE_list[i])
            first_arrival_list.append(first_arrival)
            
            if i < 100:
                plot_AE_event(AE_list[i])
        
            if first_arrival !=0:
                RMS = calculate_RMS(AE_list[i],first_arrival,300)
                RMS_list.append(RMS)
            else:
                RMS_list.append(0)
                
    
    #plot_cum(record_time_list,np.cumsum(RMS_list))
    #plot_AE_slope(record_time_list,np.cumsum(RMS_list))
    #plot_AE_slope(record_time_list,np.arange(len(record_time_list)))
    
    '''
    
    stress_file = open("C:/Users/huxxx/OneDrive/Desktop/SiC/SiC real specimen testing/11_04_2021/stress.txt",'r')
    lines = stress_file.readlines()
    stress_time = []
    axial_stress_list = []
    hoop_stress_list = []
    axial_strain_list = []
    hoop_strain_list = []
    for line in lines:
        time = float(line.split()[0])   
        #print(minute,second)
        real_time = round(time + 1437.99+208.01)
        stress_time.append(real_time)
        axial_stress_list.append(float(line.split()[7]))
        hoop_stress_list.append(float(line.split()[8]))
        axial_strain_list.append(float(line.split()[9]))
        hoop_strain_list.append(float(line.split()[10]))
    stress_file.close()
    
    
    estimate_time = real_record_time_list[3597]
    print(estimate_time)
    axial_stress,hoop_stress = find_stress(axial_stress_list,hoop_stress_list,stress_time,estimate_time)
    print(axial_stress,hoop_stress)
    
    plot_strain_slope(axial_strain_list,axial_stress_list)
    print(axial_stress_list[1152],hoop_stress_list[1152])
    
    '''

if __name__ == "__main__":
    main()