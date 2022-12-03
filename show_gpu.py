import torch
import GPUtil
from os import system, name
import time


def sleep_time(hour, min, sec):
    return hour * 3600 + min * 60 + sec


def clear():
 
    # for windows
    if name == 'nt':
        _ = system('cls')
 
    # for mac and linux(here, os.name is 'posix')
    else:
        _ = system('clear')


def main():
    Gpus = GPUtil.getGPUs()
    #second = sleep_time(0, 0, 3)
    while True:
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        for gpu in Gpus:
            print("GPU: {}, Total: {}, Ued: {}, Free: {}".format(gpu.id, gpu.memoryTotal, gpu.memoryUsed, gpu.memoryFree) )
        time.sleep(5)

if __name__ == '__main__':
    main()