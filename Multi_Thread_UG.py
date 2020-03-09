"""
Multi process implement of UG in complex network
@date: 2020.3.8
@author: Tingyu Mo
"""

import os
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import threading
from UG_Complex_Network import*
# multiprocessing.set_start_method('spawn',True)


class Run(threading.Thread):
    def __init__(self,node_num = 100,network_type = "RG",update_rule ="EF",player_type = "C",
        avg_degree = 4,intensity_selection = 0.01,mutate_rate = 0.001,check_point = None,Epochs = 1000000):
        super().__init__()
        self.node_num = node_num
        self.avg_degree = avg_degree
        self.network_type = network_type # "SF" or "ER"
        self.player_type = player_type # "A" or "B" "C"
        self.update_rule = update_rule # "SP" or "SP"
        self.max_weight = 0.4
        self.intensity_selection = intensity_selection
        self.mutate_rate = mutate_rate
        self.check_point = check_point
        self.Epochs = Epochs
    def run(self):
        info = "{}_{}_{}_{}_{}".format(self.network_type,self.player_type,self.update_rule,self.intensity_selection,self.mutate_rate)
        print(info," is running!")
        Game( 
                    self.node_num,
                    self.network_type,# "SF" or "ER"
                    self.update_rule, # "SP" or "SP"
                    self.player_type , # "A" or "B" "C"
                    self.avg_degree,
                    self.intensity_selection,
                    self.mutate_rate,
                    self.check_point ,
                    self.Epochs,
                    info)

        
if __name__ == '__main__':
    p1=Run(node_num = 100,network_type = "RG",update_rule ="EF",player_type = "C",
            avg_degree = 4,intensity_selection = 10,mutate_rate = 0.001,check_point = None)
    p2=Run(node_num = 100,network_type = "RG",update_rule ="EF",player_type = "C",
            avg_degree = 4,intensity_selection = 10,mutate_rate = 0.1,check_point = None)
    p3=Run(node_num = 100,network_type = "RG",update_rule ="EF",player_type = "C",
            avg_degree = 4,intensity_selection = 1,mutate_rate = 0.01,check_point = None)
    p4=Run(node_num = 100,network_type = "RG",update_rule ="EF",player_type = "C",
            avg_degree = 4,intensity_selection = 1,mutate_rate = 0.1,check_point = None)
    p1.start() #start会自动调用run
    time.sleep(3)
    p2.start()
    time.sleep(3)
    p3.start()
    time.sleep(3)
    p4.start()
  
    p1.join()#等待p1进程停止
    p2.join()
    p3.join()
    p4.join()
    print('Done!')

