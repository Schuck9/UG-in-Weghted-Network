"""
a script for extracting data
@date: 2020.4.24
@author: Tingyu Mo
"""

import os
import numpy as np
import pandas as pd

def find_NewFile(path):
        #获取文件夹中的所有文件
        lists = os.listdir(path)
        #对获取的文件根据修改时间进行排序
        lists.sort(key=lambda x:os.path.getmtime(path +'/'+x))
        # lists.sort()
        #把目录和文件名合成一个路径
        if lists:
            name= lists[-1]
            file_new = os.path.join(path,name)
            return file_new,name
        else:
            return None,None
        
def data_extract(data_path,save_path,num_extract_file=4):
    file_list = os.listdir(data_path)
    file_list.sort()
    if (file_list[-1] == "extracted_data.csv"):
        del file_list[-2:]
    elif (file_list[-1] == "data_extract.py"):
        del file_list[-1:]
    extracted_data = []
    for file_name in file_list:
        sub_dir = os.path.join(data_path,file_name)
        last_file,last_file_name = find_NewFile(sub_dir)
        if last_file:
            print(last_file)
            epoch_data = os.listdir(last_file)
            epoch_data.sort()
            avg_pq_path= os.path.join(last_file,epoch_data[1])
            avg_pq = pd.read_csv(avg_pq_path)
            avg_strategy = avg_pq.values[-1][1:]
            extracted_data.append([last_file_name,avg_strategy[0],avg_strategy[1]])
    
    extracted_data_pd = pd.DataFrame(data =extracted_data)
    extracted_data_pd.to_csv(save_path)


if __name__ == "__main__":
    num_extract_file = 4
    save_path = "./extracted_data.csv"
    data_path = os.getcwd()
    data_extract(data_path,save_path,num_extract_file)

    