"""
Ultimatum Game in complex network Visualization 
@date: 2020.3.19
@author: Tingyu Mo
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os


def pq_distribution(value_list):
    x_axis = np.arange(0,1.05,1/20) # 21 descrete points,range 0~1,step size 0.05
    y_axis = np.zeros(x_axis.size)
    for v in value_list:
        for i in range(x_axis.size):
            if abs(v-x_axis[i]) < 0.05:
                y_axis[i] += 1
    return y_axis

def pq_distribution_viz(RecordName,time_option = "all"):
    # Epoch_list = ['1','100','1000','20000']
    Epoch_list = ['100','1000','20000']
    result_dir = "./result"
    record_dir = os.path.join(result_dir,RecordName)
    checkpoint_list = os.listdir(record_dir)
    parse_str = checkpoint_list[0].split("_")
    del(parse_str[-1])
    info_str = '_'.join(parse_str)
    save_path =os.path.join(record_dir, info_str+'.jpg')
    y_axis_plist = []
    y_axis_qlist = []
    for Epoch in Epoch_list:
        info_e = info_str+"_"+Epoch
        Epoch_dir = os.path.join(record_dir,info_e )
        strategy_path = os.path.join(Epoch_dir,info_e+"_strategy.csv")
        strategy = pd.read_csv(strategy_path)
        # strategy.reset_index(drop = True)
        pq_array = strategy.values
        # np.delete(pq_array,1,axis=1)
        p = pq_array[0][1:]
        q = pq_array[1][1:]
        # del(p[0])
        # del(q[0])
        p = pq_distribution(p)
        q = pq_distribution(q)
    
        y_axis_plist.append(p/10000)
        y_axis_qlist.append(q/10000)

    plt.figure()
    x_axis = np.arange(0,1.05,1/20)
    # plt.rcParams['font.sans-serif']=['SimHei']
    # plt.rcParams['axes.unicode_minus'] = False
    # # plt.title("")
    plt.xlabel("p")#x轴p上的名字
    plt.ylabel("D(p)")#y轴上的名字
    plt.plot(x_axis, y_axis_plist[0] ,marker='^',linestyle='-',color='skyblue', label='t = 100')
    plt.plot(x_axis, y_axis_plist[1], marker='s',linestyle='-',color='green', label='t = 1000')
    plt.plot(x_axis, y_axis_plist[2], marker='*',linestyle='-',color='red', label='t = 20000')
    # plt.plot(x_axis, thresholds, color='blue', label='threshold')
    plt.legend(loc = 'upper right') # 显示图例
    plt.savefig(save_path)
    print("Figure has been saved to: ",save_path)
    plt.show()


def avg_pq_viz():
    '''
    Figure 2 like
    '''
    u = 0.1
    info = 'RG_Weighted_0.4'
    save_path = "./result/{}_u_{}.jpg".format(info,u)
    x_label = [0.001,0.01,0.1,1,10]
    x_axis = np.log10(x_label)
    avg_list = [ (0.5,0.5),
                            (0.494011917,0.496625418),(0.498278643,0.471188505),
                            (0.341997159,0.261274376),(0.124914813,0.115971024),
                            ]
    p_axis = list()
    q_axis = list()
    for stg in avg_list:
        p,q = stg
        p_axis.append(p)
        q_axis.append(q)

    plt.figure()
    # plt.rcParams['font.family'] = ['sans-serif']
    # plt.rcParams['font.sans-serif'] = ['SimHei']
    # plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.title(" {} u={}".format(info,u))
    plt.xlabel("Selection strength(w)")#x轴p上的名字
    plt.ylabel("Mean")#y轴上的名字
    plt.xticks(x_axis,x_label,fontsize=16)
    plt.plot(x_axis, p_axis,marker='^',linestyle='-',color='skyblue', label='Offer (p)')
    plt.plot(x_axis, q_axis, marker='s',linestyle='-',color='red', label='Demand (q)')
    # plt.plot(x_axis, thresholds, color='blue', label='threshold')
    plt.legend(loc = 'upper right') # 显示图例
    plt.savefig(save_path)
    print("Figure has been saved to: ",save_path)
    plt.show()


def data_loader(data_path):
    
    weight_axis = [0.25, 0.3, 0.35, 0.4, 0.55, 0.7, 0.85]
    weight_axis = [str(x )for x in weight_axis]
    u = [0.001, 0.01, 0.1]
    u = [str(i) for i in u]
    w = [0.001, 0.01, 0.1, 1, 10]
    w = [str(i) for i in w]
    data = pd.read_excel(data_path)
    data_dict = dict()
    for weight_key in weight_axis:
        data_dict[weight_key] = dict()
        for u_key in u:
            data_dict[weight_key][u_key] =dict()
            # for w_key in w:
            #     data_dict[weight_key][u_key][w_key] = np.zeros([1,2])
    
    pq_data =data[['p','q']].dropna()
    for i,weight_key in enumerate(weight_axis):
        weight_data = pq_data.iloc[15*i:15*(i+1)]
        
        for j,u_key in enumerate(u):
            u_data = weight_data.iloc[5*j:5*(j+1)].values
            for k,w_key in enumerate(w):
                # print(u_data[k])
                data_dict[weight_key][u_key][w_key] = u_data[k]
    print("data loaded!")
    return data_dict

def weighted_graph_viz():
    data_path ='./result/data.xlsx'
    data_dict= data_loader(data_path)
    weight_axis = [0.25, 0.3, 0.35, 0.4, 0.55, 0.7, 0.85]
    weight_axis_str = [str(x )for x in weight_axis]
    x_axis = weight_axis
    pq = ['Offer(p)','Demond(q)']
    u = [0.001, 0.01, 0.1]
    u = [str(i) for i in u]
    w = [0.001, 0.01, 0.1, 1, 10]
    w = [str(i) for i in w]
    
    for k ,role in enumerate(pq):
        for u_ in u:
            y_list = []
            for w_ in w:
                ls = []
                for i,weight_key in enumerate(weight_axis_str):
                    ls.append(data_dict[weight_key][u_][w_][k])
                y_list.append(ls)

            # print("y_axis done!")

            info_str = role+"_"+u_
            save_path = './result/{}.jpg'.format(info_str)
            plt.figure()
            
            # plt.rcParams['font.sans-serif']=['SimHei']
            # plt.rcParams['axes.unicode_minus'] = False
            plt.title(info_str)
            plt.xlabel("weight")#x轴p上的名字
            plt.ylabel("{}".format(role))#y轴上的名字
            plt.plot(x_axis, y_list[0] ,marker='>',linestyle='-',color='purple', label='w = 0.001')
            plt.plot(x_axis, y_list[1] ,marker='^',linestyle='-',color='skyblue', label='w = 0.01')
            plt.plot(x_axis, y_list[2], marker='s',linestyle='-',color='green', label='w = 0.1')
            plt.plot(x_axis, y_list[3], marker='*',linestyle='-',color='red', label='w = 1')
            plt.plot(x_axis, y_list[4], marker='x',linestyle='-',color='black', label='w = 10')
            # plt.plot(x_axis, thresholds, color='blue', label='threshold')
            plt.legend(loc = 'upper right') # 显示图例
            plt.savefig(save_path)
            print("Figure has been saved to: ",save_path)
            plt.show()

if __name__ == '__main__':

    # RecordName ='2020-03-03-09-14-20'   
    # time_option = "all"
    # pq_distribution_viz(RecordName,time_option)
    # avg_pq_viz()
    weighted_graph_viz()