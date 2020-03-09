"""
A simple implementation of Ultimatum Game in complex network
@date: 2020.3.8
@author: Tingyu Mo
"""

import os
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time

class UG_Complex_Network():
    def __init__(self,node_num = 10000,network_type = "SF",update_rule ="NS",player_type = "B",
        avg_degree = 4,intensity_selection = 0.01,mutate_rate = 0.001,check_point = None):
        self.node_num = node_num
        self.avg_degree = avg_degree
        self.network_type = network_type # "SF" or "ER"
        self.player_type = player_type # "A" or "B" "C"
        self.update_rule = update_rule # "SP" or "SP"
        self.max_weight = 0.25
        self.intensity_selection = intensity_selection
        self.mutate_rate = mutate_rate
        self.avg_strategy = (0,0)
        self.avg_pq_list=[]

        if not os.path.exists("./result"):
            os.mkdir('./result')

        if check_point == None:
            self.dir_str = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
            os.mkdir("./result/{}".format(self.dir_str))
        else:
            self.dir_str = check_point
            
    
    def build_network(self,network_type = None):
        '''
        building network
        '''
        print("Building network!")
        G = None
        if network_type == None:
            network_type = self.network_type
        
        if network_type == "SF":
            G = nx.random_graphs.barabasi_albert_graph(self.node_num, int(self.avg_degree/2))
            G = self.network_weights_asign(G)
        elif network_type == "ER":
            G = nx.random_graphs.erdos_renyi_graph(self.node_num, self.avg_degree/self.node_num)
            for n in G.nodes():
                if G.degree(n) == 0:
                    while True:
                        nbr = np.random.choice(G.nodes(),size = 1)[0]
                        if nbr != n:
                            break
                    G.add_edge(n, nbr)
        elif network_type == "RG":
            G = nx.random_graphs.random_regular_graph(self.avg_degree, self.node_num)
            G = self.network_weights_asign(G)

                 

        print("平均连接度为: ",self.avg_degree_caculate(G))
        return G

    def nbr_weighted_check(self,G,n):
        is_MaxWeight_exis = None
        for nbr in G.adj[n]:
            weight = G.edges[n, nbr]['weight']
            if weight == self.max_weight:
                is_MaxWeight_exis = nbr
                break
        return is_MaxWeight_exis
        
    def network_weights_asign(self,G):
        #边权重初始化
        for n in list(G.nodes()):
            nbrs = list(G.adj[n])
            for nbr in nbrs:
                G.edges[n, nbr]['weight'] = 0
        
        #检查双方是否存在紧密联系者
        for n in list(G.nodes()):
            nbrs = list(G.adj[n])
            for nbr in nbrs:
                isMaxWeightExisIn_N = self.nbr_weighted_check(G,n)
                isMaxWeightExisIn_Nbr = self.nbr_weighted_check(G,nbr)
                if (isMaxWeightExisIn_N == None) and (isMaxWeightExisIn_Nbr == None):
                    G.edges[n, nbr]['weight'] = self.max_weight
                elif (isMaxWeightExisIn_N==nbr) and (isMaxWeightExisIn_Nbr == n):
                    G.edges[n, nbr]['weight'] = self.max_weight
                elif (isMaxWeightExisIn_N != None) or (isMaxWeightExisIn_Nbr != None) :
                    G.edges[n, nbr]['weight'] = (1-self.max_weight)/(self.avg_degree-1)
                    
        # 打印输出
        # for n, nbrs in G.adjacency():
        #     for nbr, eattr in nbrs.items():
        #         data = eattr['weight']
        #         print('(%d, %d, %0.3f)' % (n,nbr,data))
        cnt = 0
        for n in list(G.nodes()):
            result = self.nbr_weighted_check(G,n)
            if result == None:
                cnt += 1
        print("无亲密关系者率:",cnt/self.node_num)

        return G

    def initialize_strategy(self,G):
        '''
        initialize every node's strategy
        '''
        self.strategy_asigned(G,list(G.nodes()),Type = self.player_type)

    def initialize_payoff(self,G):
        '''
        clear all player's payoff
        '''
        for n in list(G.nodes()):
            G.nodes[n]['payoff'] = 0 

    def fitness_calculate(self,G,n):
        '''
        f = 1-w+w*Π
        '''
        return 1 - self.intensity_selection + self.intensity_selection*G.nodes[n]['payoff']

    def strategy_asigned(self,G,node_list,Type = 'B'):
        '''
        A B C ,three types inBdividual
        '''
        if Type == 'B':
            for n in node_list:
                #Type-A player
                strategy = np.random.rand()
                G.nodes[n]['p'] = strategy 
                G.nodes[n]['q'] = 1-strategy
                G.nodes[n]['payoff'] = 0 
        elif Type == 'A':
            for n in node_list:
                #Type-A player
                strategy = np.random.rand()
                G.nodes[n]['p'] = strategy 
                G.nodes[n]['q'] = strategy
                G.nodes[n]['payoff'] = 0 
        elif Type == 'C':
            for n in node_list:
                #Type-A player
                G.nodes[n]['p'] = np.random.rand()
                G.nodes[n]['q'] = np.random.rand()
                G.nodes[n]['payoff'] = 0 

    def synchronous_play(self,G):
        '''
        using synchronous method to play ultimatum game 
        and update graph every generation
        '''
        for n, nbrs in G.adjacency():
            for nbr, _ in nbrs.items():
                # proposer = n ,responder = nbr
                offer = G.nodes[n]['p']
                demand = G.nodes[nbr]['q']
                weight = G.edges[n, nbr]['weight'] 
                if offer > demand:
                    G.nodes[n]['payoff'] += (1-offer)*weight
                    # G.nodes[nbr]['payoff'] += offer

                # proposer = nbr ,responder = n
                offer = G.nodes[nbr]['p']
                demand = G.nodes[n]['q']
                if offer > demand:
                    # G.node[nbr]['payoff'] += 1-offer
                    G.nodes[n]['payoff'] += offer*weight
            num_nbrs = G.degree(n)
            # if num_nbrs != 0:
            #     G.nodes[n]['payoff'] /= G.degree(n)
        
    def natural_selection(self,G):
        '''
        each player i in the network selects at random one neighbor j 
        and compares its payoff Πi with that of j
        '''
        cnt = 0
        for n in list(G.nodes()):
            nbrs = list(G.adj[n])
            nbr = np.random.choice(nbrs,size = 1)[0]
            n_payoff = G.nodes[n]['payoff']
            nbr_payoff = G.nodes[nbr]['payoff']
            if nbr_payoff > n_payoff:
                probs_adopt =  (nbr_payoff - n_payoff)/(2*max(G.degree(n),G.degree(nbr)))
                if np.random.rand() < probs_adopt:
                    # n adopts nbr's strategy
                    cnt += 1
                    G.nodes[n]['p'] = G.nodes[nbr]['p']
                    G.nodes[n]['q'] = G.nodes[nbr]['q']
        # print("occur:",cnt)

    def social_penalty(self,G):
        '''
        remove the player with lowest payoff and replace it with random one
        '''
        lowest_n = 0
        for n in G.nodes():
            if G.nodes[n]['payoff'] < G.nodes[lowest_n]['payoff']:
                lowest_n = n

        lowest_cluster = list(G.adj[lowest_n])
        lowest_cluster.append(lowest_n)
        
        self.strategy_asigned(G,lowest_cluster,Type = self.player_type)
        # for n in lowest_cluster:
        #     #Type-A player
        #     strategy = np.random.rand()
        #     G.nodes[n]['p'] = strategy 
        #     G.nodes[n]['q'] = strategy
        #     G.nodes[n]['payoff'] = 0 

    def death_birth_updating(self,G):
        '''
        death-birth updating process,
        choose an individual randomly over the whole population,
        reproduce the strategy with proportional to nbr's fitness
        '''
        individual_list = list(G.nodes())
        chosen_individual = np.random.choice(individual_list,size=1)[0]
        nbrs = list(G.adj[chosen_individual])
        reproduce_probability = list()
        for nbr in nbrs:
            rp = self.fitness_calculate(G,nbr)
            reproduce_probability.append(rp)
        reproduce_probability = np.array(reproduce_probability)
        reproduce_probability /= sum(reproduce_probability)
        reproduce_individual = np.random.choice(nbrs,size=1,p = reproduce_probability)[0]
        G.nodes[chosen_individual]['p'] = G.nodes[reproduce_individual]['p']
        G.nodes[chosen_individual]['q'] = G.nodes[reproduce_individual]['q']

        return chosen_individual,reproduce_individual

    def birth_death_updating(self,G):
        '''
        birth death updating process,
        choose an individual with proportional to fitnees
        replace one of its nbr randomly
        '''
        individual_list = list(G.nodes())
        fitness_list = list()
        for n in individual_list:
            fitness = self.fitness_calculate(G,n)
            fitness_list.append(fitness)
        fitness_list = np.array(fitness_list)
        fitness_list /= sum(fitness_list) 
        reproduce_individual = np.random.choice(individual_list,size = 1,p = fitness_list)[0]
        nbrs = list(G.adj[reproduce_individual])
        chosen_individual = np.random.choice(nbrs,size = 1)[0]
        G.nodes[chosen_individual]['p'] = G.nodes[reproduce_individual]['p']
        G.nodes[chosen_individual]['q'] = G.nodes[reproduce_individual]['q']

        return chosen_individual,reproduce_individual
           
    def pairwise_comparison(self,G):
        '''
        pairwise comparison process,
        choose an individual and its nbr randomlyj
        individual imitate its nbr's strategy with probability of 1/1+e^(-w*(Πi-Πj))
        '''    
        individual_list = list(G.nodes())
        chosen_individual = np.random.choice(individual_list,size=1)[0]
        nbrs = list(G.adj[chosen_individual])
        reproduce_individual = np.random.choice(nbrs,size = 1)[0]
        imitate_probability = 1/(1+np.exp(-1*self.intensity_selection*(G.nodes[chosen_individual]['payoff']-G.nodes[reproduce_individual]['payoff'])))
        if np.random.rand() < imitate_probability:
            G.nodes[chosen_individual]['p'] = G.nodes[reproduce_individual]['p']
            G.nodes[chosen_individual]['q'] = G.nodes[reproduce_individual]['q']   
        
        return chosen_individual,reproduce_individual

    def imitaion_updaing(self,G):
        '''
        imitaion updating process,
        choose an individual randomly,
        update its strategy with proportional to  its & nbrs fitness
        '''
        individual_list = list(G.nodes())
        chosen_individual = np.random.choice(individual_list,size=1)[0]
        fitness_list = list()
        nbrs = list(G.adj[chosen_individual])
        for n in nbrs:
            fitness = self.fitness_calculate(G,n)
            fitness_list.append(fitness)
        nbrs.append(chosen_individual)
        near_domain = nbrs
        fitness_ci = self.fitness_calculate(G,chosen_individual)
        fitness_list.append(fitness_ci)
        fitness_list = np.array(fitness_list)
        fitness_list /= sum(fitness_list)
        reproduce_individual = np.random.choice(near_domain,size =1,p = fitness_list)[0]
        G.nodes[chosen_individual]['p'] = G.nodes[reproduce_individual]['p']
        G.nodes[chosen_individual]['q'] = G.nodes[reproduce_individual]['q'] 

        return chosen_individual,reproduce_individual

    def evolution_fairness(self,G):
        '''
        update process in Evolution of fairness in the one-shot anonymous UG
        '''
        individual_list = list(G.nodes())
        pay_off = self.get_all_values(G,'payoff')
        pay_off = np.array(pay_off)
        effective_payoff = np.exp(self.intensity_selection*pay_off)
        effective_payoff = effective_payoff*1.0/np.sum(effective_payoff) 
        reproduce_individual = np.random.choice(a = individual_list ,size = 1,p = effective_payoff)[0]
        chosen_individual = np.random.choice(a = individual_list,size = 1)[0]
        G.nodes[chosen_individual]['p'] = G.nodes[reproduce_individual]['p']
        G.nodes[chosen_individual]['q'] = G.nodes[reproduce_individual]['q'] 
        return chosen_individual,reproduce_individual

    def mutation(self,G,chosen_individual,reproduce_individual):
        if np.random.rand(1) <= self.mutate_rate:
            G.nodes[chosen_individual]['p'],G.nodes[chosen_individual]['q'] = np.random.rand(2)
            # print("MC")
        # else:
        #     G.nodes[chosen_individual]['p'] = G.nodes[reproduce_individual]['p']
        #     G.nodes[chosen_individual]['q'] = G.nodes[reproduce_individual]['q']   

    def update(self,G):
        '''
        natural seletion an social penalty
        '''
        if self.update_rule == "NS":
            chosen_individual,reproduce_individual = self.natural_selection(G)
        elif self.update_rule == "SP":
            chosen_individual,reproduce_individual = self.social_penalty(G)
        elif self.update_rule == "EF":
            chosen_individual,reproduce_individual = self.evolution_fairness(G)
        elif self.update_rule == "DB":
            chosen_individual,reproduce_individual = self.death_birth_updating(G)
        elif self.update_rule == "BD":
            chosen_individual,reproduce_individual = self.birth_death_updating(G)
        elif self.update_rule == "PC":
            chosen_individual,reproduce_individual = self.pairwise_comparison(G)
        elif self.update_rule == "IU":
            chosen_individual,reproduce_individual = self.imitaion_updaing(G)
        
        self.mutation(G,chosen_individual,reproduce_individual)

    def avg_strategy_calculate(self,G,Epoch):
        '''
        calculate the mean strategy over arg Epoch
        '''
        p_vector = self.get_all_values(G,'p')
        q_vector = self.get_all_values(G,'q')
        p,q = self.avg_strategy #the Epoch-1's average strategy
        p = ((p*self.node_num*(Epoch-1)) + np.sum(p_vector))*1.0/(Epoch*self.node_num)
        q = ((q*self.node_num*(Epoch-1)) + np.sum(q_vector))*1.0/(Epoch*self.node_num)
        # self.avg_pq_list.append((p,q))
        return (p,q)
    
    def save(self,G,Epoch):
        #Save Graph
        result_dir = './result/'
        info = "{}_{}_{}_{}_{}_{}".format(self.network_type,self.player_type,self.update_rule,self.intensity_selection,self.mutate_rate,Epoch)
        Epoch_dir = os.path.join(result_dir,self.dir_str,info)
        if not os.path.exists(Epoch_dir):
            os.mkdir(Epoch_dir)
        graph_path = os.path.join(Epoch_dir,info+"_Graph.yaml")
        nx.write_yaml(G,graph_path)
        #Save strategy
        p_vector = self.get_all_values(G,'p')
        q_vector = self.get_all_values(G,'q')
        pq_array = np.vstack((p_vector,q_vector))
        pq_path = os.path.join(Epoch_dir,info+"_strategy.csv")
        pq = pd.DataFrame(data = pq_array)
        pq.to_csv(pq_path)
        #Save average offer/respond
        avg_pq_path = os.path.join(Epoch_dir,info+"_average_strategy.csv")
        avg_pq = pd.DataFrame(data = self.avg_pq_list)
        avg_pq.to_csv(avg_pq_path)


    def retrain(self,filepath):
        '''
        continue evolution from specific check point
        '''
        print(filepath)
        filepath = os.path.join('./result/',filepath)
        lists = os.listdir(filepath)   
        lists.sort(key=lambda fn: os.path.getmtime(filepath + "/" + fn)) 
        result_dir = os.path.join(filepath, lists[-1])      
        result_list = os.listdir(result_dir)
        result_list.sort()
        parse_str = result_list[0][:-5].split("_")
        self.network_type = parse_str[0]
        self.player_type = parse_str[1]
        self.update_rule = parse_str[2]
        self.intensity_selection =  float(parse_str[3])
        self.mutate_rate = float(parse_str[4])
        Epoch = int(parse_str[5])
        graph_path = os.path.join(result_dir,result_list[0])
        avg_pq_path = os.path.join(result_dir,result_list[1])
        avg_pq = pd.read_csv(avg_pq_path)
        self.avg_strategy = avg_pq.values[-1][1:]
        G = nx.read_yaml(graph_path)
        return G,Epoch+1
        

    def get_all_values(self,G,attr_name):
        '''
        get specific attribute values of all nodes
        '''
        value_dict = nx.get_node_attributes(G,attr_name)
        value_list = list(value_dict.values())
        return value_list

    def pq_distribution(self,G,attr_name):

        x_axis = np.arange(0,1.05,1/20) # 21 descrete points,range 0~1,step size 0.05
        y_axis = np.zeros(x_axis.size)
        value_list = self.get_all_values(G,attr_name)
        for v in value_list:
            for i in range(x_axis.size):
                if abs(v-x_axis[i]) < 0.05:
                    y_axis[i] += 1
        
        return (x_axis,y_axis)


        
    def avg_degree_caculate(self,G):
        '''
        caculate average degree of graph
        '''
        degree_total = 0
        for x in range(len(G.degree())):
            degree_total = degree_total + G.degree(x)
        return degree_total/self.node_num


def Game(node_num,network_type,update_rule,player_type,avg_degree,intensity_selection,mutate_rate,check_point,Epochs ,info):
    if check_point != None:
        UG = UG_Complex_Network(node_num,network_type,update_rule,player_type,avg_degree,intensity_selection,mutate_rate,check_point)
        G,Start  = UG.retrain(check_point)
    else:
        Start = 1
        UG = UG_Complex_Network(node_num,network_type,update_rule,player_type,avg_degree,intensity_selection,mutate_rate)
        #bulids network structure
        G = UG.build_network()
        #initialize the strategy of player in network
        UG.initialize_strategy(G)
    #play game
    for Epoch in range(Start,Epochs+1):
        UG.initialize_payoff(G)
        UG.synchronous_play(G)
        UG.update(G)
        UG.avg_strategy = UG.avg_strategy_calculate(G,Epoch)
        if Epoch % 10000 == 0:
            print("Thread info:",info)
            print("Epoch[{}]".format(Epoch))
            print("Average strategy: (p ,q)={}\n".format(UG.avg_strategy))
            UG.avg_pq_list.append(UG.avg_strategy)
            UG.save(G,Epoch)
            # UG.viz(G)



if __name__ == '__main__':

    node_num = 100
    network_type = "RG" # [SF, ER, RG]
    update_rule ='EF'   # [NS, SP, EF,DB, BD, PC, IU]
    player_type = "C" # [A=(p=q,q), B=(p,1-p), C=(p,q)]
    avg_degree = 4
    intensity_selection = 0.1
    mutate_rate = 0.01
    Epochs = pow(10,7)
    check_point = None
    # check_point = '2020-03-08-11-52-42'

    if check_point != None:
        UG = UG_Complex_Network(node_num,network_type,update_rule,player_type,avg_degree,intensity_selection,mutate_rate,check_point)
        G,Start  = UG.retrain(check_point)
    else:
        Start = 1
        UG = UG_Complex_Network(node_num,network_type,update_rule,player_type,avg_degree,intensity_selection,mutate_rate)
        #bulids network structure
        G = UG.build_network()
        #initialize the strategy of player in network
        UG.initialize_strategy(G)
    #play game
    for Epoch in range(Start,Epochs+1):
        UG.initialize_payoff(G)
        UG.synchronous_play(G)
        UG.update(G)
        UG.avg_strategy = UG.avg_strategy_calculate(G,Epoch)
        if Epoch % 10000 == 0:
            print("Epoch[{}]".format(Epoch))
            print("Average strategy: (p ,q)={}\n".format(UG.avg_strategy))
            UG.avg_pq_list.append(UG.avg_strategy)
            UG.save(G,Epoch)
            # UG.viz(G)
            
        