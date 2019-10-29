# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""
Created on Sat Oct 26 15:26:03 2019

@author: cw817615
"""

import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号


class Unit:
    '''个体类'''
    
    def __init__(self, x_min, x_max, dim):
        '''
        初始化方法，参数含义如下：
        params: x_min:某一维的最小值
        params: x_max:某一维的最大值
        params: dim:维度数
        '''
        # 将个体的信息全部设置为私有变量
        # 个体的初始化向量，random为(0,1)之间的随机数(最优解)
        self.quepos = Question()
        self.__pos = np.array([x_min + random.random()*(x_max - x_min) for i in range(dim)])
        self.__pos_decode = self.quepos.decoding(self.__pos)[0]   # 解码后的
        # 个体突变后的向量
        self.__mutation = np.array([0.0 for i in range(dim)])  
        self.__mu_decode = self.quepos.decoding(self.__mutation)[0] 
        # 个体交叉后的向量
        self.__crossover = np.array([0.0 for i in range(dim)])  
        self.__cro_decode = self.quepos.decoding(self.__crossover)[0]
        self.__spend = [0]*5
        # 个体适应度
        self.__fitnessValue = self.quepos.fit_fun1(self.__pos)  


    def set_pos(self, i, value):
        # 修改个体向量的第i个值为value
        self.__pos[i] = value
        


    def get_pos(self):
        # 添加一个获取个体向量的接口
        return self.__pos
    
    def set_pos2(self,pos_decode):
        # 修改最优解(用已经解码了的结果)
        self.__pos_decode = pos_decode
        
    def get_pos2(self):
        # 获取解码后的个体向量
        return self.__pos_decode

    def set_mutation(self, i, value):
        # 个体变异
        self.__mutation[i] = value

    def get_mutation(self):
        # 获取个体变异后的向量
        return self.__mutation   

    def set_crossover(self, i, value):
        # 个体交叉
        self.__crossover[i] = value

    def get_crossover(self):
        # 获取个体交叉后的向量
        return self.__crossover
    
    def set_crossover2(self,decode_result,spend):
        # 修改经过解码的交叉向量，主要是Shifting和kmove
        self.__cro_decode = decode_result
        self.__cro_spend = spend
        
    def get_crossover2(self):
        # 获取解码后的交叉向量
        return self.__cro_decode,self.__cro_spend

    def set_fitness_value(self, value):
        # 设置个体的适应度
        self.__fitnessValue = value

    def get_fitness_value(self):
        # 获取适应度
        return self.__fitnessValue



class DE:
    '''差分进化算法类'''
    
    def __init__(self, dim,agent_num, size, iter_num, x_min, x_max, best_fitness_value=float('Inf'), F=0.5, CR=0.8):
        '''
        参数释义如下：
        dim:维度
        agent_num:代理人的个数
        size:总群个数
        iter_num:迭代次数
        x_min:个体某一维最小值
        x_max:个体某一维最大值
        best_fitness_value:最佳的适应度值，初始设置为无穷大
        best_position:全局最优解，为1*dim维的向量
        fitness_val_list:每次迭代最优适应值
        F:缩放因子，一般在[0,2]之间选择，通常取0.5，主要影响算法的全局寻优能力
        CR:交叉概率(或称交叉系数)，取值范围为[0,1]之间的随机浮点数
        '''
        self.F = F
        self.CR = CR
        self.dim = dim
        self.agent_num = agent_num
        self.size = size  
        self.iter_num = iter_num  
        self.x_min = x_min
        self.x_max = x_max
        self.best_fitness_value = best_fitness_value
        self.best_position = np.array([0.0 for i in range(dim)])  
        self.fitness_val_list = []  
        self.ques = Question()
        self.__spend = [0]*5
        
        # 对种群进行初始化(generate a set of initial vectors)
        self.unit_list = [Unit(self.x_min, self.x_max, self.dim) for i in range(self.size)]

    ##----------------------------基础操作区-------------------------------##
    def get_kth_unit(self, k):
        # 获取种群中第k+1个个体
        return self.unit_list[k]

    def set_bestFitnessValue(self, value):
        # 设置最优适应度
        self.best_fitness_value = value

    def get_bestFitnessValue(self):
        # 获取最优适应度
        return self.best_fitness_value

    def set_bestPosition(self, pos_decode):
        # 设置全局最优解
        self.best_position = pos_decode

    def get_bestPosition(self):
        # 获取全局最优解
        return self.best_position
    
    
    ##----------------------------进化区----------------------------------##
    def mutation_fun(self):
        '''变异(mutation process)'''
        # 变异操作，对第g代随机抽取三个组成一个新的个体，对于第i个新个体来说，原来的第i个个体与它没有关系
        for i in range(self.size):
            # 获取进行交叉运算的个体，个体下标分别为r1,r2,r3
            r1 = r2 = r3 = 0
            # 保证获取不同的个体且与i不等
            while r1 == i or r2 == i or r3 == i or r2 == r1 or r3 == r1 or r3 == r2:
                r1 = random.randint(0, self.size - 1)  # 随机数范围为[0,size-1]的整数
                r2 = random.randint(0, self.size - 1)
                r3 = random.randint(0, self.size - 1)
            
            # 进行变异操作:v_i = x_r1 + F * (x_r2 - x_r3),mutation为1*dim维向量
            mutation = self.get_kth_unit(r1).get_pos() + \
                       self.F * (self.get_kth_unit(r2).get_pos() - self.get_kth_unit(r3).get_pos())
            
            for j in range(self.dim):
                #  判断变异后的值是否满足边界条件，不满足需重新生成
                if self.x_min <= mutation[j] <= self.x_max:
                    # 满足边界条件则将第i个个体的第j个值变异位mutation[j]
                    self.get_kth_unit(i).set_mutation(j, mutation[j])
                    
                else:
                    rand_value = self.x_min + random.random()*(self.x_max - self.x_min)
                    self.get_kth_unit(i).set_mutation(j, rand_value)


    def crossover(self):
        '''交叉(recombination process)'''
        # 我们的交叉是对变异种群和原始种群做交叉
        for unit in self.unit_list:
            for j in range(self.dim):
                
                rand_j = random.randint(0, self.dim - 1)  
                rand_float = random.random()   # 交叉概率
                
                if rand_float <= self.CR or rand_j == j:
                    # 交叉，即赋变异种群的值
                    unit.set_crossover(j, unit.get_mutation()[j])
                    
                # 不交叉，即赋原始种群的值
                else:
                    unit.set_crossover(j, unit.get_pos()[j])
            # 对交叉的结果解码并存储起来
            cro_decode,cro_spend = self.ques.decoding(unit.get_crossover())
            unit.set_crossover2(cro_decode,cro_spend)
                    
                    
    def Shifting(self):
        '''Shifting algorithm'''
        # 对种群每一个个体使用一次Shifting Algorithm
        cost = self.ques.get_cost()
        resource = self.ques.get_resource()
        spend_max = self.ques.get_capacity()  # 获取所有agent最大capacity
        for unit in self.unit_list:
            # agent为任务指派情况,spend为交叉解码后每一个agent capacity消耗情况
            agent,spend = unit.get_crossover2()
            # 依据lower cost来调整任务的指派
            for i in range(self.dim):
                # 将task_i指派给agent_j'，若能使花费变少
                # 先对任务i的cost排序,，因为Shifting算法是依据这个选择是否替换agent的
                agent_ind = np.argsort(cost[:,i])
                for j in agent_ind:
                    # 如果j就是最小的了
                    if j == agent[i] or cost[j,i] == cost[agent[i],i]:
                        break
                    # 花费少且j没触及capacity上限
                    elif cost[j,i] < cost[agent[i],i] and spend[j]+resource[j,i] < spend_max[j]:
                        spend[j] += resource[j,i]
                        # 之前做这个任务的就要减去resource
                        spend[agent[i]] -= resource[agent[i],i]
                        # 更换任务指派
                        agent[i] = j
            unit.set_crossover2(agent,spend)           
    
    
    def kmove(self):
        '''k-variable move algorithm'''
        #k = 3  
        #cost = self.ques.get_cost()
        resource = self.ques.get_resource()
        spend_max = self.ques.get_capacity()  # 获取所有agent最大capacity
        for unit in self.unit_list:
            agent,spend = unit.get_crossover2()
            # 随机地选择三个位置做kmove比较
            r1 = r2 = r3 = 0
            # 保证i不等
            while r2 == r1 or r3 == r1 or r3 == r2:
                r1 = random.randint(0, self.dim-1)  # 随机数范围为[0,dim-1]的整数
                r2 = random.randint(0, self.dim-1)
                r3 = random.randint(0, self.dim-1)
            # 总共会有5种情况，分别记录下来，选择合适且最优的，并更换cro_decode的值
            min_fit = self.ques.fit_fun2(agent)   # 记录最小适应度
            # 第1种，agent_r1不变，另外两个交换
            spend1 = spend.copy()
            agent1 = agent.copy()
            agent1[r2],agent1[r3] = agent1[r3],agent1[r2]   # r2与r3交换agent
            # 交换之后资源配置发生改变，之前做r2的agent消耗的资源要减去做r2所需的，加上做r3所需的
            spend1[agent[r2]] = spend[agent[r2]]-resource[agent[r2],r2]+resource[agent[r2],r3]
            spend1[agent[r3]] = spend[agent[r3]]-resource[agent[r3],r3]+resource[agent[r3],r2]
            if np.sum(spend1 <= spend_max) == self.agent_num:   # 都满足要求的情况下再做相关处理
                if self.ques.fit_fun2(agent1) < min_fit:
                    min_fit = self.ques.fit_fun2(agent1)
                    unit.set_crossover2(agent1,spend1)
            # 第2种，agent_r1做了r2，agent_r2做了r1，agent_r3不变
            spend2 = spend.copy()
            agent2 = agent.copy()
            agent2[r1],agent2[r2] = agent2[r2],agent2[r1]  
            spend2[agent[r2]] = spend[agent[r2]]-resource[agent[r2],r2]+resource[agent[r2],r1]
            spend2[agent[r1]] = spend[agent[r1]]-resource[agent[r1],r1]+resource[agent[r1],r2]
            if np.sum(spend2 <= spend_max) == self.agent_num:   # 都满足要求的情况下再做相关处理
                if self.ques.fit_fun2(agent2) < min_fit:
                    min_fit = self.ques.fit_fun2(agent2)
                    unit.set_crossover2(agent2,spend2)
            # 第3种，agent_r1做了r2，agent_r2做了r3，agent_r3做了r1
            spend3 = spend.copy()
            agent3 = agent.copy()
            agent3[r1],agent3[r2],agent3[r3] = agent3[r3],agent3[r1],agent3[r2]
            spend3[agent[r1]] = spend[agent[r1]]-resource[agent[r1],r1]+resource[agent[r1],r2]
            spend3[agent[r2]] = spend[agent[r2]]-resource[agent[r2],r2]+resource[agent[r2],r3]
            spend3[agent[r3]] = spend[agent[r3]]-resource[agent[r3],r3]+resource[agent[r3],r1]
            if np.sum(spend3 <= spend_max) == self.agent_num:   # 都满足要求的情况下再做相关处理
                if self.ques.fit_fun2(agent3) < min_fit:
                    min_fit = self.ques.fit_fun2(agent3)
                    unit.set_crossover2(agent3,spend3)
            # 第4种，agent_r1做了r3，agent_r2不变，agent_r3做了r1
            spend4 = spend.copy()
            agent4 = agent.copy()
            agent4[r1],agent4[r3] = agent4[r3],agent4[r1]  
            spend4[agent[r1]] = spend[agent[r1]]-resource[agent[r1],r1]+resource[agent[r1],r3]
            spend4[agent[r3]] = spend[agent[r3]]-resource[agent[r3],r3]+resource[agent[r3],r1]
            if np.sum(spend4 <= spend_max) == self.agent_num:   # 都满足要求的情况下再做相关处理
                if self.ques.fit_fun2(agent4) < min_fit:
                    min_fit = self.ques.fit_fun2(agent4)
                    unit.set_crossover2(agent4,spend4)
            # 第5种，agent_r1作了r3，agent_r2做了r1，agent_r3做了r2
            spend5 = spend.copy()
            agent5 = agent.copy()
            agent5[r1],agent5[r2],agent5[r3] = agent5[r2],agent5[r3],agent5[r1]
            spend5[agent[r1]] = spend[agent[r1]]-resource[agent[r1],r1]+resource[agent[r1],r3]
            spend5[agent[r2]] = spend[agent[r2]]-resource[agent[r2],r2]+resource[agent[r2],r1]
            spend5[agent[r3]] = spend[agent[r3]]-resource[agent[r3],r3]+resource[agent[r3],r2]
            if np.sum(spend5 <= spend_max) == self.agent_num:   # 都满足要求的情况下再做相关处理
                if self.ques.fit_fun2(agent5) < min_fit:
                    min_fit = self.ques.fit_fun2(agent5)
                    unit.set_crossover2(agent5,spend5)
                    
            


    def selection(self):
        '''选择(selection process)'''
        # 主要是将原始种群与经过变异交叉之后的种群做比较，选择优良个体
        for unit in self.unit_list:
            # 计算交叉个体的适应度
            decode_result,spend = unit.get_crossover2()
            new_fitness_value = self.ques.fit_fun2(decode_result)  
            
            # 修正个体最优，适应度小为优
            if new_fitness_value < unit.get_fitness_value() and np.sum(decode_result == -1) == 0 :
                # 还要保证它实际上花费的spend小于阈值
                if np.sum(self.ques.cal_agent(decode_result) <= self.ques.get_capacity()) == self.ques.agent_num:
                    unit.set_fitness_value(new_fitness_value)
                    # 修正最优解
                    unit.set_pos2(unit.get_crossover2()[0])
                    
            # 修正种群最优
            if new_fitness_value < self.get_bestFitnessValue() and np.sum(decode_result == -1) == 0 :
                if np.sum(self.ques.cal_agent(decode_result) <= self.ques.get_capacity()) == self.ques.agent_num:
                    self.set_bestFitnessValue(new_fitness_value)
                    self.set_bestPosition(unit.get_crossover2()[0])
                    self.__spend = unit.get_crossover2()[1]

    ##--------------------------运行区---------------------------------##
    def run(self):
        '''运行DE算法'''
        for i in range(self.iter_num):
            self.mutation_fun()
            self.crossover()
            self.Shifting()
            self.kmove()
            self.selection()
            # 收集每一代最优个体的适应度，以用于绘图
            self.fitness_val_list.append(self.get_bestFitnessValue())
        
    
    def show_result(self):
        # 绘制适应度随代数变化的情况
        plt.plot(range(self.iter_num),self.fitness_val_list,c="G",alpha=0.5)
        plt.title('广义指派问题最优解变化情况')
        plt.xlabel('iter_num')
        plt.ylabel('fitness')
        plt.show()
        
    def print_result(self):
        # 输出最优解
        print('运行结果......')
        print('该最小值问题的最优函数值:',1/self.best_fitness_value)
        print('该问题的最优解向量:',self.best_position+1)	
        print('最优解时spend：',self.__spend)
        print("实际spend为：",self.ques.cal_agent(self.best_position))		


class Question:
    '''问题类，存储与问题相关的数据'''
    
    def __init__(self):
        '''
        参数说明：
        cost:花费数据
        resource:资源消耗数据
        capacity:agent能力上限
        agent_num:agent人数
        task_num:任务数
        '''
        self.__cost = self.read_cost()
        self.__resource = self.read_resource()
        self.__capacity = self.read_capacity()[0]
        self.agent_num = self.get_size()[0]
        self.task_num = self.get_size()[1]
        
        
    def read_cost(self):
        return np.array(pd.read_excel(r"C:\Users\cw817615\Desktop\test_data.xlsx",sheetname="cost",header=None))
        
    def get_cost(self):
        return self.__cost
        
    def read_resource(self):
        return np.array(pd.read_excel(r"C:\Users\cw817615\Desktop\test_data.xlsx",sheetname="resource",header=None))
        
    def get_resource(self):
        return self.__resource
    
    def read_capacity(self):
        return np.array(pd.read_excel(r"C:\Users\cw817615\Desktop\test_data.xlsx",sheetname="capacity",header=None))
        
    def get_capacity(self):
        return self.__capacity
        
    def get_size(self):
        return self.__cost.shape
        
    def decoding(self,pos):
        '''解码函数，解码的结果便是NP的一个解'''
        spend = [0]*self.agent_num    # 每一个agent已经消耗的capacity
        # 将(0,1)之间数字列表转为任务指派列表,注意pos为ndarray数组
        pos_decode = np.array([-1]*len(pos))
        # 排序并返回index，取值范围为[0,14]
        index = np.argsort(pos,kind='quicksort')
        # index = np.array([5,4,1,7,8,9,12,15,14,13,3,10,2,6,11])-1   # 测试用的数据
        for i in index:
            # 找做task_i+1花费的agent的索引的排序，取值范围为[0,3]
            agent_ind = np.argsort(self.get_resource()[:,i])

            for j in agent_ind:
                # agent做了任务i之后未超出上限则可以做
                if spend[j]+self.get_resource()[j,i] <= self.__capacity[j]:
                    pos_decode[i] = j
                    # agent_j消耗的资源增加
                    spend[j] += self.get_resource()[j,i]
                    break
            if pos_decode[i] == -1:
                pass
                #print("任务没有指派出去")
        #print("这一次的实际spend值为：",self.cal_agent(pos_decode)<=self.get_capacity())  
        return pos_decode,spend
    
    
    def fit_fun1(self,pos):
        # 计算适应度
        pos_decode,spend = self.decoding(pos)
        pos_fit = 1/np.sum([self.get_cost()[pos_decode[i],i] for i in range(len(pos_decode))])
        return pos_fit
    
    def fit_fun2(self,pos_decode):
        return 1/np.sum([self.get_cost()[pos_decode[i],i] for i in range(len(pos_decode))])

    def cal_agent(self,decode_result):
        '''用来计算实际花费的spend'''
        resource = self.__resource
        spend = [0]*self.agent_num
        for i in range(self.task_num):
            spend[decode_result[i]] += resource[decode_result[i],i]
        #print("实际spend为：",spend)
        return spend

                
# 主函数
def main():

    question = Question()
    dim = question.task_num   # 粒子维度，在广义指派问题中为number of tasks
    agent_num = question.agent_num  
    
    size = 100   # 种群大小
    iter_num = 2000   #迭代次数
    
    # 初始解向量都是(0,1)之间的，当然也可以直接使用random.random()  
    x_max = 1
    x_min = 0 
    
    F = 2.0   # 缩放因子，在论文中取2.0      
    
    de = DE(dim,agent_num, size, iter_num, x_min, x_max, F=F )
    de.run()
    de.show_result()
    de.print_result()


if __name__ == "__main__":
    main()
    ## 一下代码时测试bug用的
    #randnum = np.array([0.6,0.85,0.01,0.25,0.96,0.26,0.60,0.10,0.62,0.07,0.02,0.67,0.76,0.73,0.74])
    #qus = Question()
    #cost = qus.get_cost()
    #resource = qus.get_resource()
    #print(resource)
    #decode_result,spend = qus.decoding(randnum)
		