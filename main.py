# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 11:21:35 2018

@author: hld
"""
import numpy as np
from grid import Grid

"""
cell_category: 
    0 normal
agent_category:
    10 activist, 20 hangers-on, 30 passers-by, 40 cops, （50 熟人）
"""
class Config(object):
    L = 0.82
    th1 = 0.1
    th2 = 0.1
    beta = 0.3 #传染概率
    gama = 0.7 #恢复概率
    lamda = 0.3 #再次感染概率
    J = 2
    alpha = 0.5
    delay_time = 3
    infect_time_bound = [180, 100] # mu=300, sigma2=100
    recover_time_bound = [180, 100] # mu=300, sigma2=100
    Nmax_arrest = 800
    PPower = 1
    
opt = Config()

'''
定义运行环境的网格属性cells_matrix
'''
matrix = np.zeros([40,40])


grid = Grid(cells_matrix=matrix,
            num_agents=850,
            num_per_category={10:0/850,20:8.5/850,30:792/850,40:50/850},
            opt = opt)
    
grid.count_agents_matrix()
grid.count_agents_in_cells()



grid.run(steps=3600, sleep=0.00001, rand=False, save_iter=1)
grid.count_agents_matrix()
grid.count_agents_in_cells()
grid.check()

'''
cop 三角形
viloent 红色
active 黄色
quiet 蓝色
分辨率300 
1000 步，每5步保存这一次
'''