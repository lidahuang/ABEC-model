# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 11:19:14 2018

@author: hld
"""
import numpy as np
from numpy.linalg import cholesky
import random

import copy
import math
import collections
import matplotlib  
import matplotlib.pyplot as plt
from matplotlib import colors

# make a color map of fixed colors
cmap = colors.ListedColormap(['w','g','k','b', 'r', 'c', 'm','y'])
bounds = [0,0.5,1.5,2.5,5,15,25,35,45]
norm = colors.BoundaryNorm(bounds, cmap.N)

def list_distance(list1, list2):
    return np.sqrt((list1[0]-list2[0])**2+(list1[1]-list2[1])**2)
        
class Agent(object):
    """
    category:
        10: violent
        20: active
        30: quiet
        40: cop
    heading:
        *3 *2 *1
        *4  * *0
        *5 *6 *7
    """
    def __init__(self, cell_id = None, agent_category=None, agent_id=None, heading=None):
        self.agent_id = agent_id
        self.category = agent_category
        self.cell_id = cell_id
        self.heading = heading
        self.init_H = 0.48*random.random()#[0,0.5)
        self.H = self.init_H
        self.grievance = 0 #初始愤怒值为0
        self.expression = random.random()
        self.impact = random.random()
        self.beimpact = self.init_H
        self.infect_time = np.random.normal(300,100)
        self.recover_time = 0
        self.is_recover = 0 #是否免疫
        self.delay_time_remain = 0 # 0代表delay结束
        self.is_fighting = 0 #是否在fight,>0则在打架
        self.is_arrest = 0 #是否被抓，=1则被抓

    
class Cell(object):
    """
    cell_category: 
        0 normal
    agent_id: None means no agent in this cell
    """
    def __init__(self, h=1, w=1, cell_id=None, cell_category=None, agent_id=None):
        self.h = h #网格的高、宽均设为1
        self.w = w
        self.cell_id = cell_id #[row,col]
        self.category = cell_category
        self.agent_id = agent_id
        self.is_fighting = 0
        
    
class Grid(object):
    """
    包括cells列表和agents列表，同时分别维护cells_matrix和agents_matrix用来可视化，
    cells和agents之间有双向映射，
    cell_id形如[row, col]，通过one2two和tow2one与cells列表中的idx相互转换
    """
    def __init__(self, cells_matrix=None,num_agents=None, 
                 num_per_category=None, num_friends=None, kij=None, opt=None):
        """
        cells_matrix二维数组，行列表示空间shape，行列内数值指定网格category
        num_per_category指定agents中各类别的比例,字典，可用key访问
        """
        self.nh, self.nw = cells_matrix.shape
        self.num_agents = num_agents
        self.cells_matrix = cells_matrix
        self.agents_matrix = copy.deepcopy(self.cells_matrix)
        self.N_arrest = 0
        
        #cells agents列表，里面装Cell Agent对象
        """
        for i in range(3) i=0,1,2
        list.append 在list末尾添加一个元素
        """
        self.cells = []
        self.agents = []
        for i in range(self.nh):
            for j in range(self.nw):
                self.cells.append(Cell(cell_id=[i,j], cell_category=self.cells_matrix[i][j]))
        for i in range(num_agents):
            self.agents.append(Agent(agent_id=i))
            
        self.disappear_agents = []
        self.arrest_agents = []
        self.knocked_cop = []
        self.opt=opt #initialized 参数
        
        self._init_agents(num_per_category=num_per_category)
        self._put_agents_into_cells()
        

        """the order is identical to headings:
        *3 *2 *1
        *4  * *0
        *5 *6 *7
        """
        print("initialized")
        
    def _init_agents(self, num_per_category):
        """按照比例随机初始化所有agents的类别
        随机初始化heading
        """
        categories = [] #list A=[10,5] B=[10] A+B=[10,5,10] B*2=[10,10]
        for key, value in num_per_category.items():#遍历字典的key和value
            categories += [key] * int(value * self.num_agents)
        random.shuffle(categories)#随机打断categories的顺序
       
        for i in range(self.num_agents):
            self.agents[i].category = categories[i]
            if categories[i] == 20:#active
                self.agents[i].init_H = 1
                self.agents[i].H = 1
            self.agents[i].grievance = self.agents[i].H*(1-0.8)
            #heading,初始化方向
            self.agents[i].heading = random.randint(0,7)

                
    def _one2_inner_two(self, idx):
        return idx//(self.nw-2)+1, idx%(self.nw-2)+1 
    
    def _put_agents_into_cells(self):
        """将agents随机洒在网格上，
        记录网格和agent之间的双向映射，
        生成agents_matrix，其中小于10的数字表示的是cell的类别，
            大于10的数字表示agent的类别（10*agent_category）
        """
        #随机选择网格放入agent
        chosen_cells = [1]*self.num_agents + [0]*((self.nh-2)*(self.nw-2) - self.num_agents)
        random.shuffle(chosen_cells)
               
        j=0
        for i, value in enumerate(chosen_cells):
            if value == 1:
                row_i, col_i = self._one2_inner_two(i)
                self.agents_matrix[row_i][col_i] = self.agents[j].category
                #双向映射
                self.agents[j].cell_id = [row_i, col_i]                
                self.cells[row_i*self.nw+col_i].agent_id = j
                j += 1
    
    # cells
    """in the run time, all cells in the space canbe occupied
    """
    def one2two(self, idx):
        return idx//self.nw, idx%self.nw
    def two2one(self, cell_id):
        return int(cell_id[0]*self.nw+cell_id[1])
    
    def get_8_neighbors_cell_ids(self, cell_id):
        """the order is identical to headings:
        *3 *2 *1
        *4  * *0
        *5 *6 *7
        """
        return [
                [cell_id[0], cell_id[1]+1],
                [cell_id[0]-1, cell_id[1]+1],
                [cell_id[0]-1, cell_id[1]],
                [cell_id[0]-1, cell_id[1]-1],
                [cell_id[0], cell_id[1]-1],
                [cell_id[0]+1, cell_id[1]-1],
                [cell_id[0]+1, cell_id[1]],
                [cell_id[0]+1, cell_id[1]+1]
                ]
    def get_circle_neighbors_cell_idxs(self, cell_id, radius):
        neighbors = []
        distances = []
        coss = []
        sins = []
        
        """间隔采样
        np.linspace(-1,1,5)  out<<<array([-1. , -0.5,  0. ,  0.5,  1. ])
        """
        for i in np.linspace (-radius,radius,2*radius+1):
            for j in np.linspace (-radius,radius,2*radius+1):
                if not (i==0 and j==0):
                    distance = list_distance([0,0],[i,j])
                    if distance>radius:
                        pass
                    else:
                        neighbors.append([cell_id[0]+i,cell_id[1]+j])
                        distances.append(distance)
                        coss.append(j/distance)
                        sins.append(-i/distance)
        return neighbors, distances, coss, sins
   
    """越界的处理
    """
    def out_of_boundary(self, cell_id):
        if cell_id[0]<0 or cell_id[0]>=self.nh or cell_id[1]<0 or cell_id[1]>=self.nw:
            return True
        return False       
    
    # agents
    def choose_direction_update_state(self, agent_id, rand):
        '''
        return: 将要移动的cell_id[row][col]
        '''
        """scan, update state and choose next position
        heading:
        *3 *2 *1
        *4  * *0
        *5 *6 *7     
        y
        ^
        | /
        |/theta
        ------> x
        """
        if rand:
            cell_id = self.agents[agent_id].cell_id
            neighbors = self.get_8_neighbors_cell_ids(cell_id)
            return neighbors[random.randint(0,7)]
        
        """SCAN HERE
        8邻域的agents: neighbors
        circle视域范围内
            agents: circle_neighbors
            Nvr_prosters Nvr_violent Nvr_active Nvr_cops
            最近的violent, cops 列表(可能不止一个)    
        """
        cell_id = self.agents[agent_id].cell_id
        neighbors = self.get_8_neighbors_cell_ids(cell_id)
        circle_neighbors, distances, coss, sins = self.get_circle_neighbors_cell_idxs(cell_id, 5)
        Nvr_violent = 0 # 10
        Nvr_cops = 0  # 40
        Nearest_inner_cops = []
        Nearest_cops = []
        distance_cop = 100
        Nearest_inner_violent = []
        Nearest_violent = []
        distance_violent = 100
        for i, nei in enumerate(circle_neighbors):
            if self.out_of_boundary(nei):
                pass
            else:
                nei_agent_id = self.cells[self.two2one(nei)].agent_id
                # 如果有人， 统计人的类型，并记录最近的cop和activist
                if nei_agent_id != None:
                    category = self.agents[nei_agent_id].category
                    if category == 40:
                        Nvr_cops += 1
                        if distance_cop >= distances[i]:
                            distance_cop = distances[i]
                            Nearest_inner_cops.append(i)
                    else:
                        if category == 10:
                            Nvr_violent += 1
                            if distance_violent >= distances[i]:
                                distance_violent = distances[i]
                                Nearest_inner_violent.append(i)      
        '''
        enumerate遍历list时可以同时遍历id和value
        只遍历value,可用for a in Nearest_inner_cops
        '''
        for Nearest_id in Nearest_inner_cops:
            if distances[Nearest_id] == distance_cop:
                Nearest_cops.append(Nearest_id)
        for Nearest_id in Nearest_inner_violent:
            if distances[Nearest_id] == distance_violent:
                Nearest_violent.append(Nearest_id)
                          
        """UPDATE GRIEVANCE HERE
        """
        # 只要没有recover，都要更新grievance
        if self.agents[agent_id].category != 40:
            if self.agents[agent_id].is_recover == 0:
                Hji = 0
                for i, nei in enumerate(circle_neighbors):
                    if self.out_of_boundary(nei):
                        pass
                    else:
                        nei_agent_id = self.cells[self.two2one(nei)].agent_id
                        if nei_agent_id != None:
                            category = self.agents[nei_agent_id].category
                            if category == 10 or category == 20:
                                hij = 1-1/(1+np.exp(-distances[i]))
                                Hji = Hji + hij*self.opt.beta*self.agents[nei_agent_id].expression \
                                    *self.agents[nei_agent_id].impact*self.agents[agent_id].beimpact
                self.agents[agent_id].H = self.agents[agent_id].H + Hji
                if self.agents[agent_id].H > 1:
                    self.agents[agent_id].H = 1
                self.agents[agent_id].grievance = self.agents[agent_id].H * (1-self.opt.L)             
            else:#recover了
                if self.agents[agent_id].recover_time > 0:
                    if self.agents[agent_id].H == 1:
                        self.agents[agent_id].is_recover = 0
                        self.agents[agent_id].category = 20
                    self.agents[agent_id].recover_time -= 1
                else:
                    if random.random() < self.opt.lamda: #再次感染
                        self.agents[agent_id].is_recover = 0
               
            
        """UPDATE STATE HERE
        agent_category: 10 violent, 20 active, 30 quiet, 40 cops
        """
        # N, the net risk
        Ra = random.random()
        kb = np.log(0.1)
        Pa = 1 - np.exp(kb*(Nvr_cops/(Nvr_violent + 1e-6)))
        N = Ra*Pa*self.opt.J**self.opt.alpha
        
        #quiet 30 -> active
        if self.agents[agent_id].category == 30:
            self.agents[agent_id].infect_time = 0
            if self.agents[agent_id].is_recover == 0: #未免疫
                if self.agents[agent_id].grievance > self.opt.th1:
                    self.agents[agent_id].category = 20
                    self.agents_matrix[cell_id[0]][cell_id[1]] = 20
                    self.agents[agent_id].infect_time = np.random.normal(self.opt.infect_time_bound[0], \
                               self.opt.infect_time_bound[1])
           
        # active 是否免疫
        if self.agents[agent_id].category == 20:
            # 未超过活跃时间，-1
            if self.agents[agent_id].infect_time > 0:
                self.agents[agent_id].infect_time -= 1
            else:#超过活跃时间，有一定概率变为quiet
                if random.random() < self.opt.gama: #恢复
                    self.agents[agent_id].category = 30
                    self.agents_matrix[cell_id[0]][cell_id[1]] = 30
                    #self.agents[agent_id].grievance = 0
                    self.agents[agent_id].H = self.agents[agent_id].init_H
                    self.agents[agent_id].grievance = self.agents[agent_id].H * (1-self.opt.L)
                    self.agents[agent_id].infect_time = 0
                    self.agents[agent_id].is_recover = 1
                    self.agents[agent_id].recover_time = np.random.normal(self.opt.recover_time_bound[0], \
                               self.opt.recover_time_bound[1])
        
        # active -> violent
        if self.agents[agent_id].category == 20 and self.agents[agent_id].grievance-N > self.opt.th2:
            self.agents[agent_id].category = 10
            self.agents_matrix[cell_id[0]][cell_id[1]] = 10
            #self.agents[agent_id].infect_time = 0
        
        # violent 10 ->active      
        if self.agents[agent_id].category == 10 and self.agents[agent_id].grievance-N <= self.opt.th2:
            self.agents[agent_id].category = 20
            self.agents_matrix[cell_id[0]][cell_id[1]] = 20
            self.agents[agent_id].infect_time = np.random.normal(self.opt.infect_time_bound[0],self.opt.infect_time_bound[1])
        
        """CHOOSE NEXT POSITION
        """
        # violent
        if self.agents[agent_id].category == 10:
            if Nvr_cops>0 and Nvr_violent > 2*Nvr_cops:#追逐最近的警察
                i = random.choice(Nearest_cops)
                direction_x = coss[i]
                direction_y = sins[i]
                direction_theta = np.rad2deg(math.atan2(direction_y,direction_x))
                if direction_theta < 0:
                    direction_theta +=360
                return neighbors[int(round(direction_theta / 45.0) % 8)]
            if Nvr_cops>0 and Nvr_violent <= Nvr_cops:#远离最近的警察
                forces_x = [-coss[i] for i in Nearest_cops]
                forces_y = [-sins[i] for i in Nearest_cops]
                forces_theta = np.rad2deg(math.atan2(sum(forces_y), sum(forces_x)))
                if forces_theta < 0:
                    forces_theta += 360 
                return neighbors[int(round(forces_theta / 45.0) % 8)]
            else:
                dst0 = []
                dst = []
                for nei in neighbors:
                    if self.out_of_boundary(nei):
                        pass
                    else:
                        cell = self.cells[self.two2one(nei)]
                        if cell.agent_id == None:
                            dst0.append(nei) 
                l = len(dst0)
                if random.random()<1/(1+l):
                    dst = self.agents[agent_id].cell_id
                else:
                    dst = dst0[random.randint(0,l-1)]
                return dst
        # cop
        if self.agents[agent_id].category == 40:
            if Nvr_violent > 0 and self.opt.PPower*Nvr_cops > Nvr_violent:
                i = random.choice(Nearest_violent)
                direction_x = coss[i]
                direction_y = sins[i]
                direction_theta = np.rad2deg(math.atan2(direction_y,direction_x))
                if direction_theta < 0:
                    direction_theta +=360
                return neighbors[int(round(direction_theta / 45.0) % 8)]
            if Nvr_violent > 0 and self.opt.PPower*Nvr_cops <= Nvr_violent:#远离最近的暴徒
                forces_x = [-coss[i] for i in Nearest_violent]
                forces_y = [-sins[i] for i in Nearest_violent]
                forces_theta = np.rad2deg(math.atan2(sum(forces_y), sum(forces_x)))
                if forces_theta < 0:
                    forces_theta += 360 
                return neighbors[int(round(forces_theta / 45.0) % 8)]
            else:
                dst0 = []
                dst = []
                for nei in neighbors:
                    if self.out_of_boundary(nei):
                        pass
                    else:
                        cell = self.cells[self.two2one(nei)]
                        if cell.agent_id == None:
                            dst0.append(nei) 
                l = len(dst0)
                if random.random()<1/(1+l):
                    dst = self.agents[agent_id].cell_id
                else:
                    dst = dst0[random.randint(0,l-1)]
                return dst
        
        # active or quiet
        if self.agents[agent_id].category == 20:
            if Nvr_cops>0:#远离最近的警察
                forces_x = [-coss[i] for i in Nearest_cops]
                forces_y = [-sins[i] for i in Nearest_cops]
                forces_theta = np.rad2deg(math.atan2(sum(forces_y), sum(forces_x)))
                if forces_theta < 0:
                    forces_theta += 360 
                return neighbors[int(round(forces_theta / 45.0) % 8)]
            else:
                dst0 = []
                dst = []
                for nei in neighbors:
                    if self.out_of_boundary(nei):
                        pass
                    else:
                        cell = self.cells[self.two2one(nei)]
                        if cell.agent_id == None:
                            dst0.append(nei) 
                l = len(dst0)
                if random.random()<1/(1+l):
                    dst = self.agents[agent_id].cell_id
                else:
                    dst = dst0[random.randint(0,l-1)]
                return dst
        if self.agents[agent_id].category == 30:
            dst0 = []
            dst = []
            for nei in neighbors:
                if self.out_of_boundary(nei):
                    pass
                else:
                    cell = self.cells[self.two2one(nei)]
                    if cell.agent_id == None:
                        dst0.append(nei) 
            l = len(dst0)
            if random.random()<1/(1+l):
                dst = self.agents[agent_id].cell_id
            else:
                dst = dst0[random.randint(0,l-1)]
            return dst
        
    def update(self, agent_id, rand):
        """update position 更新cells.agent_id agents.cell_id agents_matrix
        目的地有人则不动
        目的地超边界则镜像进入
        """
        dst_id = self.choose_direction_update_state(agent_id, rand)
        if self.out_of_boundary(dst_id):
            pass
        else:
            cell = self.cells[self.two2one(dst_id)] #目的cell
            if cell.agent_id != None:
                pass
            else:
                original_cell_id = self.agents[agent_id].cell_id
            
                self.agents_matrix[original_cell_id[0]][original_cell_id[1]] = \
                self.cells_matrix[original_cell_id[0]][original_cell_id[1]]
            
                self.cells[self.two2one(original_cell_id)].agent_id = None
            
                self.cells[self.two2one(dst_id)].agent_id = agent_id
                self.agents_matrix[dst_id[0]][dst_id[1]] = self.agents[agent_id].category
            
                self.agents[agent_id].cell_id = dst_id

                    
 # grid
    def run_one_step(self, rand, time_step):
        """消失的agent啥也不干
        """
        for idx in range(len(self.agents)):
            if idx in self.disappear_agents:
                pass
            else:
                self.update(idx, rand) # scan, update state and position
                
            
    def run(self, steps, sleep, rand=False, save_iter=5):
        """https://www.cnblogs.com/DHUtoBUAA/p/6619099.html
        不同身份要显示不同形状可以用scatter
        markers = ('-','-','s', 'x', 'o', '^', 'v')
        cop 三角形 40
        viloent 红色 10
        active 黄色 20
        quiet 蓝色 30
            https://matplotlib.org/examples/pylab_examples/multiple_figs_demo.html
            https://morvanzhou.github.io/tutorials/data-manipulation/plt/3-1-scatter/
        """
        
        '''
        #grievance
        plt.figure(num=1, figsize=(6, 4))
        p = plt.imshow(self.agents_matrix, cmap=cmap, norm=norm)
        plt.title("test: 1")
        plt.colorbar(orientation="horizontal")
        plt.pause(sleep)
        plt.savefig("test"+str(1)+'.tiff', dpi=300)
                
        plt.figure(num=1, figsize=(6, 4))
        #指定colormap  
        cmap = matplotlib.cm.jet  
        #设定每个图的colormap和colorbar所表示范围是一样的，即归一化  
        norm = matplotlib.colors.Normalize(vmin=0, vmax=0.2)
        p = plt.imshow(self.get_grievance_matrix(),origin='lower',cmap=cmap, norm=norm)#, cmap=plt.cm.BuPu_r)
        plt.colorbar(p)
        #cbar.set_label('Grievance') 
        plt.xlim(-1, 40)
        plt.ylim(-1, 40)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.xticks([0,5,10,15,20,25,30,35,40])
        plt.yticks([0,5,10,15,20,25,30,35,40])
        plt.title("step: 1")
        plt.pause(sleep)
        plt.savefig("step"+str(1)+'.tiff', dpi=300)
        '''
        
        plt.figure(num=2, figsize=(6, 4))
        plt.xlim(-1, 41)
        plt.ylim(-1, 41)
        plt.gca().set_aspect('equal', adjustable='box')
        #p2 = plt.imshow(self.agents_matrix, cmap=cmap, norm=norm)
        violent = np.where(self.agents_matrix == 10) # tuple(array, array)
        active = np.where(self.agents_matrix == 20)
        quiet = np.where(self.agents_matrix == 30)
        cop = np.where(self.agents_matrix == 40)
        plt.scatter(violent[1], 40-violent[0], s=9, c='r')
        plt.scatter(active[1], 40-active[0], s=9, c='y')
        plt.scatter(quiet[1], 40-quiet[0], s=9, c='b')
        plt.scatter(cop[1], 40-cop[0], s=9, c='g', marker='^')
        plt.title("step: 1")
        plt.pause(sleep)
        plt.savefig("step"+str(1)+'.tiff', dpi=300)
               

        '''
        输出文档
        '''
        #doc1 = open('StateCount-PPower4.txt','w')
        #doc2 = open('CopAroundViolent.txt','w')
        #doc3 = open('Grievance.txt','w')

        for i in range(steps):
            self.run_one_step(rand, time_step=i)
            #统计输出
            num_violent, num_active, num_quiet, num_fight, num_arrest = self.count_protest_in_cells()
            ratio_violent = num_violent/(num_violent+num_active+num_quiet)
            ratio_cop_around_violent = self.ratio_cop_around_violent()
            num_knocked_cop = len(self.knocked_cop)
            print(i,num_violent,num_active,num_quiet, num_fight, num_arrest, num_knocked_cop)
            #print(i,num_violent,num_active,num_quiet, file=doc1)
            #print(i,ratio_violent,ratio_cop_around_violent,file=doc2)
            #doc3.write("%s " %i)
      
            '''
            plt.figure(1, figsize=(6, 4))
            p.set_data(self.agents_matrix)
            plt.title("test: "+str(i))
            plt.pause(sleep)#绘图延迟
            if(i%save_iter == 0):
                plt.savefig("test"+str(i)+'.tiff', dpi=300)
            
            
            #grievance
            plt.figure(1, figsize=(6, 4))
            p.set_data(self.get_grievance_matrix())
            plt.title("step: "+str(i))
            plt.pause(sleep)#绘图延迟
            if(i%save_iter == 0):
                plt.savefig("test"+str(i)+'.tiff', dpi=300)
            
            '''
            plt.figure(2, figsize=(6, 4))
            plt.clf()
            plt.xlim(-1, 41)
            plt.ylim(-1, 41)
            plt.gca().set_aspect('equal', adjustable='box')
            
            #p2.set_data(self.agents_matrix)
            #violent = np.where(self.agents_matrix == 10) # tuple(array, array)
            #active = np.where(self.agents_matrix == 20)
            #quiet = np.where(self.agents_matrix == 30)
            #cop = np.where(self.agents_matrix == 40)
            #plt.scatter(violent[1], 40-violent[0], s=9, c='r')
            #plt.scatter(active[1], 40-active[0], s=10, c='y')
            #plt.scatter(quiet[1], 40-quiet[0], s=10, c='b')
            #plt.scatter(cop[1], 40-cop[0], s=10, c='g', marker='^')
            fight_x = []
            fight_y = []
            violent_x = []
            violent_y = []
            active_x = []
            active_y = []
            quiet_x =[]
            quiet_y=[]
            cop_x=[]
            cop_y=[]
            
            for idx in range(self.num_agents):
                if idx not in self.disappear_agents:
                    if self.agents[idx].category == 10:
                        if self.agents[idx].is_fighting > 0:
                            fight_x.append(self.agents[idx].cell_id[0])
                            fight_y.append(self.agents[idx].cell_id[1])
                        else:
                            violent_x.append(self.agents[idx].cell_id[0])
                            violent_y.append(self.agents[idx].cell_id[1])
                    if self.agents[idx].category == 20:
                        active_x.append(self.agents[idx].cell_id[0])
                        active_y.append(self.agents[idx].cell_id[1])
                    if self.agents[idx].category == 30:
                        quiet_x.append(self.agents[idx].cell_id[0])
                        quiet_y.append(self.agents[idx].cell_id[1])
                    if self.agents[idx].category == 40:
                        cop_x.append(self.agents[idx].cell_id[0])
                        cop_y.append(self.agents[idx].cell_id[1])
                        
                    if self.agents[idx].category != 40:
                        G = self.agents[idx].grievance
                        #doc3.write("%s " %G)
            #doc3.write("\n")
            
            plt.scatter(fight_x,fight_y, s=10, c='r', marker='*')
            plt.scatter(violent_x,violent_y, s=10, c='r')
            plt.scatter(active_x, active_y, s=10, c='y')
            plt.scatter(quiet_x, quiet_y, s=10, c='b')
            plt.scatter(cop_x, cop_y, s=10, c='g', marker='^')
            plt.title("step: "+str(i))
            plt.pause(sleep)#绘图延迟
            if(i%save_iter == 0):
                plt.savefig("step"+str(i)+'.tiff', dpi=300)
            
            #self.print_grievance()
            #self.print_recover()
        #doc1.close()
        #doc2.close()
        #doc3.close()
        
    def get_grievance_matrix(self):
        gri = self.agents_matrix.copy()
        gri *= 0
        for agent in self.agents:
            gri[agent.cell_id[1],agent.cell_id[0]] = agent.grievance
        return gri
    
    def plot_cells(self):
        plt.figure(figsize=(12, 8))
        plt.imshow(self.cells_matrix)
    def plot_agents(self):
        plt.figure(figsize=(24, 16))
        plt.imshow(self.agents_matrix, cmap=cmap, norm=norm)
        # 10*self.agents[j].category
    
    """统计信息
    """
    def count_agents_matrix(self):
        print(collections.Counter(self.agents_matrix.flatten()))
    def count_agents_in_cells(self):
        num = 0
        for i in self.cells:
            if i.agent_id != None:
                num += 1
        print(num)
    def count_protest_in_cells(self):
        num_violent = 0
        num_active = 0
        num_quiet = 0
        num_arrest = 0
        num_fight = 0
        for agent in self.agents:
            if agent.category == 10:
                if agent.is_arrest == 0:
                    num_violent +=1
                    if agent.is_fighting > 0:
                        num_fight +=1
                else:
                    num_arrest +=1
            if agent.category == 20:
                num_active +=1
            if agent.category == 30:
                num_quiet +=1             
        #print(num_violent, num_active,num_quiet,num_arrest)
        return [num_violent, num_active, num_quiet, num_fight, num_arrest]
    def check(self):
        for agent in self.agents:
            if agent.agent_id not in self.disappear_agents: 
                assert agent.category == self.agents_matrix[agent.cell_id[0]][agent.cell_id[1]]
                assert agent.agent_id == self.cells[self.two2one(agent.cell_id)].agent_id
    def print_recover(self):
        for agent in self.agents:
            if agent.is_recover == 1:
                i=agent.agent_id
                print(i)
    def print_grievance(self):
        for agent in self.agents:
            if agent.grievance>0:
                i=agent.agent_id
                g=agent.grievance
                print(i,g)
    def ratio_cop_around_violent(self):
        num_cop_around_agent = 0
        num_cop_around_violent = 0
        for agent in self.agents:
            if agent.category == 40:
                cop_id = agent.agent_id
                cell_id = self.agents[cop_id].cell_id
                neighbors = self.get_8_neighbors_cell_ids(cell_id)
                circle_neighbors, distances, coss, sins = self.get_circle_neighbors_cell_idxs(cell_id, 5)
                for i, nei in enumerate(circle_neighbors):
                    if self.out_of_boundary(nei):
                        pass
                    else:
                        nei_agent_id = self.cells[self.two2one(nei)].agent_id
                        # 如果有人， 统计人的类型，记录violent的数量
                        num_cop_around_agent += 1
                        if nei_agent_id != None:
                            category = self.agents[nei_agent_id].category
                            if category == 10:
                                num_cop_around_violent += 1                  
        ratio = num_cop_around_violent/num_cop_around_agent
        return(ratio)
