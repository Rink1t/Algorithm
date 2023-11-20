import numpy as np

class PSO(object):

    def __init__(self, func, lb, ub, n_dim, w=0.8, c1=0.5, c2=0.5, pop=40, v_inital=0, tol=0.1,
                 max_iter=100, vlimit=0.2, w_max=None, w_min=None, random_state=None):
        self.__func = func  # 目标函数
        self.__lb = np.full(n_dim, lb)  # 求解空间下界
        self.__ub = np.full(n_dim, ub)  # 求解空间上界
        self.__n_dim = n_dim  # 求解空间维度
        self.__w = w  # 惯性权重
        self.__c1 = c1  # 个体学习因子
        self.__c2 = c2  # 群体学习因子
        self.__pop = pop  # 种群大小
        self.__v_inital = v_inital  # 各粒子初始化速度
        self.__max_iter = max_iter  # 最大迭代次数
        self.__tol = tol  # 容忍度
        self.__vlimit = vlimit  # 速度限制
        self.__random_state = random_state  # 随机种子

        # 惯性权重线性衰减
        if self.__w == 'line_decay':
            self.__w_mode = 'line_decay'
            self.__w_max = w_max
            self.__w_min = w_min
            self.__w = self.__w_max
        else:
            self.__w_mode = 'constant'

    # 初始化信息
    def __inital_info(self):
        # 初始化粒子位置
        self.__pop_pos = np.random.RandomState(self.__random_state) \
            .uniform(low=self.__lb, high=self.__ub, size=(self.__pop, self.__n_dim))

        # 初始化其它相关信息
        self.__pbest_pos = np.zeros((self.__pop, self.__n_dim))  # 个体最优位置
        self.__gbest_pos = np.zeros(self.__n_dim)  # 全局最优位置
        self.__pbest_fitness = np.full(self.__pop, np.inf)  # 个体最优适应值
        self.__gbest_fitness = np.inf  # 全局最优适应值
        self.__v = np.full((self.__pop, self.__n_dim), self.__v_inital)  # 各粒子初始化速度
        self.__pop_fitness = None  # 各粒子对应适应值

        # 模型相关属性
        self.n_iter_ = None  # 模型真实迭代次数
        self.gbest_fitn_ = None  # 得到的解对应的目标函数值(适应值)
        self.gbest_pos_ = None  # 得到的解
        self.gbest_fitn_his_ = []  # 模型gbest替换历史

        # 初始化pbest, gbest相关信息
        self.__update_best()

    # 更新所有相关信息
    def __update_info(self, epoch):
        # 更新权重
        if self.__w_mode != 'constant':
            self.__update_w(epoch)

        # 更新速度
        if epoch == 1:
            self.__update_V(inital=True)
        else:
            self.__update_V()

        self.__update_X()  # 更新各粒子位置
        self.__update_best()  # 更新个体最优位置, 全局最优位置

    # 计算和更新 pbest, gbest位置
    def __update_best(self):
        self.__pop_fitness = np.apply_along_axis(self.__func, axis=1, arr=self.__pop_pos).flatten()  # 计算当前各粒子的适应值

        # 更新pbest信息
        exp = self.__pop_fitness < self.__pbest_fitness
        self.__pbest_fitness[exp] = self.__pop_fitness[exp]
        self.__pbest_pos[exp] = self.__pop_pos[exp]

        # 更新gbest信息
        min_pbest_fitness = np.min(self.__pbest_fitness)

        if min_pbest_fitness < self.__gbest_fitness:
            index = np.where(self.__pbest_fitness == min_pbest_fitness)[0][0]
            self.__gbest_fitness = self.__pbest_fitness[index]
            self.__gbest_pos = self.__pbest_pos[index]

            # 记录gbest的适应值信息
            self.gbest_fitn_his_.append(self.__gbest_fitness)

    # 自适应权重更新函数
    def __update_w(self, epoch):
        if self.__w_mode == 'line_decay':
            assert (self.__w_max != None) and (self.__w_min != None), 'w_max or w_min can not be "None"'

            self.__w = self.__w_max - ((self.__w_max - self.__w_min) * (epoch / self.__max_iter))

    def __update_V(self, inital=False):
        # 保存当前速度矩阵
        pre_v = self.__v

        # 增加随机性
        r1 = np.random.RandomState(self.__random_state).rand(self.__pop, self.__n_dim)
        r2 = np.random.RandomState(self.__random_state).rand(self.__pop, self.__n_dim)

        # 更新各粒子速度
        self.__v = (self.__w * self.__v) + \
                   (self.__c1 * r1 * (self.__pbest_pos - self.__pop_pos)) + \
                   (self.__c2 * r2 * (self.__gbest_pos - self.__pop_pos))

        # 由于各粒子速度初始化为0, 因此在第一轮迭代中不对速度进行检查修正
        if not inital:
            # 检查和修正粒子速度
            self.__check_V(pre_v)

    # 更新粒子位置
    def __update_X(self):
        self.__pop_pos += self.__v

        # 修正位置
        self.__check_X()

    # 检查和修正粒子位置
    def __check_X(self):
        def check_f(pop_pos_i):
            up_exp = (pop_pos_i > self.__ub)
            low_exp = (pop_pos_i < self.__lb)

            pop_pos_i[up_exp] = self.__ub[up_exp]
            pop_pos_i[low_exp] = self.__lb[low_exp]

        # 修正粒子位置
        np.apply_along_axis(check_f, axis=1, arr=self.__pop_pos)

    # 检查和修正粒子速度
    def __check_V(self, pre_v):
        up_v = pre_v + self.__vlimit * pre_v
        low_v = pre_v - self.__vlimit * pre_v

        up_exp = self.__v > up_v
        low_exp = self.__v < low_v

        # 修正速度
        self.__v[up_exp] = up_v[up_exp]
        self.__v[low_exp] = low_v[low_exp]

    # 停止准则: 若过去3次全局最优位置对应适应值的变化量的均值小于容忍度tol, 则判断模型已拟合
    def __is_stop(self):
        flag = False

        if len(self.gbest_fitn_his_) >= 4:

            if np.abs(np.mean(np.diff(self.gbest_fitn_his_[-4:]))) < self.__tol:
                flag = True

        return flag

    # 训练模型
    def fit(self):
        self.__inital_info()  # 初始化相关信息

        for epoch in range(self.__max_iter):
            # 停止准则
            if self.__is_stop():
                self.n_iter_ = epoch  # 记录真实迭代次数
                break

            self.__update_info(epoch)

        # 记录拟合结果
        if self.n_iter_ == None:
            self.n_iter_ = self.__max_iter

        self.gbest_fitn_ = self.__gbest_fitness
        self.gbest_pos_ = self.__gbest_pos

        return self

    # 模型拟合后的相关信息
    def fit_info(self):
        print('n_iter: ', self.n_iter_)
        print('gbest X: ', self.gbest_pos_)
        print('gbest Y: ', self.gbest_fitn_)
        print('gbest Y history: ', self.gbest_fitn_his_)