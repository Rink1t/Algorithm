import numpy as np
import matplotlib.pyplot as plt


class GA(object):
    # 初始化相关参数
    def __init__(self, lb, ub, n_dim, pop_size=40, max_iter=100, cross_rate=0.4, mut_rate=0.01, cross_strategy='single',
                 is_tol=False, tol=0.01, mut_strategy='simple', precision=4, code_mode='binary', random_state=None):
        self.__lb = np.array(lb)  # 下限
        self.__ub = np.array(ub)  # 上限
        self.__n_dim = n_dim  # 维数
        self.__pop_size = pop_size  # 种群大小
        self.__max_iter = max_iter  # 最大迭代数
        self.__cross_rate = cross_rate  # 交叉率
        self.__mut_rate = mut_rate  # 变异率
        self.__precision = precision  # 精度
        self.__code_mode = code_mode  # 编码方式
        self.__cross_strategy = cross_strategy  # 交叉策略
        self.__mut_strategy = mut_strategy  # 变异策略
        self.__is_tol = is_tol  # 是否使用容忍度
        self.__tol = tol  # 容忍度
        self.__random_state = random_state  # 随机种子

    # 初始化相关信息
    def __initial_info(self, func):
        self.__func = func  # 目标函数(适应度函数)
        self.__chrom_length = self.__cal_chrom_length(code_mode=self.__code_mode)  # 获取各维度的染色体长度

        if self.__code_mode == 'binary':  # 计算编解码中线性映射的范围 shape=(n_dim, 2, 2)
            self.__binary_range = np.array(
                [[(self.__lb[i], self.__ub[i]), (0, 2 ** self.__chrom_length[i] - 1)] for i in range(self.__n_dim)])

        if self.__mut_strategy == 'simple':  # 基本位变异中每次变异的位数由染色体长度来决定
            temp_bits = self.__chrom_length.min() // 10
            if temp_bits != 0:
                self.__mut_bits = temp_bits
            else:
                self.__mut_bits = 1

        # 初始化种群染色体: 此时为十进制数据, shape: (pop_size, n_dim)
        self.__pop_chrom = np.around(
            np.random.RandomState(self.__random_state).uniform(self.__lb, self.__ub, (self.__pop_size, self.__n_dim)),
            self.__precision)
        self.__pop_chrom_encode = self.__encode(self.__pop_chrom)  # 编码种群染色体

        self.fitness_best_history_ = []  # 最优个体对应适应值历史
        self.chrom_best_history_ = []  # 最优个体历史

        # 迭代更新主逻辑

    def __update_info(self):
        self.__pop_chrom_encode = self.__crossover(self.__pop_chrom_encode)  # 交叉
        self.__pop_chrom_encode = self.__mutation(self.__pop_chrom_encode)  # 变异
        self.__pop_chrom_encode = self.__selection(self.__pop_chrom_encode)  # 选择

        # 每次迭代保存当前种群最优值信息
        fitness_best, pop_chrom_best = self.__get_best(self.__pop_chrom_encode)
        self.fitness_best_history_.append(fitness_best)
        self.chrom_best_history_.append(pop_chrom_best)

    def __cal_chrom_length(self, code_mode):
        if code_mode == 'binary':
            get_L = lambda x: np.log(x * (10 ** self.__precision) + 1) / np.log(2)
            chrom_length = np.array(np.ceil(get_L((self.__ub - self.__lb))), dtype=np.int64)  # 各个维度染色体的长度

        return chrom_length

    # 编码: 目前仅支持二进制编码方式
    def __encode(self, pop_chrom):
        if self.__code_mode == 'binary':  # 二进制编码
            # shape(二进制编码): (pop_size, n_dim, chrom_length)

            # 线性映射到二进制数可表示的范围中, 由于不同维度对应的范围不同, 因此要一列一列进行处理
            pop_chrom_encode_lst = []
            for dim_i in range(self.__n_dim):
                # 线性映射到目标范围的十进制数并四舍五入得到对应整数
                pop_chrom_interp_dimi = np.interp(pop_chrom[:, dim_i], self.__binary_range[dim_i][0],
                                                  self.__binary_range[dim_i][1])
                pop_chrom_interp_dimi = np.array(np.around(pop_chrom_interp_dimi), dtype=np.int64)

                # 将每一列十进制数转换为对应位数的二进制数
                vfunc = np.vectorize(lambda x: bin(x)[2:].zfill(self.__chrom_length[dim_i]))
                pop_chrom_encode_lst.append(vfunc(pop_chrom_interp_dimi))

            pop_chrom_encode = np.vstack(pop_chrom_encode_lst).T

            return pop_chrom_encode

    # 解码: 执行编码函数的逆操作
    def __decode(self, pop_chrom_encode):
        if self.__code_mode == 'binary':  # 二进制编码

            # 将每一列二进制数转化为十进制数
            vint = np.vectorize(int)
            pop_chrom_dec = vint(pop_chrom_encode, 2)

            pop_chrom_decode_lst = []
            for dim_i in range(self.__n_dim):
                # 线性映射到原范围并保留预先设置的精度
                pop_chrom_interp_dimi = np.around(
                    np.interp(pop_chrom_dec[:, dim_i], self.__binary_range[dim_i][1], self.__binary_range[dim_i][0]),
                    self.__precision)
                pop_chrom_decode_lst.append(pop_chrom_interp_dimi)

            pop_chrom_decode = np.vstack(pop_chrom_decode_lst).T

        return pop_chrom_decode

    # 交叉函数主逻辑
    def __crossover(self, pop_chrom_encode):
        # 交叉率通常较大, 因此直接得到对应交叉次数
        for i in range(int(np.ceil(self.__pop_size * self.__cross_rate))):  # 执行(种群数 * 交叉率)次交叉操作
            # 随机选取两个父代染色体和对应索引
            parent_index1, parent_index2 = np.random.choice(np.array(range(len(pop_chrom_encode))), size=2)
            parent1, parent2 = pop_chrom_encode[parent_index1], pop_chrom_encode[parent_index2]

            # 交叉策略: 当前仅支持单点交叉
            if self.__cross_strategy == 'single':  # 单点交叉
                pop_chrom_encode[parent_index1], pop_chrom_encode[parent_index2] = self.__crossover_single(parent1,
                                                                                                           parent2)

        return pop_chrom_encode

    # 单点交叉
    def __crossover_single(self, parent1, parent2):
        for dim_i in range(self.__n_dim):
            # 随机选取一点(索引), 执行单点交叉
            index = np.random.randint(0, self.__chrom_length[dim_i])

            # 分别获取该维度下染色体的前缀和后缀
            parent1_dimi_pre, parent2_dimi_pre = parent1[dim_i][:index], parent2[dim_i][:index]
            parent1_dimi_suf, parent2_dimi_suf = parent1[dim_i][index:], parent2[dim_i][index:]

            # 将两染色体的后缀交换, 并保存在变量parent1和parent2中
            parent1[dim_i], parent2[dim_i] = ''.join([parent1_dimi_pre, parent2_dimi_suf]), ''.join(
                [parent2_dimi_pre, parent1_dimi_suf])

        return parent1, parent2

    # 变异主逻辑
    def __mutation(self, pop_chrom_encode):
        # 基于轮盘赌判断各个染色体是否发生变异, shape: (pop_size, n_dim)
        is_mut = np.random.choice([False, True], size=(self.__pop_size, self.__n_dim),
                                  p=[1 - self.__mut_rate, self.__mut_rate])
        mut_num = is_mut.sum()  # 发生变异的染色体数目

        if mut_num != 0:  # 存在发生变异的染色体, 执行变异操作
            # 变异策略: 当前仅支持基本位变异
            if self.__mut_strategy == 'simple':  # 基本位变异
                pop_chrom_encode = self.__mutation_simple(pop_chrom_encode, is_mut, mut_num)

        return pop_chrom_encode

    # 基本位变异
    def __mutation_simple(self, pop_chrom_encode, is_mut, mut_num):
        # 定义用于基本位变异的子函数
        def func(x, index):
            x_lst = list(x)
            if x_lst[index] == '0':
                x_lst[index] = '1'
            else:
                x_lst[index] = '0'

            return ''.join(x_lst)

        vfunc = np.frompyfunc(func, nin=2, nout=1)
        vlen = np.vectorize(len)

        for i in range(self.__mut_bits):
            # 随机选取各个染色体的变异1位索引
            mut_index = np.random.randint([0] * mut_num, vlen(pop_chrom_encode[is_mut]), size=mut_num)
            pop_chrom_encode[is_mut] = vfunc(pop_chrom_encode[is_mut], mut_index)

        return pop_chrom_encode

    # 选择
    def __selection(self, pop_chrom_encode):
        pop_chrom_fitness = self.__get_fitness(pop_chrom_encode)  # 各个体的适应值

        # 个体的适应值可正可负, 这里的策略为减去最小值并加上绝对值后的最小值, 然后再计算概率
        pop_chrom_fitness = pop_chrom_fitness - pop_chrom_fitness.min() + abs(pop_chrom_fitness).min()
        pop_chrom_prob = pop_chrom_fitness / pop_chrom_fitness.sum()  # 各个体被选择的概率

        # 基于轮盘赌来选择保留哪些染色体
        pop_chrom_index = np.arange(0, self.__pop_size)

        pop_chrom_select_index = np.random.choice(pop_chrom_index, size=self.__pop_size, p=pop_chrom_prob)
        pop_chrom_select = pop_chrom_encode[pop_chrom_select_index]

        return pop_chrom_select

    # 计算种群中各个体的适应值
    def __get_fitness(self, pop_chrom_encode):
        pop_chrom_decode = self.__decode(pop_chrom_encode)
        pop_chrom_fitness = np.apply_along_axis(self.__func, axis=1, arr=pop_chrom_decode)  # 计算各个体的适应值

        return pop_chrom_fitness.flatten()

    # 获取种群中的最优个体以及对应适应值
    def __get_best(self, pop_chrom_encode):
        pop_chrom_fitness = self.__get_fitness(pop_chrom_encode)
        fitness_best = pop_chrom_fitness.max()  # 当前种群中最优个体的适应值
        pop_chrom_best_index = np.where(pop_chrom_fitness == fitness_best)[0][0]  # 当前种群中的最优个体对应的索引

        pop_chrom_decode = self.__decode(pop_chrom_encode)
        pop_chrom_best = pop_chrom_decode[pop_chrom_best_index, :]  # 当前种群中的最优个体的染色体

        return fitness_best, pop_chrom_best

    # 停止准则：若种群中所有个体的适应度都是0，则停止迭代
    def __is_stop(self):
        flag = False
        if (self.__get_fitness(self.__pop_chrom_encode) == 0).sum() == self.__pop_size:
            flag = True

        if self.__is_tol == True:
            if len(self.fitness_best_history_) >= 8:  # 若过去8次最优染色体对应适应值的变化量的均值小于容忍度tol, 认为已收敛
                if np.abs(np.mean(np.diff(self.fitness_best_history_[-9:]))) < self.__tol:
                    flag = True

        return flag

    # 对外接口: 拟合数据/训练模型
    def fit(self, func):
        # 初始化信息
        self.__initial_info(func=func)

        self.n_iter_ = None
        # 迭代指定次数
        for epoch in range(self.__max_iter):
            if self.__is_stop():
                self.n_iter_ = epoch
                break

            self.__update_info()

        # 记录拟合结果
        if self.n_iter_ == None:
            self.n_iter_ = self.__max_iter

        self.chrom_best_ = self.chrom_best_history_[-1]  # 最优个体
        self.pop_chrom_fitness_ = self.__get_fitness(self.__pop_chrom_encode)  # 最终的所有个体对应的适应值
        self.fitness_best_ = self.fitness_best_history_[-1]  # 最优个体的适应值

        return self

    # 对外接口: 拟合信息
    def fit_info(self):
        print(f'n_iter: {self.n_iter_}')
        print(f'chrom best: {self.chrom_best_}')
        print(f'fitness best: {self.fitness_best_}')
        print(f'fitness best history: {self.fitness_best_history_}')

    # 对外接口: 可视化结果
    def plot_info(self):
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        axes[0].scatter(range(self.__pop_size), self.pop_chrom_fitness_, color='green', alpha=0.6, s=30,
                        edgecolor='white', linewidth=1)
        axes[0].scatter(np.where(self.pop_chrom_fitness_ == self.fitness_best_)[0][0], self.fitness_best_, s=30,
                        color='red', alpha=1, edgecolor='white', linewidth=1, zorder=10)
        axes[0].set_xlabel('pop id')
        axes[0].set_ylabel('fitness')
        axes[0].grid(color='black', linestyle='--', alpha=0.2)

        axes[1].plot(range(1, self.n_iter_ + 1), self.fitness_best_history_, color='blue', alpha=0.6)
        axes[1].scatter(self.n_iter_, self.fitness_best_, s=30, color='red', alpha=1, edgecolor='white', linewidth=1,
                        zorder=10)
        axes[1].grid(color='black', linestyle='--', alpha=0.2)
        axes[1].set_xlabel('n_iter')
        axes[1].set_ylabel('best fitness')
        plt.show()
