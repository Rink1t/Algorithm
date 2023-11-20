# PSO: Particle Swarm Optimization

## Parameters
- `lb`: tuple; 无默认值; 限定求解空间各维度的下界
- `ub`: tuple; 无默认值; 限定求解空间各维度的上界
- `n_dim`: integer; 无默认值; 求解空间维度
- `w`: constant/str; 默认为0.8; 惯性权重,
  - constant: 惯性权重为定值, 不发生改变
  - "line_decay": 惯性权重随着迭代次数的增加线性衰减, 若为"line_decay", 则需指定w_max, w_min
- `c1`: constant; 默认为0.5; 个体最优位置学习因子
- `c2`: constant, 默认为0.5; 全局最优位置学习因子
- `pop`: integer; 默认为40; 种群大小(粒子个数)
- `tol`: constant; 默认为0.1; 容忍度, 若最近3次epoch中全局最优位置对应适应值变化量的均值小于tol, 则认为模型已收敛, 停止训练
- `max_iter`: integer; 默认为100; 最大迭代次数
- `vlimit`: float in [0, 1]; 默认为0.2; 限制速度变化的大小, 即:
  $$
  pre_v(1-vlimit)\le cur_v \le pre_v(1+vlimit)
  $$
- `w_max`: constant; 默认为None; 当参数w设置为"line_decay"时需要设置该参数, 表示惯性权重的最大值
- `w_min`: constant; 默认为None; 当参数w设置为"line_decay"时需要设置该参数, 表示惯性权重的最小值
- `random_state`: integer; 默认为None; 随机数种子, 控制随机数生成

## Attributes
- `n_iter_`: 模型的真实迭代次数
- `gbest_pos_`: 求得的最优解(最终种群的全局最优位置)
- `gbest_fitn_`: 求得的最优解对应的目标函数值(最终种群的全局最优位置对应的适应值)
- `gbest_fitn_his_`: 全局最优位置对应适应值的替换记录, 模型迭代过程中的各个全局最优位置对应适应值

## Methods
- `fit(func)`: 传入求解的目标函数, 进行模型训练, 函数值越小越好
- `fit_info()`: 模型训练后打印输出相关结果信息
