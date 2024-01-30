# PSO: Particle Swarm Optimization

## Description
支持一般函数的最优值求解问题
适应度函数(目标函数)值越小越好

## Parameters
- `lb`: tuple; 无默认值; 限定求解空间各维度的下界
- `ub`: tuple; 无默认值; 限定求解空间各维度的上界
- `n_dim`: integer; 无默认值; 求解空间维度
- `pop_size`: integer; 默认为40; 种群大小(粒子个数)
- `w`: constant/str; 默认为0.8; 惯性权重,
  - constant: 惯性权重为定值, 不发生改变
  - "line_decay": 惯性权重随着迭代次数的增加线性衰减, 若为"line_decay", 则需指定w_max, w_min
- `c1`: constant; 默认为0.5; 个体最优位置学习因子
- `c2`: constant; 默认为0.5; 全局最优位置学习因子
- `is_tol`: boolean, 默认为False; 是否使用容忍度, 若该值为True, 则参数`tol`有效
- `tol`: constant; 默认为0.01; 容忍度, 若最近8次迭代中的最优适应度变化量的均值小于tol, 则认为模型已收敛, 停止训练
- `max_iter`: integer; 默认为100; 最大迭代次数
- `vlimit`: float in [0, 1]; 默认为0.2; 限制速度变化的大小, 即 $V_{pre}(1-vlimit)\le V_{cur} \le V_{pre}(1+vlimit)$
- `w_max`: constant; 默认为None; 当参数w设置为"line_decay"时需要设置该参数, 表示惯性权重的最大值
- `w_min`: constant; 默认为None; 当参数w设置为"line_decay"时需要设置该参数, 表示惯性权重的最小值
- `random_state`: integer; 默认为None; 随机数种子, 控制初始化时的随机数生成

## Attributes
- `.n_iter_`: 模型的真实迭代次数
- `.gbest_pos_`: 求得的最优解(最终种群的全局最优位置)
- `.gbest_fitness_`: 求得的最优解对应的目标函数值(最终种群的全局最优位置对应的适应值)
- `.gbest_fitness_history_`: 每次迭代中记录的全局最优位置对应适应值

## Methods
- `.fit(func)`: 传入求解的目标函数, 进行模型训练, 函数值越小越好
- `.fit_info()`: 模型训练后打印输出相关结果信息
- `.plot_info()`: 可视化模型拟合结果

## Example
``` python
pso = PSO(
    lb=[0],
    ub=[40],
    n_dim=1,
    w=0.8,
    is_tol=True,
    tol=0.01,
    max_iter=100,
    pop_size=40).fit(func=lambda x: x * np.sin(10) * x + x * np.cos(2) * x)

pso.fit_info()
```
**Output**: 
```
n_iter:  12
gbest X:  [40.]
gbest Y:  -1536.2687158984195
gbest Y history:  [-1481.8661325328146, -1481.8661325328146, -1490.2356171334081, -1536.2687158984195, -1536.2687158984195, -1536.2687158984195, -1536.2687158984195, -1536.2687158984195, -1536.2687158984195, -1536.2687158984195, -1536.2687158984195, -1536.2687158984195]
```


