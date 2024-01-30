# GA: Genetic Algorithm

## Description
支持一般函数的最优值求解问题
适应度函数(目标函数)值越大越好

## Parameters
- `lb`: tuple; 无默认值; 限定求解空间各维度的下界
- `ub`: tuple; 无默认值; 限定求解空间各维度的上界
- `n_dim`: integer; 无默认值; 求解空间维度
- `pop_size`: integer; 默认为40; 种群大小
- `max_iter`: integer; 默认为100; 最大迭代次数
- `cross_rate`: float; 默认为0.5; 交叉率
- `mut_rate`: float; 默认为0.01; 变异率
- `precision`: integer; 默认为4; 精度, 即保留小数个数
- `code_mode`: string; 默认为"binary"; 编码方式
  - `binary`: 二进制编码
- `cross_strategy`: string; 默认为 "single"; 交叉策略
  - `single`: 单点交叉
- `mut_strategy`: string; 默认为 "simple"; 变异策略
  - `simple`: 基本位变异
- `is_tol`: boolearn; 默认为False; 是否使用容忍度, 若该值为True, 则参数`tol`有效
- `tol`: float; 默认为0.01; 容忍度, 若最近8次迭代中的最优适应度变化量的均值小于tol, 则认为模型已收敛, 停止训练
- `random_state`: integer; 默认为None; 随机数种子, 控制初始化时的随机数生成

## Attribute
- `.n_iter_`: 模型的真实迭代次数
- `.chrom_best_`: 求得的最优解(最终种群中的最优个体)
- `.fitness_best_`: 求得的最优解对应的目标函数值
- `.fitness_best_history_`: 每次迭代中记录的最优个体对应适应值
- `.chrom_best_history_`: 每次迭代中记录的最优个体

## Example
``` python
ga = GA(lb=[0],
        ub=[40],
        n_dim=1,
        max_iter=10,
        is_tol=True,
        tol=0.01,
        mut_rate=0.005,
        cross_rate=0.4,
        pop_size=50,).fit(func=lambda x: -(x * np.sin(10) * x + x * np.cos(2) * x))

ga.fit_info()
```
**Outputs**: 
```
n_iter: 10
chrom best: [39.6399]
fitness best: 1508.73270457635
fitness best history: [1477.8498611787336, 1499.4752986964786, 1480.6990401178587, 1480.6990401178587, 1447.50259084949, 1508.1770657069737, 1508.961078986379, 1509.1209513567983, 1509.1209513567983, 1508.73270457635]
```
