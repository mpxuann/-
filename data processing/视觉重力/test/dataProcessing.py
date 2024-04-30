
import numpy as np
from statsmodels.stats.proportion import proportions_ztest

# 假设我们有两个样本，每个样本都进行了500次试验
# 在第一个样本中，有200次试验成功
# 在第二个样本中，有150次试验成功
count1 = 446
nobs1 = 905
count2 = 546
nobs2 = 1082

# 将成功次数和试验次数放入numpy数组中
counts = np.array([count1, count2])
nobs = np.array([nobs1, nobs2])

# 进行z检验
z, p = proportions_ztest(counts, nobs)

print(f'z-statistic: {z:.2f}')
print(f'p-value: {p:.4f}')