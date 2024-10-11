import pandas as pd
import numpy as np

# 示例數據
data = {
    'C3': [0.00, 0.00, 3.00, 0.00, 4.00, 4.00, -1.00, 2.00, 6.00, 0.00, 1.00, 0.00, 3.00, 5.00, 2.00, 3.00, 2.00, 3.00, 0.00, 1.00, 0.00, 0.00, 1.00, -1.00, 1.00, 0.00, 1.00, 1.00, 8.00, 6.00, 1.00, 1.00],
    'C4': [-1.00, 1.00, 3.00, 2.00, 5.00, 0.00, -3.00, 3.00, 5.00, 0.00, 1.00, 2.00, 3.00, 3.00, 1.00, 2.00, 2.00, 1.00, 0.00, -1.00, 0.00, -1.00, 0.00, 1.00, 0.00, 0.00, -1.00, 0.00, 4.00, 7.00, 1.00, 2.00]
} 
  

# 將數據轉換為 DataFrame
df = pd.DataFrame(data)

# 計算協方差矩陣
cov_matrix = df.cov()

# 將協方差矩陣轉換為 NumPy 陣列
cov_matrix_np = cov_matrix.to_numpy()

# 計算 lambda 1
def calculate_lambda_1(cov_matrix):
    n = cov_matrix.shape[0]
    sum_of_main_diagonal = np.sum(np.diag(cov_matrix))  # 對角線元素之和
    sum_of_cov_matrix_elements = np.sum(cov_matrix)  # 總和
    lambda_1 = 1 - (sum_of_main_diagonal / sum_of_cov_matrix_elements)
    return lambda_1

# 計算 lambda 2
def calculate_lambda_2(cov_matrix):
    n = len(cov_matrix)
    l1 = calculate_lambda_1(cov_matrix)
    
    # 計算非對角線元素平方和
    sum_of_squares = np.sum(cov_matrix[np.tril_indices(n, -1)] ** 2)
    under_square_root = ((n / (n - 1)) * sum_of_squares) ** 0.5
    lambda_2 = l1 + (under_square_root / np.sum(cov_matrix))
    
    return lambda_2

# 進行 10000 次重抽樣並計算 lambda 2
lambda_2_samples = []

for _ in range(10000):
    # 從數據中進行重抽樣（bootstrap sampling）
    bootstrap_sample = df.sample(n=len(df), replace=True)
    
    # 計算重抽樣數據的協方差矩陣
    cov_matrix_bootstrap = bootstrap_sample.cov().to_numpy()
    
    # 計算 lambda 2
    lambda_2_value = calculate_lambda_2(cov_matrix_bootstrap)
    
    # 保存 lambda 2 的值
    lambda_2_samples.append(lambda_2_value)

# 將結果轉換為 NumPy 陣列以便進行計算
lambda_2_samples = np.array(lambda_2_samples)

# 計算 95% 信賴區間
lower_bound = np.percentile(lambda_2_samples, 2.5)
upper_bound = np.percentile(lambda_2_samples, 97.5)

# 打印結果
print(f"95% 信賴區間: ({lower_bound}, {upper_bound})")
print(f"前 10 筆 Lambda 2 值: {lambda_2_samples[:10]}")
