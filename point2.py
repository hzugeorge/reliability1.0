import pandas as pd
import numpy as np

# 示例數據
data = {
    'C3': [32.00, 32.00, 31.00, 23.00, 32.00, 32.00, 30.00, 32.00, 31.00, 32.00, 32.00, 32.00, 32.00, 32.00, 32.00, 32.00, 28.00, 31.00, 32.00, 32.00, 32.00, 32.00, 32.00, 31.00, 32.00, 32.00, 32.00, 32.00, 32.00, 32.00, 31.00, 32.00],
    'C4': [31.00, 32.00, 30.00, 24.00, 32.00, 31.00, 29.00, 32.00, 32.00, 32.00, 31.00, 32.00, 32.00, 29.00, 32.00, 32.00, 27.00, 32.00, 32.00, 31.00, 32.00, 31.00, 32.00, 32.00, 31.00, 32.00, 31.00, 32.00, 28.00, 32.00, 32.00, 32.00],
    'C7': [32.00, 32.00, 28.00, 23.00, 28.00, 28.00, 31.00, 30.00, 25.00, 32.00, 31.00, 32.00, 29.00, 27.00, 30.00, 29.00, 26.00, 28.00, 32.00, 31.00, 32.00, 32.00, 31.00, 26.00, 31.00, 32.00, 31.00, 31.00, 24.00, 26.00, 30.00, 31.00],
    'C8': [32.00, 31.00, 27.00, 22.00, 27.00, 31.00, 32.00, 29.00, 27.00, 32.00, 30.00, 30.00, 29.00, 26.00, 31.00, 30.00, 25.00, 31.00, 32.00, 32.00, 32.00, 32.00, 32.00, 25.00, 31.00, 32.00, 32.00, 32.00, 24.00, 25.00, 31.00, 30.00]
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
    sum_of_main_diagonal = np.sum(np.diag(cov_matrix))
    sum_of_cov_matrix_elements = np.sum(cov_matrix)
    lambda_1 = 1 - (sum_of_main_diagonal / sum_of_cov_matrix_elements)
    return lambda_1

# 計算非對角線元素平方和
def sum_of_squares_of_non_diagonal_matrix_elements(cov_matrix):
    n = len(cov_matrix)
    summ = 0
    for i in range(n):
        for j in range(i):
            summ += cov_matrix[i, j] ** 2
    return 2 * summ

# 計算 lambda 2
def calculate_lambda_2(cov_matrix):
    n = len(cov_matrix)
    l1 = calculate_lambda_1(cov_matrix)
    sum_of_squares = sum_of_squares_of_non_diagonal_matrix_elements(cov_matrix)
    under_square_root = ((n / (n - 1)) * sum_of_squares) ** (1 / 2)
    lambda_2 = l1 + (under_square_root / np.sum(cov_matrix))
    return lambda_2

# 打印協方差矩陣
print("協方差矩陣：")
print(cov_matrix)

# 打印 lambda 1 的值
lambda_1 = calculate_lambda_1(cov_matrix_np)
print(f"Lambda 1: {lambda_1}")

# 打印 lambda 2 的值
lambda_2 = calculate_lambda_2(cov_matrix_np)
print(f"Lambda 2: {lambda_2}")

# 進行 1000 次重抽樣並計算 lambda 2
lambda_2_samples = []
sampled_data = []

for i in range(10000):
    # 從數據中進行重抽樣（bootstrap sampling）
    bootstrap_sample = df.sample(n=len(df), replace=True)
    
    # 儲存抽取的數據
    sampled_data.append(bootstrap_sample)

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
print(f"95% 信賴區間: ({lower_bound:.4f}, {upper_bound:.4f})")

# 將抽取的數據轉換為單個 DataFrame 以便保存
sampled_df = pd.concat(sampled_data, ignore_index=True)
sampled_df.to_csv('data_for_analysis.csv', index=False)

print("前1000筆抽取的數據已儲存到 data_for_analysis.csv。")

# 打印前10筆抽取的 lambda 2 值
print("前10筆抽取的 lambda 2 值:")
for i in range(10):
    print(f"Sample {i + 1} 的 lambda 2: {lambda_2_samples[i]:.4f}")
