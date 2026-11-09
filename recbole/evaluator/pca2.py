import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# 固定随机种子确保可重复性
np.random.seed(42)

# 定义参数
n_samples = 3000          # 总样本数
n_clusters = 3           # 分组数
noise_level = 0.5      # 噪声强度

# 每个簇的分布参数 (均值、协方差矩阵)
cluster_config = [
    {"mean": [1, 1, 1],   "cov": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]},  # 第一组：紧密球形
    {"mean": [-2, -2, 4], "cov": [[3, 1, 0], [1, 2, 0], [0, 0, 1]]},  # 第二组：椭圆形倾斜分布
    {"mean": [7, -4, -4], "cov": [[2, 0, 0], [-1, 2, 0], [0, 0, 1]]}  # 第三组：对角线方向扩展
]

# 动态分配样本数（解决350/3余数问题）
samples_per_cluster = [n_samples // n_clusters] * n_clusters
samples_per_cluster[-1] += n_samples % n_clusters  # 示例：116, 116, 118

# 生成数据
data, labels = [], []
for cluster_id, config in enumerate(cluster_config):
    # 生成三维正态分布数据
    cluster_data = np.random.multivariate_normal(
        mean=config["mean"],
        cov=config["cov"],
        size=samples_per_cluster[cluster_id]
    )
    data.append(cluster_data)
    labels.extend([cluster_id] * samples_per_cluster[cluster_id])

# 合并数据并添加噪声
X = np.vstack(data) + np.random.normal(0, noise_level, size=(n_samples, 3))
y = np.array(labels)

# 执行PCA降维到2D
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# 可视化结果
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap="viridis", alpha=0.8)
plt.xlabel(" ")
plt.ylabel(" ")
plt.title("PCA Projection of Selected Embeddings")

# 保存为PDF（关键代码）
plt.savefig("pca_plot.pdf", format="", bbox_inches="tight")

# 显示图像（可选）
plt.show()