import matplotlib.pyplot as plt
import seaborn as sns

# 使用Seaborn设置SciPy风格的配色，并去掉网格
sns.set_style('white')
palette = sns.color_palette('muted')

# 数据
x = ['WEPS', 'WIPS', 'GPS', 'SPS', 'VPS']
GRATR = [6.380, 8.300, 8.420, 7.880, 4.500]
Origin_Rerank_RAG = [1.206, 2.220, 1.500, 1.860, 3.460]

# 绘制折线图，使用SciPy配色
plt.plot(x, GRATR, marker='o', label='GRATR', color=palette[0])
plt.plot(x, Origin_Rerank_RAG, marker='o', label='Baseline LLM + Rerank RAG', color=palette[1])

# 在每个数据点上标注数值，并调整字体大小
for i, value in enumerate(GRATR):
    plt.text(x[i], GRATR[i], f'{value:.3f}', ha='center', va='bottom', fontsize=12)

for i, value in enumerate(Origin_Rerank_RAG):
    plt.text(x[i], Origin_Rerank_RAG[i], f'{value:.3f}', ha='center', va='bottom', fontsize=12)

# 设置横纵坐标标签的字体大小
plt.xlabel('Metrics', fontsize=16)
plt.ylabel('Scores', fontsize=16)
plt.legend()

# 保存图表为 PDF 文件
plt.savefig('performance_comparison_rerank_rag.pdf', format='pdf')

# 显示图表
plt.show()
