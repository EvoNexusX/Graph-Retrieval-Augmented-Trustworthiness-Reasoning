import matplotlib.pyplot as plt
import seaborn as sns

# 使用Seaborn设置SciPy风格的配色，并去掉网格
sns.set_style('white')  # 设置背景为白色，不包含网格
palette = sns.color_palette('muted')

# 数据
x = ['WEPS', 'WIPS', 'GPS', 'SPS', 'VPS']
GRAR = [4.653, 7.080, 7.560, 6.180, 4.860]
Origin_Native_RAG = [2.047, 3.980, 3.500, 3.380, 4.000]

# 绘制折线图，使用SciPy配色
plt.plot(x, GRAR, marker='o', label='GRATR', color=palette[0])
plt.plot(x, Origin_Native_RAG, marker='o', label='Baseline LLM + Native RAG', color=palette[1])

# 在每个数据点上标注数值，并调整字体大小
for i, value in enumerate(GRAR):
    plt.text(x[i], GRAR[i], f'{value:.3f}', ha='center', va='bottom', fontsize=12)

for i, value in enumerate(Origin_Native_RAG):
    plt.text(x[i], Origin_Native_RAG[i], f'{value:.3f}', ha='center', va='bottom', fontsize=12)

# 设置横纵坐标标签的字体大小
plt.xlabel('Metrics', fontsize=16)
plt.ylabel('Scores', fontsize=16)
plt.legend()

# 保存图表为 PDF 文件
plt.savefig('performance_comparison_native_rag.pdf', format='pdf')

# 显示图表
plt.show()
