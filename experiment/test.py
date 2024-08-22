import matplotlib.pyplot as plt

# 数据
x = ['WEPS', 'WIPS', 'GPS', 'SPS', 'VPS']
GRATR = [5.487, 7.120, 7.240, 6.040, 4.460]
Origin = [2.200, 2.780, 2.420, 2.480, 3.600]

# 绘制折线图
plt.plot(x, GRATR, marker='o', label='GRATR')
plt.plot(x, Origin, marker='o', label='Baseline LLM')

# 在每个数据点上标注数值，并调整字体大小
for i, value in enumerate(GRATR):
    plt.text(x[i], GRATR[i], f'{value:.3f}', ha='center', va='bottom', fontsize=12)

for i, value in enumerate(Origin):
    plt.text(x[i], Origin[i], f'{value:.3f}', ha='center', va='bottom', fontsize=12)

# 图表标题和标签，并调整标题字体大小
# plt.title('Performance comparison between GRATR and the original algorithm', fontsize=16)
plt.xlabel('Metrics', fontsize=16)
plt.ylabel('Scores', fontsize=16)
plt.legend()

# 保存图表为 PDF 文件
plt.savefig('performance_comparison.pdf', format='pdf')

# 显示图表
plt.show()
