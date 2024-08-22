import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# 使用Seaborn设置颜色和样式，采用 Color Brewer 的 "Set2" 方案，颜色柔和且对比鲜明
sns.set_style('white')
palette = sns.color_palette("Set2", 2)  # Set2 是 Color Brewer 中的柔和配色方案

# 数据
categories = ['Total', 'WE', 'WI', 'GU', 'SE', 'VI']
GRATR = [0.751, 0.868, 0.686, 0.669, 0.771,0.661]
Origin_Rearnk_RAG = [0.295, 0.297, 0.321, 0.262, 0.280, 0.321]

# 设置柱状图的宽度和位置
bar_width = 0.35
index = np.arange(len(categories))

# 绘制分组柱状图，使用更柔和的 Set2 配色
bars1 = plt.bar(index, GRATR, bar_width, label='GRATR', color=palette[0])
bars2 = plt.bar(index + bar_width, Origin_Rearnk_RAG, bar_width, label='Baseline LLM + Rerank RAG', color=palette[1])

# 在每个数据点上标注数值，并调整字体大小
def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.02,
            f'{height:.3f}',
            ha='center',
            va='bottom',
            fontsize=12
        )

add_labels(bars1)
add_labels(bars2)

# 设置横纵坐标标签的字体大小
plt.xlabel('Identity', fontsize=16)
plt.ylabel('Successful Ratio', fontsize=16)

# 设置x轴类别标签
plt.xticks(index + bar_width / 2, categories, fontsize=12)

# 去掉图表边框以使图表更简洁
sns.despine()

# 添加图例
plt.legend()

# 保存图表为 PDF 文件
plt.savefig('successful_ratio_comparison_rerank_rag.pdf', format='pdf')

# 显示图表
plt.show()
