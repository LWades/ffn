import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

data = {
    'Physical error rate': ['0.01', '0.01', '0.01', '0.01', '0.05', '0.05', '0.05', '0.05', '0.07', '0.07', '0.07', '0.07', '0.10', '0.10', '0.10', '0.10', '0.15', '0.15', '0.15', '0.15'],
    'Geometric symmetry': ['all', 'tl', 'rt', 'rf', 'all', 'tl', 'rt', 'rf', 'all', 'tl', 'rt', 'rf', 'all', 'tl', 'rt', 'rf', 'all', 'tl', 'rt', 'rf'],
    'Equivalent samples': [9998834, 9996790, 9996673, 9996681, 9974339, 9977949, 9959277, 9962185, 9972738, 9971890,
                           9961347, 9961212, 9970381, 9968575, 9962301, 9961029, 9968934, 9968136, 9951389, 9948298]
}

df = pd.DataFrame(data)

# plaette = 'deep'
plaette = 'PuBu_r'
# plaette = 'Blues_r'
# plaette = 'GnBu_r'

# plaette = 'pastel'

# 创建柱状图
plt.figure(figsize=(10, 6), dpi=600)
sns.barplot(data=df, x='Physical error rate', y='Equivalent samples', hue='Geometric symmetry', palette=plaette)
plt.title('d = 3')
plt.xlabel('Physical error rate')
plt.ylabel('Equivalent samples')
plt.legend(title='Geometric symmetry', loc='upper center', bbox_to_anchor=(0.5, 0.15), ncol=4)
plt.show()
