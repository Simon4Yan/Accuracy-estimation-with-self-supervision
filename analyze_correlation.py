import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter
from scipy.stats import pearsonr

cls_acc = np.load('./accuracy_cls_dense_aug.npy')
ssh_acc = np.load('./accuracy_ss_dense_aug.npy')

majorFormatter = FormatStrFormatter('%0.1f')

palette = sns.color_palette("Set2")
plt.rcParams['xtick.labelsize'] = 30
plt.rcParams['ytick.labelsize'] = 30

robust = True
sns.set()
sns.set(font_scale=1.3)
sns.set_style('darkgrid', {'axes.facecolor': '0.96', 'axes.linewidth': 20, 'axes.edgecolor': '0.15'})

f, ax1 = plt.subplots(1, 1, tight_layout=True)

sns.regplot(ax=ax1, color=palette[4], x=ssh_acc, y=cls_acc, robust=robust, scatter_kws={'alpha': 0.5, 's': 30},
            label='{:<2}\n{:>8}'.format('CIFAR-10', r'$r$' + '={:.3f}'.format(pearsonr(ssh_acc, cls_acc)[0])))
ax1.legend(loc=2, shadow=True, labelspacing=-0.0, handletextpad=0, borderpad=0.5, markerscale=2,
           prop={'weight': 'medium', 'size': '16'})
ax1.set_xlim(30, 90)
ax1.set_ylim(0, 100)

ax1.xaxis.set_major_formatter(majorFormatter)
ax1.yaxis.set_major_formatter(majorFormatter)
f.savefig('correlation.pdf')

from scipy import stats

rho, pval = stats.spearmanr(ssh_acc, cls_acc)
print('\nRank correlation-rho', rho)
print('Rank correlation-pval', pval)

rho, pval = stats.pearsonr(ssh_acc, cls_acc)
print('\nPearsons correlation-rho', rho)
print('Pearsons correlation-pval', pval)
