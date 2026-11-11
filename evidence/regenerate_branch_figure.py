"""Regenerate manuscript Figure 2 from the audited Table 2 data.

Run from any directory: python evidence/regenerate_branch_figure.py
No random data, fitting, or PDE rerun is performed by this plotting script.
"""
from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[1]
with (root/'evidence/table2_offset_recomputed.csv').open() as f:
    rows=list(csv.DictReader(f))
x=np.array([float(r['delta_chi']) for r in rows])
a=np.array([float(r['A_numeric']) for r in rows])
chi2=2.1143925003845543
plt.rcParams.update({'font.family':'serif','font.serif':['STIXGeneral'],
                     'mathtext.fontset':'stix','font.size':10,'axes.labelsize':12,
                     'axes.spines.top':False,'axes.spines.right':False,
                     'savefig.dpi':300})
fig,ax=plt.subplots(figsize=(7.4,5.1),layout='constrained')
xx=np.linspace(0,0.011,300)
ax.plot(xx,xx/chi2,color='0.25',lw=1.5,label=r'Leading-order law $A^2=\delta/\chi_2$')
ax.plot(x,a*a,'o-',color='#23699b',lw=1.0,ms=5,label=r'Finite-time amplitudes, $T=500$')
ax.set(xlabel=r'$\delta=\chi-\chi_T$',ylabel=r'$A_{\rm num}^{2}$',
       xlim=(0,.0113),ylim=(0,.0054))
ax.grid(True,color='0.92',lw=.6)
ax.legend(loc='upper left',frameon=False,fontsize=9)
ins=ax.inset_axes([.55,.16,.39,.32])
ins.plot(x,100*(chi2*a*a/x-1),'o-',color='#d06115',ms=3,lw=1)
ins.set_title('Finite-time deviation',fontsize=9)
ins.set_xlabel(r'$\delta$',fontsize=9)
ins.set_ylabel(r'$100(\chi_2 A^2/\delta-1)$',fontsize=8)
ins.tick_params(labelsize=8)
ins.grid(True,color='0.92',lw=.5)
for ext in ('png','pdf'):
    fig.savefig(root/f'figures/figure4_bifurcation_branch.{ext}')
plt.close(fig)
print('Figure 2 regenerated from table2_offset_recomputed.csv')
