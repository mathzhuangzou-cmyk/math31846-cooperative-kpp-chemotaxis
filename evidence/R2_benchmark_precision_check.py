#!/usr/bin/env python3
"""Recompute Math-31846 R2 benchmarks and regenerate Figure 1.

Usage: python R2_benchmark_precision_check.py --output .
Dependencies: numpy, scipy, mpmath, matplotlib.
The equations and parameters are those in manuscript Sections 4.1 and 5.1.
No PDE simulation, fitted coefficient, or imported rounded constant is used.
"""
import argparse
import inspect
import json
from pathlib import Path
import numpy as np
import scipy
from scipy.optimize import brentq
import mpmath as mp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def calculate(cast, pi):
    d1,d2,Dw,r1,r2,k12,k21,alpha,beta=map(cast,['0.2','0.1','1','1','0.8','0.3','0.2','1','1'])
    a=r1+k12; b=r2+k21; K=k12*k21; D=a*b-K
    def threshold(mu):
        return (Dw*mu+alpha)*((d1*mu+a)*(d2*mu+b)-K)/(beta*mu*(d2*mu+b))
    def polynomial(mu):
        return Dw*d1*d2*d2*mu**4+2*Dw*b*d1*d2*mu**3+(Dw*K*d2+Dw*b*b*d1-a*alpha*d2*d2)*mu**2-2*alpha*d2*D*mu-alpha*b*D
    if cast is float:
        muc=brentq(polynomial,0.001,100.,xtol=1e-14)
        sqrt=np.sqrt
    else:
        muc=mp.findroot(polynomial,(cast('2'),cast('3')))
        sqrt=mp.sqrt
    mu=(5*pi/10)**2; chi=threshold(mu)
    A=d1*mu+a; B=d2*mu+b; C=Dw*mu+alpha
    eta=k21/B; zeta=beta/C
    U0=-(b*r1+k12*r2*eta**2)/(2*D)
    V0=(k21*U0-r2*eta**2/2)/b; W0=beta/alpha*U0
    A2=4*d1*mu+a; B2=4*d2*mu+b; C2=4*Dw*mu+alpha
    D2=A2-K/B2-4*chi*beta*mu/C2
    U2=-(r1/2-chi*zeta*mu+k12/B2*r2*eta**2/2)/D2
    V2=(k21*U2-r2*eta**2/2)/B2; W2=beta/C2*U2
    N1=chi*mu*(W2+zeta*U0-zeta*U2/2)-2*r1*U0-r1*U2
    N2=-2*r2*eta*V0-r2*eta*V2
    chi2=-C/(beta*mu)*(N1+k12/B*N2)
    values={'mu_c':muc,'chi_c':threshold(muc),'k_c':sqrt(muc),'Lambda_c':2*pi/sqrt(muc),'mu_5':mu,'chi_T':chi,'chi2':chi2}
    return values,threshold,polynomial

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=Path('.'));args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    vals,threshold,poly=calculate(float,np.pi)
    mp.mp.dps=60
    high,_,_=calculate(mp.mpf,mp.pi)
    report={'purpose':'Reviewer 2 comment 4 benchmark arithmetic check and Figure 1 regeneration',
      'numpy_version':np.__version__,'scipy_version':scipy.__version__,'mpmath_version':mp.__version__,
      'arithmetic':'IEEE 754 binary64','root_solver':'scipy.optimize.brentq','bracket':[0.001,100.0],
      'xtol':1e-14,'rtol':float(inspect.signature(brentq).parameters['rtol'].default),
      'binary64':vals,'display_12_decimal_places':{k:f'{v:.12f}' for k,v in vals.items()},
      'high_precision_decimal_digits':60,'high_precision_exact_decimal_inputs':True,
      'high_precision_values':{k:mp.nstr(v,55) for k,v in high.items()},
      'absolute_difference_from_high_precision':{k:float(abs(mp.mpf(vals[k])-high[k])) for k in vals},
      'quartic_residual_binary64':float(poly(vals['mu_c'])),
      'correction':{'quantity':'mu_c','old_display':'2.468635740034','new_display':f'{vals["mu_c"]:.12f}'},
      'pde_simulations_rerun':False,'note':'Root tolerances and high-precision arithmetic checks do not certify PDE discretization error.'}
    assert all(v<2e-13 for v in report['absolute_difference_from_high_precision'].values())
    (args.output/'R2_benchmark_precision_check.json').write_text(json.dumps(report,indent=2)+'\n')
    plt.rcParams.update({'font.family':'serif','mathtext.fontset':'stix','font.size':10,'axes.labelsize':12,'axes.titlesize':11,'legend.fontsize':9})
    fig,ax=plt.subplots(figsize=(7.2,4.8),layout='constrained')
    mu=np.linspace(.05,8,2000); y=threshold(mu)
    ax.plot(mu,y,color='#237bb6',lw=2,label=r'$\chi(\mu)$')
    ax.fill_between(mu,y,vals['chi_c'],color='#237bb6',alpha=.08)
    n=np.arange(1,9);mus=(n*np.pi/10)**2;ys=threshold(mus)
    ax.scatter(mus,ys,c='#2d9a3e',s=35,zorder=4,label='Neumann modes')
    ax.scatter(vals['mu_c'],vals['chi_c'],s=90,facecolors='none',edgecolors='#ef7d0b',lw=1.8,zorder=5,label='Continuous minimum')
    ax.annotate('Critical mode 5',xy=(vals['mu_5'],vals['chi_T']),xytext=(2.95,4.2),arrowprops={'arrowstyle':'->','lw':.9},fontsize=11)
    ax.annotate(r'$\mu_c=2.4686$'+'\n'+r'$\chi_c=2.4527$',xy=(vals['mu_c'],vals['chi_c']),xytext=(1.1,8),arrowprops={'arrowstyle':'->','lw':.9},fontsize=10)
    ins=ax.inset_axes([.56,.43,.41,.32])
    iz=np.linspace(2.2,2.8,500)
    ins.plot(iz,threshold(iz),color='#237bb6',lw=1.4)
    ins.scatter(vals['mu_c'],vals['chi_c'],color='#ef7d0b',s=24,zorder=4)
    ins.scatter(vals['mu_5'],vals['chi_T'],facecolors='none',edgecolors='#2d9a3e',s=46,lw=1.2,zorder=5)
    ins.set_xlim(2.2,2.8);ins.set_ylim(2.44,2.50);ins.set_xticks([2.2,2.4,2.6,2.8]);ins.set_yticks([2.44,2.46,2.48,2.50]);ins.tick_params(labelsize=8)
    ins.set_title('Near the critical minimum',fontsize=9)
    ax.set(xlabel=r'$\mu$',ylabel=r'$\chi(\mu)$',xlim=(0,8.2),ylim=(1.5,28),title='Modal-threshold curve and mode selection at onset')
    ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.18);ax.legend(loc='upper right',frameon=False)
    fig.savefig(args.output/'figure1_dispersion_real_high.png',dpi=400)
    fig.savefig(args.output/'figure1_modal_threshold.pdf')
    plt.close(fig)
    print(json.dumps({'values':report['display_12_decimal_places'],'max_absolute_arithmetic_difference':max(report['absolute_difference_from_high_precision'].values()),'figure':'figure1_dispersion_real_high.png'}))

if __name__=='__main__':main()
