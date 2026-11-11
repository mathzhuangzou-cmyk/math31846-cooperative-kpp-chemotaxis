"""Small executable checks for the corrected code, not a full PDE proof."""
from pathlib import Path
import ast, importlib.util, json, sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]

def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);sys.modules[name]=m
    spec.loader.exec_module(m);return m

def main():
    torch.set_num_threads(1)
    files=list((ROOT/'patched_project').glob('*.py'))
    for file in files: ast.parse(file.read_text(encoding='utf-8-sig'))
    m=load(ROOT/'patched_project/1.py','patched_solver_check')
    m7=load(ROOT/'patched_project/7.1.py','patched_audit_check')
    p=m.ModelParams();c=m.continuous_critical_data(p)
    s=m.select_mode(10,p,c['mu_c'])
    cc=m.chi2_data(10,p,s['n_star'],s['mu_star'],s['chi_T'])
    solver=m.SpectralIMEX1D(p,10.,257,.02,torch.device('cpu'),torch.float64)
    short=m.stationary_branch_run(solver,p,s['n_star'],s['chi_T'],cc['chi2'],cc,.01,.1,0.,1.)
    assert short['sim_time']==.1
    assert np.isfinite(short['residual']) and short['residual']>0
    assert not short['residual_tolerance_met']
    fields=[torch.tensor(np.r_[short[k],short[k][-2:0:-1]],dtype=torch.float64) for k in ['u','v','w']]
    true_residual=solver.steady_residual(*fields,s['chi_T']+.01)
    assert abs(true_residual-short['residual'])<1e-10
    x=np.linspace(0,10,65);y=np.linspace(0,10,257)
    profile=1+.1*np.cos(5*np.pi*x/10)+.01*np.cos(10*np.pi*x/10)
    target=1+.1*np.cos(5*np.pi*y/10)+.01*np.cos(10*np.pi*y/10)
    err=float(abs(m7.prolong_even_half(profile,512)-target).max())
    assert err<2e-14
    # Finite-difference derivative of the real time-step routine, independent
    # of the helper's matrix construction.
    eq=[torch.ones_like(solver.x)*v for v in [1.,1.,p.beta/p.alpha]]
    mode=torch.cos(s['n_star']*np.pi*solver.x/10);eps=1e-5
    J=np.zeros((3,3));chi=1.02*s['chi_T']
    for j in range(3):
        plus=[v.clone() for v in eq];minus=[v.clone() for v in eq]
        plus[j]+=eps*mode;minus[j]-=eps*mode
        up=solver.step(*plus,chi);um=solver.step(*minus,chi)
        for i in range(3):J[i,j]=solver.cosine_amplitude((up[i]-um[i])/(2*eps),s['n_star'],0.)
    measured=float(np.log(abs(np.linalg.eigvals(J)).max())/solver.dt)
    reported=m.actual_imex_growth_rate(p,s['mu_star'],chi,solver.dt)
    assert abs(measured-reported)<2e-8
    result={'syntax_files':len(files),'terminal_time':short['sim_time'],
      'terminal_residual':short['residual'],'prolongation_max_error':err,
      'reported_growth_rate':reported,'numerically_linearized_growth_rate':measured,
      'growth_rate_difference':abs(measured-reported)}
    out=ROOT/'audit_evidence/rechecked';out.mkdir(parents=True,exist_ok=True)
    (out/'patched_checks.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
