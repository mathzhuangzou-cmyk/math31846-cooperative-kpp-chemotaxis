from pathlib import Path
import importlib.util, sys, json
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'audit_evidence/rechecked'
OUT.mkdir(parents=True,exist_ok=True)
def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

def main():
    torch.set_num_threads(1)
    m=load(ROOT/'original_code/1.py','original_solver_check')
    m7=load(ROOT/'original_code/7.1.py','original_audit_check')
    p=m.ModelParams();c=m.continuous_critical_data(p);s=m.select_mode(10,p,c['mu_c'])
    cc=m.chi2_data(10,p,s['n_star'],s['mu_star'],s['chi_T'])
    solver=m.SpectralIMEX1D(p,10.,257,.02,torch.device('cpu'),torch.float64)
    eq=[torch.ones_like(solver.x)*v for v in [1.,1.,p.beta/p.alpha]]
    stepped=solver.step(*eq,1.02*s['chi_T'])
    eq_error=max(float((a-b).abs().max()) for a,b in zip(eq,stepped))
    eps=1e-5;mode=torch.cos(s['n_star']*np.pi*solver.x/10)
    J=np.zeros((3,3))
    for j in range(3):
        plus=[v.clone() for v in eq];minus=[v.clone() for v in eq]
        plus[j]+=eps*mode;minus[j]-=eps*mode
        up=solver.step(*plus,1.02*s['chi_T']);um=solver.step(*minus,1.02*s['chi_T'])
        for i in range(3):J[i,j]=solver.cosine_amplitude((up[i]-um[i])/(2*eps),s['n_star'],0.)
    G=m7.imex_amplification_matrix(p,s['mu_star'],1.02*s['chi_T'],.02)
    jacerr=float(abs(J-G).max())
    rows=[]
    for dtype in [torch.float32,torch.float64]:
        sol=m.SpectralIMEX1D(p,10.,257,.02,torch.device('cpu'),dtype)
        for ratio in [.98,1.02]:
            r=m.threshold_crossing_run(sol,p,s['n_star'],s['mu_star'],s['chi_T'],ratio,80.,1e-4,.2)
            g=m7.imex_amplification_matrix(p,s['mu_star'],ratio*s['chi_T'],.02)
            exact_discrete=float(np.log(abs(np.linalg.eigvals(g)).max())/.02)
            rows.append({'dtype':str(dtype),'ratio':ratio,'continuous':r['lambda_continuous'],
                         'old_reported_discrete':r['lambda_imex'],'actual_discrete':exact_discrete,
                         'observed_fit':r['fit_rate']})
    short=m.stationary_branch_run(solver,p,s['n_star'],s['chi_T'],cc['chi2'],cc,.01,.1,0.,1.)
    actual_res=float(solver.steady_residual(*[torch.tensor(np.concatenate([short[k],short[k][-2:0:-1]]),dtype=torch.float64) for k in ['u','v','w']],s['chi_T']+.01))
    result={'torch_version':torch.__version__,'device':'cpu','equilibrium_one_step_error':eq_error,
      'linearized_step_vs_actual_IMEX_matrix_error':jacerr,'growth_rate_checks':rows,
      'short_run_metadata_bug':{'requested_T':.1,'dt':.02,'old_sim_time':short['sim_time'],
        'old_residual':str(short['residual']),'recomputed_terminal_residual':actual_res}}
    assert eq_error<1e-12 and jacerr<1e-8
    (OUT/'solver_checks.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
