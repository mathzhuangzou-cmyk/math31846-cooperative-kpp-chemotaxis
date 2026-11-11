#!/usr/bin/env python3
"""Generate tables from model calculations and the actual branch CSV."""
from pathlib import Path
import csv, importlib.util, sys

HERE=Path(__file__).resolve().parent
def main():
    spec=importlib.util.spec_from_file_location('table_model',HERE/'1.py')
    m=importlib.util.module_from_spec(spec);sys.modules['table_model']=m;spec.loader.exec_module(m)
    p=m.ModelParams();c=m.continuous_critical_data(p);s=m.select_mode(10.,p,c['mu_c'])
    branch=Path('results_task17B_advanced/branch_continuation_advanced.csv')
    if not branch.exists():raise FileNotFoundError('Run 3.1.py first: '+str(branch))
    rows=list(csv.DictReader(branch.open(encoding='utf-8')))
    out=Path('tables_task18');out.mkdir(exist_ok=True)
    with (out/'critical_parameters.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(['parameter','value'])
        w.writerows([(k,c[k]) for k in ['mu_c','k_c','chi_c','Lambda_c']]+[('n_star',s['n_star']),('chi_T(L)',s['chi_T'])])
    with (out/'branch_validation.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(['delta_chi','A_numeric','delta_over_A2','chi2_theory'])
        for r in rows:
            delta=float(r['delta_chi']);a=float(r['A_numeric'])
            w.writerow([delta,a,delta/(a*a),float(r['chi2_theory'])])
    print('Generated tables from model and',branch)

if __name__=='__main__':main()
