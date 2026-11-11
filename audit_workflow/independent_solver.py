#!/usr/bin/env python3
"""Independent reconstruction of Math-31846 Section 5; not the original code.

Run: python reproduce_concern1.py --output results
M is the number of points on [0,2L), with no duplicated endpoint.
Both critical-point and current-parameter eigenvector initializations are run.
The implicit operator contains diffusion, linear growth/exchange, and the
signal equation. The explicit terms are -r_i*u_i**2 and -chi*(u*w_x)_x.
All quadratic products are projected to Fourier indices n < M/3.
No fitted amplitudes, inserted errors, or reviewer values enter the solver.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, platform, time
from pathlib import Path
import numpy as np
import scipy
from scipy.fft import rfft, irfft
from scipy.optimize import brentq

P = dict(d1=.2, d2=.1, Dw=1., r1=1., r2=.8, k12=.3, k21=.2,
         alpha=1., beta=1., tau=1., L=10.)

def threshold(mu):
    p=P; A=p['d1']*mu+p['r1']+p['k12']; B=p['d2']*mu+p['r2']+p['k21']
    C=p['Dw']*mu+p['alpha']
    return C*(A*B-p['k12']*p['k21'])/(p['beta']*mu*B)

def coefficients():
    p=P; mu=(5*np.pi/p['L'])**2; chi=threshold(mu)
    A=p['d1']*mu+p['r1']+p['k12']; B=p['d2']*mu+p['r2']+p['k21']
    C=p['Dw']*mu+p['alpha']; eta=p['k21']/B; zeta=p['beta']/C
    a=p['r1']+p['k12']; b=p['r2']+p['k21']; K=p['k12']*p['k21']
    U0=-(b*p['r1']+p['k12']*p['r2']*eta**2)/(2*(a*b-K))
    V0=(p['k21']*U0-p['r2']*eta**2/2)/b; W0=p['beta']/p['alpha']*U0
    A2=4*p['d1']*mu+a; B2=4*p['d2']*mu+b; C2=4*p['Dw']*mu+p['alpha']
    D2=A2-K/B2-4*chi*p['beta']*mu/C2
    fu=p['r1']/2-chi*zeta*mu; fv=p['r2']*eta**2/2
    U2=-(fu+p['k12']/B2*fv)/D2
    V2=(p['k21']*U2-fv)/B2; W2=p['beta']/C2*U2
    N1=chi*mu*(W2+zeta*U0-zeta*U2/2)-2*p['r1']*U0-p['r1']*U2
    N2=-2*p['r2']*eta*V0-p['r2']*eta*V2
    chi2=-C/(p['beta']*mu)*(N1+p['k12']/B*N2)
    sigma=(mu*p['beta']/C)/(1+K/B**2+p['tau']*chi*mu*p['beta']/C**2)
    def quartic(m):
        return (p['Dw']*p['d1']*p['d2']**2*m**4
          +2*p['Dw']*b*p['d1']*p['d2']*m**3
          +(p['Dw']*K*p['d2']+p['Dw']*b*b*p['d1']-a*p['alpha']*p['d2']**2)*m*m
          -2*p['alpha']*p['d2']*(a*b-K)*m-p['alpha']*b*(a*b-K))
    muc=brentq(quartic, .001, 100., xtol=1e-14)
    return dict(mu=mu, chi_T=chi, mu_c=muc, chi_c=threshold(muc), chi2=chi2,
                sigma=sigma, eta=eta, zeta=zeta, zero=[U0,V0,W0], second=[U2,V2,W2])

def operators(M, dt):
    p=P; n=np.arange(M//2+1); k=np.pi*n/p['L']; mask=n<M/3
    a=1+dt*(p['d1']*k*k-p['r1']+p['k12'])
    b=1+dt*(p['d2']*k*k-p['r2']+p['k21'])
    det=a*b-dt*dt*p['k12']*p['k21']
    c=1+dt*(p['Dw']*k*k+p['alpha'])/p['tau']
    return n,k,mask,a,b,det,c

def explicit(H, M, k, mask, chi):
    U=irfft(H,n=M,axis=1)
    wx=irfft(1j*k*H[2],n=M)
    # Project logistic products and the flux before the outer derivative.
    squares=rfft(U[:2]**2,axis=1)*mask
    flux=rfft(U[0]*wx)*mask
    N=-np.array([P['r1'],P['r2']])[:,None]*squares
    N[0]-=chi*1j*k*flux
    return N

def modal_matrix(mu,chi,tau=1.):
    p=P
    return np.array([[-p['d1']*mu-p['r1']-p['k12'],p['k12'],chi*mu],
      [p['k21'],-p['d2']*mu-p['r2']-p['k21'],0.],
      [p['beta']/tau,0.,-(p['Dw']*mu+p['alpha'])/tau]])

def run(M, delta, dt=.01, T=500., output=None, initialization='critical'):
    c0=coefficients(); p=P; chi=c0['chi_T']+delta
    n,k,mask,a,b,det,c=operators(M,dt)
    amp0=np.sqrt(delta/c0['chi2'])
    H=np.zeros((3,M//2+1),dtype=np.complex128)
    H[:,0]=M*(np.array([1.,1.,p['beta']/p['alpha']])+amp0**2*np.array(c0['zero']))
    vec=np.array([1.,c0['eta'],c0['zeta']])
    if initialization=='offset':
        values,vectors=np.linalg.eig(modal_matrix(c0['mu'],chi))
        vec=vectors[:,np.argmax(values.real)].real
        vec/=vec[0]
    elif initialization!='critical': raise ValueError('unknown initialization')
    H[:,5]=M*amp0*vec/2
    H[:,10]=M*amp0**2*np.array(c0['second'])/2
    initial=H.copy(); steps=round(T/dt)
    if abs(steps*dt-T)>1e-10: raise ValueError('T must be a multiple of dt')
    trace=[]; start=time.perf_counter()
    for j in range(steps):
        N=explicit(H,M,k,mask,chi)
        R=H[:2]+dt*N
        h0=(b*R[0]+dt*p['k12']*R[1])/det
        h1=(dt*p['k21']*R[0]+a*R[1])/det
        h2=(H[2]+dt*p['beta']/p['tau']*h0)/c
        # Real cosine coefficients enforce the invariant even subspace.
        H=np.array([h0.real,h1.real,h2.real],dtype=np.complex128)*mask
        if (j+1)%max(1,steps//10)==0:
            trace.append(dict(t=(j+1)*dt,amplitude=float(2*H[0,5].real/M)))
        if (j+1)%5000==0 and not np.isfinite(H).all(): raise FloatingPointError('nonfinite solution')
    U=irfft(H,n=M,axis=1); x=np.arange(M)*2*p['L']/M
    N=explicit(H,M,k,mask,chi); F=np.zeros_like(H)
    F[0]=(-p['d1']*k*k+p['r1']-p['k12'])*H[0]+p['k12']*H[1]+N[0]
    F[1]=p['k21']*H[0]+(-p['d2']*k*k+p['r2']-p['k21'])*H[1]+N[1]
    F[2]=(p['beta']*H[0]-(p['Dw']*k*k+p['alpha'])*H[2])/p['tau']
    F*=mask; res=irfft(F,n=M,axis=1)
    # Independent endpoint-weighted quadrature on [0,L], including L once.
    half=M//2+1
    amplitude_fft=abs(2*H[0,5].real/M)
    amplitude_quad=abs(2/p['L']*np.trapezoid((U[0,:half]-1)*np.cos(5*np.pi*x[:half]/p['L']),x[:half]))
    row=dict(M=M,delta_chi=delta,dt=dt,T=T,initialization=initialization,A_numeric=float(amplitude_fft),
       coefficient_ratio=float(delta/amplitude_fft**2),amplitude_trapezoid=float(amplitude_quad),
       amplitude_measurement_difference=float(abs(amplitude_fft-amplitude_quad)),
       terminal_residual_max=float(np.max(np.abs(res))),
       terminal_amplitude_derivative=float(2*F[0,5].real/M),
       min_u=float(U[0].min()),min_v=float(U[1].min()),
       runtime_seconds=time.perf_counter()-start,trace=trace)
    if output:
        path=Path(output)/f'state_{initialization}_M{M}_delta{delta:g}_dt{dt:g}_T{T:g}.npz'
        np.savez_compressed(path,normalized_rfft=H/M,initial_normalized_rfft=initial/M,x=x,state=U)
        row['state_file']=path.name
    return row,H/M

def profile_errors(h,reference):
    diff=-reference.copy(); diff[:,:h.shape[1]]+=h
    weights=np.full(reference.shape[1],2.); weights[0]=1.; weights[-1]=1.
    def norm(z): return np.sqrt(np.sum(weights*np.abs(z)**2))
    pert=reference[0].copy(); pert[0]-=1
    return dict(relative_u_L2=float(norm(diff[0])/norm(reference[0])),
       relative_u_perturbation_L2=float(norm(diff[0])/norm(pert)),
       relative_joint_L2=float(norm(diff)/norm(reference)))

def linear_checks():
    coeff=coefficients(); mu=coeff['mu']; chi=coeff['chi_T']
    taurows=[]
    for tau in [.25,.5,1.,2.,4.]:
        row={'tau':tau}
        for key,mult in [('below',.98),('at',1.),('above',1.02)]:
            row[key]=float(np.linalg.eigvals(modal_matrix(mu,chi*mult,tau)).real.max())
        taurows.append(row)
    J=modal_matrix(mu,1.02*chi)
    E=np.array([[-2*P['r1'],0,1.02*chi*mu],[0,-2*P['r2'],0],[0,0,0]])
    I=J-E; lam=np.linalg.eigvals(J).real.max(); rows=[]; prev=None
    for dt in [.04,.02,.01,.005]:
        G=np.linalg.solve(np.eye(3)-dt*I,np.eye(3)+dt*E)
        value=np.log(np.abs(np.linalg.eigvals(G)).max())/dt
        err=abs(value-lam)
        rows.append(dict(dt=dt,lambda_exact=float(lam),lambda_imex=float(value),
            absolute_error=float(err),observed_rate=None if prev is None else float(np.log2(prev/err))))
        prev=err
    return taurows,rows

def write_csv(path,rows,exclude=()):
    keys=[k for k in rows[0] if k not in exclude]
    with Path(path).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');w.writeheader();w.writerows(rows)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='results'); args=parser.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    result={'provenance':'New independent reconstruction. Archived task16 and task17B sources examined; exact origin of the old Table 3 remains unresolved.',
       'parameters':P,'coefficients':coefficients(),
       'environment':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,
                      'platform':platform.platform()},
       'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    result['table1'],result['table4']=linear_checks()
    jobs=[(M,.01) for M in [128,512,1024,2048]]+[(1024,d) for d in [.00125,.0025,.005]]
    allrows=[]
    for init in ['critical','offset']:
        states={}; spatial=[]; branch=[]
        for M,d in jobs:
            row,h=run(M,d,output=out,initialization=init)
            if d==.01: spatial.append(row);states[M]=h
            if M==1024:branch.append(row)
            print(json.dumps(row,ensure_ascii=False),flush=True)
            allrows.append(row)
        for row in spatial: row.update(profile_errors(states[row['M']],states[2048]))
        result['table2_'+init]=sorted(branch,key=lambda r:r['delta_chi'])
        result['table3_'+init]=spatial
    result['verification']={
        'positive_u_v':all(r['min_u']>0 and r['min_v']>0 for r in allrows),
        'amplitude_quadrature_agreement':max(r['amplitude_measurement_difference'] for r in allrows),
        'spatial_amplitude_range_critical':float(np.ptp([r['A_numeric'] for r in result['table3_critical']])),
        'spatial_amplitude_range_offset':float(np.ptp([r['A_numeric'] for r in result['table3_offset']])),
        'rounding_match_table2_amplitudes':all(round(r['A_numeric'],6)==v for r,v in zip(result['table2_offset'],[.024304,.034345,.048481,.068317])),
        'rounding_match_table2_ratios':all(round(r['coefficient_ratio'],6)==v for r,v in zip(result['table2_offset'],[2.116112,2.119441,2.127320,2.142633])),
        'rounding_match_table4':all(abs(r['absolute_error']/v-1)<.005 for r,v in zip(result['table4'],[1.82e-4,9.04e-5,4.51e-5,2.25e-5]))}
    for name in ['table1','table2_critical','table2_offset','table3_critical','table3_offset','table4']:
        write_csv(out/f'{name}_recomputed.csv',result[name],exclude=['trace'])
    (out/'audit_results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result['verification'],indent=2),flush=True)

if __name__=='__main__':main()
