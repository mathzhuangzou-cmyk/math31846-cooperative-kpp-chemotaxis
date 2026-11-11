"""Additional nonlinear terminal-time checks for the manuscript benchmark.
Run each job 0..3 with --job N; outputs are independent of one another.
All values are computed, without prescribed final amplitudes.
"""
from pathlib import Path
import argparse,json,time
import numpy as np
from scipy.fft import irfft
from independent_solver import P,coefficients,operators,explicit,modal_matrix

JOBS=[(1024,.01,8000.,None),(1024,.005,8000.,None),
      (128,.01,8000.,None),(128,.01,16000.,.01)]

def run(job,out):
    M,dt,T,initial_amplitude=JOBS[job];p=P;c0=coefficients();delta=.00125
    chi=c0['chi_T']+delta;n,k,mask,a,b,det,c=operators(M,dt)
    amp=np.sqrt(delta/c0['chi2']) if initial_amplitude is None else initial_amplitude
    val,vec=np.linalg.eig(modal_matrix(c0['mu'],chi));v=vec[:,np.argmax(val.real)].real;v/=v[0]
    H=np.zeros((3,M//2+1),dtype=complex)
    H[:,0]=M*(np.array([1.,1.,p['beta']/p['alpha']])+amp**2*np.array(c0['zero']))
    H[:,5]=M*amp*v/2;H[:,10]=M*amp**2*np.array(c0['second'])/2
    targets={round(t/dt):t for t in [500,1000,2000,4000,8000,16000] if t<=T}
    rows=[];start=time.perf_counter()
    for j in range(1,round(T/dt)+1):
        N=explicit(H,M,k,mask,chi);R=H[:2]+dt*N
        h0=(b*R[0]+dt*p['k12']*R[1])/det
        h1=(dt*p['k21']*R[0]+a*R[1])/det
        h2=(H[2]+dt*p['beta']/p['tau']*h0)/c
        H=np.array([h0.real,h1.real,h2.real],dtype=complex)*mask
        if j in targets:
            N=explicit(H,M,k,mask,chi);F=np.zeros_like(H)
            F[0]=(-p['d1']*k*k+p['r1']-p['k12'])*H[0]+p['k12']*H[1]+N[0]
            F[1]=p['k21']*H[0]+(-p['d2']*k*k+p['r2']-p['k21'])*H[1]+N[1]
            F[2]=(p['beta']*H[0]-(p['Dw']*k*k+p['alpha'])*H[2])/p['tau']
            A=float(2*H[0,5].real/M)
            row={'M':M,'dt':dt,'T':targets[j],'delta_chi':delta,'initial_amplitude':float(amp),
              'A':A,'ratio':delta/A**2,'terminal_residual':float(abs(irfft(F*mask,n=M,axis=1)).max()),
              'amplitude_derivative':float(2*F[0,5].real/M)}
            rows.append(row);print(json.dumps(row),flush=True)
    dest=Path(out);dest.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(dest/f'extended_job{job}.npz',normalized_rfft=H/M)
    (dest/f'extended_job{job}.json').write_text(json.dumps({'job':job,'parameters':P,'runtime_seconds':time.perf_counter()-start,'rows':rows},indent=2)+'\n')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--job',type=int,choices=range(4),required=True);ap.add_argument('--output',default='extended_results');a=ap.parse_args();run(a.job,a.output)
