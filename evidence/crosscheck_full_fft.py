"""Cross-check the real-FFT solver with full complex FFT and matrix inversion.

This follows the archived Task 1.6B update algebra in NumPy. It does not run
the archived PyTorch program and does not force evenness after each step.
"""
import json
from pathlib import Path
import numpy as np
from reproduce_concern1 import P, coefficients, modal_matrix

def full_fft_run(initialization, M=128, dt=.01, T=500.):
    c=coefficients(); p=P; delta=.01;chi=c['chi_T']+delta
    x=np.arange(M)*2*p['L']/M;k=2*np.pi*np.fft.fftfreq(M,d=2*p['L']/M)
    mask=np.abs(np.fft.fftfreq(M,d=1/M))<M/3
    a=np.sqrt(delta/c['chi2']);v=np.array([1.,c['eta'],c['zeta']])
    if initialization=='offset':
        val,vec=np.linalg.eig(modal_matrix(c['mu'],chi));v=vec[:,np.argmax(val.real)].real;v/=v[0]
    U=(np.array([1.,1.,p['beta']/p['alpha']])[:,None]
       +a*v[:,None]*np.cos(5*np.pi*x/p['L'])
       +a*a*(np.array(c['zero'])[:,None]+np.array(c['second'])[:,None]*np.cos(10*np.pi*x/p['L'])))
    L=np.zeros((M,3,3))
    L[:,0,0]=-p['d1']*k*k+p['r1']-p['k12'];L[:,0,1]=p['k12']
    L[:,1,1]=-p['d2']*k*k+p['r2']-p['k21'];L[:,1,0]=p['k21']
    L[:,2,2]=-(p['Dw']*k*k+p['alpha'])/p['tau'];L[:,2,0]=p['beta']/p['tau']
    inverse=np.linalg.inv(np.eye(3)-dt*L)
    for _ in range(round(T/dt)):
        H=np.fft.fft(U,axis=1);wx=np.fft.ifft(1j*k*H[2]).real
        R=H.copy()
        R[0]+=dt*(-p['r1']*np.fft.fft(U[0]**2)*mask-chi*1j*k*np.fft.fft(U[0]*wx)*mask)
        R[1]+=dt*(-p['r2']*np.fft.fft(U[1]**2)*mask)
        H=np.einsum('kij,jk->ik',inverse,R)
        U=np.fft.ifft(H,axis=1).real
    return 2*np.fft.fft(U[0]-1)[5].real/M

if __name__=='__main__':
    result=json.loads(Path('final_results/audit_results.json').read_text())
    rows=[]
    for init in ['critical','offset']:
        value=full_fft_run(init)
        expected=result['table3_'+init][0]['A_numeric']
        rows.append(dict(initialization=init,full_fft_amplitude=float(value),
                         real_fft_amplitude=expected,absolute_difference=float(abs(value-expected))))
    assert max(r['absolute_difference'] for r in rows)<1e-10
    Path('final_results/full_fft_crosscheck.json').write_text(json.dumps(rows,indent=2)+'\n')
    print(json.dumps(rows,indent=2))
