# Copyright (c) 2024-2026 Kevin T. Procopio and contributors.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# AI-assisted experimental code; full human and mathematical review is not established.

"""Bounded pendulum and free-rotor demonstration with saved plots."""
import argparse
from pathlib import Path
import torch
import matplotlib.pyplot as plt
from .discrete_control_lagrange import (
    Pendulum, FreeRotor, VariationalIntegrator, Symmetry, StepRecorder,
    pendulum_observables_from_records,
)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps',type=int,default=500,help='number of stored positions (at least 3)')
    parser.add_argument('--step-size',type=float,default=.01)
    parser.add_argument('--output-dir',type=Path,default=Path('outputs/variational'))
    args = parser.parse_args()
    if args.steps < 3:
        parser.error('--steps must be at least 3')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    for name,system,velocity in [
        ('pendulum',Pendulum({'mass':1.,'length':1.,'gravity':9.81}),0.),
        ('rotor',FreeRotor({'mass':1.,'length':1.}),1.),
    ]:
        recorder = StepRecorder()
        vi = VariationalIntegrator(system,args.step_size,on_step=recorder.on_step)
        if name == 'rotor':
            vi.register_noether_charge('angular_momentum',Symmetry(system.model.layout['theta'][0],torch.tensor([1.],dtype=torch.float64)))
        # Initialize the second position by q1=q0+h*v0 (first-order accurate).
        a = torch.tensor([.8],dtype=torch.float64)
        b = a + args.step_size*velocity
        positions = [a.item(),b.item()]
        for _ in range(args.steps-2):
            c,ok = vi.step(a,b)
            if not ok:
                raise RuntimeError(f'{name}: Newton equation did not converge')
            positions.append(c.item())
            a,b = b,c
        times = torch.arange(args.steps,dtype=torch.float64).numpy()*args.step_size
        fig,ax = plt.subplots(figsize=(7,4))
        ax.plot(times,positions)
        ax.set(xlabel='Time',ylabel='Angle (rad)',title=name.capitalize())
        ax.grid(alpha=.3)
        fig.tight_layout()
        fig.savefig(args.output_dir/f'{name}.png',dpi=150)
        plt.close(fig)
        print(f'{name}: {args.steps} positions, final angle {b.item():.10f}')
        if name == 'rotor':
            charges = [float(r['noether_charges']['angular_momentum']) for r in recorder.records]
            print(f'angular momentum range: {min(charges):.8f} to {max(charges):.8f}')
        else:
            _,_,_,energies = pendulum_observables_from_records(system,recorder,args.step_size)
            print(f'backward-difference energy range: {min(energies):.8f} to {max(energies):.8f}')

if __name__ == '__main__':
    main()
