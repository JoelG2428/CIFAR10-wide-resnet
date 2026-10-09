#!/usr/bin/env python3
"""Isolated expanded-rules WRN workflow. No official-test stage exists."""
import argparse
import json
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'outputs/deadline_wrn'
CONFIG=dict(architecture='WRN-28-10',depth=28,widen_factor=10,dropout=0.3,
            batch_size=128,drop_last=False,epochs=200,seed=42,lr=0.1,momentum=0.9,
            nesterov=True,weight_decay=5e-4,cosine_t_max=200,eta_min=0.0,
            augmentation='REFLECT_CROP_FLIP_AUTOAUGMENT_CIFAR10_CUTOUT',cutout_length=16,autoaugment=True,
            scheduler='cosine_epoch',channels_last=True,cudnn_benchmark=True,deterministic=True,
            initialization='Kaiming normal fan_in ReLU; BN ones/zeros; FC uniform +-1/sqrt(fan_in)',
            workers=2,mixed_precision=False,official_test_enabled=False)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['verify','benchmark','train','final'],default='verify')
    parser.add_argument('--name',default='wrn28_10_aa_verify')
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--select-run',help='Completed expanded-rules holdout used to freeze final epoch; no weights transferred')
    args=parser.parse_args()
    if not args.execute:
        print('DISABLED: no ML imports, CUDA access, data access, or training. No official-test stage is supported.')
        return
    if not args.name or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in args.name):parser.error('Use a simple unique output name')
    if not os.environ.get('SLURM_JOB_ID'):parser.error('Execution requires a Slurm GPU allocation')
    if args.resume and args.stage not in ('train','final'):parser.error('Resume only the same holdout/final run')
    if args.select_run and (args.stage!='final' or args.resume):parser.error('Selection applies only to a new final run')
    if args.stage=='final' and not args.resume and not args.select_run:parser.error('Final requires a completed holdout --select-run')
    import fcntl
    OUTPUT.mkdir(parents=True,exist_ok=True)
    lock=(OUTPUT/'.execution.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    run_dir=OUTPUT/args.stage/args.name
    if args.resume:
        if (run_dir/'summary.json').exists() and json.loads((run_dir/'summary.json').read_text()).get('completed'):
            raise RuntimeError('Completed runs cannot be resumed or overwritten')
        if not (run_dir/'latest.pt').exists():raise FileNotFoundError('Same run latest.pt required')
    else:run_dir.mkdir(parents=True,exist_ok=False)
    import workflow
    workflow.execute(args,run_dir)

if __name__=='__main__':main()
