"""GPU-only training/validation implementation; never reads official test data."""
import copy
import csv
import hashlib
import json
import math
import random
import sys
import tempfile
import time
import datetime
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.datasets.utils import check_integrity
from torchvision.utils import save_image
from model import WideResNet, WideBlock, expected_parameters
from run import ROOT, OUTPUT, CONFIG

class TrainingCIFAR10(datasets.CIFAR10):
    """torchvision loader restricted to original training batches.
    Stock torchvision hashes test_batch even with train=True; avoid that access.
    Metadata integrity is still checked by torchvision's _load_meta.
    """
    def __init__(self,root):super().__init__(root=str(root),train=True,download=False)
    def _check_integrity(self):
        return all(check_integrity(str(Path(self.root)/self.base_folder/name),md5) for name,md5 in self.train_list)

class Cutout:
    def __init__(self,length=16):self.length=length
    def apply_at(self,image,y,x):
        result=image.clone();h,w=image.shape[-2:];half=self.length//2
        y1,y2=max(0,y-half),min(h,y+half);x1,x2=max(0,x-half),min(w,x+half)
        result[:,y1:y2,x1:x2]=0 # normalized zero is raw training-channel mean
        return result,(y1,y2,x1,x2)
    def __call__(self,image):
        h,w=image.shape[-2:]
        return self.apply_at(image,int(np.random.randint(h)),int(np.random.randint(w)))[0]

class SubsetImages(Dataset):
    def __init__(self,raw,indices,transform):self.raw,self.indices,self.transform=raw,np.asarray(indices),transform
    def __len__(self):return len(self.indices)
    def __getitem__(self,i):
        original=int(self.indices[i])
        return self.transform(Image.fromarray(self.raw.data[original])),int(self.raw.targets[original])

def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=True;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False

def statistics(images,indices):
    total=np.zeros(3);squared=np.zeros(3);pixels=0
    for start in range(0,len(indices),512):
        x=images[indices[start:start+512]].astype(np.float64)/255
        total+=x.sum((0,1,2));squared+=(x*x).sum((0,1,2));pixels+=x.shape[0]*1024
    mean=total/pixels;std=np.sqrt(squared/pixels-mean**2)
    return dict(mean=mean.tolist(),std=std.tolist(),source_count=len(indices))

def make_transform(normalization,training=False):
    steps=[]
    if training:steps=[transforms.RandomCrop(32,padding=4,padding_mode='reflect'),transforms.RandomHorizontalFlip(0.5),
        transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10,fill=tuple(round(255*c) for c in normalization['mean']))]
    steps += [transforms.ToTensor(),transforms.Normalize(normalization['mean'],normalization['std'])]
    if training:steps.append(Cutout(16))
    return transforms.Compose(steps)

def seed_worker(worker_id):
    seed=torch.initial_seed()%2**32;random.seed(seed);np.random.seed(seed)

def loaders(raw,train_indices,valid_indices,normalization,config):
    generator=torch.Generator().manual_seed(config['seed'])
    train=DataLoader(SubsetImages(raw,train_indices,make_transform(normalization,True)),batch_size=config['batch_size'],
                     shuffle=True,drop_last=False,num_workers=config["workers"],persistent_workers=False,worker_init_fn=seed_worker,pin_memory=True,generator=generator)
    valid=None if valid_indices is None else DataLoader(SubsetImages(raw,valid_indices,make_transform(normalization)),
                     batch_size=config['batch_size'],shuffle=False,drop_last=False,num_workers=config["workers"],persistent_workers=False,worker_init_fn=seed_worker,pin_memory=True)
    return train,valid

def optimizer_scheduler(model,config):
    optimizer=torch.optim.SGD(model.parameters(),lr=config['lr'],momentum=config['momentum'],nesterov=config['nesterov'],weight_decay=config['weight_decay'])
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=config['cosine_t_max'],eta_min=config['eta_min'])
    return optimizer,scheduler

def sync():torch.cuda.synchronize()

def epoch(model,loader,optimizer=None):
    training=optimizer is not None;model.train(training);loss_sum=correct=count=0
    sync();started=time.perf_counter()
    with torch.set_grad_enabled(training):
        for images,labels in loader:
            if datetime.datetime.now(datetime.timezone.utc)>=datetime.datetime(2026,10,2,12,30,tzinfo=datetime.timezone.utc):
                raise RuntimeError('08:30 EDT GPU cutoff reached; preserve latest completed epoch and package backup')
            images=images.cuda(non_blocking=True).contiguous(memory_format=torch.channels_last);labels=labels.cuda(non_blocking=True)
            if training:optimizer.zero_grad(set_to_none=True)
            logits=model(images);loss=nn.functional.cross_entropy(logits,labels)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss; stop without recipe changes')
            if training:loss.backward();optimizer.step()
            loss_sum+=loss.item()*len(labels);correct+=(logits.detach().argmax(1)==labels).sum().item();count+=len(labels)
    sync()
    return dict(loss=loss_sum/count,accuracy=correct/count,count=count,seconds=time.perf_counter()-started)

def rng(loader):
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all(),loader=loader.generator.get_state())

def restore_rng(state,loader):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch'])
    torch.cuda.set_rng_state_all(state['cuda']);loader.generator.set_state(state['loader'])

def save_checkpoint(path,model,optimizer,scheduler,completed,history,config,normalization,train,train_indices,valid_indices,classes,kind):
    state=dict(kind=kind,configuration=config,epoch=completed,history=history,normalization=normalization,class_names=classes,
               model_state=model.state_dict(),optimizer_state=optimizer.state_dict(),scheduler_state=scheduler.state_dict(),
               rng=rng(train),train_indices=train_indices,validation_indices=valid_indices,
               gpu=torch.cuda.get_device_name(0),slurm_job=os.environ['SLURM_JOB_ID'])
    temporary=path.with_suffix('.tmp');torch.save(state,temporary);temporary.replace(path)

# os is only used for the allocated job identity, never configuration overrides.
import os

def write_json(path,value):
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value,indent=2));temporary.replace(path)

def write_history(run_dir,history):
    write_json(run_dir/'history.json',history)
    if history:
        with (run_dir/'history.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=history[0].keys());w.writeheader();w.writerows(history)

def check_scheduler():
    parameter=nn.Parameter(torch.zeros((),device='cuda'))
    optimizer,scheduler=optimizer_scheduler(nn.ParameterList([parameter]),CONFIG)
    observed=[];saved=None
    for completed in range(1,201):
        observed.append(optimizer.param_groups[0]['lr'])
        assert math.isclose(observed[-1],0.05*(1+math.cos(math.pi*(completed-1)/200)),rel_tol=1e-9,abs_tol=1e-12)
        parameter.grad=torch.zeros_like(parameter);optimizer.step();scheduler.step()
        if completed==100:saved=copy.deepcopy((optimizer.state_dict(),scheduler.state_dict()))
    assert optimizer.param_groups[0]['lr']==0.0
    optimizer.load_state_dict(saved[0]);scheduler.load_state_dict(saved[1])
    assert scheduler.last_epoch==100 and math.isclose(optimizer.param_groups[0]['lr'],0.05)
    for completed in range(101,201):
        assert math.isclose(optimizer.param_groups[0]['lr'],observed[completed-1],rel_tol=1e-9,abs_tol=1e-12)
        parameter.grad=torch.zeros_like(parameter);optimizer.step();scheduler.step()
    return {str(e):observed[e-1] for e in (1,50,100,101,150,200)}

def verify(model,optimizer,scheduler,train,valid,raw,config,normalization,train_indices,valid_indices,run_dir):
    count=sum(p.numel() for p in model.parameters())
    assert count==expected_parameters()==36479194
    assert len(model.groups)==3 and all(len(group)==4 for group in model.groups)
    convs=[m for m in model.modules() if isinstance(m,nn.Conv2d)]
    assert len(convs)==28 and all(m.bias is None for m in convs)
    init=[]
    for m in convs:
        expected=math.sqrt(2/(m.in_channels*m.kernel_size[0]*m.kernel_size[1]));actual=m.weight.detach().std().item()
        assert abs(actual/expected-1)<0.13
        init.append(dict(shape=list(m.weight.shape),expected_std=expected,observed_std=actual))
    for m in model.modules():
        if isinstance(m,nn.BatchNorm2d):
            assert torch.all(m.weight==1) and torch.all(m.bias==0) and torch.all(m.running_mean==0) and torch.all(m.running_var==1)
    assert torch.all(model.classifier.bias==0)
    assert model.classifier.weight.abs().max()<=1/math.sqrt(640)
    model.eval();shapes=[]
    with torch.no_grad():
        x=torch.randn(2,3,32,32,device='cuda');x=model.stem(x);assert x.shape==(2,16,32,32)
        for group,channels,side in zip(model.groups,[160,320,640],[32,16,8]):
            first=group[0];activated=first.relu1(first.bn1(x))
            assert first.projected and first.shortcut.stride==first.conv1.stride
            residual=first.conv2(first.dropout(first.relu2(first.bn2(first.conv1(activated)))))
            expected=first.shortcut(activated)+residual
            assert torch.equal(first(x),expected)
            identity=group[1];assert not identity.projected and isinstance(identity.shortcut,nn.Identity)
            x=group(x);assert tuple(x.shape)==(2,channels,side,side);shapes.append(list(x.shape))
        logits=model(torch.randn(2,3,32,32,device='cuda'));assert logits.shape==(2,10) and torch.isfinite(logits).all()
        assert not torch.allclose(logits.sum(1),torch.ones(2,device='cuda'))
        probe=torch.randn(2,3,32,32,device='cuda');before={k:v.clone() for k,v in model.named_buffers()}
        assert torch.equal(model(probe),model(probe))
        assert all(torch.equal(v,before[k]) for k,v in model.named_buffers())
    example=make_transform(normalization)(Image.fromarray(raw.data[int(train_indices[0])]))
    masked,bounds=Cutout().apply_at(example,16,16);assert bounds==(8,24,8,24)
    assert torch.all(masked[:,8:24,8:24]==0)
    mean=torch.tensor(normalization['mean'])[:,None,None];std=torch.tensor(normalization['std'])[:,None,None]
    assert torch.allclose((masked*std+mean)[:,8:24,8:24],mean.expand(3,16,16))
    edge,edge_bounds=Cutout().apply_at(example,0,0);assert edge_bounds==(0,8,0,8)
    transform=make_transform(normalization,True)
    examples=[torch.from_numpy(raw.data[int(i)].copy()).permute(2,0,1).float()/255 for i in train_indices[:8]]
    augmented=[transform(Image.fromarray(raw.data[int(i)]))*std+mean for i in train_indices[:8]]
    save_image(examples+augmented,run_dir/'augmentation_examples.png',nrow=8)
    save_image(torch.stack([example*std+mean,masked*std+mean,edge*std+mean]),run_dir/'cutout_fill_examples.png',nrow=3)
    schedule=check_scheduler()
    images,labels=next(iter(train));assert len(labels)==128
    model.train();optimizer.zero_grad(set_to_none=True)
    logits=model(images.cuda());loss=nn.functional.cross_entropy(logits,labels.cuda());assert torch.isfinite(loss)
    loss.backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    initial=model.stem.weight.detach().clone();optimizer.step();assert not torch.equal(initial,model.stem.weight)
    assert all(m.training for m in model.modules() if isinstance(m,(nn.Dropout,nn.BatchNorm2d)))
    optimizer.zero_grad(set_to_none=True)
    with tempfile.TemporaryDirectory(dir=run_dir) as temporary:
        checkpoint=Path(temporary)/'verification.pt'
        save_checkpoint(checkpoint,model,optimizer,scheduler,0,[],config,normalization,train,train_indices,valid_indices,raw.classes,'verification')
        state=torch.load(checkpoint,map_location='cpu',weights_only=False)
        expected_rng=(random.random(),float(np.random.random()),torch.rand(1),torch.rand(1,device='cuda').cpu(),torch.rand(1,generator=train.generator))
        model.load_state_dict(state['model_state']);optimizer.load_state_dict(state['optimizer_state']);scheduler.load_state_dict(state['scheduler_state']);restore_rng(state['rng'],train)
        actual_rng=(random.random(),float(np.random.random()),torch.rand(1),torch.rand(1,device='cuda').cpu(),torch.rand(1,generator=train.generator))
        assert expected_rng[:2]==actual_rng[:2] and all(torch.equal(a,b) for a,b in zip(expected_rng[2:],actual_rng[2:]))
        assert all(torch.equal(v.cpu(),state['model_state'][k]) for k,v in model.state_dict().items())
        assert scheduler.state_dict()==state['scheduler_state']
        for group in optimizer.param_groups:
            for p in group['params']:assert 'momentum_buffer' in optimizer.state[p]
        # Verify exact continuation from the same checkpoint on a small synthetic batch.
        probe=torch.randn(2,3,32,32,device='cuda');targets=torch.tensor([0,1],device='cuda')
        def update():
            model.train();optimizer.zero_grad(set_to_none=True);loss=nn.functional.cross_entropy(model(probe),targets);loss.backward();optimizer.step();scheduler.step()
            return loss.detach().cpu(),{k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        restore_rng(state['rng'],train);loss_a,continuation=update()
        model.load_state_dict(state['model_state']);optimizer.load_state_dict(state['optimizer_state']);scheduler.load_state_dict(state['scheduler_state']);restore_rng(state['rng'],train)
        loss_b,continued=update();assert torch.equal(loss_a,loss_b)
        assert all(torch.equal(v,continued[k]) for k,v in continuation.items())
    validation=epoch(model,valid);assert validation['count']==5000 and not model.training
    assert all(not m.training for m in model.modules() if isinstance(m,(nn.Dropout,nn.BatchNorm2d)))
    return dict(passed=True,parameters=count,group_shapes=shapes,initialization=init,scheduler_learning_rates=schedule,
                validation_sample_count=5000,finite_real_batch_loss=loss.item(),cutout_center_bounds=bounds,cutout_edge_bounds=edge_bounds,
                checkpoint_resume_exact=True,benchmark_weights_retained=False)

def execute(args,run_dir):
    started=time.perf_counter()
    if not torch.cuda.is_available():raise RuntimeError('Allocated CUDA GPU required; no CPU fallback')
    torch.set_num_threads(min(4,os.cpu_count() or 1));seed_all(42)
    config=copy.deepcopy(CONFIG);saved=None
    if args.resume:
        saved=torch.load(run_dir/'latest.pt',map_location='cpu',weights_only=False)
        if saved['kind']!=args.stage:raise ValueError('Checkpoint stage mismatch')
        config=saved['configuration']
    if args.stage=='final' and not args.resume:
        selected_dir=OUTPUT/'train'/args.select_run
        selected=json.loads((selected_dir/'summary.json').read_text())
        if not selected['completed'] or selected['epochs']!=200:raise ValueError('Require completed 200-epoch holdout selection')
        config=copy.deepcopy(selected['configuration']);config['epochs']=selected['best_epoch']
        write_json(run_dir/'selection_plan.json',dict(configuration=config,source=args.select_run,best_validation_accuracy=selected['best_validation_accuracy'],fixed_final_epochs=config['epochs'],weights_transferred=False))
    split_path=ROOT/'outputs/split_indices.npz'
    with np.load(split_path) as split:train_indices,valid_indices=split['train'].copy(),split['validation'].copy()
    expected=np.random.default_rng(42).permutation(50000)
    assert np.array_equal(train_indices,expected[:45000]) and np.array_equal(valid_indices,expected[45000:])
    raw=TrainingCIFAR10(ROOT/'data');assert raw.data.shape==(50000,32,32,3)
    if args.stage=='final':train_indices=np.arange(50000);valid_indices=None
    normalization=statistics(raw.data,train_indices)
    if saved is not None:
        assert normalization==saved['normalization'] and np.array_equal(train_indices,saved['train_indices'])
        assert (valid_indices is None and saved['validation_indices'] is None) or np.array_equal(valid_indices,saved['validation_indices'])
    train,valid=loaders(raw,train_indices,valid_indices,normalization,config)
    model=WideResNet(config['depth'],config['widen_factor'],config['dropout']).cuda().to(memory_format=torch.channels_last)
    optimizer,scheduler=optimizer_scheduler(model,config)
    if saved is not None:
        model.load_state_dict(saved['model_state']);optimizer.load_state_dict(saved['optimizer_state']);scheduler.load_state_dict(saved['scheduler_state']);restore_rng(saved['rng'],train)
    metadata=dict(configuration=config,normalization=normalization,class_names=raw.classes,split_sha256=hashlib.sha256(split_path.read_bytes()).hexdigest(),
                  gpu=torch.cuda.get_device_name(0),gpu_total_mib=torch.cuda.get_device_properties(0).total_memory/2**20,
                  torch_version=str(torch.__version__),slurm_job=os.environ['SLURM_JOB_ID'],official_test_access=False,parameters=sum(p.numel() for p in model.parameters()))
    if not args.resume:write_json(run_dir/'configuration.json',metadata)
    print(json.dumps(metadata,indent=2),flush=True)
    sync();setup_seconds=time.perf_counter()-started
    if args.stage=='verify':
        torch.cuda.reset_peak_memory_stats();verification=verify(model,optimizer,scheduler,train,valid,raw,config,normalization,train_indices,valid_indices,run_dir)
        verification.update(metadata,setup_seconds=setup_seconds,total_seconds=time.perf_counter()-started,peak_gpu_mib=torch.cuda.max_memory_allocated()/2**20,weights_discarded=True)
        write_json(run_dir/'verification.json',verification);print('PASS: complete WRN GPU verification; disposable weights discarded',flush=True)
        return
    history=[] if saved is None else saved['history']
    if saved is not None and (run_dir/'history.json').exists():
        numerical=json.loads((run_dir/'history.json').read_text())
        if len(numerical)==saved['epoch']:history=numerical
    start_epoch=1 if saved is None else saved['epoch']+1
    best_accuracy=None if args.stage=='final' else max((row['validation_accuracy'] for row in history),default=None)
    best_epoch=None if best_accuracy is None else max(history,key=lambda row:row['validation_accuracy'])['epoch']
    if saved is not None and args.stage=='train' and best_epoch is not None:
        best_path=run_dir/'best_validation.pt'
        best_saved=torch.load(best_path,map_location='cpu',weights_only=False) if best_path.exists() else None
        if best_saved is None or best_saved['epoch']!=best_epoch:
            if best_epoch!=saved['epoch']:raise RuntimeError('Best checkpoint/history mismatch; preserve run and inspect before resume')
            temporary=best_path.with_suffix('.tmp');torch.save(saved,temporary);temporary.replace(best_path)
        del best_saved
    limit=3 if args.stage=='benchmark' else config['epochs']
    with tempfile.TemporaryDirectory(dir=run_dir) as transient:
        checkpoint=Path(transient)/'benchmark_latest.pt' if args.stage=='benchmark' else run_dir/'latest.pt'
        if not args.resume and args.stage!='benchmark':
            save_checkpoint(checkpoint,model,optimizer,scheduler,0,[],config,normalization,train,train_indices,valid_indices,raw.classes,args.stage)
        for completed in range(start_epoch,limit+1):
            torch.cuda.reset_peak_memory_stats();lr=optimizer.param_groups[0]['lr']
            trained=epoch(model,train,optimizer);assert trained['count']==len(train_indices)
            validated=epoch(model,valid) if valid is not None else None
            if validated is not None:assert validated['count']==5000
            row=dict(epoch=completed,learning_rate=lr,train_loss=trained['loss'],train_accuracy=trained['accuracy'],train_count=trained['count'],train_seconds=trained['seconds'],
                     validation_loss=None if validated is None else validated['loss'],validation_accuracy=None if validated is None else validated['accuracy'],
                     validation_count=0 if validated is None else validated['count'],validation_seconds=0 if validated is None else validated['seconds'],peak_gpu_mib=torch.cuda.max_memory_allocated()/2**20)
            scheduler.step();row['next_learning_rate']=optimizer.param_groups[0]['lr'];history.append(row)
            row['checkpoint_seconds']=0.0
            io_started=time.perf_counter()
            save_checkpoint(checkpoint,model,optimizer,scheduler,completed,history,config,normalization,train,train_indices,valid_indices,raw.classes,args.stage)
            if validated is not None and (best_accuracy is None or validated['accuracy']>best_accuracy):
                best_accuracy,best_epoch=validated['accuracy'],completed
                if args.stage=='train':save_checkpoint(run_dir/'best_validation.pt',model,optimizer,scheduler,completed,history,config,normalization,train,train_indices,valid_indices,raw.classes,args.stage)
            row['checkpoint_seconds']=time.perf_counter()-io_started
            # Numerical history records measured I/O; checkpoint has a zero placeholder for its own write.
            write_history(run_dir,history)
            print(json.dumps(row),flush=True)
            if completed>=5 and max(h['train_accuracy'] for h in history)<0.15:raise RuntimeError('Clearly broken training; stop')
        final_checkpoint_seconds=0
        if args.stage=='final':
            io_started=time.perf_counter();save_checkpoint(run_dir/'final_model.pt',model,optimizer,scheduler,limit,history,config,normalization,train,train_indices,None,raw.classes,args.stage);final_checkpoint_seconds=time.perf_counter()-io_started
    summary=dict(metadata,completed=True,epochs=limit,setup_seconds=setup_seconds,total_seconds=time.perf_counter()-started,history=history,
                 best_epoch=best_epoch,best_validation_accuracy=best_accuracy,final_checkpoint_seconds=final_checkpoint_seconds,weights_discarded=args.stage=='benchmark')
    if args.stage=='benchmark':
        train_worst=max(h['train_seconds'] for h in history);valid_worst=max(h['validation_seconds'] for h in history);io_worst=max(h['checkpoint_seconds'] for h in history)
        summary['estimate']=dict(holdout_200_training_seconds=200*train_worst,holdout_200_validation_seconds=200*valid_worst,
             holdout_200_checkpoint_seconds=400*io_worst,holdout_200_total_seconds=setup_seconds+200*(train_worst+valid_worst+2*io_worst),
             full_data_200_training_seconds=200*train_worst*50000/45000,full_data_200_checkpoint_seconds=201*io_worst,
             full_data_200_total_seconds=2*setup_seconds+200*train_worst*50000/45000+201*io_worst,
             assumptions='Slowest of three measured epochs; full-data training scales 50000/45000; holdout budgets latest+best every epoch; runtime uncertainty and queue buffers additional')
    write_json(run_dir/'summary.json',summary);print('COMPLETE',json.dumps({k:v for k,v in summary.items() if k not in ('history','configuration','normalization','class_names')}),flush=True)
