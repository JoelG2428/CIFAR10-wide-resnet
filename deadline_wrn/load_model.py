"""Load a trusted local WRN checkpoint; no dataset access or training."""
from pathlib import Path
import torch
from model import WideResNet

def load_model(path,device='cpu'):
    checkpoint=torch.load(Path(path),map_location='cpu',weights_only=False)
    config=checkpoint['configuration']
    if config['architecture']!='WRN-28-10':raise ValueError('Unexpected architecture')
    model=WideResNet(config['depth'],config['widen_factor'],config['dropout'])
    model.load_state_dict(checkpoint['model_state'],strict=True)
    model.to(device).eval()
    return model,checkpoint['normalization'],checkpoint['class_names']

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('checkpoint');args=parser.parse_args()
    torch.set_num_threads(2)
    model,normalization,classes=load_model(args.checkpoint)
    with torch.no_grad():output=model(torch.zeros(2,3,32,32))
    assert output.shape==(2,10) and torch.isfinite(output).all()
    print('Strict checkpoint loading and synthetic inference passed; no dataset accessed.')
