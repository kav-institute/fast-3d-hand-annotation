#!/usr/bin/env python3
"""Train the HRNet-W48/SimCC hand model on THAB and optionally Re:InterHand."""
import argparse
from pathlib import Path


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-root', type=Path, required=True, help='THAB training root with annotations__train.json, annotations__val.json and skeleton_thab.txt.')
    p.add_argument('--reinterhand-root', type=Path, help='Optional Re:InterHand download root.')
    p.add_argument('--capture-ids', nargs='+', help='Re:InterHand capture directory names; required with --reinterhand-root.')
    p.add_argument('--thab-unit-scale', type=float, default=10., help='THAB world units to millimeters: 10 for legacy centimeters, 1 for millimeters.')
    p.add_argument('--output-dir', type=Path, default=Path('outputs/training'))
    p.add_argument('--resume', type=Path)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--epochs', type=int, default=1000)
    p.add_argument('--steps-per-epoch', type=int, default=1000)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--max-val-batches', type=int, help='Limit validation only for smoke tests.')
    p.add_argument('--pretrained-backbone', action='store_true', help='Allow timm to download ImageNet backbone initialization for fresh training.')
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args(argv)
    if bool(args.reinterhand_root) != bool(args.capture_ids):
        p.error('--reinterhand-root and --capture-ids must be supplied together')
    if min(args.epochs, args.steps_per_epoch, args.batch_size) < 1 or args.workers < 0:
        p.error('epochs, steps and batch size must be positive; workers must be nonnegative')
    if args.thab_unit_scale <= 0:
        p.error('--thab-unit-scale must be positive')
    if args.max_val_batches is not None and args.max_val_batches < 1:
        p.error('--max-val-batches must be positive')
    return args


def train(args):
    import random
    import numpy as np
    import torch
    from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler
    from torchvision.transforms import ToTensor
    from dataset import THABDataset, ReInterHandDataset
    from model import MultiHandKeypoint
    from training_utils import mixed_triplet_collate

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    transform = ToTensor()
    thab = THABDataset(transform, 'train', 'annotations__', str(args.dataset_root), world_scale=args.thab_unit_scale)
    datasets = [thab]
    if args.reinterhand_root:
        for rig in ['Ego', 'Mugsy']:
            for capture in args.capture_ids:
                datasets.append(ReInterHandDataset(str(args.reinterhand_root), capture,
                    rig=rig, transform=transform,
                    skeleton_path=str(args.dataset_root/'skeleton_thab.txt')))
    if any(len(ds) == 0 for ds in datasets):
        raise ValueError('Training datasets must not be empty')
    weights = torch.cat([torch.full((len(ds),), 1 / (len(datasets)*len(ds)), dtype=torch.double) for ds in datasets])
    sampler = WeightedRandomSampler(weights, args.steps_per_epoch*args.batch_size, replacement=True)
    kwargs = dict(batch_size=args.batch_size, num_workers=args.workers,
                  pin_memory=device.type == 'cuda', collate_fn=mixed_triplet_collate)
    loader = DataLoader(ConcatDataset(datasets), sampler=sampler, **kwargs)
    val_loader = DataLoader(THABDataset(transform, 'val', 'annotations__', str(args.dataset_root), world_scale=args.thab_unit_scale), shuffle=False, **kwargs)
    if len(val_loader) == 0:
        raise ValueError('Validation dataset must not be empty')
    model = MultiHandKeypoint(21, pretrained_backbone=args.pretrained_backbone and args.resume is None).to(device)
    # Preserve the source optimizer's backbone LR multiplier and norm/bias exclusions.
    groups = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        lr = 2e-4 if name.startswith('backbone_net.') else 1e-3
        decay = 0.0 if parameter.ndim == 1 or name.endswith('.bias') else 5e-2
        groups.setdefault((lr,decay), []).append(parameter)
    optimizer = torch.optim.AdamW([dict(params=params, lr=lr, weight_decay=wd) for (lr,wd),params in groups.items()])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=.5, patience=10)
    best, start = float('inf'), 0
    if args.resume:
        state = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(state['model_state_dict'])
        optimizer.load_state_dict(state['optimizer_state_dict'])
        scheduler.load_state_dict(state['scheduler_state_dict'])
        start, best = state['epoch'], state['val_loss']
    args.output_dir.mkdir(parents=True, exist_ok=True)

    def to_device(value):
        if torch.is_tensor(value):
            return value.to(device, non_blocking=True)
        if isinstance(value, dict):
            return {k:to_device(v) for k,v in value.items()}
        if isinstance(value, (list,tuple)):
            return type(value)(to_device(v) for v in value)
        return value

    def run_epoch(batches, training):
        model.train(training)
        total, count = 0., 0.
        with torch.set_grad_enabled(training):
            for index, batch in enumerate(batches):
                if not training and args.max_val_batches is not None and index >= args.max_val_batches:
                    break
                inputs, targets, meta = to_device(batch)
                if training:
                    optimizer.zero_grad(set_to_none=True)
                losses = model(inputs, targets=targets, meta_info=meta)
                loss = losses['loss_xy'] + .3*losses['loss_z'] + losses['loss_shifter']
                if not torch.isfinite(loss):
                    raise RuntimeError('Non-finite training/validation loss')
                if training:
                    loss.backward()
                    optimizer.step()
                total += loss.item()
                count += targets['hand_presence'].sum().item()
        return total/(count+1e-6)

    for epoch in range(start,args.epochs):
        train_loss = run_epoch(loader, True)
        val_loss = run_epoch(val_loader, False)
        scheduler.step(val_loss)
        print(f'Epoch {epoch+1}: train={train_loss:.7f}, validation={val_loss:.7f}', flush=True)
        if val_loss < best:
            best = val_loss
            torch.save(dict(epoch=epoch+1, model_state_dict=model.state_dict(),
                optimizer_state_dict=optimizer.state_dict(), scheduler_state_dict=scheduler.state_dict(),
                val_loss=best), args.output_dir/'best_model.pt')


if __name__ == '__main__':
    train(parse_args())
