from torch.utils.data import DataLoader
from datasets.vg_dataset import vg_collate_fn, build_vg_dsets

def build_vg_loaders(args):

    vocab, train_dset, val_dset = build_vg_dsets(args)
    collate_fn = vg_collate_fn

    loader_kwargs = {
        'batch_size': args.batch_size,
        'num_workers': args.workers,
        'shuffle': True,
        'collate_fn': collate_fn,
    }
    train_loader = DataLoader(train_dset, **loader_kwargs)
    train_loader.num_samples = len(train_dset)

    loader_kwargs['batch_size'] = args.val_batch_size
    loader_kwargs['shuffle'] = False
    val_loader = DataLoader(val_dset, **loader_kwargs)
    val_loader.num_samples = len(val_dset)
    return vocab, train_loader, val_loader
