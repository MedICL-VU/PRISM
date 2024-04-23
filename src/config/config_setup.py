from src.models.build_sam3D import sam_model_registry3D
from src.dataset.dataloader import Dataset_promise, Dataloader_promise
import torchio as tio
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch
def get_dataloader(args, split='', use_small=False):
    transforms_list = [tio.ToCanonical(), tio.Resample(1), ]
    if split == 'train':
        transforms_list.append(tio.RandomFlip(axes=(0, 1, 2)))

    transforms = tio.Compose(transforms_list)

    dataset = Dataset_promise(
        data=args.data,
        data_dir=args.data_dir,
        split=split,
        transform=transforms,
        image_size=args.image_size,
        args=args,
    )

    batch_size = args.batch_size if split == 'train' else 1

    if split == 'train':
        train_sampler = None
        shuffle = True
        if args.ddp:
            train_sampler = DistributedSampler(dataset)
            shuffle = False
    else:
        train_sampler = None
        shuffle = False

    dataloader = Dataloader_promise(
        dataset=dataset,
        sampler=train_sampler,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    return dataloader



def build_model(args, checkpoint=None):
    sam_model = sam_model_registry3D[args.model_type](checkpoint=checkpoint, args=args).to(args.device)
    if args.ddp:
        sam_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(sam_model)
        sam_model = DDP(sam_model, device_ids=[args.rank], output_device=args.rank)
    return sam_model