from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchio as tio
import pickle
import numpy as np
import os
import torch
import SimpleITK as sitk
from prefetch_generator import BackgroundGenerator
from monai.transforms import (
    Compose,
    RandCropByPosNegLabeld,
    ScaleIntensityRanged,
    NormalizeIntensityd,
    RandShiftIntensityd,
    RandZoomd,
)
import cc3d, math

class Dataset_promise(Dataset):
    def __init__(self, data, data_dir, split='train', image_size=128, transform=None, pcc=False, args=None):
        self.args = args
        self.data = data
        self.paths = data_dir

        self._set_file_paths(self.paths, split)
        self._set_dataset_stat()

        self.image_size = (image_size, image_size, image_size)
        self.transform = transform
        self.threshold = 0
        self.split = split
        self.pcc = pcc
        self.monai_transforms = self._get_transforms(split=split)

        self.cc = 1

    def __len__(self):
        return len(self.label_paths)

    def __getitem__(self, index):
        sitk_image = sitk.ReadImage(self.image_paths[index])
        sitk_label = sitk.ReadImage(self.label_paths[index])

        if sitk_image.GetOrigin() != sitk_label.GetOrigin():
            sitk_image.SetOrigin(sitk_label.GetOrigin())
        if sitk_image.GetDirection() != sitk_label.GetDirection():
            sitk_image.SetDirection(sitk_label.GetDirection())

        if sitk_image.GetSpacing() != sitk_label.GetSpacing():
            sitk_label.SetSpacing(sitk_image.GetSpacing())

        subject = tio.Subject(
            image=tio.ScalarImage.from_sitk(sitk_image),
            label=tio.LabelMap.from_sitk(sitk_label),
        )

        subject_save = tio.Subject(
            image=tio.ScalarImage.from_sitk(sitk_image),
            label=tio.LabelMap.from_sitk(sitk_label),
        )


        if self.data == 'lits':
            b = subject.label.data
            a = tio.CropOrPad._bbox_mask(b[0].cpu().numpy())
            w, h, d = a[1][0] - a[0][0], a[1][1] - a[0][1], a[1][2] - a[0][2]
            w, h, d = max(w + 20, 128), max(h + 20, 128), max(d + 20, 128)
            crop_transform = tio.CropOrPad(mask_name='label', target_shape=(w, h, d))
            subject = crop_transform(subject)
            subject_save = crop_transform(subject_save)



        if self.target_label != 0:
            subject = self._binary_label(subject)
            subject_save = self._binary_label(subject_save)

        if self.transform:
            try:
                subject = self.transform(subject)
                subject_save = self.transform(subject_save)
            except:
                print(self.image_paths[index])

        if (self.pcc):
            subject = self._pcc(subject)


        if subject.label.data.sum() <= self.threshold:
            print(self.image_paths[index], 'label volume too small')
            if self.split == 'train':
                return self.__getitem__(np.random.randint(self.__len__()))
                #return self.__getitem__(0)
            else:
                if self.data == 'lits':
                    return subject, self.image_paths[index]
                else:
                    return subject.image.data.clone().detach(), subject.label.data.clone().detach(), self.image_paths[index]


        if self.split == "train":
            trans_dict = self.monai_transforms({"image": subject.image.data.clone().detach(),
                                                "label": subject.label.data.clone().detach()})[0]
            img_aug, seg_aug = trans_dict["image"], trans_dict["label"]
            return img_aug.float(), seg_aug.float(), self.image_paths[index]
        else:
            if self.data == 'lits':
                trans_dict = self.monai_transforms({"image": subject.image.data.clone().detach()})
                subject.image.data = trans_dict["image"]
                return subject, self.image_paths[index], subject_save

            if self.data == 'kits':
                subject = self._separate_crop(subject)

            crop_transform = tio.CropOrPad(mask_name='label', target_shape=self.image_size)
            subject = crop_transform(subject)
            subject_save = crop_transform(subject_save)

            trans_dict = self.monai_transforms({"image": subject.image.data.clone().detach()})
            img_aug = trans_dict["image"]
            return img_aug, subject.label.data.clone().detach(), self.image_paths[index], subject_save


    def _separate_crop(self, subject):
        label = subject.label.data
        labels_out, N = cc3d.connected_components(label[0].cpu().numpy(), return_N=True)
        crop_transform = tio.CropOrPad(mask_name='label', target_shape=self.image_size)
        mid_cut = 0

        if N > 1:
            label_1, label_2 = torch.zeros_like(label), torch.zeros_like(label)

            # left, right
            mid_cut = math.ceil(label.size(1) / 2)
            label_1[0, 0: mid_cut, :], label_2[0, mid_cut: -1, :] = label[0, 0: mid_cut, :], label[0, mid_cut: -1, :] # left, right


            image_1, image_2 = subject.image.data, subject.image.data

            subject_1 = tio.Subject(image=tio.ScalarImage(tensor=image_1), label=tio.LabelMap(tensor=label_1))
            subject_2 = tio.Subject(image=tio.ScalarImage(tensor=image_2), label=tio.LabelMap(tensor=label_2))

            subject_1, subject_2 = crop_transform(subject_1), crop_transform(subject_2)

            # found 2 connected components for some cases (e.g. case 289), use below to eliminate
            # however, this will bring warnings, but it's okay
            if torch.unique(subject_2.label.data).size(0) == 1:
                subject.image.data, subject.label.data = subject_1.image.data, subject_1.label.data
            elif torch.unique(subject_1.label.data).size(0) == 1:
                subject.image.data, subject.label.data = subject_2.image.data, subject_2.label.data
            else:
                subject.image.data = torch.cat([subject_1.image.data, subject_2.image.data], dim=0)
                subject.label.data = torch.cat([subject_1.label.data, subject_2.label.data], dim=0)
        else:
            subject = crop_transform(subject)

        return subject

    def _set_file_paths(self, data_dir, split):
        self.image_paths = []
        self.label_paths = []
        split_file = "split.pkl"
        dataset_split = os.path.join(data_dir, split_file)
        with open(dataset_split, "rb") as f:
            d = pickle.load(f)[0][split]
        self.image_paths = [os.path.join(data_dir, d[i][0].strip("/")) for i in list(d.keys())]
        self.label_paths = [os.path.join(data_dir, d[i][1].strip("/")) for i in list(d.keys())]

    def _set_dataset_stat(self):
        self.target_label = 0
        if self.data == 'colon':
            self.intensity_range, self.global_mean, self.global_std = (-57, 175), 65.175035, 32.651197

        elif self.data == 'pancreas':
            self.intensity_range, self.global_mean, self.global_std = (-39, 204), 68.45214, 63.422806
            self.target_label = 2

        elif self.data == 'lits':
            self.intensity_range, self.global_mean, self.global_std = (-48, 163), 60.057533, 40.198017
            self.target_label = 2

        elif self.data == 'kits':
            self.intensity_range, self.global_mean, self.global_std = (-54, 247), 59.53867, 55.457336
            self.target_label = 2


    def _get_transforms(self, split):
        if split == "train":
            transforms = Compose(
                [
                    ScaleIntensityRanged(
                        keys=["image"],
                        a_min=self.intensity_range[0],
                        a_max=self.intensity_range[1],
                        b_min=self.intensity_range[0],
                        b_max=self.intensity_range[1],
                        clip=True,
                    ),
                    RandCropByPosNegLabeld(
                        keys=["image", "label"],
                        spatial_size=(128, 128, 128),
                        label_key="label",
                        pos=2,
                        neg=0,
                        num_samples=1,
                    ),
                    RandShiftIntensityd(keys=["image"], offsets=20, prob=0.5),
                    NormalizeIntensityd(keys=["image"], subtrahend=self.global_mean, divisor=self.global_std),

                    RandZoomd(keys=["image", "label"], prob=0.8, min_zoom=0.85, max_zoom=1.25,
                              mode=["trilinear", "nearest"]),
                    ])
        else:
            transforms = Compose(
                [
                    ScaleIntensityRanged(
                        keys=["image"],
                        a_min=self.intensity_range[0],
                        a_max=self.intensity_range[1],
                        b_min=self.intensity_range[0],
                        b_max=self.intensity_range[1],
                        clip=True,
                    ),
                    NormalizeIntensityd(keys=["image"], subtrahend=self.global_mean, divisor=self.global_std),
                ]
            )
        return transforms

    def _binary_label(self, subject):
        label = subject.label.data
        label = (label == self.target_label)
        subject.label.data = label.float()
        return subject

    def _pcc(self, subject):
        print("using pcc setting")
        # crop from random click point
        random_index = torch.argwhere(subject.label.data == 1)
        if (len(random_index) >= 1):
            random_index = random_index[np.random.randint(0, len(random_index))]
            # print(random_index)
            crop_mask = torch.zeros_like(subject.label.data)
            # print(crop_mask.shape)
            crop_mask[random_index[0]][random_index[1]][random_index[2]][random_index[3]] = 1
            subject.add_image(tio.LabelMap(tensor=crop_mask, affine=subject.label.affine), image_name="crop_mask")
            subject = tio.CropOrPad(mask_name='crop_mask', target_shape=self.image_size)(subject)

        return subject


class Dataloader_promise(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())




