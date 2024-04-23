import torch
import torch.nn as nn
import numpy as np
import nibabel as nib
import os
def get_boundary(seg, kernel_size):
    pad_size = int((kernel_size - 1) / 2)
    m_xy = nn.AvgPool3d((kernel_size, kernel_size, 1), stride=1, padding=(pad_size, pad_size, 0)).cuda()
    output_xy = m_xy(seg)
    edge_xy = abs(seg - output_xy)
    # edge = edge_xy[0, :]
    edge_locations = torch.multiply(edge_xy, seg)
    edge_locations[edge_locations > 0] = 1
    edge_mask = edge_locations.squeeze(0)

    return edge_mask


def find_boundary_map(seg, boundary_kernel=3, margin_kernel=7):
    boundary = get_boundary(seg, kernel_size=boundary_kernel).unsqueeze(0)
    margin = get_boundary(seg, kernel_size=margin_kernel).unsqueeze(0) - boundary
    content = seg - margin - boundary
    return boundary.squeeze(0), margin.squeeze(0), content.squeeze(0)


def get_points(seg, sample):
    x = torch.where(seg == 1)[2][sample]
    y = torch.where(seg == 1)[3][sample]
    z = torch.where(seg == 1)[4][sample]  # --> tensor([[x_value]]) instead of tensor([x_value])
    return x, y, z


def get_points_location(seg, num=1, use_seed=True, seed=0):
    """
    use this to get anchor points
    """
    l = len(torch.where(seg == 1)[0])
    if use_seed:
        np.random.seed(seed)
    else:
        np.random.seed(None)
    sample = np.random.choice(np.arange(l), num, replace=True)
    x, y, z = get_points(seg, sample)


    points_dict = {'x_location': x, 'y_location': y, 'z_location': z}
    return points_dict

if __name__ == '__main__':
    seg_data = nib.load('./example_label_cropped.nii.gz')
    seg = seg_data.get_fdata()
    seg = torch.from_numpy(seg).float().cuda().unsqueeze(0).unsqueeze(0)

    boundary, margin, content = find_boundary_map(seg)

    points_dict = get_points_location(seg)

    boundary = boundary.squeeze(0).squeeze(0).cpu().detach().numpy()
    margin = margin.squeeze(0).squeeze(0).cpu().detach().numpy()
    content = content.squeeze(0).squeeze(0).cpu().detach().numpy()

    nib.save(nib.Nifti1Image(boundary, seg_data.affine, seg_data.header), os.path.join(os.getcwd(), 'boundary.nii.gz'))
    nib.save(nib.Nifti1Image(margin, seg_data.affine, seg_data.header), os.path.join(os.getcwd(), 'margin.nii.gz'))
    nib.save(nib.Nifti1Image(content, seg_data.affine, seg_data.header), os.path.join(os.getcwd(), 'content.nii.gz'))
    print('points location: {}'.format(points_dict))

    # seg_data = nib.load('./example_label.nii.gz')
    # seg = seg_data.get_fdata()
    # seg_crop = seg[280:280+200, 200:200+200, :]
    # nib.save(nib.Nifti1Image(seg_crop, seg_data.affine, seg_data.header), os.path.join(os.getcwd(), 'example_label_cropped.nii.gz'))
    #
    # seg_data = nib.load('./example_image.nii.gz')
    # seg = seg_data.get_fdata()
    # seg_crop = seg[280:280+200, 200:200+200, :]
    # nib.save(nib.Nifti1Image(seg_crop, seg_data.affine, seg_data.header), os.path.join(os.getcwd(), 'example_image_cropped.nii.gz'))