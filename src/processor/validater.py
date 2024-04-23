import torch
import torch.nn.functional as F
# from src.utils.util import get_points
import numpy as np


def validater(args, val_data, logger, epoch_num, sam,
          loss_validation):
    patch_size = args.rand_crop_size[0]
    device = args.device
    with torch.no_grad():
        loss_summary = []
        #for idx, data in enumerate(val_data):
            #img, label = data['image'].to(device), data['label'].to(device)
        for idx, (image, label, _) in enumerate(val_data):
            # import torchio as tio
            # norm_transform = tio.ZNormalization(masking_method=lambda x: x > 0)
            # image = norm_transform(image.squeeze(dim=1))  # (N, C, W, H, D)
            # image = image.unsqueeze(dim=1)

            image, label = image.to(device), label.to(device)

            image_embedding = sam.image_encoder(image)
            prev_masks = interaction(args, sam, image_embedding, label, num_clicks=11)

            masks = prev_masks
            loss = loss_validation(masks, label)
            loss_summary.append(loss.detach().cpu().numpy())
            logger.info(
                'epoch: {}/{}, iter: {}/{}'.format(epoch_num, args.max_epoch, idx, len(val_data)) + ": loss:" + str(
                    loss_summary[-1].flatten()[0]))
        logger.info("- Val metrics: " + str(np.mean(loss_summary)))
    return loss_summary


def get_next_click3D_torch_2(prev_seg, gt_semantic_seg):

    mask_threshold = 0.5

    batch_points = []
    batch_labels = []
    # dice_list = []

    pred_masks = (prev_seg > mask_threshold)
    true_masks = (gt_semantic_seg > 0)
    fn_masks = torch.logical_and(true_masks, torch.logical_not(pred_masks))
    fp_masks = torch.logical_and(torch.logical_not(true_masks), pred_masks)

    to_point_mask = torch.logical_or(fn_masks, fp_masks)

    for i in range(gt_semantic_seg.shape[0]):

        points = torch.argwhere(to_point_mask[i])
        point = points[np.random.randint(len(points))]
        # import pdb; pdb.set_trace()
        if fn_masks[i, 0, point[1], point[2], point[3]]:
            is_positive = True
        else:
            is_positive = False

        bp = point[1:].clone().detach().reshape(1, 1, 3)
        bl = torch.tensor([int(is_positive), ]).reshape(1, 1)
        batch_points.append(bp)
        batch_labels.append(bl)

    return batch_points, batch_labels
def get_points(args, prev_masks, gt3D, click_points, click_labels):
    batch_points, batch_labels = get_next_click3D_torch_2(prev_masks, gt3D)

    points_co = torch.cat(batch_points, dim=0).to(args.device)
    points_la = torch.cat(batch_labels, dim=0).to(args.device)

    click_points.append(points_co)
    click_labels.append(points_la)

    points_multi = torch.cat(click_points, dim=1).to(args.device)
    labels_multi = torch.cat(click_labels, dim=1).to(args.device)

    # if self.args.multi_click:
    #     points_input = points_multi
    #     labels_input = labels_multi
    # else:
    points_input = points_co
    labels_input = points_la
    return points_input, labels_input, click_points, click_labels

def batch_forward(args, sam_model, image_embedding, gt3D, low_res_masks, points=None):

    sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
        points=points,
        boxes=None,
        masks=low_res_masks,
    )
    low_res_masks, iou_predictions = sam_model.mask_decoder(
        image_embeddings=image_embedding.to(args.device),  # (B, 256, 64, 64)
        image_pe=sam_model.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
        sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
        dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
        multimask_output=False,
    )
    prev_masks = F.interpolate(low_res_masks, size=gt3D.shape[-3:], mode='trilinear', align_corners=False)
    return low_res_masks, prev_masks

def interaction(args, sam_model, image_embedding, gt3D, num_clicks):
    # return_loss = 0
    prev_masks = torch.zeros_like(gt3D).to(gt3D.device)
    random_insert = np.random.randint(2, 9)

    click_points, click_labels = [], []
    for num_click in range(num_clicks):
        points_input, labels_input, click_points, click_labels = get_points(args, prev_masks, gt3D, click_points, click_labels)

        if num_click == random_insert or num_click == num_clicks - 1:
            prev_masks = batch_forward(args, sam_model, image_embedding, gt3D, prev_masks, points=None)
        else:
            prev_masks = batch_forward(args, sam_model, image_embedding, gt3D, prev_masks, points=[points_input, labels_input])
        # loss = self.seg_loss(prev_masks, gt3D)
        # return_loss += loss
    #return prev_masks, return_loss
    return prev_masks