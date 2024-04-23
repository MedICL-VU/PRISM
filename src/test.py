import logging
import os.path
import torch
from utils.util import setup_logger
from config.config_args import *
import numpy as np
from torch.backends import cudnn
from src.config.config_setup import build_model, get_dataloader
import time, random
import torch.nn.functional as F
from src.utils.util import _bbox_mask
from src.utils import scribble, boundary_selection
import torchio as tio
import surface_distance
from surface_distance import metrics

def init_seeds(seed=0, cuda_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if cuda_deterministic:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True


class Tester(object):
    def __init__(self, args, logger, ckpt):
        self.args = args
        self.logger = logger

        self.val_data = get_dataloader(args, split='test')

        a = time.time()
        print('loading models and setting up')
        self.sam = build_model(args, checkpoint=ckpt)

        self.image_encoder = self.sam.image_encoder
        self.prompt_encoder = self.sam.prompt_encoder
        self.mask_decoder = self.sam.mask_decoder

        # self._load_pretrain_model(ckpt)

    def _load_pretrain_model(self, ckpt):
        model_dict = torch.load(ckpt, map_location=self.args.device)
        state_dict = model_dict
        self.sam.load_state_dict(state_dict['model_state_dict'])

    def validate(self, epoch_num):
        self.image_encoder.eval()
        self.prompt_encoder.eval()
        self.mask_decoder.eval()

        if self.args.data == 'lits':
            loss = self.validater_sliding_window(epoch_num)
        else:
            loss = self.validater(epoch_num)
        return loss


    def validater_sliding_window(self, epoch_num):
        with torch.no_grad():
            dice_summary, nsd_summary = [], []
            for idx, (subject_dict, image_path, subject_dict_save) in enumerate(self.val_data):
                if subject_dict['label']['data'][0].sum() <= 0:
                    self.logger.info(image_path, 'label volume too small, and it has been skipped for validation')
                    continue
                mean_dice = 0
                subject = tio.Subject(image=tio.ScalarImage(tensor=subject_dict['image']['data'][0].float(), affine=subject_dict['image']['affine'][0]),
                                      label=tio.LabelMap(tensor=subject_dict['label']['data'][0].float(), affine=subject_dict['label']['affine'][0]))
                grid_sampler = tio.inference.GridSampler(subject, 128, 16)
                patch_loader = torch.utils.data.DataLoader(grid_sampler, batch_size=1)
                aggregator = tio.inference.GridAggregator(grid_sampler, overlap_mode='average')


                for idx_patch, patches_batch in enumerate(patch_loader):
                    image, label = patches_batch['image'][tio.DATA].to(self.args.device), patches_batch['label'][tio.DATA].to(self.args.device)
                    print(torch.count_nonzero(label))
                    print('how many voxels')
                    locations = patches_batch[tio.LOCATION]

                    if torch.count_nonzero(label) == 0:
                        print('found empty patch')
                        masks = torch.zeros([1, 1, 128, 128, 128])
                    else:
                        # _, masks = self._interaction(self.sam, image, label, iter_nums=self.args.iter_nums, train=False, return_each_iter=True)
                        _, masks = self._interaction(self.sam, image, label, iter_nums=self.args.iter_nums, train=False)

                    aggregator.add_batch(masks, locations)
                masks_iter_final = aggregator.get_output_tensor()
                mean_dice_sub = self.get_dice_score(torch.sigmoid(masks_iter_final), subject.label.data)

                mean_dice += mean_dice_sub
                dice_summary.append(mean_dice)

                ssd = surface_distance.compute_surface_distances(
                    (subject.label.data == 1)[0].cpu().numpy(),
                    (torch.sigmoid(masks_iter_final) > 0.5)[0].cpu().numpy(),
                    spacing_mm=(1,1,1)
                )
                nsd = metrics.compute_surface_dice_at_tolerance(ssd, 5)

                nsd_summary.append(nsd)
                print(mean_dice_sub)

                if self.args.save_predictions:
                    save_test_dir = os.path.join(self.args.save_test_dir, 'prism_prediction', self.args.data, self.args.save_name, str(self.args.iter_nums))
                    if not os.path.exists(save_test_dir):
                        os.makedirs(save_test_dir)
                    a = torch.sigmoid(masks_iter_final) > 0.5
                    a = a[0].float().cpu().numpy()
                    import SimpleITK as sitk
                    prediction = sitk.GetImageFromArray(a)
                    if self.args.data == 'lits':
                        base_name = image_path[0].split('/')[-2] + '_' +image_path[0].split('/')[-1]
                    if self.args.refine_test:
                        pred_name = base_name.replace('.nii.gz', '._pred.nii.gz')
                    else:
                        pred_name = base_name.replace('.nii.gz', '._pred_no_refine.nii.gz')
                    save_path = os.path.join(save_test_dir, pred_name)
                    sitk.WriteImage(prediction, save_path)

                    if self.args.iter_nums == 1:
                        if self.args.refine_test:
                            image_name = base_name.replace('.nii.gz', '._image.nii.gz')
                        else:
                            image_name = base_name.replace('.nii.gz', '._image_no_refine.nii.gz')
                        b = subject_dict_save['image']['data'][0][0].float().cpu().numpy()
                        image_save = sitk.GetImageFromArray(b)
                        sitk.WriteImage(image_save, os.path.join(save_test_dir, image_name))

                        if self.args.refine_test:
                            label_name = base_name.replace('.nii.gz', '._label.nii.gz')
                        else:
                            label_name = base_name.replace('.nii.gz', '._label_no_refine.nii.gz')
                        c = subject_dict_save['label']['data'][0][0].float().cpu().numpy()
                        label_save = sitk.GetImageFromArray(c)
                        sitk.WriteImage(label_save, os.path.join(save_test_dir, label_name))



                self.logger.info(
                    'epoch: {}/{}, iter: {}/{}'.format(epoch_num, self.args.max_epoch, idx, len(self.val_data)) +
                    ' subject: ' + str(image_path) + ' mean nsd over clicks:' + str(nsd) + ' mean dice over clicks:' + str(mean_dice) +
                    ' stich left and right side (total size): ' + str(label.size(1)))
            self.logger.info("- Val metrics mean dice: " + str(np.mean(dice_summary)) + "- Val metrics nsd: " + str(np.mean(nsd_summary)))

            from scipy import stats
            data = dice_summary
            # Calculate mean
            mean = np.mean(data)
            # Calculate standard error of the mean (SEM)
            sem = stats.sem(data)
            # Determine the t-value for the 95% confidence interval
            # Degrees of freedom
            df = len(data) - 1
            # t-value for 95% CI
            t_value = stats.t.ppf(0.975, df)
            # Calculate the margin of error
            margin_of_error = sem * t_value
            # Calculate the 95% CI
            ci_lower = mean - margin_of_error
            ci_upper = mean + margin_of_error
            self.logger.info("- ci_lower dice: " + str(ci_lower) + "- ci_lower dice: " + str(ci_upper))

        return dice_summary

    def validater(self, epoch_num):
        device = self.args.device
        with torch.no_grad():
            loss_summary, nsd_summary = [], []
            # for idx, data in enumerate(val_data):
            # img, label = data['image'].to(device), data['label'].to(device)
            for idx, (image, label, image_path, subject_dict_save) in enumerate(self.val_data):

                image, label = image.to(device), label.to(device)

                if self.args.data == 'kits' and image.size(1) > 1:
                    label_final, masks_final = torch.zeros([1, 1, int(image.size(2) * 2), image.size(3), image.size(4)]), torch.zeros([self.args.iter_nums, 1, int(image.size(2) * 2), image.size(3), image.size(4)])

                    for channel_num in range(image.size(1)):
                        masks = self.interaction(self.sam, image[:, channel_num, :].unsqueeze(1), label[:, channel_num, :].unsqueeze(1))
                        start_point, end_pont = 0 + channel_num * image.size(2), image.size(2) + channel_num * image.size(2)

                        masks_final[:, 0, start_point: end_pont, :] = masks[:, 0, :]
                        label_final[0, 0, start_point: end_pont, :] = label[0, channel_num, :]

                    masks, label = masks_final, label_final
                else:
                    masks = self.interaction(self.sam, image, label)

                # masks = self.interaction(self.sam, image, label)

                dice = self.get_dice_score(torch.sigmoid(masks), label)
                loss_summary.append(dice)

                ssd = surface_distance.compute_surface_distances(
                    (label == 1)[0][0].cpu().numpy(),
                    (torch.sigmoid(masks) > 0.5)[0][0].cpu().numpy(),
                    spacing_mm=(1, 1, 1)
                )
                nsd = metrics.compute_surface_dice_at_tolerance(ssd, 5)

                nsd_summary.append(nsd)

                if self.args.save_predictions:
                    save_test_dir = os.path.join(self.args.save_test_dir, 'prism_prediction', self.args.data,
                                                 self.args.save_name, str(self.args.iter_nums))
                    if not os.path.exists(save_test_dir):
                        os.makedirs(save_test_dir)
                    a = torch.sigmoid(masks) > 0.5
                    a = a.float().cpu().numpy()
                    import SimpleITK as sitk
                    prediction = sitk.GetImageFromArray(a)
                    if self.args.data == 'colon':
                        base_name = image_path[0].split('/')[-1]
                    else:
                        base_name = image_path[0].split('/')[-2] + '_' + image_path[0].split('/')[-1]
                    if self.args.refine_test:
                        pred_name = base_name.replace('.nii.gz', '._pred.nii.gz')
                    else:
                        pred_name = base_name.replace('.nii.gz', '._pred_no_refine.nii.gz')
                    save_path = os.path.join(save_test_dir, pred_name)
                    sitk.WriteImage(prediction, save_path)

                    if self.args.iter_nums == 1:
                        if self.args.refine_test:
                            image_name = base_name.replace('.nii.gz', '._image.nii.gz')
                        else:
                            image_name = base_name.replace('.nii.gz', '._image_no_refine.nii.gz')
                        b = subject_dict_save['image']['data'][0][0].float().cpu().numpy()
                        image_save = sitk.GetImageFromArray(b)
                        sitk.WriteImage(image_save, os.path.join(save_test_dir, image_name))

                        if self.args.refine_test:
                            label_name = base_name.replace('.nii.gz', '._label.nii.gz')
                        else:
                            label_name = base_name.replace('.nii.gz', '._label_no_refine.nii.gz')
                        c = subject_dict_save['label']['data'][0][0].float().cpu().numpy()
                        label_save = sitk.GetImageFromArray(c)
                        sitk.WriteImage(label_save, os.path.join(save_test_dir, label_name))

                self.logger.info(
                    'epoch: {}/{}, iter: {}/{}'.format(epoch_num, self.args.max_epoch, idx, len(self.val_data)) +
                    ' subject: ' + str(image_path) + ' mean nsd over clicks:' + str(nsd) + ' mean dice over clicks:' + str(dice) +
                    ' stich left and right side (total size): ' + str(label.size(1)))

            self.logger.info("- Val metrics mean dice: " + str(np.mean(loss_summary)) + "- Val metrics nsd: " + str(np.mean(nsd_summary)))
            from scipy import stats
            data = loss_summary
            # Calculate mean
            mean = np.mean(data)
            # Calculate standard error of the mean (SEM)
            sem = stats.sem(data)
            # Determine the t-value for the 95% confidence interval
            # Degrees of freedom
            df = len(data) - 1
            # t-value for 95% CI
            t_value = stats.t.ppf(0.975, df)
            # Calculate the margin of error
            margin_of_error = sem * t_value
            # Calculate the 95% CI
            ci_lower = mean - margin_of_error
            ci_upper = mean + margin_of_error
            self.logger.info("- ci_lower dice: " + str(ci_lower) + "- ci_lower dice: " + str(ci_upper))

        return loss_summary

    def get_next_click3D_torch_2(self, prev_seg, gt_semantic_seg):

        mask_threshold = 0.5

        batch_points = []
        batch_labels = []
        # dice_list = []

        pred_masks = (prev_seg > mask_threshold)
        true_masks = (gt_semantic_seg > 0)
        fn_masks = torch.logical_and(true_masks, torch.logical_not(pred_masks))
        fp_masks = torch.logical_and(torch.logical_not(true_masks), pred_masks)
        print('fn: {}, fp: {}'.format(torch.count_nonzero(fn_masks) / torch.count_nonzero(true_masks),
                                      torch.count_nonzero(fp_masks) / torch.count_nonzero(true_masks)))
        to_point_mask = torch.logical_or(fn_masks, fp_masks)
        #to_point_mask = fn_masks
        for i in range(gt_semantic_seg.shape[0]):
            bp_list, bl_list = [], []
            points = torch.argwhere(to_point_mask[i])
            if self.args.num_clicks > len(points):
                click_size = len(points)
            else:
                click_size = self.args.num_clicks

            dynamic_size = random.randint(1, click_size) if self.args.dynamic else click_size

            point_index = np.random.choice(len(points), size=dynamic_size, replace=False)
            points_select = points[point_index]  # each row tensor([0, x, y, z]), size --> num_clicks x 4
            # point = points[np.random.randint(len(points))] # tensor([0, x, y, z])
            for click_index in range(dynamic_size):
                point = points_select[click_index]
                if fn_masks[i, 0, point[1], point[2], point[3]]:
                    is_positive = True
                else:
                    is_positive = False

                bp = point[1:].clone().detach().reshape(1, 1, 3)
                bl = torch.tensor([int(is_positive), ]).reshape(1, 1)
                bp_list.append(bp)
                bl_list.append(bl)

            if self.args.use_scribble:
                #sample_method = random.choice(['line', 'center', 'default'])
                sample_method = 'center'
                scribble_types = {
                    'line': 'LineScribble',
                    'center': 'CenterlineScribble',
                    'default': 'ContourScribble'
                }

                def create_scribble_mask(scribble_type, data):
                    scribble_object = getattr(scribble, scribble_type)()
                    scribble_mask = scribble_object.batch_scribble(data).permute(1, 2, 3, 0)
                    return scribble_mask > 0

                # fg = gt_semantic_seg[i].permute(3, 0, 1, 2).float()
                # bg = (torch.ones_like(pred_masks[i, :]).float() - gt_semantic_seg[i].float()).permute(3, 0, 1, 2)
                fg, bg = fn_masks[0].permute(3, 0, 1, 2).float(), fp_masks[0].permute(3, 0, 1, 2).float()

                scribble_type = scribble_types.get(sample_method, scribble_types['default'])

                scribble_mask_fg = create_scribble_mask(scribble_type, fg)
                #fg_coors = torch.argwhere(scribble_mask_fg)[:, 1:].unsqueeze(0)[:, 0: 100, :]  # for computation only
                fg_coors = torch.argwhere(scribble_mask_fg)[:, 1:].unsqueeze(0)
                if self.args.efficient_scribble:
                    fg_coors = fg_coors[:, 0: 10000, :]  # for computation only# for computation only
                fg_coors_label = torch.ones(1, fg_coors.size(1))
                bp_list.append(fg_coors)
                bl_list.append(fg_coors_label)
                # x,y,z = bp_list[-1][0, 99, 0], bp_list[-1][0, 99, 1], bp_list[-1][0, 99, 2]
                # print(gt_semantic_seg[i, 0, x,y,z])

                #if sample_method == 'default':
                if torch.count_nonzero(fp_masks) > 0:
                    scribble_mask_bg = create_scribble_mask(scribble_type, bg)
                    bg_coors = torch.argwhere(scribble_mask_bg)[:, 1:].unsqueeze(0)
                    if self.args.efficient_scribble:
                        bg_coors = bg_coors[:, 0: 10000, :]
                    bg_coors_label = torch.zeros(1, bg_coors.size(1))
                    bp_list.append(bg_coors)
                    bl_list.append(bg_coors_label)

            batch_points.append(torch.cat(bp_list, dim=1))
            batch_labels.append(torch.cat(bl_list, dim=1))

            smallest_n = min(tensor.size(1) for tensor in batch_labels)
            batch_points = [tensor[:, :smallest_n] if tensor.size(1) > smallest_n else tensor for tensor in
                            batch_points]
            batch_labels = [tensor[:, :smallest_n] if tensor.size(1) > smallest_n else tensor for tensor in
                            batch_labels]

            # Check the shapes of the adjusted tensors
            for i, tensor in enumerate(batch_points):
                print(f"Tensor {i + 1} shape: {tensor.shape}")


        return batch_points, batch_labels

    def get_points(self, prev_masks, label):
        batch_points, batch_labels = self.get_next_click3D_torch_2(prev_masks, label)

        points_co = torch.cat(batch_points, dim=0).to(self.args.device)
        points_la = torch.cat(batch_labels, dim=0).to(self.args.device)

        self.click_points.append(points_co)
        self.click_labels.append(points_la)

        points_input = points_co
        labels_input = points_la

        bbox_coords = _bbox_mask(label[:, 0, :]).to(self.args.device) if self.args.use_box else None
        return points_input, labels_input, bbox_coords


    def batch_forward(self, sam_model, features, image_embedding, image, prev_masks, points=None, boxes=None):
        prev_masks = F.interpolate(prev_masks, scale_factor=0.25)
        features = [features[i].to(self.args.device) for i in range(0, len(features))]

        # sparse_embeddings --> (B, 2, embed_dim) 2 represents concat of coordination and its label
        # dense_embeddings --> (B, embed_dim, W, H, D), whd values are customized
        new_point_embedding, new_image_embedding = sam_model.prompt_encoder(
            points=points,
            boxes=boxes,
            masks=prev_masks,
            image_embeddings=image_embedding.to(self.args.device)
        )

        mask, pred_dice = sam_model.mask_decoder(
            prompt_embeddings=new_point_embedding,  # (B, 2, 256)
            image_embeddings=new_image_embedding,  # (B, 256, 64, 64)
            feature_list=features,
        )

        return mask, pred_dice

    def interaction(self, sam_model, image, label):
        image_embedding, feature_list = self.sam.image_encoder(image)

        self.click_points = []
        self.click_labels = []
        prev_masks = torch.zeros_like(label).to(label.device)
        for iter_num in range(self.args.iter_nums):
            prev_masks_sigmoid = torch.sigmoid(prev_masks) if iter_num > 0 else prev_masks

            points_input, labels_input, bbox_input = self.get_points(prev_masks_sigmoid, label)
            mask, pred_dice = self.batch_forward(sam_model, feature_list, image_embedding, image, prev_masks, points=[points_input, labels_input], boxes=bbox_input)

            if self.args.multiple_outputs:
                pred_best_dice, pred_dice_max_index = torch.max(pred_dice, dim=1)
                mask_best = mask[:, pred_dice_max_index, :]
            else:
                mask_best, pred_best_dice = mask, pred_dice
            # FIXME refine or not
            if self.args.refine and self.args.refine_test:
                mask_refine, error_map = self.sam.mask_decoder.refine(image, mask_best, [self.click_points, self.click_labels], mask_best.detach())
                print('dice before refine {} and after {}'.format(
                    self.get_dice_score(torch.sigmoid(mask_best), label),
                    self.get_dice_score(torch.sigmoid(mask_refine), label))
                )
                mask_best = mask_refine

            prev_masks = mask_best
            dice = self.get_dice_score(torch.sigmoid(prev_masks).cpu().numpy(), label.cpu().numpy())
            print('---')
            print(f'Dice: {dice:.4f}, pred_dice: {pred_best_dice}, label: {labels_input}')

        return prev_masks



    def _interaction(self, sam_model, image, label, iter_nums, train=False, return_each_iter=False):
        if return_each_iter:
            return_mask_total_iter = torch.zeros([iter_nums, 1, image.size(2), image.size(3), image.size(4)])

        image_embedding, feature_list = self.sam.image_encoder(image)
        self.click_points = []
        self.click_labels = []
        return_loss = 0
        prev_masks = torch.zeros_like(label, dtype=torch.float).to(label.device)
        for iter_num in range(iter_nums):
            prev_masks_sigmoid = torch.sigmoid(prev_masks) if iter_num > 0 else prev_masks

            if self.args.init_learning and iter_num == 0:
                boundary, margin, content = boundary_selection.find_boundary_map(label)
                use_content = True
                for batch_index in range(label.size(0)):
                    if torch.count_nonzero(content[batch_index]) < self.args.num_clicks:
                        use_content = False
                if use_content:
                    label_sample = content
                else:
                    label_sample = label
            else:
                label_sample = label

            points_input, labels_input, box_input = self.get_points(prev_masks_sigmoid, label_sample, label)
            mask, dice_pred = self.batch_forward(sam_model, feature_list, image_embedding, image, prev_masks, points=[points_input, labels_input], boxes=box_input)

            # ========================================================
            if self.args.multiple_outputs:
                dice_pred_best, max_label_index = torch.max(dice_pred, dim=1)
                mask_list = [mask[i, max_label_index[i], :].unsqueeze(0) for i in range(mask.size(0))]
                mask_best = torch.stack(mask_list, dim=0)
            else:
                mask_best = mask

            # ========================================================

            if self.args.refine and self.args.refine_test:
                mask_refine, error_map = self.sam.mask_decoder.refine(image, mask_best, [self.click_points, self.click_labels], mask_best.detach())
                self.logger.info('dice before refine {} and after {}, label 0: {}, label 1: {}'.format(
                    self.get_dice_score(torch.sigmoid(mask_best), label), self.get_dice_score(torch.sigmoid(mask_refine), label),
                    str(labels_input.numel() - torch.count_nonzero(labels_input)), str(torch.count_nonzero(labels_input)) ) )
                mask_best = mask_refine  # FIXME refine or not

            loss = self.get_dice_score(torch.sigmoid(mask_best), label) # dice

            return_loss += loss
            prev_masks = mask_best

            if return_each_iter:
                return_mask_total_iter[iter_num, :] = mask_best
        if return_each_iter:
            print(return_mask_total_iter.shape)
            return return_loss / iter_nums, return_mask_total_iter
        else:
            return return_loss / iter_nums, prev_masks

    def get_dice_score(self, prev_masks, label):
        def compute_dice(mask_pred, mask_gt):
            mask_threshold = 0.5

            mask_pred = (mask_pred > mask_threshold)
            mask_gt = (mask_gt > 0)

            volume_sum = mask_gt.sum() + mask_pred.sum()
            if volume_sum == 0:
                return np.NaN
            volume_intersect = (mask_gt & mask_pred).sum()
            return 2 * volume_intersect / volume_sum

        pred_masks = (prev_masks > 0.5)
        true_masks = (label > 0)
        dice_list = []
        for i in range(true_masks.shape[0]):
            dice_list.append(compute_dice(pred_masks[i], true_masks[i]))
        return (sum(dice_list) / len(dice_list)).item()




def main():
    init_seeds()
    args = parser.parse_args()
    check_and_setup_parser(args)

    log_name = 'test_' + args.save_name
    setup_logger(logger_name=log_name, root=args.save_dir, screen=True, tofile=True)
    logger = logging.getLogger(log_name)
    logger.info(str(args))

    #ckpt = '/home/hao/Hao/3D_medical_foundation_model/src/implementation/log/colon/3DSAM/best.pth.tar'
    ckpt = os.path.join(args.save_dir, args.checkpoint + '.pth.tar')
    with torch.no_grad():
        tester = Tester(args, logger, ckpt)
        loss = tester.validate(epoch_num=0)

        print(loss)

    logger.info("- Test done")

if __name__ == "__main__":
    main()