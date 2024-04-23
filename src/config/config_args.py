import argparse
import os
import warnings
parser = argparse.ArgumentParser()


# data
parser.add_argument("--data", default=None, type=str, choices=["kits", "pancreas", "lits", "colon"])
parser.add_argument("--save_dir", default="./implementation/", type=str)
parser.add_argument("--data_dir", default="", type=str)
parser.add_argument("--num_workers", default=2, type=int)
parser.add_argument("--split", default="train", type=str)
parser.add_argument('--use_small_dataset', action="store_true")


# network
parser.add_argument('--model_type', type=str, default='vit_b_ori')
parser.add_argument("--lr", default=4e-5, type=float)
parser.add_argument("--lr_scheduler", default='linear', type=str, choices=["linear", "exp"])
parser.add_argument('--warm_up', action="store_true")
parser.add_argument("--device", default="cuda:0", type=str)
parser.add_argument("--max_epoch", default=200, type=int)
parser.add_argument("--image_size", default=128, type=int)
parser.add_argument("--batch_size", default=1, type=int)
parser.add_argument("--checkpoint", default="best", type=str)
parser.add_argument("--checkpoint_sam", default="./checkpoint_sam/sam_vit_b_01ec64.pth", type=str,
                    help='path of pretrained SAM')
parser.add_argument("--num_classes", default=2, type=int)
parser.add_argument("--tolerance", default=5, type=int)
parser.add_argument("--boundary_kernel_size", default=5, type=int,
                    help='an integer for kernel size of avepooling layer for boundary generation')
parser.add_argument("--use_pretrain", action="store_true")
parser.add_argument("--pretrain_path", default="", type=str)
parser.add_argument("--resume", action="store_true")
parser.add_argument("--resume_best", action="store_true")
parser.add_argument("--ddp", action="store_true")
parser.add_argument('--gpu_ids', type=int, nargs='+', default=[0, 1])
parser.add_argument('--accumulation_steps', type=int, default=20)

parser.add_argument('--iter_nums', type=int, default=11)
parser.add_argument('--num_clicks', type=int, default=50)
parser.add_argument('--num_clicks_validation', type=int, default=10)
parser.add_argument('--use_box', action="store_true")
parser.add_argument('--dynamic_box', action="store_true")
parser.add_argument('--use_scribble', action="store_true")


parser.add_argument('--num_multiple_outputs', type=int, default=3)
parser.add_argument('--multiple_outputs', action="store_true")
parser.add_argument('--refine', action="store_true")
parser.add_argument('--no_detach', action="store_true")
parser.add_argument('--refine_test', action="store_true")

parser.add_argument('--dynamic', action="store_true")
parser.add_argument('--efficient_scribble', action="store_true")
parser.add_argument("--use_sam3d_turbo", action="store_true")



# saving
parser.add_argument("--save_predictions", action="store_true")
parser.add_argument("--save_csv", action="store_true")
parser.add_argument("--save_test_dir", default='./', type=str)
parser.add_argument("--save_name", default='testing_only', type=str)






def check_and_setup_parser(args):
    if args.save_name == 'testing_only':
        warnings.warn("[save_name] (--save_name) should be a real name, currently is for testing purpose (--save_name=testing_only)")


    args.save_dir = os.path.join(args.save_dir, args.data, args.save_name)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
