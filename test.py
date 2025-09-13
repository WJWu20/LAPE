import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset
import torchvision
from torchvision import transforms
import torchvision.transforms.functional as TF
import torchvision.utils as vutils

import timm

# assert timm.__version__ == "0.3.2"  # version check

import util.misc as misc
import models.LAPE as CntViT

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
# matplotlib.use('TkAgg')
plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']

def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=1, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=1, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='mae_vit_base_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=10, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--data_path', default='/media/cs4007/adfc2692-0951-4a9b-8ea6-ebfa0e11323b/wwj/FSC147/', type=str,
                        help='dataset path')

    parser.add_argument('--output_dir', default='./output/',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output/',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='/media/cs4007/adfc2692-0951-4a9b-8ea6-ebfa0e11323b/wwj/CLIPCAC/output/checkpoint-185.pth',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    return parser


os.environ["CUDA_LAUNCH_BLOCKING"] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = '2'


# load data from FSC147
data_path = '/media/lcc/DATA/wwj/datasets/FSC147/'
anno_file = data_path + 'annotation_FSC147_384.json'
data_split_file = data_path + 'Train_Test_Val_FSC_147.json'
im_dir = data_path + 'images_384_VarV2'
gt_dir = data_path + 'gt_density_map_adaptive_384_VarV2'
class_file = data_path + 'ImageClasses_FSC147.txt'


with open(anno_file) as f:
    annotations = json.load(f)

with open(data_split_file) as f:
    data_split = json.load(f)

class_dict = {}
with open(class_file) as f:
    for line in f:
        key = line.split()[0]
        val = line.split()[1:]
        class_ = ""
        for s in val:
            class_ += s + ' '
        class_dict[key] = class_.rstrip()

class TestData(Dataset):
    def __init__(self, dataset='test'):
        self.img = data_split[dataset]
        self.img_dir = im_dir
        with open("text_embedding/clip_text_embedding_oc.json", "r") as f:
            self.text_tokens = json.load(f)
        with open("text_embedding/clip_text_embedding.json", "r") as f:
            self.description_tokens = json.load(f)

    def __len__(self):
        return len(self.img)

    def __getitem__(self, idx):
        im_id = self.img[idx]
        anno = annotations[im_id]

        dots = np.array(anno['points'])

        image = Image.open('{}/{}'.format(im_dir, im_id))
        image.load()
        W, H = image.size

        new_H = 16 * int(H / 16)
        new_W = 16 * int(W / 16)
        scale_factor = float(new_W) / W
        image = transforms.Resize((new_H, new_W))(image)
        Normalize = transforms.Compose([transforms.ToTensor()])
        image = Normalize(image)

        # Only for visualisation purpose, no need for ground truth density map indeed.
        gt_map = np.zeros((image.shape[1], image.shape[2]), dtype='float32')
        for i in range(dots.shape[0]):
            gt_map[min(new_H - 1, int(dots[i][1]))][min(new_W - 1, int(dots[i][0] * scale_factor))] = 1
        gt_map = ndimage.gaussian_filter(gt_map, sigma=(1, 1), order=0)
        gt_map = torch.from_numpy(gt_map)
        gt_map = gt_map * 60
        sample = {'image': image, 'dots': dots, 'gt_map': gt_map}
        ## text embedding
        sample['text'] = torch.tensor(self.text_tokens[im_id])
        sample['description'] = torch.tensor(self.description_tokens[class_dict[im_id]])
        return sample['image'], sample['text'], sample['description'], sample['dots'], sample['gt_map'], im_id


def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    dataset_test = TestData()
    print(dataset_test)

    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_test = torch.utils.data.DistributedSampler(
            dataset_test, num_replicas=num_tasks, rank=global_rank, shuffle=False
        )
        print("Sampler_test = %s" % str(sampler_test))
    else:
        sampler_test = torch.utils.data.RandomSampler(dataset_test)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, sampler=sampler_test,
        batch_size=1,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    # define the model
    model = CntViT.__dict__[args.model](norm_pix_loss=args.norm_pix_loss)

    model.to(device)

    model_without_ddp = model

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    misc.load_model_FSC(args=args, model_without_ddp=model_without_ddp)

    print(f"Start testing.")
    start_time = time.time()

    # test
    epoch = 0
    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    # some parameters in training
    train_mae = 0
    train_rmse = 0
    pred_cnt = 0
    gt_cnt = 0

    loss_array = []
    gt_array = []
    wrong_id = []

    for data_iter_step, (samples, text, desp, gt_dots, gt_map, im_id) in enumerate(
            metric_logger.log_every(data_loader_test, print_freq, header)):
        samples = samples.to(device, non_blocking=True)
        gt_dots = gt_dots.to(device, non_blocking=True).half()
        text = text.to(device, non_blocking=True)
        desp = desp.to(device, non_blocking=True)
        gt_map = gt_map.to(device, non_blocking=True)

        _, _, h, w = samples.shape

        density_map = torch.zeros([h, w])
        density_map = density_map.to(device, non_blocking=True)
        start = 0
        prev = -1
        with torch.no_grad():
            while start + 383 < w:
                input_x = []
                a = samples[:, :, :, start:start + 384]
                input_x.append(a)
                input_x.append(text)
                input_x.append(desp)
                output, indices, text_feature, vis_feature, query = model(input_x)
                generate_tSNE(text_feature.squeeze(0).detach().cpu().numpy(), vis_feature.squeeze(0).detach().cpu().numpy(), query.squeeze(0).detach().cpu().numpy(), im_id)
                import matplotlib.pyplot as plt
                x = (indices // 24).cpu().numpy() * 24
                y = (indices % 24).cpu().numpy() * 24
                plt.figure(figsize=(6, 6))
                plt.imshow(a[0].permute(1, 2, 0).detach().cpu())
                plt.scatter(x, y, c='blue', marker='o')
                plt.title("Scatter Plot of (x, y)")
                plt.xlabel("x")
                plt.ylabel("y")
                plt.gca()
                plt.savefig(f"{im_id}.jpg")

                output = output.squeeze(0)
                b1 = nn.ZeroPad2d(padding=(start, w - prev - 1, 0, 0))
                d1 = b1(output[:, 0:prev - start + 1])
                b2 = nn.ZeroPad2d(padding=(prev + 1, w - start - 384, 0, 0))
                d2 = b2(output[:, prev - start + 1:384])

                b3 = nn.ZeroPad2d(padding=(0, w - start, 0, 0))
                density_map_l = b3(density_map[:, 0:start])
                density_map_m = b1(density_map[:, start:prev + 1])
                b4 = nn.ZeroPad2d(padding=(prev + 1, 0, 0, 0))
                density_map_r = b4(density_map[:, prev + 1:w])

                density_map = density_map_l + density_map_r + density_map_m / 2 + d1 / 2 + d2

                prev = start + 383
                start = start + 128
                if start + 383 >= w:
                    if start == w - 384 + 128:
                        break
                    else:
                        start = w - 384

            pred_cnt = torch.sum(density_map / 60).item()

        gt_cnt = gt_dots.shape[1]
        cnt_err = abs(pred_cnt - gt_cnt)
        # if cnt_err < 1:
            # plt.figure()
            # plt.imshow(samples.squeeze(0).permute(1, 2, 0).detach().cpu())
            # plt.imshow(density_map.squeeze(0).detach().cpu() / 60, cmap="jet", alpha=0.5)
            # plt.axis('off')
            # plt.title("GT:{:.2f}, Pred:{:.2f}".format(gt_cnt, pred_cnt))
            # plt.savefig("/media/cs4007/adfc2692-0951-4a9b-8ea6-ebfa0e11323b/wwj/CLIPCAC/img/" + im_id[0])
        # import matplotlib.pyplot as plt
        # density_map = density_map - density_map.min() / (density_map.max() - density_map.min())
        #
        # plt.figure()
        # plt.imshow(samples.squeeze(0).permute(1, 2, 0).detach().cpu())
        # plt.imshow(density_map.squeeze(0).detach().cpu() / 60, cmap="jet", alpha=0.5)
        # plt.axis('off')
        # plt.title("{} GT:{:.2f}   Pred:{:.2f}".format(im_id, gt_cnt, pred_cnt))
        #
        # # 去掉白边保存
        # plt.savefig(
        #     "/media/cs4007/adfc2692-0951-4a9b-8ea6-ebfa0e11323b/wwj/CLIPCAC/img_val/" + im_id[0].split('.')[0] + ".png",
        #     bbox_inches='tight',  # 紧凑边界
        #     pad_inches=0,  # 不留边距
        # )
        # plt.close()  # 关闭 figure，避免内存占用

        train_mae += cnt_err
        train_rmse += cnt_err ** 2

        print(
            f'{data_iter_step}/{len(data_loader_test)}: pred_cnt: {pred_cnt},  gt_cnt: {gt_cnt},  error: {cnt_err},  AE: {cnt_err},  SE: {cnt_err ** 2} ')

        loss_array.append(cnt_err)
        gt_array.append(gt_cnt)

        torch.cuda.synchronize(device=0)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    print('Current MAE: {:5.2f}, RMSE: {:5.2f} '.format(train_mae / (len(data_loader_test)),
                                                        (train_rmse / (len(data_loader_test))) ** 0.5))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
