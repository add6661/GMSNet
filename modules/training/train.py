# The code is heavily borrowed from [XFeat](https://github.com/verlab/accelerated_features)

import argparse
import os
import time
import sys
import glob
import tqdm

def parse_arguments():
    parser = argparse.ArgumentParser(description="GMSNet training script.")

    parser.add_argument('--megadepth_root_path', type=str, default='/ssd/guipotje/Data/MegaDepth',
                        help='Path to the MegaDepth dataset root directory.')
    parser.add_argument('--synthetic_root_path', type=str, default='/homeLocal/guipotje/sshfs/datasets/coco_20k',
                        help='Path to the synthetic dataset root directory.')
    parser.add_argument('--ckpt_save_path', type=str, required=True,
                        help='Path to save the checkpoints.')
    parser.add_argument('--training_type', type=str, default='GMSNet_default',
                        choices=['GMSNet_default', 'GMSNet_synthetic', 'GMSNet_megadepth'],
                        help='Training scheme. GMSNet_default uses both megadepth & synthetic warps.')
    parser.add_argument('--batch_size', type=int, default=10,
                        help='Batch size for training. Default is 10.')
    parser.add_argument('--n_steps', type=int, default=160_000,
                        help='Number of training steps. Default is 160000.')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate. Default is 0.0003.')
    parser.add_argument('--gamma_steplr', type=float, default=0.5,
                        help='Gamma value for StepLR scheduler. Default is 0.5.')
    parser.add_argument('--training_res', type=lambda s: tuple(map(int, s.split(','))),
                        default=(800, 608), help='Training resolution as width,height. Default is (800, 608).')
    parser.add_argument('--device_num', type=str, default='0',
                        help='Device number to use for training. Default is "0".')
    parser.add_argument('--dry_run', action='store_true',
                        help='If set, perform a dry run training with a mini-batch for sanity check.')
    parser.add_argument('--save_ckpt_every', type=int, default=500,
                        help='Save checkpoints every N steps. Default is 500.')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of worker processes for data loading. Default is 0.')

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.device_num

    return args

args = parse_arguments()

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import numpy as np

from modules.model import *
from modules.dataset.augmentation import *
from modules.training.utils import *
from modules.training.losses import *

from modules.dataset.megadepth.megadepth import MegaDepthDataset
from modules.dataset.megadepth import megadepth_warper
from torch.utils.data import Dataset, DataLoader


class Trainer():


    def __init__(self, megadepth_root_path,
                       synthetic_root_path,
                       ckpt_save_path,
                       model_name='GMSNet_default',
                       batch_size=10, n_steps=160_000, lr=3e-4, gamma_steplr=0.5,
                       training_res=(800, 608), device_num="0", dry_run=False,
                       save_ckpt_every=500, num_workers=0):

        self.dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net = GMSNetModel().to(self.dev)

        # Setup optimizer
        self.batch_size = batch_size
        self.steps = n_steps
        self.opt = optim.Adam(filter(lambda x: x.requires_grad, self.net.parameters()), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.opt, step_size=30_000, gamma=gamma_steplr)

        if model_name in ('GMSNet_default', 'GMSNet_synthetic'):
            self.augmentor = AugmentationPipe(
                                        img_dir=synthetic_root_path,
                                        device=self.dev, load_dataset=True,
                                        batch_size=int(self.batch_size * 0.4 if model_name == 'GMSNet_default' else batch_size),
                                        out_resolution=training_res,
                                        warp_resolution=training_res,
                                        sides_crop=0.1,
                                        max_num_imgs=3_000,
                                        num_test_imgs=5,
                                        photometric=True,
                                        geometric=True,
                                        reload_step=4_000
                                        )
        else:
            self.augmentor = None

        if model_name in ('GMSNet_default', 'GMSNet_megadepth'):
            TRAIN_BASE_PATH = f"{megadepth_root_path}/train_data/megadepth_indices"
            TRAINVAL_DATA_SOURCE = f"{megadepth_root_path}/MegaDepth_v1"

            TRAIN_NPZ_ROOT = f"{TRAIN_BASE_PATH}/scene_info_0.1_0.7"

            npz_paths = glob.glob(TRAIN_NPZ_ROOT + '/*.npz')[:]
            data = torch.utils.data.ConcatDataset([MegaDepthDataset(root_dir=TRAINVAL_DATA_SOURCE,
                                                                    npz_path=path) for path in tqdm.tqdm(npz_paths, desc="[MegaDepth] Loading metadata")])

            self.data_loader = DataLoader(data,
                                          batch_size=int(self.batch_size * 0.6 if model_name == 'GMSNet_default' else batch_size),
                                          shuffle=True,
                                          num_workers=num_workers)
            self.data_iter = iter(self.data_loader)

        else:
            self.data_iter = None

        os.makedirs(ckpt_save_path, exist_ok=True)
        os.makedirs(ckpt_save_path + '/logdir', exist_ok=True)

        self.dry_run = dry_run
        self.save_ckpt_every = save_ckpt_every
        self.ckpt_save_path = ckpt_save_path
        self.writer = SummaryWriter(ckpt_save_path + f'/logdir/{model_name}_' + time.strftime("%Y_%m_%d-%H_%M_%S"))
        self.model_name = model_name


    def train(self):

        self.net.train()

        difficulty = 0.10

        p1s, p2s, H1, H2 = None, None, None, None
        d = None

        if self.augmentor is not None:
            p1s, p2s, H1, H2 = make_batch(self.augmentor, difficulty)

        if self.data_iter is not None:
            d = next(self.data_iter)

        with tqdm.tqdm(total=self.steps) as pbar:
            for i in range(self.steps):
                if not self.dry_run:
                    if self.data_iter is not None:
                        try:
                            # Get the next MD batch
                            d = next(self.data_iter)

                        except StopIteration:
                            print("End of DATASET!")
                            # If StopIteration is raised, create a new iterator.
                            self.data_iter = iter(self.data_loader)
                            d = next(self.data_iter)

                    if self.augmentor is not None:
                        # Grab synthetic data
                        p1s, p2s, H1, H2 = make_batch(self.augmentor, difficulty)

                if d is not None:
                    for k in d.keys():
                        if isinstance(d[k], torch.Tensor):
                            d[k] = d[k].to(self.dev)

                    p1, p2 = d['image0'], d['image1']
                    positives_md_coarse = megadepth_warper.spvs_coarse(d, 8)

                if self.augmentor is not None:
                    h_coarse, w_coarse = p1s[0].shape[-2] // 8, p1s[0].shape[-1] // 8
                    _ , positives_s_coarse = get_corresponding_pts(p1s, p2s, H1, H2, self.augmentor, h_coarse, w_coarse)

                # Join megadepth & synthetic data
                with torch.inference_mode():

                    if d is not None:
                        p1 = p1.mean(1, keepdim=True)
                        p2 = p2.mean(1, keepdim=True)
                    if self.augmentor is not None:
                        p1s = p1s.mean(1, keepdim=True)
                        p2s = p2s.mean(1, keepdim=True)

                    # Cat two batches
                    if self.model_name in ('GMSNet_default'):
                        p1 = torch.cat([p1s, p1], dim=0)
                        p2 = torch.cat([p2s, p2], dim=0)
                        positives_c = positives_s_coarse + positives_md_coarse
                    elif self.model_name in ('GMSNet_synthetic'):
                        p1 = p1s ; p2 = p2s
                        positives_c = positives_s_coarse
                    else:
                        positives_c = positives_md_coarse

                # Check if batch is corrupted with too few correspondences
                is_corrupted = False
                for p in positives_c:
                    if len(p) < 30:
                        is_corrupted = True

                if is_corrupted:
                    continue

                # Forward pass
                feats1, kpts1, hmap1 = self.net(p1)
                feats2, kpts2, hmap2 = self.net(p2)

                loss_items = []

                for b in range(len(positives_c)):
                    # Get positive correspondencies
                    pts1, pts2 = positives_c[b][:, :2], positives_c[b][:, 2:]

                    # Grab features at corresponding idxs
                    m1 = feats1[b, :, pts1[:,1].long(), pts1[:,0].long()].permute(1,0)
                    m2 = feats2[b, :, pts2[:,1].long(), pts2[:,0].long()].permute(1,0)

                    # Grab heatmaps at corresponding idxs
                    h1 = hmap1[b, 0, pts1[:,1].long(), pts1[:,0].long()]
                    h2 = hmap2[b, 0, pts2[:,1].long(), pts2[:,0].long()]
                    coords1 = self.net.fine_matcher(torch.cat([m1, m2], dim=-1))

                    # Compute losses
                    loss_ds, conf = dual_softmax_loss(m1, m2)
                    try:
                        loss_coords, acc_coords = coordinate_classification_loss(coords1, pts1, pts2, conf)
                        if torch.isnan(acc_coords) or torch.isinf(acc_coords):
                            acc_coords = torch.tensor(0.000, device=self.dev)
                    except Exception as e:
                        print(f"Error in coordinate_classification_loss: {e}")
                        acc_coords = torch.tensor(0.000, device=self.dev)

                    loss_kp_pos1, acc_pos1 = alike_distill_loss(kpts1[b], p1[b])
                    loss_kp_pos2, acc_pos2 = alike_distill_loss(kpts2[b], p2[b])
                    loss_kp_pos = (loss_kp_pos1 + loss_kp_pos2)*2.0
                    acc_pos = (acc_pos1 + acc_pos2)/2

                    loss_kp =  keypoint_loss(h1, conf) + keypoint_loss(h2, conf)

                    loss_items.append(loss_ds.unsqueeze(0))
                    loss_items.append(loss_coords.unsqueeze(0))
                    loss_items.append(loss_kp.unsqueeze(0))
                    loss_items.append(loss_kp_pos.unsqueeze(0))

                    if b == 0:
                        acc_coarse_0 = check_accuracy(m1, m2)

                acc_coarse = check_accuracy(m1, m2)

                nb_coarse = len(m1)
                loss = torch.cat(loss_items, -1).mean()
                loss_coarse = loss_ds.item()
                loss_coord = loss_coords.item()
                loss_kp_pos = loss_kp_pos.item()
                loss_l1 = loss_kp.item()

                # Compute Backward Pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.)
                self.opt.step()
                self.opt.zero_grad()
                self.scheduler.step()

                if (i+1) % self.save_ckpt_every == 0:
                    print('saving iter ', i+1)
                    torch.save(self.net.state_dict(), self.ckpt_save_path + f'/{self.model_name}_{i+1}.pth')

                pbar.set_description( 'Loss: {:.4f} acc_c0 {:.3f} acc_c1 {:.3f} acc_f: {:.3f} loss_c: {:.3f} loss_f: {:.3f} loss_kp: {:.3f} #matches_c: {:d} loss_kp_pos: {:.3f} acc_kp_pos: {:.3f}'.format(
                                                                        loss.item(), acc_coarse_0, acc_coarse, acc_coords, loss_coarse, loss_coord, loss_l1, nb_coarse, loss_kp_pos, acc_pos) )
                pbar.update(1)

                # Log metrics
                self.writer.add_scalar('Loss/total', loss.item(), i)
                self.writer.add_scalar('Accuracy/coarse_synth', acc_coarse_0, i)
                self.writer.add_scalar('Accuracy/coarse_mdepth', acc_coarse, i)
                self.writer.add_scalar('Accuracy/fine_mdepth', acc_coords, i)
                self.writer.add_scalar('Accuracy/kp_position', acc_pos, i)
                self.writer.add_scalar('Loss/coarse', loss_coarse, i)
                self.writer.add_scalar('Loss/fine', loss_coord, i)
                self.writer.add_scalar('Loss/reliability', loss_l1, i)
                self.writer.add_scalar('Loss/keypoint_pos', loss_kp_pos, i)
                self.writer.add_scalar('Count/matches_coarse', nb_coarse, i)



if __name__ == '__main__':
    args = parse_arguments()

    trainer = Trainer(
        megadepth_root_path=args.megadepth_root_path,
        synthetic_root_path=args.synthetic_root_path,
        ckpt_save_path=args.ckpt_save_path,
        model_name=args.training_type,
        batch_size=args.batch_size,
        n_steps=args.n_steps,
        lr=args.lr,
        gamma_steplr=args.gamma_steplr,
        training_res=args.training_res,
        device_num=args.device_num,
        dry_run=args.dry_run,
        save_ckpt_every=args.save_ckpt_every,
        num_workers=args.num_workers
    )

    trainer.train()