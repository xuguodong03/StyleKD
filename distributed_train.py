import argparse
import os
import os.path as osp
import random
import time

import numpy as np
import torch
from torch import autograd, nn, optim
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, utils

import lpips
from dataset import FFHQ_Dataset
from Evaluation.fid import Get_Model_FID_Score
from Miscellaneous.distributed import (get_world_size, reduce_loss_dict,
                                       reduce_sum)
from model import Discriminator, EqualLinear
from op import conv2d_gradfix
from Util.content_aware_pruning import (Batch_Img_Parsing, Get_Masked_Tensor,
                                        Get_Parsing_Net)
from Util.network_util import Build_Generator_From_Dict


def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)
    if shuffle:
        return data.RandomSampler(dataset)
    else:
        return data.SequentialSampler(dataset)

def calc_direction_split(model, args):

    vectors = []
    for i in range(max(args.mimic_layer)):
        w1 = model.convs[2*i].conv.modulation.weight.data.cpu().numpy()
        w2 = model.convs[2*i+1].conv.modulation.weight.data.cpu().numpy()
        w = np.concatenate((w1,w2), axis=0).T
        w /= np.linalg.norm(w, axis=0, keepdims=True)
        _, eigen_vectors = np.linalg.eig(w.dot(w.T))
        vectors.append(torch.from_numpy(eigen_vectors[:,:5].T))
    return torch.cat(vectors, dim=0)   # (5*L) * 512

def calc_direction_global(model, args):

    k = 0
    pool = []
    for i in range(max(args.mimic_layer)):
        w1 = model.convs[2*i].conv.modulation.weight.data.cpu().numpy()
        w2 = model.convs[2*i+1].conv.modulation.weight.data.cpu().numpy()
        w = np.concatenate((w1,w2), axis=0).T
        pool.append(w)
        k += 5
    w = np.hstack(pool)
    w /= np.linalg.norm(w, axis=0, keepdims=True)
    _, eigen_vectors = np.linalg.eig(w.dot(w.T))
    return torch.from_numpy(eigen_vectors[:,:k].T)

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1-decay)

def cycle(loader):
    while True:
        for batch in loader:
            yield batch

def Downsample_Image(im_tensor, size):
    im_tensor = F.interpolate(im_tensor, size=(size, size), mode='bilinear', align_corners=False)
    return im_tensor

def KD_loss(args, teacher_g, noise, inject_index, fake_img, fake_img_list, percept_loss, parsing_net, offsets, student_feat_list, idx, student_w, device):

    with torch.no_grad():
        fake_img_teacher_list, teacher_feat_list, teacher_w = teacher_g(noise, \
                offsets=offsets, return_rgb_list=True, inject_index=inject_index, \
                return_feat=True, return_style=True)
    fake_img_teacher = fake_img_teacher_list[-1]

    # Content-Aware Adjustment for fake_img and fake_img_teacher
    if parsing_net is not None:
        with torch.no_grad():
            teacher_img_parsing = Batch_Img_Parsing(fake_img_teacher, parsing_net, device)
        fake_img_teacher = Get_Masked_Tensor(fake_img_teacher, teacher_img_parsing, device, mask_grad=False)
        fake_img = Get_Masked_Tensor(fake_img, teacher_img_parsing, device, mask_grad=True)
    #fake_img_teacher.requires_grad = True

    # kd_l1_loss
    if args.kd_mode == 'Output_Only':
        kd_l1_loss = torch.mean(torch.abs(fake_img_teacher - fake_img))
    elif args.kd_mode == 'Intermediate':
        #for fake_img_teacher in fake_img_teacher_list:
        #    fake_img_teacher.requires_grad = True
        loss_list = [torch.mean(torch.abs(fake_img_teacher - fake_img)) for (fake_img_teacher, fake_img) in zip(fake_img_teacher_list, fake_img_list)]
        kd_l1_loss = sum(loss_list)

    kd_style_loss = F.l1_loss(student_w, teacher_w)

    # kd_lpips_loss
    if percept_loss is None:
        kd_lpips_loss = torch.tensor(0.0, device=device)
    else:
        if args.size > args.lpips_image_size: # pooled the image for LPIPS for memory saving
            pooled_fake_img = Downsample_Image(fake_img, args.lpips_image_size)
            pooled_fake_img_teacher = Downsample_Image(fake_img_teacher, args.lpips_image_size)
            kd_lpips_loss = torch.mean(percept_loss(pooled_fake_img, pooled_fake_img_teacher))

        else:
            kd_lpips_loss = torch.mean(percept_loss(fake_img, fake_img_teacher))

    # relation loss
    kd_simi_loss = torch.zeros_like(kd_style_loss)
    batch = args.batch
    for i in args.mimic_layer:
        f1 = student_feat_list[i-1][:batch]
        f2 = student_feat_list[i-1][:batch] if args.single_view \
                else student_feat_list[i-1][batch:]
        s_simi = F.cosine_similarity(f1[:,None,:], f2[None,:,:], dim=2)

        f1 = teacher_feat_list[i-1][:batch]
        f2 = teacher_feat_list[i-1][:batch] if args.single_view \
                else teacher_feat_list[i-1][batch:]
        t_simi = F.cosine_similarity(f1[:,None,:], f2[None,:,:], dim=2)
        if args.simi_loss == 'mse':
            kd_simi_loss += F.mse_loss(s_simi, t_simi)
        elif args.simi_loss == 'kl':
            s_simi = F.log_softmax(s_simi, dim=1)
            t_simi = F.softmax(t_simi, dim=1)
            kd_simi_loss += F.kl_div(s_simi, t_simi, reduction='batchmean')

    return kd_l1_loss, kd_lpips_loss, kd_simi_loss, kd_style_loss


def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)
    return real_loss.mean() + fake_loss.mean()

def d_r1_loss(real_pred, real_img):
    with conv2d_gradfix.no_weight_gradients():
        grad_real, = autograd.grad(
            outputs=real_pred.sum(), inputs=real_img, create_graph=True)
    grad_penalty = grad_real.pow(2).view(grad_real.shape[0], -1).sum(1).mean()

    return grad_penalty

def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()
    return loss

def make_noise(batch, latent_dim, n_noise, device):
    if n_noise == 1:
        return torch.randn(batch, latent_dim, device=device)
    noises = torch.randn(n_noise, batch, latent_dim, device=device).unbind(0)
    return noises

def mixing_noise(batch, latent_dim, prob, device):
    if prob > 0 and random.random() < prob:
        return make_noise(batch, latent_dim, 2, device)
    else:
        return [make_noise(batch, latent_dim, 1, device)]

def index_aware_mixing_noise(batch, latent_dim, prob, n_latent, device):
    if prob > 0 and random.random() < prob:
        mixed_noises = make_noise(batch, latent_dim, 2, device)
        inject_index = random.randint(1, n_latent - 1)
        return mixed_noises, inject_index
    else:
        return [make_noise(batch, latent_dim, 1, device)], None

def G_Loss_BackProp(generator, discriminator, args, loss_dict, g_optim, teacher_g, percept_loss, parsing_net, vectors, idx, device):

    requires_grad(generator, True)
    requires_grad(discriminator, False)

    # compute offset
    batch = args.batch
    dim = vectors.size(1)
    if args.offset_mode == 'random':
        offsets = torch.randn(batch,dim, device=device)
    elif args.offset_mode == 'main':
        num_direction = vectors.size(0)
        index = np.random.choice(np.arange(num_direction), size=(batch,)).astype(np.int64)
        offsets = vectors[index].to(device)
    norm = torch.norm(offsets, dim=1, keepdim=True)
    offsets = offsets / norm
    weight = torch.randn(batch,1, device=device) * args.offset_weight
    offsets = offsets * weight
    offsets = offsets[:,None,:]

    # GAN Loss
    noise, inject_index = index_aware_mixing_noise(args.batch, args.latent, args.mixing, args.n_latent, device)
    fake_img_list, student_feat_list, student_w = generator(noise, offsets=offsets, return_rgb_list=True, inject_index=inject_index, return_feat=True, return_style=True)
    fake_img = fake_img_list[-1]
    fake_pred = discriminator(fake_img)
    g_loss = g_nonsaturating_loss(fake_pred)
    loss_dict['g'] = g_loss

    total_loss = g_loss if idx >= args.g_step else 0

    # KD Loss
    kd_l1_loss, kd_lpips_loss, kd_simi_loss, kd_style_loss = KD_loss(args, teacher_g, noise, inject_index, fake_img, fake_img_list, percept_loss, parsing_net, offsets, student_feat_list, idx, student_w, device)
    loss_dict['kd_l1_loss'] = kd_l1_loss
    loss_dict['kd_lpips_loss'] = kd_lpips_loss
    loss_dict['kd_simi_loss'] = kd_simi_loss
    loss_dict['kd_style_loss'] = kd_style_loss
    total_loss = total_loss + args.kd_l1_lambda * kd_l1_loss \
                    + args.kd_lpips_lambda * kd_lpips_loss \
                    + args.kd_simi_lambda * kd_simi_loss \
                    + args.kd_style_lambda * kd_style_loss

    g_optim.zero_grad()
    total_loss.backward()
    g_optim.step()

def G_Reg_BackProp(generator, args, mean_path_length, g_optim):

    path_batch_size = max(1, args.batch // args.path_batch_shrink)
    noise = mixing_noise(path_batch_size, args.latent, args.mixing, device)

    fake_img, path_lengths = generator(noise, PPL_regularize=True)
    decay = 0.01
    path_mean = mean_path_length + decay * (path_lengths.mean() - mean_path_length)
    path_loss = (path_lengths - path_mean).pow(2).mean()
    mean_path_length = path_mean.detach()

    g_optim.zero_grad()
    weighted_path_loss = args.path_regularize * args.g_reg_every * path_loss

    if args.path_batch_shrink:
        weighted_path_loss += 0 * fake_img[0, 0, 0, 0]

    weighted_path_loss.backward()

    g_optim.step()

    mean_path_length_avg = (
        reduce_sum(mean_path_length).item() / get_world_size()
    )
    return path_loss, path_lengths, mean_path_length, mean_path_length_avg


def train(args, loader, generator, discriminator, g_optim, d_optim, g_ema, teacher_g, percept_loss, parsing_net, exp_dir, logger, vectors, device):

    loader = cycle(loader)

    if args.local_rank == 0:
        sample_dir = exp_dir + '/sample/'
        ckpt_dir = exp_dir + '/ckpt/'
        os.makedirs(sample_dir, exist_ok=True)
        os.makedirs(ckpt_dir, exist_ok=True)

    # Experiment Statistics Setup

    r1_loss = torch.tensor(0.0, device=device)
    path_loss = torch.tensor(0.0, device=device)
    path_lengths = torch.tensor(0.0, device=device)
    mean_path_length = 0
    mean_path_length_avg = 0
    loss_dict = {}

    accum = 0.5 ** (32 / (10 * 1000))

    sample_z = torch.load('noise.pth').to(device)

    for iter_idx in range(args.start_iter, args.iter):

        real_img = next(loader).to(device)
        real_img.requires_grad_()

        requires_grad(generator, False)
        requires_grad(discriminator, True)

        # Use GAN loss to train the discriminator
        if iter_idx >= args.g_step:

            noise = mixing_noise(args.batch, args.latent, args.mixing, device)
            fake_img = generator(noise)
            real_pred = discriminator(real_img)
            fake_pred = discriminator(fake_img)
            d_loss = d_logistic_loss(real_pred, fake_pred)

            loss_dict['d'] = d_loss
            loss_dict['real_score'] = real_pred.mean()
            loss_dict['fake_score'] = fake_pred.mean()

            # Discriminator regularization
            if iter_idx % args.d_reg_every == 0:

                r1_loss = d_r1_loss(real_pred, real_img)
                loss_dict['r1'] = r1_loss

                d_reg_loss = args.r1 / 2 * r1_loss * args.d_reg_every + 0 * real_pred[0]
                d_loss = d_loss + d_reg_loss

            d_optim.zero_grad()
            d_loss.backward()
            d_optim.step()

        # Use GAN loss to train the generator
        G_Loss_BackProp(generator, discriminator, args, loss_dict, g_optim, teacher_g, percept_loss, parsing_net, vectors, iter_idx, device)

        # Generator regularization
        if iter_idx % args.g_reg_every == 0 and (iter_idx >= args.g_step):
            path_loss, path_lengths, mean_path_length, mean_path_length_avg = G_Reg_BackProp(generator, args, mean_path_length, g_optim)

            loss_dict['path'] = path_loss
            loss_dict['path_length'] = path_lengths.mean()
        time3 = time.time()

        accumulate(g_ema, generator.module, accum)

        loss_reduced = reduce_loss_dict(loss_dict)

        if iter_idx > args.g_step + 20:
            d_loss_val = loss_reduced['d'].mean().item()
            g_loss_val = loss_reduced['g'].mean().item()
            r1_val = loss_reduced['r1'].mean().item()
            path_loss_val = loss_reduced['path'].mean().item()
            real_score_val = loss_reduced['real_score'].mean().item()
            fake_score_val = loss_reduced['fake_score'].mean().item()
            path_length_val = loss_reduced['path_length'].mean().item()
        else:
            d_loss_val = 0
            g_loss_val = 0
            r1_val = 0
            path_loss_val = 0
            real_score_val = 0
            fake_score_val = 0
            path_length_val = 0

        kd_l1_loss_val = loss_reduced['kd_l1_loss'].mean().item()
        kd_lpips_loss_val = loss_reduced['kd_lpips_loss'].mean().item()
        kd_simi_loss_val = loss_reduced['kd_simi_loss'].mean().item()
        kd_style_loss_val = loss_reduced['kd_style_loss'].mean().item()

        if iter_idx % 10 == 0:
            if args.local_rank == 0:
                logger.add_scalar('train/D_loss', round(d_loss_val,3), iter_idx)
                logger.add_scalar('train/G_loss', round(g_loss_val,3), iter_idx)
                logger.add_scalar('train/KD_L1_loss', round(kd_l1_loss_val,3), iter_idx)
                logger.add_scalar('train/KD_LPIPS_loss', round(kd_lpips_loss_val,3), iter_idx)
                logger.add_scalar('train/KD_SIMI_loss', round(kd_simi_loss_val,3), iter_idx)
                logger.add_scalar('train/KD_style_loss', round(kd_style_loss_val,3), iter_idx)
                logger.add_scalar('train/D_reg', round(r1_val,3), iter_idx)
                logger.add_scalar('train/G_reg', round(path_loss_val,3), iter_idx)
                logger.add_scalar('train/G_mean_path', round(mean_path_length_avg,4), iter_idx)

        if iter_idx % args.val_sample_freq == 0:
            with torch.no_grad():
                sample = g_ema([sample_z])
                t_sample = teacher_g([sample_z])
                if args.local_rank == 0:
                    utils.save_image(
                        sample,
                        sample_dir + f'{str(iter_idx).zfill(6)}.png',
                        nrow=int(args.n_sample ** 0.5),
                        normalize=True,
                        range=(-1, 1),
                    )
                    utils.save_image(t_sample, osp.join(exp_dir, 'teacher.jpg'), \
                            nrow=int(args.n_sample ** 0.5), \
                            normalize=True, range=(-1,1))

        if (iter_idx % args.model_save_freq == 0) and (iter_idx > 0):
            with torch.no_grad():
                g_ema_fid = Get_Model_FID_Score(generator=g_ema, \
                    batch_size=args.fid_batch, num_sample=args.fid_n_sample,
                    device=device, gpu_device_ids=[args.local_rank], info_print=False)

            if args.local_rank == 0:
                logger.add_scalar('val/FID', g_ema_fid, iter_idx)
                torch.save(
                    {
                        'g': generator.module.state_dict(),
                        'd': discriminator.module.state_dict(),
                        'g_ema': g_ema.state_dict(),
                    },
                    ckpt_dir + f'{str(iter_idx).zfill(6)}.pt'
                )

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='/kaggle/input/ffhq-256x256')
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--ckpt', type=str, default='550000.pt')
    parser.add_argument('--channel_multiplier', type=int, default=2)
    parser.add_argument('--latent', type=int, default=512)
    parser.add_argument('--n_mlp', type=int, default=8)

    parser.add_argument('--iter', type=int, default=450001)
    parser.add_argument('--lr', type=float, default=0.002)
    parser.add_argument('--r1', type=float, default=10)
    parser.add_argument('--path_regularize', type=float, default=2)
    parser.add_argument('--path_batch_shrink', type=int, default=2)
    parser.add_argument('--d_reg_every', type=int, default=16)
    parser.add_argument('--g_reg_every', type=int, default=4)
    parser.add_argument('--mixing', type=float, default=0.9)

    parser.add_argument('--n_sample', type=int, default=25)
    parser.add_argument('--val_sample_freq', type=int, default=1000)
    parser.add_argument('--model_save_freq', type=int, default=10000)
    parser.add_argument('--fid_n_sample', type=int, default=50000)
    parser.add_argument('--fid_batch', type=int, default=32)

    parser.add_argument('--teacher_ckpt', type=str, default='550000.pt')
    parser.add_argument('--kd_mode', type=str, default='Output_Only')
    parser.add_argument('--content_aware_KD', action='store_false')
    parser.add_argument('--lpips_image_size', type=int, default=256)

    parser.add_argument('--mimic-layer', type=int, nargs='*', default=[2,3,4,5])
    parser.add_argument('--kd_l1_lambda', type=float, default=0)
    parser.add_argument('--kd_lpips_lambda', type=float, default=0)
    parser.add_argument('--kd_simi_lambda', type=float, default=0)
    parser.add_argument('--kd_style_lambda', type=float, default=0.0)

    parser.add_argument('--name', type=str)
    parser.add_argument('--load', type=int)
    parser.add_argument('--load_style', type=int)
    parser.add_argument('--g_step', type=int)
    parser.add_argument('--batch', type=int)
    parser.add_argument('--worker', type=int)
    parser.add_argument('--start_iter', type=int, default=0)
    parser.add_argument('--kernel_size', type=int, default=3)

    parser.add_argument('--lr_mlp', type=float, default=0.01)
    parser.add_argument('--fix_w', type=int)

    parser.add_argument('--simi_loss', type=str, choices=['mse', 'kl'], default='mse')
    parser.add_argument('--single_view', type=int, default=0)
    parser.add_argument('--offset_mode', type=str, choices=['random', 'main'], default='main')
    parser.add_argument('--main_direction', type=str, choices=['split', 'global'], default='split')
    parser.add_argument('--offset_weight', type=float, default=5.0)

    parser.add_argument('--mlp_cfg', type=int, nargs='*', default=None)
    parser.add_argument('--mlp_loss', type=str, default='L1')
    parser.add_argument('--mlp_pretrain', action='store_true')

    parser.add_argument('--local_rank', type=int, )

    args = parser.parse_args()
    n_gpu = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = False

    device = 'cuda'

    # ============================== Setting All Hyperparameters ==============================
    g_reg_ratio = args.g_reg_every / (args.g_reg_every + 1)
    d_reg_ratio = args.d_reg_every / (args.d_reg_every + 1)

    # ============================== Building Dataset ==============================
    transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize(args.size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)])

    train_dataset = FFHQ_Dataset(args.path, transform)
    loader = data.DataLoader(
            train_dataset,
            num_workers=args.worker,
            batch_size = args.batch,
            sampler=data_sampler(train_dataset, shuffle=True, distributed=args.distributed),
            drop_last=True,
            pin_memory=True
    )

    # ============================== Building Network Model ==============================

    # Building target compressed model
    discriminator = Discriminator(args.size, \
            channel_multiplier=args.channel_multiplier).to(device)
    ckpt = torch.load(args.ckpt, map_location=lambda storage, loc: storage)
    generator = Build_Generator_From_Dict(ckpt['g'], size=args.size, \
                    lr_mlp=args.lr_mlp, load=args.load, kernel_size=args.kernel_size).to(device)
    g_ema = Build_Generator_From_Dict(ckpt['g_ema'], size=args.size, \
                    lr_mlp=args.lr_mlp, load=args.load, kernel_size=args.kernel_size).to(device)
    if args.load:
        discriminator.load_state_dict(ckpt['d'])
    else:
        if args.load_style:
            tmp = {k[6:]:v for k,v in ckpt['g'].items() if 'style' in k}
            generator.style.load_state_dict(tmp)
        accumulate(g_ema, generator, 0)
    args.n_latent = g_ema.n_latent

    # replace style module
    if args.mlp_cfg is not None:
        style = []
        tmp = args.latent
        for channel in args.mlp_cfg:
            style.append(EqualLinear(in_dim=tmp, out_dim=channel, activation='fused_lrelu'))
            tmp = channel
        style = nn.Sequential(*style).to(device)
        generator.style = style

        style = []
        tmp = args.latent
        for channel in args.mlp_cfg:
            style.append(EqualLinear(in_dim=tmp, out_dim=channel, activation='fused_lrelu'))
            tmp = channel
        style = nn.Sequential(*style).to(device)
        g_ema.style = style

    g_ema.eval()

    # Building the teacher model
    teacher = torch.load(args.teacher_ckpt, map_location=lambda storage, loc: storage)
    teacher_g = Build_Generator_From_Dict(teacher['g_ema'], size=args.size).to(device)
    teacher_g.eval()
    requires_grad(teacher_g, False)

    # calc main direction
    vectors = eval(f'calc_direction_{args.main_direction}')(teacher_g, args)

    # LPIPS KD
    percept_loss = lpips.PerceptualLoss(model='net-lin', net='vgg', \
                    use_gpu=True, gpu_ids=[args.local_rank])

    # Content aware KD
    parsing_net, _ = Get_Parsing_Net(device)

    # Parallelize the model
    generator = DDP(generator, device_ids=[args.local_rank], output_device=args.local_rank, broadcast_buffers=False)
    discriminator = DDP(discriminator, device_ids=[args.local_rank], output_device=args.local_rank, broadcast_buffers=False)

    # ============================== Initializing Optimizers ==============================
    if args.fix_w:
        tmp = [{'params':generator.style.parameters(), 'lr':0},
                {'params':generator.input.parameters()},
                {'params':generator.conv1.parameters()},
                {'params':generator.to_rgb1.parameters()},
                {'params':generator.convs.parameters()},
                {'params':generator.to_rgbs.parameters()}]
        g_optim = optim.Adam(
            tmp,
            lr=args.lr * g_reg_ratio,
            betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio),
        )
    else:
        g_optim = optim.Adam(
            generator.parameters(),
            lr=args.lr * g_reg_ratio,
            betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio),
        )
    d_optim = optim.Adam(
        discriminator.parameters(),
        lr=args.lr * d_reg_ratio,
        betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
    )

    # ============================== Training Start ==============================

    # Experiment Saving Directory
    exp_dir = f'experiments/{args.name}'
    if args.local_rank == 0:
        os.makedirs(exp_dir, exist_ok=True)
        os.makedirs(osp.join(exp_dir, 'events'), exist_ok=True)
        logger = SummaryWriter(osp.join(exp_dir, 'events'))
    else:
        logger = None

    # train MLP
    if args.mlp_pretrain:
        optimizer = optim.Adam(generator.style.parameters(), lr=0.05)
        for idx in range(50000):
            z = torch.randn(4096, args.latent).to(device)
            with torch.no_grad():
                wt = teacher_g.style(z)
            ws = generator.style(z)
            if args.mlp_loss == 'L2':
                loss = F.mse_loss(ws, wt)
            elif args.mlp_loss == 'L1':
                loss = F.l1_loss(ws, wt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if logger is not None:
                logger.add_scalar('MLP_loss', loss.item(), idx)
        accumulate(g_ema, generator, 0)

    train(args, loader, generator, discriminator, g_optim, d_optim, g_ema, teacher_g, percept_loss, parsing_net, exp_dir, logger, vectors, device)

