import argparse
import datetime
import logging
import math
import random
import time
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
#os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32' 
#os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512,expandable_segments:True'
import torch
#torch.cuda.set_per_process_memory_fraction(0.87)
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
#torch.cuda.memory.set_per_process_memory_fraction(0.87)
from os import path as osp
from tqdm import tqdm
from basicsr.data import create_dataloader, create_dataset
from basicsr.data.data_sampler import EnlargedSampler
from basicsr.data.prefetch_dataloader import CPUPrefetcher, CUDAPrefetcher
from basicsr.models import create_model
from basicsr.utils import (MessageLogger, check_resume, get_env_info,
                           get_root_logger, get_time_str, init_tb_logger,
                           init_wandb_logger, make_exp_dirs, mkdir_and_rename,
                           set_random_seed)
from basicsr.utils.dist_util import get_dist_info, init_dist
from basicsr.utils.options import dict2str, parse

import numpy as np


def parse_options(is_train=True):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-opt', type=str, default='./Dehazing/Options/MB-TaylorFormerV2-Haze4K-L.yml',
        help='Path to option YAML file')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    opt = parse(args.opt, is_train=is_train)

    # distributed settings
    if args.launcher == 'none':
        opt['dist'] = False
        print('Disable distributed.', flush=True)
    else:
        opt['dist'] = True
        if args.launcher == 'slurm' and 'dist_params' in opt:
            init_dist(args.launcher, **opt['dist_params'])
        else:
            init_dist(args.launcher)
            print('init dist .. ', args.launcher)

    opt['rank'], opt['world_size'] = get_dist_info()

    # random seed
    seed = opt.get('manual_seed')
    if seed is None:
        seed = random.randint(1, 10000)
        opt['manual_seed'] = seed
    set_random_seed(seed + opt['rank'])

    return opt


def init_loggers(opt):
    log_file = osp.join(opt['path']['log'],
                        f"train_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(
        logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(opt))

    # initialize wandb logger before tensorboard logger to allow proper sync:
    if (opt['logger'].get('wandb')
            is not None) and (opt['logger']['wandb'].get('project')
                              is not None) and ('debug' not in opt['name']):
        assert opt['logger'].get('use_tb_logger') is True, (
            'should turn on tensorboard when using wandb')
        init_wandb_logger(opt)
    tb_logger = None
    if opt['logger'].get('use_tb_logger') and 'debug' not in opt['name']:
        tb_logger = init_tb_logger(log_dir=osp.join('tb_logger', opt['name']))
    return logger, tb_logger


def create_train_val_dataloader(opt, logger):
    # create train and val dataloaders
    train_loader, val_loader = None, None
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            dataset_enlarge_ratio = dataset_opt.get('dataset_enlarge_ratio', 1)
            train_set = create_dataset(dataset_opt)
            train_sampler = EnlargedSampler(train_set, opt['world_size'],
                                            opt['rank'], dataset_enlarge_ratio)
            train_loader = create_dataloader(
                train_set,
                dataset_opt,
                num_gpu=opt['num_gpu'],
                dist=opt['dist'],
                sampler=train_sampler,
                seed=opt['manual_seed'])

            num_iter_per_epoch = math.ceil(
                len(train_set) * dataset_enlarge_ratio /
                (dataset_opt['batch_size_per_gpu'] * opt['world_size']))
            total_iters = int(opt['train']['total_iter'])
            total_epochs = math.ceil(total_iters / (num_iter_per_epoch))
            logger.info(
                'Training statistics:'
                f'\n\tNumber of train images: {len(train_set)}'
                f'\n\tDataset enlarge ratio: {dataset_enlarge_ratio}'
                f'\n\tBatch size per gpu: {dataset_opt["batch_size_per_gpu"]}'
                f'\n\tWorld size (gpu number): {opt["world_size"]}'
                f'\n\tRequire iter number per epoch: {num_iter_per_epoch}'
                f'\n\tTotal epochs: {total_epochs}; iters: {total_iters}.')

        elif phase == 'val':
            val_set = create_dataset(dataset_opt)
            #val_set = create_dataset_with_crop(dataset_opt, crop_region=(100, 100, 1950, 2048))#use for OHAZE
            val_loader = create_dataloader(
                val_set,
                dataset_opt,
                num_gpu=opt['num_gpu'],
                dist=opt['dist'],
                sampler=None,
                seed=opt['manual_seed'])
            logger.info(
                f'Number of val images/folders in {dataset_opt["name"]}: '
                f'{len(val_set)}')
        else:
            raise ValueError(f'Dataset phase {phase} is not recognized.')

    return train_loader, train_sampler, val_loader, total_epochs, total_iters
def create_dataset_with_crop(dataset_opt, crop_region):
    
    dataset_type = dataset_opt.get('type', 'SingleImageDataset')
    
    if dataset_type == 'PairedImageDataset':
        from basicsr.data.paired_image_dataset import Dataset_PairedImage
        dataset = Dataset_PairedImage(dataset_opt)
      
        original_getitem = dataset.__getitem__
        def new_getitem(index):
            data = original_getitem(index)
            if 'lq' in data:
                data['lq'] = data['lq'][:, crop_region[0]:crop_region[2], crop_region[1]:crop_region[3]]
            if 'gt' in data:
                data['gt'] = data['gt'][:, crop_region[0]:crop_region[2], crop_region[1]:crop_region[3]]
            return data
        dataset.__getitem__ = new_getitem
    else:
      
        dataset = create_dataset(dataset_opt)
    
    return dataset

def main():
    # parse options, set distributed setting, set ramdom seed
    opt = parse_options(is_train=True)

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    # automatic resume ..
    state_folder_path = '/home/ubuntu/zengmaowei/MB-TaylorFormerV2-main/MB-TaylorFormerV2-main/experiments/{}/training_states/'.format(opt['name'])
    import os
    try:
        states = os.listdir(state_folder_path)
    except:
        states = []
    #print(states,"11111111111111111111111111")

    resume_state = None
    #print("USE GPU:", torch.cuda.get_device_name(0))
    if len(states) > 0:
        max_state_file = '{}.state'.format(max([int(x[0:-6]) for x in states]))
        resume_state = os.path.join(state_folder_path, max_state_file)
        opt['path']['resume_state'] = resume_state

    # load resume states if necessary
    if opt['path'].get('resume_state'):
        # device_id = torch.cuda.current_device()
        # resume_state = torch.load(
        #     opt['path']['resume_state'],
        #     map_location=lambda storage, loc: storage.cuda(device_id))
        
        map_location = 'cuda:{}'.format(torch.cuda.current_device()) if torch.cuda.is_available() else 'cpu'
        resume_state = torch.load(opt['path']['resume_state'], map_location=map_location)
    else:
        resume_state = None

    # mkdir for experiments and logger
    if resume_state is None:
        make_exp_dirs(opt)
        if opt['logger'].get('use_tb_logger') and 'debug' not in opt[
                'name'] and opt['rank'] == 0:
            mkdir_and_rename(osp.join('tb_logger', opt['name']))

    # initialize loggers
    logger, tb_logger = init_loggers(opt)

    # create train and validation dataloaders
    result = create_train_val_dataloader(opt, logger)

    train_loader, train_sampler, val_loader, total_epochs, total_iters = result

    # create model

    if resume_state:  # resume training
        check_resume(opt, resume_state['iter'])
        model = create_model(opt)

        model.resume_training(resume_state)  # handle optimizers and schedulers
        logger.info(f"Resuming training from epoch: {resume_state['epoch']}, "
                    f"iter: {resume_state['iter']}.")
        start_epoch = resume_state['epoch']
        current_iter = resume_state['iter']
    else:
        model = create_model(opt)

        start_epoch = 0
        current_iter = 0

    # create message logger (formatted outputs)
    msg_logger = MessageLogger(opt, current_iter, tb_logger)
    
    # dataloader prefetcher
    prefetch_mode = opt['datasets']['train'].get('prefetch_mode')
    if prefetch_mode is None or prefetch_mode == 'cpu':
        prefetcher = CPUPrefetcher(train_loader)
    elif prefetch_mode == 'cuda':
        prefetcher = CUDAPrefetcher(train_loader, opt)
        logger.info(f'Use {prefetch_mode} prefetch dataloader')
        if opt['datasets']['train'].get('pin_memory') is not True:
            raise ValueError('Please set pin_memory=True for CUDAPrefetcher.')
    else:
        raise ValueError(f'Wrong prefetch_mode {prefetch_mode}.'
                         "Supported ones are: None, 'cuda', 'cpu'.")

    # training
    logger.info(
        f'Start training from epoch: {start_epoch}, iter: {current_iter}')
    data_time, iter_time = time.time(), time.time()
    start_time = time.time()

    # for epoch in range(start_epoch, total_epochs + 1):

    iters = opt['datasets']['train'].get('iters')
    batch_size = opt['datasets']['train'].get('batch_size_per_gpu')
    mini_batch_sizes = opt['datasets']['train'].get('mini_batch_sizes')
    gt_size = opt['datasets']['train'].get('gt_size')
    mini_gt_sizes = opt['datasets']['train'].get('gt_sizes')

    groups = np.array([sum(iters[0:i + 1]) for i in range(0, len(iters))])

    logger_j = [True] * len(groups)

    scale = opt['scale']

    epoch = start_epoch
    initial_iter = current_iter if resume_state else 0
    total_pbar = tqdm(
    total=total_iters, 
    desc='Total Progress', 
    position=0,
    initial=initial_iter  
)
    best_psnr=0
    while current_iter <= total_iters:
        epoch_start_time = time.time()
        train_sampler.set_epoch(epoch)
        prefetcher.reset()
        
        train_data = prefetcher.next()

        while train_data is not None:
            data_time = time.time() - data_time

            current_iter += 1
            if current_iter > total_iters:
                break
            # update learning rate
            model.update_learning_rate(
                current_iter, warmup_iter=opt['train'].get('warmup_iter', -1))

            
            ### ------Progressive learning ---------------------
            j = ((current_iter>groups) !=True).nonzero()[0]
            if len(j) == 0:
                bs_j = len(groups) - 1
            else:
                bs_j = j[0]

            mini_gt_size = mini_gt_sizes[bs_j]
            mini_batch_size = mini_batch_sizes[bs_j]
            
            if logger_j[bs_j]:
                logger.info('\n Updating Patch_Size to {} and Batch_Size to {} \n'.format(mini_gt_size, mini_batch_size*torch.cuda.device_count())) 
                logger_j[bs_j] = False

            lq = train_data['lq']
            gt = train_data['gt']
            name = train_data['lq_path']
            

            if mini_batch_size < batch_size:
                indices = random.sample(range(0, batch_size), k=mini_batch_size)
                lq = lq[indices]
                gt = gt[indices]

            if mini_gt_size < gt_size:
                x0 = int((gt_size - mini_gt_size) * random.random())
                y0 = int((gt_size - mini_gt_size) * random.random())
                x1 = x0 + mini_gt_size
                y1 = y0 + mini_gt_size
                lq = lq[:,:,x0:x1,y0:y1]
                gt = gt[:,:,x0*scale:x1*scale,y0*scale:y1*scale]
            ###-------------------------------------------

           
            model.feed_train_data({'lq': lq, 'gt':gt},name)
            
            model.optimize_parameters(current_iter)
            if tb_logger is not None and (current_iter % 5000 == 0):
                    net = model.get_bare_model(model.net_g)
                    for name, p in net.named_parameters():
                        tb_logger.add_histogram(f'weights/{name}', p.detach().float().cpu(), current_iter)
                        if p.grad is not None:
                            tb_logger.add_histogram(f'grads/{name}', p.grad.detach().float().cpu(), current_iter)
            iter_time = time.time() - iter_time
            # log
            if current_iter % opt['logger']['print_freq'] == 0:
                log_vars = {'epoch': epoch, 'iter': current_iter}
                log_vars.update({'lrs': model.get_current_learning_rate()})
                log_vars.update({'time': iter_time, 'data_time': data_time})
                log_vars.update(model.get_current_log())
                msg_logger(log_vars)
                if tb_logger is not None:
                 tb_logger.add_scalar('lr', model.get_current_learning_rate()[0], current_iter)
            # save models and training states
            if current_iter % opt['logger']['save_checkpoint_freq'] == 0:
                logger.info('Saving models and training states.')
                model.save(epoch, current_iter)

            # validation
            if opt.get('val') is not None and (current_iter %
                                               opt['val']['val_freq'] == 0):
                rgb2bgr = opt['val'].get('rgb2bgr', True)
                # wheather use uint8 image to compute metrics
                use_image = opt['val'].get('use_image', True)
                psnr=model.validation(val_loader, current_iter, tb_logger,
                                 opt['val']['save_img'], rgb2bgr, use_image )
                if(best_psnr<psnr):
                    best_psnr=psnr
                    model.save(epoch=epoch, current_iter=current_iter)
                logger.info(f'epoch:{epoch}/{total_epochs}  best: {best_psnr:.4f} lr: {model.get_current_learning_rate()[0]:.2e}')
            data_time = time.time()
            iter_time = time.time()
            train_data = prefetcher.next()
            total_pbar.update(1)
            l_g_pix_value = model.get_current_log().get('l_g_pix', 'N/A')
            l_pix = model.log_dict.get('l_pix', 0)
            l_fft = model.log_dict.get('l_fft', 0)
            l_ca = model.log_dict.get('l_ca', 0)         
            total_loss = l_pix+ l_fft + l_ca
            total_pbar.set_postfix({
               'epoch': f'{epoch}/{total_epochs}',
               'loss_pix': l_pix if l_pix == 'N/A' else f"{float(l_pix):.4f}",
               'loss_fft': l_fft if l_fft == 'N/A' else f"{float(l_fft):.4f}",
               'loss_ca': l_ca if l_ca == 'N/A' else f"{float(l_ca):.4f}",
               'lr': f"{model.get_current_learning_rate()[0]:.2e}"
                 })
        # end of iter
        epoch += 1
        rgb2bgr = opt['val'].get('rgb2bgr', True)
                # wheather use uint8 image to compute metrics
        use_image = opt['val'].get('use_image', True)
        psnr=model.validation(val_loader, current_iter, tb_logger,opt['val']['save_img'], rgb2bgr, use_image )
        if(best_psnr<psnr):
                    best_psnr=psnr
                    model.save(epoch=epoch, current_iter=current_iter)
        #use for densehaze            
        #if epoch > 1500 or epoch % 100 == 0:
          #model.save(epoch=epoch, current_iter=current_iter)
        model.save(epoch=epoch, current_iter=-1)
        logger.info(f'epoch:{epoch}/{total_epochs}  best: {best_psnr:.4f} lr: {model.get_current_learning_rate()[0]:.2e}')
    total_pbar.close()
        # epoch_time=time.time()-epoch_start_time
        # total_time=time.time()-start_time
        # logger.info(f'epoch:{epoch}.Epoch time:{str(datetime.timedelta(seconds=int(epoch_time)))},'
        #             f'total time:{str(datetime.timedelta(seconds=int(total_time)))}')


    # end of epoch

    consumed_time = str(
        datetime.timedelta(seconds=int(time.time() - start_time)))
    logger.info(f'End of training. Time consumed: {consumed_time}')
    #logger.info('Save the latest model.')
    model.save(epoch=-1, current_iter=-1)  # -1 stands for the latest
    #if opt.get('val') is not None:
     #   model.validation(val_loader, current_iter, tb_logger,
      #                   opt['val']['save_img'])

    if tb_logger:
        tb_logger.close()


if __name__ == '__main__':
    main()
