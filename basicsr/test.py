import logging
import torch
from os import path as osp

from basicsr.data import create_dataloader, create_dataset
from basicsr.models import create_model
from basicsr.train import parse_options
from basicsr.utils import (get_env_info, get_root_logger, get_time_str,
                           make_exp_dirs)
from basicsr.utils.options import dict2str
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
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
    opt = parse_options(is_train=False)

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    # mkdir and initialize loggers
    make_exp_dirs(opt)
    log_file = osp.join(opt['path']['log'],
                        f"test_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(
        logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(opt))

    # create test dataset and dataloader
    test_loaders = []
    for phase, dataset_opt in sorted(opt['datasets'].items()):
        test_set = create_dataset(dataset_opt)#ԭʼ
        #test_set = create_dataset_with_crop(dataset_opt, crop_region=(100, 100, 1950, 2048))# use for OHAZE
        # 每边缩小300像素
        # 每边缩小400像素
        #test_set = create_dataset_with_crop(dataset_opt, crop_region=(500, 500, 1550, 1648))

        test_loader = create_dataloader(
            test_set,
            dataset_opt,
            num_gpu=opt['num_gpu'],
            dist=opt['dist'],
            sampler=None,
            seed=opt['manual_seed'])
        logger.info(
            f"Number of test images in {dataset_opt['name']}: {len(test_set)}")
        test_loaders.append(test_loader)

    # create model
    model = create_model(opt)

    for test_loader in test_loaders:
        test_set_name = test_loader.dataset.opt['name']
        logger.info(f'Testing {test_set_name}...')
        rgb2bgr = opt['val'].get('rgb2bgr', True)
        # wheather use uint8 image to compute metrics
        use_image = opt['val'].get('use_image', True)
        model.validation(
            test_loader,
            current_iter=opt['name'],
            tb_logger=None,
            save_img=opt['val']['save_img'],
            rgb2bgr=rgb2bgr, use_image=use_image)


if __name__ == '__main__':
    main()
