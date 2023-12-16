"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from utils import get_all_data_loaders, prepare_sub_folder, write_html, write_loss, get_config, write_2images, Timer
import argparse
from torch.autograd import Variable
from trainer import MUNIT_Trainer, UNIT_Trainer
import torch.backends.cudnn as cudnn
import torch
try:
    from itertools import izip as zip
except ImportError: # will be 3.x series
    pass
import os
import sys
import tensorboardX
import shutil
# linux与win的不同， win需要加上main函数的接口
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/edges2handbags_folder.yaml', help='Path to the config file.')
    parser.add_argument('--output_path', type=str, default='.', help="outputs path")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument('--trainer', type=str, default='MUNIT', help="MUNIT|UNIT")
    opts = parser.parse_args()

    cudnn.benchmark = True

    # Load experiment setting，获取环境配置
    config = get_config(opts.config)
    # 最大迭代次数
    max_iter = config['max_iter']
    # 显示图片大小
    display_size = config['display_size']
    # vgg模型的路径
    config['vgg_model_path'] = opts.output_path

    # Setup model and data loader，根据配置创建模型
    if opts.trainer == 'MUNIT':
        trainer = MUNIT_Trainer(config)
    elif opts.trainer == 'UNIT':
        trainer = UNIT_Trainer(config)
    else:
        sys.exit("Only support MUNIT|UNIT")
    trainer.cuda()

    # 创建训练以及测试得数据迭代器,同时取出对每个迭代器取出display_size张图片，水平拼接到一起，
    # 后续会一直拿这些图片作为生成图片的演示，当作一个标本即可
    train_loader_a, train_loader_b, test_loader_a, test_loader_b = get_all_data_loaders(config)
    train_display_images_a = torch.stack([train_loader_a.dataset[i] for i in range(display_size)]).cuda()
    train_display_images_b = torch.stack([train_loader_b.dataset[i] for i in range(display_size)]).cuda()
    test_display_images_a = torch.stack([test_loader_a.dataset[i] for i in range(display_size)]).cuda()
    test_display_images_b = torch.stack([test_loader_b.dataset[i] for i in range(display_size)]).cuda()

    # Setup logger and output folders, 设置打印信息以及输出目录
    # 获得模型的名字
    model_name = os.path.splitext(os.path.basename(opts.config))[0]
    # 创建一个 tensorboardX，记录训练过程中的信息
    train_writer = tensorboardX.SummaryWriter(os.path.join(opts.output_path + "/logs", model_name))
    # 准备并且创建好输出目录，同时拷贝对应的config.yaml文件
    output_directory = os.path.join(opts.output_path + "/outputs", model_name)
    checkpoint_directory, image_directory = prepare_sub_folder(output_directory)
    shutil.copy(opts.config, os.path.join(output_directory, 'config.yaml')) # copy config file to output folder

    # Start training，开始训练模型，如果设置opts.resume=Ture,表示接着之前得训练
    iterations = trainer.resume(checkpoint_directory, hyperparameters=config) if opts.resume else 0
    while True:
        # 获取训练数据
        for it, (images_a, images_b) in enumerate(zip(train_loader_a, train_loader_b)):
            # 更新学习率
            trainer.update_learning_rate()
            # 指定数据存储计算的设备
            images_a, images_b = images_a.cuda().detach(), images_b.cuda().detach()

            with Timer("Elapsed time in update: %f"):
                # Main training code，主要的训练代码
                trainer.dis_update(images_a, images_b, config)
                trainer.gen_update(images_a, images_b, config)
                torch.cuda.synchronize()

            # Dump training stats in log file，记录训练过程中的信息
            if (iterations + 1) % config['log_iter'] == 0:
                print("Iteration: %08d/%08d" % (iterations + 1, max_iter))
                write_loss(iterations, trainer, train_writer)

            # Write images，到达指定次数后，把生成的样本图片写入到输出文件夹，方便观察生成效果，重新保存
            if (iterations + 1) % config['image_save_iter'] == 0:
                with torch.no_grad():
                    test_image_outputs = trainer.sample(test_display_images_a, test_display_images_b)
                    train_image_outputs = trainer.sample(train_display_images_a, train_display_images_b)
                write_2images(test_image_outputs, display_size, image_directory, 'test_%08d' % (iterations + 1))
                write_2images(train_image_outputs, display_size, image_directory, 'train_%08d' % (iterations + 1))
                # HTML
                write_html(output_directory + "/index.html", iterations + 1, config['image_save_iter'], 'images')

            # Write images，到达指定次数后，把生成的样本图片写入到输出文件夹，方便观察生成效果，覆盖上一次结果
            if (iterations + 1) % config['image_display_iter'] == 0:
                with torch.no_grad():
                    image_outputs = trainer.sample(train_display_images_a, train_display_images_b)
                write_2images(image_outputs, display_size, image_directory, 'train_current')

            # Save network weights，保存训练的模型
            if (iterations + 1) % config['snapshot_save_iter'] == 0:
                trainer.save(checkpoint_directory, iterations)

            # 如果超过最大迭代次数，则退出训练
            iterations += 1
            if iterations >= max_iter:
                sys.exit('Finish training')

