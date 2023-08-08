import numpy as np
import torch
import os
import sys
import ntpath
import time
import torchvision
from PIL import Image
from . import util
from torch.utils.tensorboard import SummaryWriter
writer =  SummaryWriter('logdir')


def save_images(visuals, aspect_ratio=1.0, width=256):
    image_dir = os.path.join('')
    idx = 0
    for img, label in visuals.items():
        img.thumbnail((width,width), Image.BICUBIC)
        img_name = '%s.png' % label
        img.save(img_name)


class Visualizer():
    def __init__(self, opt):
        self.opt = opt  # cache the option
        #self.display_id = opt.display_id
        #self.use_html = opt.isTrain and not opt.no_html
        #self.win_size = opt.display_winsize
        self.name = opt.name
        self.port = opt.display_port
        self.saved = False
        #self.use_wandb = opt.use_wandb
        #self.wandb_project_name = opt.wandb_project_name
        self.current_epoch = 0
        self.ncols = opt.display_ncols
        self.writer = SummaryWriter('logdir')

    def display_current_results(self, visuals, epoch, save_result):
        ncols = self.ncols
        if ncols > 0:
            ncols = min(ncols, len(visuals))
            h, w = next(iter(visuals.values())).shape[:2]
            title = self.name
            idx = 0
            for label, image in visuals.items():

                image = torch.squeeze(image)


                if idx%8 == 0:
                    img_grid_realA = torchvision.utils.make_grid(image)
                    writer.add_image('realA image', img_grid_realA, global_step=idx + ncols * epoch)

                elif idx%8 == 1:
                    img_grid_fakeB = torchvision.utils.make_grid(image)
                    writer.add_image('fakeB image', img_grid_fakeB, global_step=idx + ncols * epoch)

                elif idx%8 == 2:
                    img_grid_recA = torchvision.utils.make_grid(image)

                elif idx%8 == 3:
                    img_grid_idtB = torchvision.utils.make_grid(image)

                elif idx%8 == 4:
                    img_grid_realB = torchvision.utils.make_grid(image)
                    writer.add_image('realB image', img_grid_realB, global_step=idx + ncols * epoch)

                elif idx%8 == 5:
                    img_grid_fakeA = torchvision.utils.make_grid(image)
                    writer.add_image('fakeA image', img_grid_fakeA, global_step=idx + ncols * epoch)

                elif idx%8 == 6:
                    img_grid_recB = torchvision.utils.make_grid(image)

                elif idx%8 == 7:
                    img_grid_idtA = torchvision.utils.make_grid(image)

                idx += 1


    def plot_current_losses(self, epoch, counter_ratio, losses):

        idx = 0
        for label, loss in losses.items():

            if idx%8 == 0:
                writer.add_scalar(label, loss, counter_ratio + epoch)
            elif idx%8 == 1:
                writer.add_scalar(label, loss, counter_ratio + epoch)
            elif idx%8 == 2:
                writer.add_scalar(label, loss, counter_ratio + epoch)
            elif idx%8 == 3:
                writer.add_scalar(label, loss, counter_ratio + epoch)
            elif idx%8 == 4:
                writer.add_scalar(label, loss, counter_ratio + epoch)
            elif idx%8 == 5:
                writer.add_scalar(label, loss, counter_ratio + epoch)
            elif idx%8 == 6:
                writer.add_scalar(label, loss, counter_ratio + epoch)
            elif idx%8 == 7:
                writer.add_scalar(label, loss, counter_ratio + epoch)
            idx += 1
        """for label, loss in losses.items():
            self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B']
            writer.add_scalar('D_A', loss, counter_ratio + epoch)
            writer.add_scalar('G_A', loss, counter_ratio + epoch)
            writer.add_scalar('cycle_A', loss, counter_ratio + epoch)
            writer.add_scalar('idt_A', loss, counter_ratio + epoch)
            writer.add_scalar('D_B', loss, counter_ratio + epoch)
            writer.add_scalar('G_B', loss, counter_ratio + epoch)
            writer.add_scalar('cycle_B', loss, counter_ratio + epoch)
            writer.add_scalar('idt_B', loss, counter_ratio + epoch)
        """

    def reset(self):
        pass
        #writer.close()
        #self.saved = False


    def print_current_losses(self, epoch, iters, losses, t_comp, t_data):
        message = '(epoch: %d, iters: %d, time: %.3f, data: %.3f) ' % (epoch, iters, t_comp, t_data)
        for k, v in losses.items():
            message += '%s: %.3f ' % (k, v)

        print(message)