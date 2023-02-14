import time
from options.train_options import TrainOptions
from data import DataLoaderWrapper
from models import create_model
from util.util import confusion_matrix, get_scores, tensor2labelim, tensor2im, print_current_losses
import numpy as np
import random
import torch
import cv2
from tensorboardX import SummaryWriter

if __name__ == '__main__':
    train_opt = TrainOptions()

    np.random.seed(train_opt.seed)
    random.seed(train_opt.seed)
    torch.manual_seed(train_opt.seed)
    torch.cuda.manual_seed(train_opt.seed)

    train_data_loader = DataLoaderWrapper(train_opt)
    train_dataset = train_data_loader
    train_dataset_size = len(train_data_loader)
    print('Training images count: %d' % train_dataset_size)

    valid_opt = TrainOptions()
    valid_opt.phase = 'val'
    valid_opt.threads = 1
    valid_opt.serial_batches = True
    valid_opt.train = False
    valid_data_loader = DataLoaderWrapper(valid_opt)
    valid_dataset = valid_data_loader
    valid_dataset_size = len(valid_data_loader)
    print('Validation images count: %d' % valid_dataset_size)

    writer = SummaryWriter()

    model = create_model(train_opt, train_dataset.dataset)
    model.setup()
    total_steps = 0
    tfcount = 0
    F_score_max = 0
    for epoch in range(1, train_opt.epoch_count + 1):
        model.train()
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0
        train_loss_iter = []
        for i, data in enumerate(train_dataset):
            iter_start_time = time.time()
            if total_steps % train_opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            total_steps += train_opt.batch_size
            epoch_iter += train_opt.batch_size
            model.set_input(data)
            model.optimize_parameters()

            if total_steps % train_opt.print_freq == 0:
                tfcount = tfcount + 1
                losses = model.get_current_losses()
                train_loss_iter.append(losses["cross_entropy"])
                t = (time.time() - iter_start_time) / train_opt.batch_size
                print_current_losses(epoch, epoch_iter, losses, t, t_data)
                # There are several whole_loss values shown in tensorboard in one epoch,
                # to help better see the optimization phase
                writer.add_scalar('train/whole_loss', losses["cross_entropy"], tfcount)

            iter_data_time = time.time()

        mean_loss = np.mean(train_loss_iter)
        writer.add_scalar('train/mean_loss', mean_loss, epoch)

        palette = 'datasets/palette.txt'
        impalette = list(np.genfromtxt(palette, dtype=np.uint8).reshape(3 * 256))
        visuals = model.get_current_visuals()
        rgb = tensor2im(visuals['rgb_image'])
        if train_opt.use_sn:
            another = tensor2im((visuals['another_image'] + 1) / 2)  # color normal images
        else:
            another = tensor2im(visuals['another_image'])
        label = tensor2labelim(visuals['label'], impalette)
        output = tensor2labelim(visuals['output'], impalette)
        image_numpy = np.concatenate((rgb, another, label, output), axis=1)
        image_numpy = image_numpy.astype(np.float64) / 255
        writer.add_image('Epoch' + str(epoch), image_numpy, dataformats='HWC')  # show training images in tensorboard

        print('End of epoch %d / %d \n Time taken: %d sec' % (epoch, train_opt.epoch_count, time.time() - epoch_start_time))
        model.update_learning_rate()

        model.eval()
        valid_loss_iter = []
        epoch_iter = 0
        conf_mat = np.zeros((valid_dataset.dataset.num_labels, valid_dataset.dataset.num_labels), dtype=float)
        with torch.no_grad():
            for i, data in enumerate(valid_dataset):
                model.set_input(data)
                model.forward()
                model.get_loss()
                epoch_iter += valid_opt.batch_size
                gt = model.label.cpu().int().numpy()
                _, pred = torch.max(model.output.data.cpu(), 1)
                pred = pred.float().detach().int().numpy()

                image_size = model.image_orig_size
                orig_size = (image_size[0].item(), image_size[1].item())
                gt = np.expand_dims(cv2.resize(np.squeeze(gt, axis=0), orig_size, interpolation=cv2.INTER_NEAREST),
                                    axis=0)
                pred = np.expand_dims(cv2.resize(np.squeeze(pred, axis=0), orig_size, interpolation=cv2.INTER_NEAREST),
                                      axis=0)

                conf_mat += confusion_matrix(gt, pred, valid_dataset.dataset.num_labels)
                losses = model.get_current_losses()
                valid_loss_iter.append(model.loss_cross_entropy)
                print('Validation epoch {0:}, iterations: {1:}/{2:} '.format(epoch, epoch_iter,
                                                                             len(valid_dataset) * valid_opt.batch_size),
                      end='\r')

        avg_valid_loss = torch.mean(torch.stack(valid_loss_iter))
        global_acc, precision, recall, F_score, iou = get_scores(conf_mat)
        print(
            'Epoch {0:} - Global accuracy: {1:.3f}, Precision: {2:.3f}, Recall: {3:.3f}, F-score: {4:.3f}, '
            'IoU: {5:.3f}'.format(
                epoch, global_acc, precision, recall, F_score, iou))

        writer.add_scalar('valid/loss', avg_valid_loss, epoch)
        writer.add_scalar('valid/global_acc', global_acc, epoch)
        writer.add_scalar('valid/precision', precision, epoch)
        writer.add_scalar('valid/recall', recall, epoch)
        writer.add_scalar('valid/F_score', F_score, epoch)
        writer.add_scalar('valid/iou', iou, epoch)

        if F_score > F_score_max:
            print('Saving the best model so far, at the end of epoch %d, iterations %d' % (epoch, total_steps))
            model.save_network('best')
            F_score_max = F_score
            writer.add_text('best model', str(epoch))
