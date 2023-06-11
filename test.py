import os

from data import DataLoaderWrapper
from options.test_options import TestOptions
from models import create_model
from util import util
from util.util import get_confusion_matrix, get_metrics, save_images
import torch
import numpy as np
import cv2

if __name__ == '__main__':
    opt = TestOptions()
    opt.threads = 1
    opt.batch_size = 1
    opt.serial_batches = True
    opt.isTrain = False

    save_dir = os.path.join(opt.results_root, opt.dataset, 'test_' + opt.epoch)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    data_loader = DataLoaderWrapper(opt)
    dataset = data_loader
    model = create_model(opt, dataset.dataset)
    model.setup()
    model.eval()

    test_loss_iter = []
    epoch_iter = 0
    conf_mat = np.zeros((dataset.dataset.num_labels, dataset.dataset.num_labels), dtype=float)
    with torch.no_grad():
        for i, data in enumerate(dataset):
            model.set_input(data)
            model.forward()
            model.get_loss()
            epoch_iter += opt.batch_size
            gt = model.label.cpu().int().numpy()
            _, pred = torch.max(model.output.data.cpu(), 1)
            pred = pred.float().detach().int().numpy()
            save_images(save_dir, model.get_current_visuals(), model.image_names, model.image_orig_size,
                        opt.prob_map)

            image_size = model.image_orig_size
            orig_size = (image_size[0].item(), image_size[1].item())
            if opt.batch_size == 1:
                gt_res = np.expand_dims(
                    cv2.resize(np.squeeze(gt, axis=0), orig_size, interpolation=cv2.INTER_NEAREST), axis=0)
                pred_res = np.expand_dims(
                    cv2.resize(np.squeeze(pred, axis=0), orig_size, interpolation=cv2.INTER_NEAREST), axis=0)
            else:
                gt_res = np.zeros((opt.batch_size, orig_size[1], orig_size[0]), dtype=int)
                pred_res = np.zeros((opt.batch_size, orig_size[1], orig_size[0]), dtype=int)
                for idx in range(opt.batch_size):
                    gt_img = gt[idx, :, :]
                    gt_img = cv2.resize(gt_img, orig_size, interpolation=cv2.INTER_NEAREST)
                    gt_res[idx, :, :] = gt_img
                    pred_img = pred[idx, :, :]
                    pred_img = cv2.resize(pred_img, orig_size, interpolation=cv2.INTER_NEAREST)
                    pred_res[idx, :, :] = pred_img
            conf_mat += get_confusion_matrix(gt_res, pred_res, dataset.dataset.num_labels)

            test_loss_iter.append(model.loss_cross_entropy)
        avg_test_loss = torch.mean(torch.stack(test_loss_iter))
        message = ''
        message += 'Epoch {0:} -  Test loss: {1:.3f}\n'.format(opt.epoch, avg_test_loss)
        acc, pre, recall, F_score, iou = get_metrics(conf_mat)
        message += 'Epoch {0:} - Global accuracy: {1:.3f}, Precision: {2:.3f}, Recall: {3:.3f}, F_score: {4:.3f}, ' \
                   'IoU: {5:.3f}\n'.format(opt.epoch, acc, pre, recall, F_score, iou)
        print(message)
        expr_dir = os.path.join(save_dir, "res")
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'res.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')
