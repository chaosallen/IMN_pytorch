import torch
import torch.nn as nn
import logging
import sys
import os
import model
import numpy as np
import scipy.misc as misc
from options.test_options import TestOptions
import natsort
import evaluation
from scipy import io

def test_net(net,device):
    DATA_SIZE=opt.image_size
    BLOCK_SIZE=opt.input_size
    test_dice_sum=0
    test_results = os.path.join(opt.saveroot, 'test_results')
    test_results_prob = os.path.join(opt.saveroot, 'test_results_prob')
    net.eval()
    test_images = np.zeros((1, 1, opt.input_size[0], opt.input_size[1]))
    imglist = os.listdir(os.path.join(opt.testroot,'img'))
    imglist = natsort.natsorted(imglist)

    for kk, name in enumerate(imglist):
        label = misc.imread(os.path.join(opt.testroot, 'gt', name))
        img = misc.imread(os.path.join(opt.testroot, 'img', name))
        result=np.zeros((opt.image_size[0], opt.image_size[1]))
        result_prob = np.zeros((opt.image_size[0], opt.image_size[1]))
        votemap = np.zeros((DATA_SIZE[0], DATA_SIZE[1]))
        votetime=1
        for i in range(0, DATA_SIZE[0], BLOCK_SIZE[0]//votetime):
            for j in range(0, DATA_SIZE[1], BLOCK_SIZE[1]//votetime):
                if i+BLOCK_SIZE[0]>=DATA_SIZE[0]:
                    i=DATA_SIZE[0]-BLOCK_SIZE[0]
                if j+BLOCK_SIZE[1]>=DATA_SIZE[1]:
                    j=DATA_SIZE[1]-BLOCK_SIZE[1]
                test_images[0, 0, :, :] = img[i:i + BLOCK_SIZE[0], j:j + BLOCK_SIZE[1]]
                images = torch.from_numpy(test_images)
                images = images.to(device=device, dtype=torch.float32)
                pred = net(images)
                pred_softmax=torch.softmax(pred,dim=1)
                pred_argmax = torch.argmax(pred, dim=1)
                result[i:i + BLOCK_SIZE[0], j:j + BLOCK_SIZE[1]] = result[i:i + BLOCK_SIZE[0],j:j + BLOCK_SIZE[1]] + pred_argmax[0,:,:].cpu().detach().numpy()
                result_prob[i:i + BLOCK_SIZE[0], j:j + BLOCK_SIZE[1]] = result_prob[i:i + BLOCK_SIZE[0],j:j + BLOCK_SIZE[1]] + pred_softmax[0,1, :,:].cpu().detach().numpy()
                votemap[i:i + BLOCK_SIZE[0], j:j + BLOCK_SIZE[1]] = votemap[i:i + BLOCK_SIZE[0],j:j + BLOCK_SIZE[1]] + 1
        result=result/votemap
        result_prob=result_prob/votemap
        misc.imsave(os.path.join(test_results, name), (result*255).astype(np.uint8))
        misc.imsave(os.path.join(test_results_prob, name), (result_prob*255).astype(np.uint8))
        test_dice = evaluation.calc_dice(result, label//255)
        print("Valid_dice:{:.4f}".format(test_dice))
        test_dice_sum += test_dice
    val_dice = test_dice_sum / (kk + 1)
    print("Valid_dice:{:.4f}".format(val_dice))





if __name__ == '__main__':
    #setting logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    #loading options
    opt = TestOptions().parse()
    #setting GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_ids
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    #loading network
    net=model.IMN(in_channels=opt.in_channels, n_classes=opt.n_classes, channels=opt.channels)

    #load trained model
    restore_path = os.path.join(opt.saveroot, 'checkpoints', '1070.pth')
    net.load_state_dict(
        torch.load(restore_path, map_location=device)
    )
    #input the model into GPU
    net.to(device=device)
    try:
        test_net(net=net,device=device)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
