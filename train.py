import torch
import torch.nn as nn
import numpy as np
import logging
import sys
import os
import model
import evaluation
import shutil
import natsort
from options.train_options import TrainOptions
import scipy.misc as misc
import Batchreader



def train_net(net,device):
    #train setting
    DATA_SIZE=opt.image_size
    BLOCK_SIZE=opt.input_size
    interval=opt.save_interval
    best_valid_dice=0
    val_images = np.zeros((1,1, opt.input_size[0], opt.input_size[1]))
    model_save_path = os.path.join(opt.saveroot, 'checkpoints')
    best_model_save_path = os.path.join(opt.saveroot, 'best_model')
    # Read Data
    Trainreader=Batchreader.Batchreader(opt.trainroot)
    # Setting Optimizer
    if opt.optimizer == 'SGD':
        optimizer = torch.optim.SGD(net.parameters(), opt.lr, momentum=0.9, weight_decay=1e-6)
    elif opt.optimizer == 'Adam':
        optimizer = torch.optim.Adam(net.parameters(), opt.lr, betas=(0.9, 0.99))
    elif opt.optimizer == 'RMS':
        optimizer = torch.optim.RMSprop(net.parameters(), opt.lr, weight_decay=1e-8)
    #Setting Loss
    criterion = nn.CrossEntropyLoss()
    #Start train
    for itr in range(0, opt.max_iteration):
        net.train()
        train_images, train_annotations = Trainreader.next_batch(batch_size=opt.batch_size,image_size=opt.image_size,input_size=opt.input_size)
        train_images =train_images.to(device=device, dtype=torch.float32)
        train_annotations = train_annotations.to(device=device, dtype=torch.long)
        pred= net(train_images)
        loss = criterion(pred, train_annotations)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if itr % interval == 0:
            print(itr,loss.item())
        #Start Val
        with torch.no_grad():
            if itr % interval==0:
                #Save model
                torch.save(net.module.state_dict(),
                           os.path.join(model_save_path,f'{itr}.pth'))
                logging.info(f'Checkpoint {itr} saved !')
                #Calculate validation Dice
                val_dice_sum = 0
                net.eval()
                featurelist = os.listdir(os.path.join(opt.valroot,'img'))
                featurelist = natsort.natsorted(featurelist)

                for kk, name in enumerate(featurelist):
                    result = np.zeros((DATA_SIZE[0], DATA_SIZE[1]))
                    result_prob = np.zeros((DATA_SIZE[0], DATA_SIZE[1]))
                    label = misc.imread(os.path.join(opt.valroot,'gt', name))
                    img = misc.imread(os.path.join(opt.valroot,'img', name))
                    for i in range(0, DATA_SIZE[0],BLOCK_SIZE[0]):
                        for j in range(0, DATA_SIZE[1],BLOCK_SIZE[1]):
                            val_images[0,0, :,:]  = img[i:i + BLOCK_SIZE[0],j:j + BLOCK_SIZE[1]]
                            images = torch.from_numpy(val_images)
                            images = images.to(device=device, dtype=torch.float32)
                            pred = net(images)
                            pred_softmax = torch.softmax(pred, dim=1)
                            pred_argmax = torch.argmax(pred, dim=1)
                            result[i:i + BLOCK_SIZE[0], j:j + BLOCK_SIZE[1]] = result[i:i + BLOCK_SIZE[0],j:j + BLOCK_SIZE[1]] + pred_argmax[0,:,:].cpu().detach().numpy()
                            result_prob[i:i + BLOCK_SIZE[0], j:j + BLOCK_SIZE[1]] = result_prob[i:i + BLOCK_SIZE[0],j:j + BLOCK_SIZE[1]] + pred_softmax[0, 1, :,:].cpu().detach().numpy()
                    val_dice_sum+= evaluation.cal_dice(result,label)

                    #if kk<=9:
                    #    misc.imsave(os.path.join(opt.saveroot, 'itr',str(kk), str(itr)+'.bmp'), (result_prob * 255).astype(np.uint8))
                val_dice=val_dice_sum/(kk+1)
                print("Step:{}, Valid_dice:{}".format(itr,val_dice))
                #save best model
                if val_dice > best_valid_dice:
                    temp = '{:.6f}'.format(val_dice)
                    os.mkdir(os.path.join(best_model_save_path,temp))
                    temp2= f'{itr}.pth'
                    shutil.copy(os.path.join(model_save_path,temp2),os.path.join(best_model_save_path,temp,temp2))

                    model_names = natsort.natsorted(os.listdir(best_model_save_path))
                    #print(len(model_names))
                    if len(model_names) == 4:
                        shutil.rmtree(os.path.join(best_model_save_path,model_names[0]))
                    best_valid_dice = val_dice



if __name__ == '__main__':

    #setting logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    #loading options
    opt = TrainOptions().parse()
    #setting GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_ids
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    #loading network
    net=model.IMN(in_channels=opt.in_channels, n_classes=opt.n_classes, channels=opt.channels)
    net=torch.nn.DataParallel(net,[0,1]).cuda()
    #load trained model
    if opt.load:
        net.load_state_dict(
            torch.load(opt.load, map_location=device)
        )
        logging.info(f'Model loaded from {opt.load}')
    #input the model into GPU
    #net.to(device=device)
    try:
        train_net(net=net,device=device)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)




