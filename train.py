import argparse, os, random, sys
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from dataloader.dataio import PlyDataSet
from networks.THISNet import THISNet
from utils.instanceloss import Cal_Loss
from utils.read_gpu_free import sleep_until_gpu_free
from utils.creat_loggings import creat_loggings

def get_args():
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--save_path', default='./experiments/', type=str,
                        help='path to the experiments saved')
    parser.add_argument('--experiment_name', default='THISNet', type=str,
                        help='the experiment model name')
    parser.add_argument('--train_path', default='/raid/lipengcheng/Data/mesh_data_221204/all_data/TRAINT/', type=str,
                        help='path to the training data')
    parser.add_argument('--val_path', default='/raid/lipengcheng/Data/mesh_data_221204/all_data/VALT/', type=str,
                        help='path to the validation data')
    parser.add_argument('--batch_size', default=8, type=int, help='batch size in all gpu')
    parser.add_argument('--input_channel', default=24, type=int, help='input channels of the model')
    parser.add_argument('--num_classes', default=14, type=int, help='class numbers of the model')
    parser.add_argument('--epochs', default=400, type=int, help='epoch numbers of the model')
    parser.add_argument('--lr', default=1e-3, type=float, help='initial learning rate')
    parser.add_argument('--early_stop_patience', default=25, type=int, help='early stopped epochs')
    parser.add_argument('--gpu_ids', default='0,1', type=str, help='gpu_ids: e.g. 0  0,1,2  0,2  2,3')
    parser.add_argument('--setup_seed', default='42', help='set up seed or not')
    parser.add_argument('--k', default=32, help='set nodes of knn')

    return parser.parse_args()

def setup_seed(seed):
    seed = int(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def run_training(model, train_loader, test_loader, loss_fn, optimizer, scheduler, early_stop_patience: int = None,
                 EPOCHS=100, writer_path=None, num_classes=15, logger=None,log_dir=None, ckpt_path=None):
    eval_metric_score = 1000
    early_stopping = 0
    writer = SummaryWriter(os.path.join(writer_path,'tensorboard'))

    for epoch in range(EPOCHS):
        train(model, train_loader, loss_fn, optimizer, scheduler, epoch, writer, num_classes, logger)
        with torch.no_grad():
            val_metric = evaluate(model, test_loader, loss_fn, epoch, writer, num_classes, logger)
        if scheduler:
            scheduler.step(val_metric)

        if val_metric < eval_metric_score:
            eval_metric_score = val_metric
            tqdm.write("best loss: %f" % (eval_metric_score))
            tqdm.write('Best model saved!')
            torch.save(model.state_dict(), '%s/ckpt_%s_%3s.pth' % (ckpt_path, str(epoch).zfill(2), eval_metric_score))
            early_stopping = 0
        else:
            early_stopping += 1
            tqdm.write("best loss: %f" % (eval_metric_score))
            tqdm.write('Early stopping {}'.format(early_stopping))

        if early_stop_patience and early_stop_patience == early_stopping:
            tqdm.write('Early stopping stopped')
            torch.save(model.state_dict(), '%s/ckpt_early_stop.pth' % (ckpt_path))
            break
    return eval_metric_score

def train(model, train_loader, loss_fn, optimizer, scheduler, epoch, tensorboard_writer, num_classes, logger):
    epoch_train_loss = 0
    for meshdata in tqdm(train_loader, file=sys.stdout, desc='training'):
        faces, faces_points, faces_center_points, faces_normals, faces_center_normal, gt_faces, names = meshdata
        data = torch.cat([faces_center_points, faces_points, faces_center_normal, faces_normals], dim=2)
        data = data.transpose(2, 1)

        data, gt = data.float(), gt_faces.long()

        # background label->14
        gt = gt - 1
        gt[gt<0] = 14
        gt = gt.long()

        data, gt = data.cuda(), gt.cuda()



        optimizer.zero_grad()

        output, pred = model(data)
        loss_instance = loss_fn(output, gt)

        loss = loss_instance
        loss.backward()
        optimizer.step()

        epoch_train_loss += loss.item()

    train_epoch_loss = epoch_train_loss / len(train_loader)
    tqdm.write(f'Epoch {epoch}')
    tqdm.write(f'Train Epoch loss {train_epoch_loss}')
    logger.info("Epoch: %d, Train Epoch loss= %f " % (epoch, train_epoch_loss))
    logger.info("Epoch: %d, lr= %f " % (epoch, optimizer.param_groups[0]["lr"]))
    tensorboard_writer.add_scalar("LR/train", optimizer.param_groups[0]["lr"], epoch)
    tensorboard_writer.add_scalar("Loss/train", train_epoch_loss, epoch)

    return train_epoch_loss

def evaluate(model, test_loader, loss_fn, epoch, tensorboard_writer, num_classes, logger):
    epoch_eval_loss = 0
    for meshdata in tqdm(test_loader,file=sys.stdout, desc='eval'):
        faces, faces_points, faces_center_points, faces_normals, faces_center_normal, gt_faces, names = meshdata
        points = torch.cat((faces_center_points, faces_points, faces_center_normal, faces_normals), dim=2)
        points = points.transpose(2, 1)
        data, gt = points.float(), gt_faces.long()

        # background label->14
        gt = gt - 1
        gt[gt<0] = 14
        gt = gt.long()

        data, gt = data.cuda(), gt.cuda()

        batchsize, _, num_point = data.size()

        output, pred = model(data)

        with torch.no_grad():
            loss_instance = loss_fn(output, gt)

        epoch_eval_loss += loss_instance.item()

    eval_epoch_loss = epoch_eval_loss / len(test_loader)
    return eval_epoch_loss

if __name__ == '__main__':
    sleep_until_gpu_free([0,0,0.2,0.2,], sleep_interval=180) # Detects graphics cards with free memory
    """-------------------------- parameters --------------------------------------"""
    torch.cuda.empty_cache()
    opt = get_args()
    if opt.setup_seed is not None:
        setup_seed(opt.setup_seed)
    logger, log_dir, checkpoints_path = creat_loggings(experiment_dir=opt.save_path, experiment_name=opt.experiment_name)
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_ids
    writer_path = os.path.join(opt.save_path, opt.experiment_name)
    input_channel = opt.input_channel
    num_classes = opt.num_classes
    batch_size = opt.batch_size
    epochs = opt.epochs
    early_stop_patience = opt.early_stop_patience
    train_plypath = opt.train_path
    val_plypath = opt.val_path
    k = opt.k
    """-------------------------------- PlyDataloader --------------------------------"""
    train_dataset = PlyDataSet(train_plypath)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=24, pin_memory=True)
    val_dataset = PlyDataSet(val_plypath)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=4, pin_memory=True)
    """--------------------------- Build Network and optimizer----------------------"""
    model = THISNet(in_channels=input_channel, output_channels=num_classes, k=k) # coord or normal channels
    device_ids = [0]
    if (len(opt.gpu_ids)+1) // 2 == 2:
        device_ids=[0,1]
    if (len(opt.gpu_ids)+1) // 2 == 4:
        device_ids=[0,1,2,3]
    model = torch.nn.DataParallel(model,device_ids=device_ids)
    model.to(torch.device("cuda:0"))

    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                           factor=0.5, patience=10, min_lr=1e-5,
                                                           verbose=True)
    """--------------------------- Define loss function and training----------------------"""
    loss_fn = Cal_Loss().cuda()

    run_training(model=model, train_loader=train_loader, test_loader=val_loader, loss_fn=loss_fn,
                 optimizer=optimizer, scheduler=scheduler, early_stop_patience=early_stop_patience,
                 EPOCHS=epochs, writer_path=writer_path, num_classes=num_classes,
                 logger=logger, log_dir=log_dir, ckpt_path = checkpoints_path)