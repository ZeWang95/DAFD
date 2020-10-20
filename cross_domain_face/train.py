import  torch, os
import  numpy as np
from    dataset import *
import  scipy.stats
from    torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from    torch.optim import lr_scheduler
import  random, sys, pickle
import  argparse
import pdb

import shutil
from my_utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig, MovingAverage, AverageMeter_Mat, worker_init_fn, Timer

import vgg_n

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight.data)
        if not m.bias is None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight.data, mean=0, std=0.01)
        m.bias.data.zero_()


def main(args):

    # torch.manual_seed(222)
    # torch.cuda.manual_seed_all(222)
    # np.random.seed(222)

    print(args)
    
    device = torch.device('cuda')
    trush_num = len(os.listdir('/data/dataset/NIR-VIS-2.0/trush'))
    net = vgg_n.vgg16_dcf_db(num_classes=725-trush_num).cuda()
    # pdb.set_trace()
    net.load_state_dict(torch.load('/home/jacobwang/torch_model/DCFNet/vgg13_face_dcf.pth'), strict=False)

    tmp = filter(lambda x: x.requires_grad, net.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    log_string(str(net.extra_repr))
    log_string('Total trainable tensors: %d' %num)

    # optimizer = torch.optim.Adam(net.parameters(), args.lr, weight_decay=0.00025)
    optimizer = torch.optim.SGD(net.parameters(), args.lr, momentum=0.9, weight_decay=0.00025)
    # optimizer = torch.optim.RMSprop(net.parameters(), args.lr, weight_decay=0.0001)
    
    loss_ma = MovingAverage(100)
    acc_ma = MovingAverage(100)
    timer = Timer()

    crite = nn.CrossEntropyLoss()
    transform_train = transforms.Compose([
        # random_rot(60),
        # torchvision.transforms.Resize(size=256),  # Let smaller edge match
        # torchvision.transforms.Resize(size=512),  # Let smaller edge match
        # random_resize(0.8, 1.2, 224),
        torchvision.transforms.RandomHorizontalFlip(),
        # torchvision.transforms.RandomCrop(size=256),
        # torchvision.transforms.RandomCrop(size=448),
        torchvision.transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    transform_test = transforms.Compose([
        # torchvision.transforms.Resize(size=256),
        # torchvision.transforms.CenterCrop(size=256),
        # torchvision.transforms.Resize(size=512),
        torchvision.transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    root_path = '/data/dataset/NIR-VIS-2.0/merged_imgs'
    coco = NIRVIS(root_path, 'train', transform=transform_train)
    coco_test = NIRVIS(root_path, 'test', transform=transform_test)


    for epoch in range(args.epoch):
        # log_string('Epoch %d' %epoch)

        if epoch == 10000:
             optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * 0.1
             log_string('Decaying learning rate.')
        # fetch meta_batchsz num of episode each time
        db = DataLoader(coco, args.batch_size, shuffle=True, num_workers=8, pin_memory=True, worker_init_fn=worker_init_fn)

        timer.reset()
        for step, x in enumerate(db):

            x = [xx.cuda() for xx in x]
            [nir, vis, label] = x

            pred = net(torch.cat([nir, vis], 0))
            loss = crite(pred, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_ma.update(loss.item())

            acc = accuracy(pred, label)[0].item()
            acc_ma.update(acc)

            # if step % 100 == 0:
        TI = timer.time(True)
        log_string('Epoch: %d, loss: %f, acc: %2.2f%%, time: %2.2f' %(epoch, loss_ma.avg, acc_ma.avg, TI))

        if (epoch+1) % 20 == 0:  # evaluation
            # pdb.set_trace()
            test_size = args.batch_size
            db_test = DataLoader(coco_test, test_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)

            net.eval()
            cc = 0
            log_string('Validating at Epoch: %d' %epoch)

            acc = []
            acc_nir = []
            acc_vis = []

            with torch.no_grad():
                for jj, x in enumerate(db_test):
                    x = [xx.cuda() for xx in x]
                    [nir, vis, label] = x
                    # pdb.set_trace()
                    pred, p_nir, p_vis = net.infer(torch.cat([nir, vis], 0))

                    acc.append(accuracy(pred, label)[0].item())
                    acc_nir.append(accuracy(p_nir, label)[0].item())
                    acc_vis.append(accuracy(p_vis, label)[0].item())

            log_string('TEsting at Epoch: %d, acc: %2.2f%%, acc: %2.2f%%, acc: %2.2f%%' %(epoch, np.mean(acc), np.mean(acc_nir), np.mean(acc_vis)))
            timer.reset()

if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='epoch number', default=6000)
    argparser.add_argument('--n_way', type=int, help='n way', default=5)
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=1)
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=15)
    argparser.add_argument('--imgsz', type=int, help='imgsz', default=84)
    argparser.add_argument('--imgc', type=int, help='imgc', default=3)
    argparser.add_argument('--batch_size', type=int, help='imgc', default=32)
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=4)
    argparser.add_argument('--lr', type=float, help='meta-level outer learning rate', default=1e-3)
    argparser.add_argument('--no_ole', action='store_true')
    argparser.add_argument('--no_prob', action='store_true')
    argparser.add_argument('--gpu', default='0', help='GPU to use [default: GPU 0]')
    argparser.add_argument('--log_dir', default='log1', help='Log dir [default: log]')
    argparser.add_argument('--bases', default='6', type=int, help='Log dir [default: log]')


    args = argparser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    name_file = sys.argv[0]
    LOG_DIR = args.log_dir
    LOG_DIR = 'logs/'+LOG_DIR
    args.log_dir = LOG_DIR
    if os.path.exists(LOG_DIR): shutil.rmtree(LOG_DIR)
    os.mkdir(LOG_DIR)
    os.mkdir(LOG_DIR + '/train_img')
    os.mkdir(LOG_DIR + '/test_img')
    os.mkdir(LOG_DIR + '/files')
    os.system('cp %s %s' % (name_file, LOG_DIR))
    os.system('cp %s %s' % ('*.py', os.path.join(LOG_DIR, 'files')))
    LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
    LOG_FOUT.write(str(args)+'\n')
    # print(str(args))

    def log_string(out_str):
        LOG_FOUT.write(out_str+'\n')
        LOG_FOUT.flush()
        print(out_str)


    st = ' '
    log_string(st.join(sys.argv))
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    main(args)
    # main()
