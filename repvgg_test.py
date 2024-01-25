import os
import sys
import time
import yaml
import argparse
import importlib
import numpy as np
from tqdm import tqdm
from scipy import misc
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
sys.path.append("..")
from data import PolypDataset, get_datatrans, get_dataset, get_dataloader
from utils import set_seed, create_path, get_mdice, clip_gradient, get_mosaic_data, CalParams
import warnings

warnings.filterwarnings('ignore')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def run(opt):
    set_seed(opt['Seed'])
    opt['Solver']['Batch_Size'] *= len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    opt['Solver']['Learning_Rate'] *= len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))

    is_yndet = False
    datatrans = get_datatrans(opt['Solver']['Resize'], is_yndet=is_yndet)
    dataset = get_dataset(Dataset=PolypDataset, test_root=opt['Path']['TestData_Path'], transform=datatrans,
                          augmentations=False, is_yndet=is_yndet)
    dataloader = get_dataloader(dataset=dataset, batch_size=opt['Solver']['Batch_Size'])

    device = 'cuda:{}'.format(opt['Solver']['Device']) if torch.cuda.is_available() else 'cpu'
    create_model = getattr(importlib.import_module('model'), opt['Model']['Module'])
    model = create_model(module=opt['Model']['Module'],
                         model_name=opt['Model']['Model_Name'],
                         pretrained_weight=None,
                         deploy=False).to(device)
    # print(model)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Number of Model Params (M): %.2f' % (n_parameters / 1.e6))
    print("Let's use {} GPUs!".format(torch.cuda.device_count()))
    # ---- flops and params ----
    # x = torch.randn(1, 3, 224, 224).to(device)
    # CalParams(model, x)
    # input()
    if opt['Solver']['Distributed']:
        model = nn.DataParallel(model)
    model.load_state_dict(torch.load(os.path.join(opt['Path']['Model_Path'], 'best_mdice_val_model.pth'), map_location=torch.device(device)))
    criterion = getattr(importlib.import_module('criterion'), opt['Solver']['Criterion'])()

    model.eval()
    total_loss, total_mdice = {'test': 0.0}, {'test': 0.0}
    total_num, bar, start_time = {'test': 0}, {'test': tqdm(dataloader['test'])}, time.time()
    with torch.no_grad():
        for step, sample in enumerate(bar['test']):
            for key in sample.keys():
                if torch.is_tensor(sample[key]):
                    sample[key] = sample[key].to(device)

            outputs = model(sample['inputs'])
            if not torch.is_tensor(outputs):
                outputs = outputs[0]
            total_loss['test'] += criterion(outputs, sample['targets'])
            total_mdice['test'] += get_mdice(outputs, sample['targets'])
            total_num['test'] += 1

            bar['test'].desc = "{} [Test] Step [{:04d}/{:04d}], Loss:{:.5f}, mDice:{:.5f}".format(datetime.now(),
                                                                            step + 1, len(dataloader['test']),
                                                                            total_loss['test'] / total_num['test'],
                                                                            total_mdice['test'] / total_num['test'])
        print('*****************************************************')
        avg_loss = {'test': total_loss['test'] / total_num['test']}
        avg_mdice = {'test': total_mdice['test'] / total_num['test']}
        print('Test Loss: {:.5f} | Test mDice: {:.5f}'.format(avg_loss['test'], avg_mdice['test']))
        print('*****************************************************')
    print('Model Test Time: {:.5f}s'.format(time.time() - start_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='../config_kxdet.yaml', help='base config file')
    args = parser.parse_args()
    print('Job Dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("Args: {}".format(args).replace(', ', ',\n'))
    with open(args.config, 'r') as load_yaml:
        opt = yaml.load(load_yaml, Loader=yaml.FullLoader)
    run(opt)
