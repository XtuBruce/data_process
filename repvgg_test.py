import os
import sys
import csv
import time
import yaml
import argparse
import importlib
import numpy as np
from tqdm import tqdm
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
sys.path.append("..")
from data import CustomDataset, get_datatrans, get_dataset, get_dataloader
from utils import create_path, CalParams
import warnings

warnings.filterwarnings('ignore')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def run(opt):
    create_path(os.path.dirname(os.path.realpath(opt['Path']['Error_Path'])), is_remove=True)
    datatrans = get_datatrans(opt['Solver']['Resize'])
    class_indices = None
    try:
        if "None" != opt['Solver']['Select']:
            class_indices = opt['Solver']['Select']
            if not isinstance(class_indices[0], int):
               class_indices = [opt['Label_List'].index(label) for label in class_indices]
    except:
        pass
    dataset = get_dataset(Dataset=CustomDataset, test_root=opt['Path']['TestData_Path'], transform=datatrans, augmentations=False, class_indices=class_indices)
    dataloader = get_dataloader(dataset=dataset, batch_size=opt['Solver']['Batch_Size'])

    device = 'cuda:{}'.format(opt['Solver']['Device']) if torch.cuda.is_available() else 'cpu'
    create_model = getattr(importlib.import_module('model'), opt['Model']['Module'])
    model = create_model(num_classes=len(opt['Label_List']),
                         module=opt['Model']['Module'],
                         model_name=opt['Model']['Model_Name'],
                         pretrained_weight=None,
                         deploy=opt['Solver']['Deploy']).to(device)
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
    model.load_state_dict(torch.load(os.path.join(opt['Path']['Model_Path'], 'best_model.pth'), map_location=torch.device(device)))
    criterion = getattr(importlib.import_module('criterion'), opt['Solver']['Criterion'])()
    csv_file = open(opt['Path']['Error_Path'], 'w', encoding='utf-8')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['path', 'label', 'pred'])
    label_list = sorted(opt['Label_List'])
    total_loss, total_acc, total_num = {'test': 0.0}, {'test': 0.0}, {'test': 0}
    bar, start_time = {'test': tqdm(dataloader['test'])}, time.time()
    model.eval()
    with torch.no_grad():
        for step, sample in enumerate(bar['test']):
            for key in sample.keys():
                if torch.is_tensor(sample[key]):
                    sample[key] = sample[key].to(device, non_blocking=True)

            outputs = model(sample['inputs'])

            total_loss['test'] += criterion(outputs, sample['targets']).item()
            total_acc['test'] += (outputs.argmax(dim=1) == sample['targets']).float().sum().item()
            total_num['test'] += sample['targets'].shape[0]

            bar['test'].desc = "{} [Test], Step [{:04d}/{:04d}], Loss:{:.5f}, Acc:{:.5f}". \
                format(datetime.now(), step + 1, len(dataloader['test']),
                       total_loss['test'] / total_num['test'],
                       total_acc['test'] / total_num['test'], )

            for i in range(len(sample['targets'])):
                if (outputs[i].argmax() != sample['targets'][i]):
                    csv_writer.writerow([sample['paths'][i],
                                         label_list[sample['targets'][i]],
                                         label_list[outputs[i].argmax()]])
    print('Model Test Time: {:.5f}s, Loss: {:.5f}, Acc: {:.5f}'.format(time.time() - start_time,
                                                                       total_loss['test'] / total_num['test'],
                                                                       total_acc['test'] / total_num['test']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='../config_yndet.yaml', help='config path')
    args = parser.parse_args()
    print('Job Dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("Args: {}".format(args).replace(', ', ',\n'))
    with open(args.config, 'r') as load_yaml:
        opt = yaml.load(load_yaml, Loader=yaml.FullLoader)
    run(opt)
