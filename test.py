#  Torch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
# Python imports
import tqdm
from tqdm import tqdm
import os
from os.path import join as ospj
import csv
import torch.optim as optim
from data import dataset as dset
from models.common import Evaluator
from utils.utils import save_args, load_args
from utils.config_model import configure_model
from flags import parser, DATA_FOLDER
import random
import numpy as np
#import math
best_auc = 0
best_hm = 0
compose_switch = True





device = 'cuda' if torch.cuda.is_available() else 'cpu'
random.seed(3407)
torch.manual_seed(3407)
np.random.seed(3407)
torch.cuda.manual_seed_all(3407)

args = parser.parse_args()
load_args(args.config, args)
trainset = dset.CompositionDataset(
    root=os.path.join(DATA_FOLDER, args.data_dir),
    phase='train',
    split=args.splitname,
    model=args.image_extractor,
    num_negs=args.num_negs,
    pair_dropout=args.pair_dropout,
    update_features=args.update_features,
    train_only=args.train_only,
    open_world=args.open_world
)
trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.workers)
testset = dset.CompositionDataset(
    root=os.path.join(DATA_FOLDER, args.data_dir),
    phase=args.test_set,
    split=args.splitname,
    model=args.image_extractor,
    subset=args.subset,
    update_features=args.update_features,
    open_world=args.open_world
)
testloader = torch.utils.data.DataLoader(
    testset,
    batch_size=args.test_batch_size,
    shuffle=False,
    num_workers=args.workers)
image_extractor, model, optimizer = configure_model(args, trainset)
args.extractor = image_extractor
evaluator_val =  Evaluator(testset, model)

state_dict = torch.load(args.test_weights_path)
model.load_state_dict(state_dict)
print(model)

def test(epoch, image_extractor, model, testloader, evaluator, args):
    '''
    Runs testing for an epoch
    '''
    global best_auc, best_hm
    if image_extractor:
        image_extractor.eval()

    model.eval()

    accuracies, all_sub_gt, all_attr_gt, all_obj_gt, all_pair_gt, all_pred = [], [], [], [], [], []

    for idx, data in tqdm(enumerate(testloader), total=len(testloader), desc='Testing'):
        data = [d.to(device) for d in data]

        if image_extractor:
            data[0] = image_extractor(data[0])

        _, predictions = model(data)
        attr_truth, obj_truth, pair_truth = data[1], data[2], data[3]

        all_pred.append(predictions)
        all_attr_gt.append(attr_truth)
        all_obj_gt.append(obj_truth)
        all_pair_gt.append(pair_truth)

    if args.cpu_eval:
        all_attr_gt, all_obj_gt, all_pair_gt = torch.cat(all_attr_gt), torch.cat(all_obj_gt), torch.cat(all_pair_gt)
    else:
        all_attr_gt, all_obj_gt, all_pair_gt = torch.cat(all_attr_gt).to('cpu'), torch.cat(all_obj_gt).to(
            'cpu'), torch.cat(all_pair_gt).to('cpu')

    all_pred_dict = {}
    # Gather values as dict of (attr, obj) as key and list of predictions as values
    if args.cpu_eval:
        for k in all_pred[0].keys():
            all_pred_dict[k] = torch.cat(
                [all_pred[i][k].to('cpu') for i in range(len(all_pred))])
    else:
        for k in all_pred[0].keys():
            all_pred_dict[k] = torch.cat(
                [all_pred[i][k] for i in range(len(all_pred))])

    # Calculate best unseen accuracy
    results = evaluator.score_model(all_pred_dict, all_obj_gt, bias=args.bias, topk=args.topk)
    stats = evaluator.evaluate_predictions(results, all_attr_gt, all_obj_gt, all_pair_gt, all_pred_dict, topk=args.topk)

    stats['a_epoch'] = epoch

    result = ''
    # write to Tensorboard
    for key in stats:
        result = result + key + '  ' + str(round(stats[key], 4)) + '| '

    result = result + args.name
    print(f'Test Epoch: {epoch}')
    print(result)

epoch = 0
with torch.no_grad():  # todo: might not be needed
    test(epoch, image_extractor, model, testloader, evaluator_val, args)