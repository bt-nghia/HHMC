# coding: utf-8
# @email: enoche.chow@gmail.com

"""
Run application
##########################
"""
from logging import getLogger
from itertools import product
from utils.dataset import RecDataset
from utils.dataloader import TrainDataLoader, EvalDataLoader
from utils.configurator import Config
from utils.utils import init_seed, get_model, get_trainer, dict2str
import platform
import os


def quick_start(model, dataset, config_dict, save_model=True):
    # merge config dict
    config = Config(model, dataset, config_dict)
    # print config infor

    # load data
    dataset = RecDataset(config)
    # print dataset statistics

    train_dataset, valid_dataset, test_dataset = dataset.split()

    # wrap into dataloader
    train_data = TrainDataLoader(config, train_dataset, batch_size=config['train_batch_size'], shuffle=True)
    # TODO(bt-nghia): load test data, valid data
    test_data = EvalDataLoader(config, test_dataset, additional_dataset=train_dataset, batch_size=config['eval_batch_size'])
    valid_data = None

    ############ Dataset loadded, run model
    hyper_ret = []
    val_metric = config['valid_metric'].lower()
    best_test_value = 0.0
    idx = best_test_idx = 0


    # hyper-parameters
    hyper_ls = []
    if "seed" not in config['hyper_parameters']:
        config['hyper_parameters'] = ['seed'] + config['hyper_parameters']
    for i in config['hyper_parameters']:
        hyper_ls.append(config[i] or [None])
    # combinations
    combinators = list(product(*hyper_ls))
    total_loops = len(combinators)
    for hyper_tuple in combinators:
        # random seed reset
        for j, k in zip(config['hyper_parameters'], hyper_tuple):
            config[j] = k
        init_seed(config['seed'])

        # set random state of dataloader
        train_data.pretrain_setup()
        # model loading and initialization
        model = get_model(config['model'])(config, train_data).to(config['device'])

        # trainer loading and initialization
        trainer = get_trainer()(config, model)
        # debug
        # model training
        best_test = trainer.fit(train_data, valid_data = valid_data, test_data=test_data, saved=save_model)