import argparse
from utils.getter import get_model

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import pytorch_lightning as pl  

from utils.dataset import QaTa,MosMedPlus,SegmentData
import utils.config as config


def get_parser():
    parser = argparse.ArgumentParser(
        description='Language-guide Medical Image Segmentation')
    parser.add_argument('--config',
                        default='./config/training.yaml',
                        type=str,
                        help='config file')

    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)

    return cfg

if __name__ == '__main__':

    args = get_parser()

    # load model
    model = get_model(args)

    torch.serialization.add_safe_globals([config.CfgNode])

    checkpoint = torch.load(args.best_model_path,map_location='cpu',weights_only=False)["state_dict"]
    model.load_state_dict(checkpoint,strict=False)
    ds_test = SegmentData(
                    csv_path=args.test_csv_path,
                    root_path=args.test_root_path,
                    tokenizer=args.bert_type,
                    image_size=args.image_size,
                    mode='test',
                    text_length=args.text_length,
                    name=args.data)
    
    dl_test = DataLoader(ds_test, batch_size=args.valid_batch_size, shuffle=False, num_workers=8)

    trainer = pl.Trainer(accelerator='gpu',devices=1) 
    model.eval()
    trainer.test(model, dl_test) 
