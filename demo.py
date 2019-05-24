import os

import torch
import torch.nn
from torch.utils.data import DataLoader
import argparse

import models
import datasets
import losses
import eval
import util_plot
import loggers


def main(args):
    args_dict = vars(args)

    if args_dict['saved_model'] is None or not os.path.exists(args_dict['saved_model']):
        print("No valid checkpoint path provided. EXITING...")
        exit()

    if args_dict['logger_csv_path'] is not None and os.path.exists(args_dict['logger_csv_path']):
        logger = loggers.CSVLogger(args_dict['logger_csv_path'])
        logger.plot_data()
    else:
        print('No valid logger csv file provided. CONTINUING...')

    device = torch.device(args_dict['device'])

    model, saved_args_dict = models.load_demo_model(args_dict['saved_model'], device)
    args_dict = {**saved_args_dict, **args_dict}

    mpii_valid = datasets.MPII_dataset(
        dataset_type='valid',
        images_dir=args_dict['images_dir'],
        annots_json_filename=args_dict['annots_path'],
        mean_path=args_dict['mean_path'],
        std_path=args_dict['std_path'],
        input_shape=args_dict['input_dim'],
        output_shape=args_dict['output_dim']
    )

    valid_dataloader = DataLoader(dataset=mpii_valid, batch_size=args_dict['batch_size'], shuffle=False, num_workers=0)

    criterion = losses.JointsMSELoss().to(device)

    with torch.no_grad():

        for i, (input_batch, output_batch, meta_batch) in enumerate(valid_dataloader):
            x = input_batch.to(device)
            y_kappa = output_batch.to(device, non_blocking=True)
            weights = meta_batch['label_weights'].to(device, non_blocking=True)

            y = model(x)

            loss = 0
            for _y in y:
                loss += criterion(_y, y_kappa, weights)

            joint_distances, accuracy_per_joint, average_accuracy = eval.output_accuracy(
                y=y[-1],
                y_kappa=y_kappa,
                threshold=args_dict['threshold']
            )

            print('DEMO: Step=[{}/{}], Loss={:.8f}, Avg_Acc: {:.5f}'
                  .format(i + 1, len(valid_dataloader), loss.item(), average_accuracy))

            util_plot.plot_demo_results(meta_batch, y[-1], joint_distances, args_dict['images_dir'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--device', type=str, default='cpu', help='device (cpu/cuda)')
    parser.add_argument('--images_dir', type=str, default='images', help='dataset (images) dir')
    parser.add_argument('--annots_path', type=str, default='annotations.json', help='annotations path')
    parser.add_argument('--mean_path', type=str, default='mean_total.npy', help='train dataset mean values file')
    parser.add_argument('--std_path', type=str, default='std_total.npy', help='train dataset std values file')
    parser.add_argument('--logger_csv_path', type=str, default='logger.csv', help='logger csv path')
    parser.add_argument('--saved_model', type=str, default='checkpoint.pth', help='model checkpoint')

    main(parser.parse_args())

