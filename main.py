import os

import torch
import torch.nn
from torch.utils.data import DataLoader
from torch.optim.rmsprop import RMSprop
from torch.optim.adam import Adam
from torchsummary import summary
import argparse

import models
import datasets
import losses
import eval
import util_plot
import loggers


def main(args):
    args_dict = vars(args)

    if args_dict['saved_model'] is not None and os.path.exists(args_dict['saved_model']):
        device, model, optimizer, saved_args_dict = models.load_model(args_dict['saved_model'])
        args_dict = saved_args_dict
    else:
        model = models.HourglassNetwork(
            num_channels=args.channels,
            num_stacks=args.stacks,
            num_classes=args.joints,
            input_shape=(args.input_dim, args.input_dim, 3)
        )
        device = torch.device(args_dict['device'])
        model = torch.nn.DataParallel(model).to(device).double()
        optimizer = Adam(model.parameters(), lr=args.lr)

    mpii_train = datasets.MPII_dataset(
        dataset_type='train',
        images_dir=args_dict['images_dir'],
        annots_json_filename=args_dict['annots_path'],
        mean_path=args_dict['mean_path'],
        std_path=args_dict['std_path'],
        input_shape=args_dict['input_dim'],
        output_shape=args_dict['output_dim']
    )

    mpii_valid = datasets.MPII_dataset(
        dataset_type='valid',
        images_dir=args_dict['images_dir'],
        annots_json_filename=args_dict['annots_path'],
        mean_path=args_dict['mean_path'],
        std_path=args_dict['std_path'],
        input_shape=args_dict['input_dim'],
        output_shape=args_dict['output_dim']
    )

    train_dataloader = DataLoader(dataset=mpii_train, batch_size=args_dict['batch_size'], shuffle=True, num_workers=0)
    valid_dataloader = DataLoader(dataset=mpii_valid, batch_size=args_dict['batch_size'], shuffle=False, num_workers=0)

    criterion = losses.JointsMSELoss().to(device)
    logger = loggers.CSVLogger(args_dict['logger_csv_path'])

    for epoch in range(args_dict.get('epoch_to_start', 0), args_dict['epochs']):

        model.train()

        for i, (input_batch, output_batch, meta_batch) in enumerate(train_dataloader):

            # TODO adjust learning rate

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

            print('TRAIN: Epoch=[{}/{}], Step=[{}/{}], Loss={:.8f}, Avg_Acc: {:.5f}'
                  .format(epoch + 1, args_dict['epochs'], i + 1, len(train_dataloader), loss.item(), average_accuracy))

            logger.log(epoch+1, args_dict['epochs'], i, len(train_dataloader), 'train', loss, accuracy_per_joint, average_accuracy)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()

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

                print('VALID: Epoch=[{}/{}], Step=[{}/{}], Loss={:.8f}, Avg_Acc: {:.5f}'
                      .format(epoch + 1, args_dict['epochs'], i + 1, len(valid_dataloader), loss.item(),
                              average_accuracy))

                logger.log(epoch+1, args_dict['epochs'], i, len(train_dataloader), 'valid', loss, accuracy_per_joint, average_accuracy)

        args_dict['epoch_to_start'] = epoch+1

        models.save_model(
            model=model,
            optimizer=optimizer,
            args_dict=args_dict
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--device', type=str, default='cpu', help='device (cpu/cuda)')
    parser.add_argument('--lr', type=float, default=2.5e-4, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--input_dim', type=int, default=256, help='input dimension')
    parser.add_argument('--output_dim', type=int, default=64, help='output dimension')
    parser.add_argument('--epochs', type=int, default=100, help='epochs')
    parser.add_argument('--stacks', type=int, default=2, help='number of hourglass stacks')
    parser.add_argument('--channels', type=int, default=256, help='number of channels')
    parser.add_argument('--joints', type=int, default=16, help='number of classes(joints)')
    parser.add_argument('--threshold', type=float, default=0.5, help='joints distance threshold')
    parser.add_argument('--images_dir', type=str, default='images', help='dataset (images) dir')
    parser.add_argument('--annots_path', type=str, default='annotations.json', help='annotations path')
    parser.add_argument('--mean_path', type=str, default='mean_total.npy', help='train dataset mean values file')
    parser.add_argument('--std_path', type=str, default='std_total.npy', help='train dataset std values file')
    parser.add_argument('--logger_csv_path', type=str, default='logger.csv', help='logger csv path')
    parser.add_argument('--saved_model', type=str, default='checkpoint.pth', help='model checkpoint')

    main(parser.parse_args())

