import os

import torch
import torch.nn
from torch.utils.data import DataLoader
from torch.optim.rmsprop import RMSprop
from torchsummary import summary
import argparse

import models
import datasets
import losses
import eval
import util_plot
import loggers

device = torch.device('cpu')

BATCH_SIZE = 8
INPUT_DIM = 256
OUTPUT_DIM = 64
EPOCHS = 1
NUM_STACKS = 2
NUM_CLASSES = 16
NUM_CHANNELS = 256
THRESHOLD = 0.5

IMAGES_DIR = 'images'
ANNOTS_PATH = 'annotations.json'
MEAN_PATH = 'mean_total.npy'
STD_PATH = 'std_total.npy'

mpii_train = datasets.MPII_dataset(
    dataset_type='train',
    images_dir=IMAGES_DIR,
    annots_json_filename=ANNOTS_PATH,
    mean_path=MEAN_PATH,
    std_path=STD_PATH,
    input_shape=INPUT_DIM,
    output_shape=OUTPUT_DIM
)

mpii_valid = datasets.MPII_dataset(
    dataset_type='valid',
    images_dir=IMAGES_DIR,
    annots_json_filename=ANNOTS_PATH,
    mean_path=MEAN_PATH,
    std_path=STD_PATH,
    input_shape=INPUT_DIM,
    output_shape=OUTPUT_DIM
)

train_dataloader = DataLoader(dataset=mpii_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
valid_dataloader = DataLoader(dataset=mpii_valid, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

model = models.HourglassNetwork(
    num_channels=NUM_CHANNELS,
    num_stacks=NUM_STACKS,
    num_classes=NUM_CLASSES,
    input_shape=(256, 256, 3)
)
model = torch.nn.DataParallel(model).to(device).double()

criterion = losses.JointsMSELoss().to(device)
optimizer = RMSprop(model.parameters(), lr=2.5e-4)
logger = loggers.CSVLogger()

total_step = len(train_dataloader)
for epoch in range(EPOCHS):

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

        joint_distances, accuracy_per_joint, average_accuracy = eval.output_accuracy(y[-1], y_kappa, threshold=THRESHOLD)

        print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy per joint: {}, Avg Accuracy: {:.4f}'
              .format(epoch + 1, EPOCHS, i + 1, total_step, loss.item(), accuracy_per_joint, average_accuracy))

        logger.log(epoch, i, loss, accuracy_per_joint, average_accuracy)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # if average_accuracy >= 0.0233:
        #     for j in range(input_batch.shape[0]):
        #         util_plot.plot_result_by_input(j, input_batch, meta_batch, joint_distances, y[-1], THRESHOLD)

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

            joint_distances, accuracy_per_joint, average_accuracy = eval.output_accuracy(y[-1], y_kappa,
                                                                                         threshold=THRESHOLD)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--device', type=str, default='cpu', help='device (cpu/cuda)')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--input_dim', type=int, default=256, help='input dimension')
    parser.add_argument('--output_dim', type=int, default=64, help='output dimension')
    parser.add_argument('--epochs', type=int, default=100, help='epochs')
    parser.add_argument('--stacks', type=int, default=2, help='number of hourglass stacks')
    parser.add_argument('--channels', type=int, default=256, help='number of channels')
    parser.add_argument('--joints', type=int, default=16, help='number of classes(joints)')
    parser.add_argument('--threshold', type=float, default=0.5, help='joints distance threshold')
    parser.add_argument('--image_dir', type=str, default='images', help='dataset (images) dir')
    parser.add_argument('--annots_path', type=str, default='annotations.json', help='annotations path')
    parser.add_argument('--mean_path', type=str, default='mean_total.npy', help='train dataset mean values file')
    parser.add_argument('--std_path', type=str, default='std_total.npy', help='train dataset std values file')

    main(parser.parse_args())

