import os
import re

import matplotlib.pyplot as plt
import numpy as np
import scipy.misc

import eval
import util_image
import datasets


def plot_demo_results(batch_meta, batch_pred, batch_joint_distances, images_dir):
    rows = 3
    columns = 3

    fig, ax = plt.subplots(nrows=rows, ncols=columns)

    batch_size = batch_pred.shape[0]

    num_joints = batch_joint_distances.shape[1]
    output_res = batch_pred.shape[-1]

    batch_pred_joints = eval.heatmap_argmax(batch_pred)

    for row in range(rows):
        for column in range(columns):
            if row * columns + column >= batch_size: continue
            image_filename = batch_meta['img_paths'][row * columns + column]

            filename_search = re.search('(\d\d).jpg$', image_filename, re.IGNORECASE)
            image_last_digits = filename_search.group(1)
            image = scipy.misc.imread(
                os.path.join(images_dir, '_' + str(image_last_digits), image_filename))

            center = batch_meta['center'][row * columns + column]
            scale = batch_meta['scale'][row * columns + column]
            pred_joints = batch_pred_joints[row * columns + column].data.numpy()
            joint_distances = batch_joint_distances[row * columns + column]

            ax[row, column].imshow(image)

            for j in range(num_joints):
                c = 'b' if joint_distances[j] <= 0.5 else 'r'

                if pred_joints[j, 0] > 0:
                    pred_joints[j, 0:2] = util_image.transform(pred_joints[j, 0:2], center, scale, [output_res, output_res], invert=1)
                    ax[row, column].scatter(pred_joints[j, 0], pred_joints[j, 1], s=10, c=c)

            bones = get_skeleton_bones(pred_joints, joint_distances)

            for bone in bones:
                if bone[-1] == 'line':
                    ax[row, column].plot(bone[0], bone[1], color='white', markersize=0)
                elif bone[-1] == 'circle':
                    circle = plt.Circle((bone[0], bone[1]), bone[2], color='white', fill=False)
                    ax[row, column].add_patch(circle)

    plt.show()


def get_skeleton_bones(joints, distances):
    bones = []

    corrects = distances < 0.5

    for pair in datasets.MPII_dataset.bones_pairs:
        if corrects[pair[0]] == 1 and corrects[pair[1]] == 1:
            if pair[0] == 9 or pair[1] == 9:
                bones.append([(joints[pair[0]][0] + joints[pair[1]][0]) / 2,
                              (joints[pair[0]][1] + joints[pair[1]][1]) / 2,
                              np.linalg.norm(joints[pair[0]] - joints[pair[1]]) / 2,
                              'circle'])
            else:
                bones.append([[joints[pair[0]][0], joints[pair[1]][0]],
                              [joints[pair[0]][1], joints[pair[1]][1]],
                              'line'])
    return bones


def plot_result_by_input(batch_index, batch_input, batch_meta, batch_joint_distances, batch_pred, threshold):
    fig, ax = plt.subplots(nrows=4, ncols=5)

    image = batch_input[batch_index].data.numpy()
    transformed_joints = batch_meta['transformed_joints'][batch_index]
    heatmaps_joints = batch_meta['heatmaps_joints'][batch_index]
    pred = batch_pred[batch_index].data.numpy()
    joint_distances = batch_joint_distances[batch_index]

    ax[0, 0].imshow(np.transpose(image, (1, 2, 0)))
    ax[0, 0].scatter(transformed_joints[:, 0], transformed_joints[:, 1])

    pred_joints = eval.heatmap_argmax(batch_pred)[batch_index]

    for i in range(0, 4):
        for j in range(1, 5):
            ax[i, j].imshow(pred[i * 4 + j - 1, :, :])

            c = 'g' if joint_distances[i * 4 + j - 1] <= threshold else 'r'

            if pred_joints[i * 4 + j - 1, 0] > 0:
                ax[i, j].scatter(pred_joints[i * 4 + j - 1, 0], pred_joints[i * 4 + j - 1, 1], s=5, c=c)
            else:
                ax[i, j].title.set_text('No predicted')

            if heatmaps_joints[i * 4 + j - 1, 0] > 0:
                ax[i, j].scatter(heatmaps_joints[i * 4 + j - 1, 0], heatmaps_joints[i * 4 + j - 1, 1], s=5, c='w')
            else:
                ax[i, j].title.set_text('No ground truth joint')

    plt.show()


def plot_dataset_example(input_image, metadata, heatmaps):
    fig, ax = plt.subplots(nrows=4, ncols=5)

    image = input_image.data.numpy()
    transformed_joints = metadata['transformed_joints']
    heatmaps_joints = metadata['heatmaps_joints']

    ax[0, 0].imshow(np.transpose(image, (1, 2, 0)))
    ax[0, 0].scatter(transformed_joints[:, 0], transformed_joints[:, 1])

    for i in range(0, 4):
        for j in range(1, 5):
            ax[i, j].imshow(heatmaps[i * 4 + j - 1, :, :])
            ax[i, j].scatter(heatmaps_joints[i * 4 + j - 1, 0], heatmaps_joints[i * 4 + j - 1, 1], s=5, c='w')

    plt.show()
