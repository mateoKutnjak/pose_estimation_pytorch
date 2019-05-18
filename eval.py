import torch
import numpy as np


# COMMENT change
def heatmap_argmax(heatmap):
    maxs, max_positions = torch.max(heatmap.reshape(heatmap.shape[0], heatmap.shape[1], -1), dim=2)

    max_positions = max_positions.unsqueeze(2).repeat(1, 1, 2).float()
    max_positions[:, :, 0] = torch.fmod(max_positions[:, :, 0], heatmap.shape[3]).float()
    max_positions[:, :, 1] = torch.floor(max_positions[:, :, 1].float() / heatmap.shape[3])

    # COMMENT position of max is ['1', '0']
    # COMMENT if max is smaller then zero set pixels to zero

    mask_le_0 = maxs.le(0).unsqueeze(2).repeat(1, 1, 2).float() * -heatmap.shape[3]
    max_positions += mask_le_0
    return max_positions


def joint_distances(pred_pos, real_pos, norm):
    distances = torch.ones(pred_pos.shape[0], pred_pos.shape[1]) * -1  # 8 x 16

    for batch_index in range(distances.shape[0]):
        for joint_index in range(distances.shape[1]):

            if pred_pos[batch_index, joint_index, 0] >= 0 and pred_pos[batch_index, joint_index, 1] >= 0:

                distances[batch_index, joint_index] = torch.dist(
                    pred_pos[batch_index, joint_index, :],
                    real_pos[batch_index, joint_index, :]) / norm

    return distances


def joint_distance_accuracy(distances, threshold):

    # COMMENT counts distances for each joint which are le(threshold) and > 0
    inside_threshold = torch.sum(distances.le(threshold) * distances.ge(0), dim=0)
    # COMMENT counts valid distances for each joint (where value is > 0)

    valid = distances.shape[0] - torch.sum(distances.lt(0), dim=0)

    # COMMENT percentage of valid batch joints inside threshold
    accuracy_per_joints = inside_threshold.float() / valid.float()
    # COMMENT set Nan values to -1.0 when no valid joints are present
    accuracy_per_joints[accuracy_per_joints != accuracy_per_joints] = -1.0

    valid_accuracies = accuracy_per_joints[accuracy_per_joints >= 0]
    average_accuracy = torch.sum(valid_accuracies) / valid_accuracies.shape[0]

    return accuracy_per_joints, average_accuracy.item()


def output_accuracy(y, y_kappa, threshold=0.5, norm=6.4):

    # TODO add joint which are used to compute accuracy

    pred_pos = heatmap_argmax(y)
    real_pos = heatmap_argmax(y_kappa)

    distances = joint_distances(pred_pos, real_pos, norm)

    accuracy_per_joint, average_accuracy = joint_distance_accuracy(distances, threshold)
    return distances, accuracy_per_joint, average_accuracy

