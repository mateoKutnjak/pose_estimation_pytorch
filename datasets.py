import os
import json
import re
import scipy.misc
import numpy as np
import torch
import util
import util_image
import util_plot
import pdb
import matplotlib.pyplot as plt
import torch.utils.data as data


class MPII_dataset(data.Dataset):

    bones_pairs = [
        (0, 1),     # COMMENT r-ankle <-> r-knee
        (1, 2),     # COMMENT r-knee <-> r-hip
        (2, 6),     # COMMENT r-hip <-> pelvis
        (6, 3),     # COMMENT pelvis <-> l-hip
        (3, 4),     # COMMENT l-hip <-> l-knee
        (4, 5),     # COMMENT l-knee <-> l-ankle

        (6, 7),     # COMMENT pelvis <-> thorax
        (7, 8),     # COMMENT thorax <-> upper neck
        (8, 9),     # COMMENT upper neck <-> head top

        (10, 11),   # COMMENT r-wrist <-> r-elbow
        (11, 12),   # COMMENT r-elbow <-> r-shoulder
        (12, 7),    # COMMENT r-shoulder <-> thorax
        (7, 13),    # COMMENT thorax <-> l-shoulder
        (13, 14),   # COMMENT l-shoulder <-> l-elbow
        (14, 15)    # COMMENT l-elbow <-> l-wrist
    ]

    def __init__(self, dataset_type, images_dir, annots_json_filename, mean_path, std_path, input_shape, output_shape, rotation_factor=30, scale_factor=0.25, sigma=1, **kwargs):
        self.dataset_type = dataset_type
        self.annots_json_filename = annots_json_filename
        self.images_dir = images_dir
        self.mean_path = mean_path
        self.std_path = std_path
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.rotation_factor = rotation_factor
        self.scale_factor = scale_factor
        self.sigma = sigma

        self.train_annots = []
        self.valid_annots = []
        self.mean_total = np.zeros(3)
        self.std_total = np.zeros(3)

        self.num_joints = 16
        self.joint_pairs = (
            [0, 5],     # COMMENT ankles
            [1, 4],     # COMMENT knees
            [2, 3],     # COMMENT hips
            [10, 15],   # COMMENT wrists
            [11, 14],   # COMMENT elbows
            [12, 13]    # COMMENT shoulders
        )

        self.create_dataset()
        self.calculate_mean_std()

    def create_dataset(self):
        with open(self.annots_json_filename) as f:
            json_parsed = json.loads(f.read())

        for index, value in enumerate(json_parsed):
            if value['isValidation'] == 1.0:
                self.valid_annots.append(value)
            elif value['isValidation'] == 0.0:
                self.train_annots.append(value)

    def calculate_mean_std(self):
        if os.path.exists(self.mean_path) and os.path.exists(self.std_path):
            self.mean_total = np.load(self.mean_path)
            self.std_total = np.load(self.std_path)

            print('Train dataset mean and std information found')
        else:
            print('Calculating dataset mean and std...')
            for i, annot in enumerate(self.train_annots):

                image_filename = annot['img_paths']
                filename_search = re.search('(\d\d).jpg$', image_filename, re.IGNORECASE)
                image_last_digits = filename_search.group(1)
                image = scipy.misc.imread(os.path.join(self.images_dir, '_' + str(image_last_digits), image_filename))

                self.mean_total += np.mean(image, axis=(0, 1))
                self.std_total += np.std(image, axis=(0, 1))

                util.print_progress_bar(i, len(self.train_annots))

            self.mean_total /= len(self.train_annots)
            self.std_total /= len(self.train_annots)

        np.save('mean_total', self.mean_total)
        np.save('std_total', self.std_total)

        print('Mean = {}'.format(self.mean_total))
        print('Std = {}'.format(self.std_total))

    def __len__(self):
        if self.dataset_type == 'train':
            return len(self.train_annots)
        elif self.dataset_type == 'valid':
            return len(self.valid_annots)

    def __getitem__(self, index):
        annot = self.train_annots[index] if self.dataset_type == 'train' else self.valid_annots[index]

        image_filename = annot['img_paths']
        center = np.array(annot['objpos'])
        original_joints = np.array(annot['joint_self'])
        scale = annot['scale_provided']
        rotation = 0

        center[1] += 15 * scale
        scale *= 1.25

        filename_search = re.search('(\d\d).jpg$', image_filename, re.IGNORECASE)
        image_last_digits = filename_search.group(1)
        image = scipy.misc.imread(os.path.join(self.images_dir, '_' + str(image_last_digits), image_filename)) / 255.0

        if self.dataset_type == 'train':
            flip = np.random.rand() < 0.5
            scale *= np.clip(np.random.normal() * self.scale_factor + 1, 1-self.scale_factor, 1+self.scale_factor)
            rotation = np.clip(np.random.normal() * self.rotation_factor, -2*self.rotation_factor, 2*self.rotation_factor) if np.random.rand() < 0.6 else 0

            if flip:
                image = np.fliplr(image)
                original_joints = util_image.flip_joints(original_joints, image.shape[1])
                center[0] = image.shape[1] - center[0]

            image[:, :, 0] *= np.random.uniform(0.8, 1.2)
            image[:, :, 1] *= np.random.uniform(0.8, 1.2)
            image[:, :, 2] *= np.random.uniform(0.8, 1.2)
            image = np.clip(image, 0, 1)

        _input = util_image.crop(image, center, scale, [self.input_shape, self.input_shape], rot=rotation).astype(np.float)

        transformed_joints = original_joints.copy()
        for i in range(self.num_joints):
            if transformed_joints[i, 0] > 0 and transformed_joints[i, 1] > 0:
                transformed_joints[i, 0:2] = util_image.transform(transformed_joints[i, 0:2] + 1, center, scale, [self.input_shape, self.input_shape], rot=rotation)

        for i in range(_input.shape[-1]):
            _input[:, :, i] -= self.mean_total[i]
            _input[:, :, i] /= self.std_total[i]
        _input = np.clip(_input, -1, 1)
        _input = (_input + 1.0) / 2.0

        _input = np.transpose(_input, (2, 0, 1))

        # TODO change range from [0, 255] to [0, 1] ???

        heatmaps_joints = original_joints.copy()
        joint_visibility = original_joints[:, 2].copy().reshape(self.num_joints, 1)
        heatmaps = np.zeros((self.num_joints, self.output_shape, self.output_shape))

        for i in range(self.num_joints):
            if heatmaps_joints[i, 0] > 0 and heatmaps_joints[i, 1] > 0:
                heatmaps_joints[i, 0:2] = util_image.transform(heatmaps_joints[i, 0:2] + 1, center, scale, [self.output_shape, self.output_shape], rot=rotation)
                heatmaps[i, :, :], vis = util_image.draw_labelmap(heatmaps[i, :, :], heatmaps_joints[i, 0:2],
                                                                        self.sigma, type='Gaussian')
                joint_visibility[i, 0] *= vis

        _input = torch.from_numpy(_input)
        heatmaps = torch.from_numpy(heatmaps)

        meta = {
            'img_paths': image_filename,
            'index': index,
            'center': center,
            'scale': scale,
            'original_joints': original_joints,
            'transformed_joints': transformed_joints,
            'heatmaps_joints': heatmaps_joints,
            'label_weights': joint_visibility
        }

        # util_plot.plot_dataset_example(_input, meta , heatmaps)

        return _input, heatmaps, meta