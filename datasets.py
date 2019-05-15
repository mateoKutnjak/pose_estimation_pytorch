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
                heatmaps[i, :, :], vis = util_image.draw_labelmap(heatmaps[i, :, :], heatmaps_joints[i, 0:2] - 1,
                                                                        self.sigma, type='Gaussian')
                joint_visibility[i, 0] *= vis

        # util_plot.plot_input(_input, center, transformed_joints, heatmaps)

        _input = torch.from_numpy(_input)
        heatmaps = torch.from_numpy(heatmaps)

        meta = {
            'index': index,
            'center': center,
            'scale': scale,
            'original_joints': original_joints,
            'transformed_joints': transformed_joints,
            'heatmaps_joints': heatmaps_joints,
            'label_weights': joint_visibility
        }

        return _input, heatmaps, meta

    # # TODO shuffle
    # def generate_batches(self, batch_size, stacks_num, metadata_flag=False):
    #     input_batch = np.zeros(shape=(batch_size,) + self.input_shape)
    #     output_batch = np.zeros(shape=(batch_size,) + self.output_shape)
    #     metadatas = []
    #
    #     while True:
    #         if self.type == 'train':
    #             shuffle(self.annots)
    #
    #         for index, annotation in enumerate(self.annots):
    #             batch_index = index % batch_size
    #
    #             input_image, output_labelmaps, metadata = self.process_image(
    #                 annotation=annotation,
    #                 flip_flag=False,
    #                 scale_flag=False,
    #                 rotation_flag=False,
    #                 metadata_flag=True
    #             )
    #
    #             input_batch[batch_index, :, :, :] = input_image
    #             output_batch[batch_index, :, :, :] = output_labelmaps
    #             metadatas.append(metadata)
    #
    #             if batch_index == batch_size-1:
    #                 output_batch_total = []
    #                 # _metadatas = metadata.copy()
    #
    #                 for _ in range(stacks_num):
    #                     output_batch_total.append(output_batch)
    #
    #                 if metadata_flag:
    #                     yield input_batch, output_batch_total, metadatas
    #                     metadatas = []
    #                 else:
    #                     yield input_batch, output_batch_total
    #
    # # TODO send original non modified imaga and create one image as data augmentation
    # # COMMENT now it is only data augmentation with random modifications
    # def process_image(self, annotation, flip_flag, scale_flag, rotation_flag, metadata_flag=False):
    #     # image_filename = annotation['img_paths']
    #     #
    #     # filename_search = re.search('(\d\d).jpg$', image_filename, re.IGNORECASE)
    #     # image_last_digits = filename_search.group(1)
    #     #
    #     # image = scipy.misc.imread(os.path.join(self.images_dir, '_' + str(image_last_digits), image_filename))
    #     #
    #     # obj_center = np.array(annotation['objpos'])
    #     # obj_joints = np.array(annotation['joint_self'])
    #     # scale = annotation['scale_provided']
    #     #
    #     # obj_joints_visibilities = obj_joints[:, 2]
    #     # obj_joints = obj_joints[:, :2]
    #     #
    #     # # COMMENT To avoid joints cropping
    #     # scale *= 1.25
    #     # angle = 0
    #     #
    #     # # TODO change values for probs
    #     # if flip_flag and np.random.sample() > 0:
    #     #     image, obj_center, obj_joints = self.flip(
    #     #         original_image=image,
    #     #         obj_center=obj_center,
    #     #         obj_joints=obj_joints
    #     #     )
    #     # if scale_flag and np.random.sample() > 0:
    #     #     scale *= np.random.uniform(0.75, 1.25)
    #     # if rotation_flag and np.random.sample() > 0:
    #     #     angle = np.random.randint(-30, 30)
    #     #     image, obj_center, obj_joints = preprocessing.rotate(
    #     #         original_image=image,
    #     #         obj_center=obj_center,
    #     #         obj_joints=obj_joints,
    #     #         angle=angle
    #     #     )
    #     #
    #     # # plot_utils.plot_processed_image(image, obj_center, obj_joints, scale, angle)
    #     #
    #     # image, obj_center, obj_joints = preprocessing.crop(
    #     #     original_image=image,
    #     #     obj_center=obj_center,
    #     #     obj_joints=obj_joints,
    #     #     scale=scale
    #     # )
    #     #
    #     # # plot_utils.plot_processed_image(image, obj_center, obj_joints, scale, angle)
    #     #
    #     # image, obj_center, obj_joints = preprocessing.resize(
    #     #     original_image=image,
    #     #     obj_center=obj_center,
    #     #     obj_joints=obj_joints,
    #     #     shape=self.input_shape[:-1]
    #     # )
    #     #
    #     # # plot_utils.plot_processed_image(image, obj_center, obj_joints, scale, angle, draw_bbox=False)
    #     #
    #     # image = self.normalize(original_image=image)
    #     #
    #     # # plot_utils.plot_processed_image(image, obj_joints, obj_center, scale, angle, draw_bbox=False)
    #     #
    #     # labelmap_joints = preprocessing.scale_points(
    #     #     input_res=image.shape[:-1],
    #     #     output_res=self.output_shape[:-1],
    #     #     points=obj_joints,
    #     # )
    #     #
    #     # labelmaps = self.generate_labelmaps(
    #     #     obj_joints=labelmap_joints,
    #     #     obj_joints_visibilities=obj_joints_visibilities,
    #     #     sigma=self.sigma)
    #     #
    #     # # plot_utils.plot_labelmaps(image, obj_joints, labelmaps, labelmap_joints)
    #     #
    #     # metadata = {
    #     #     'obj_center': obj_center,
    #     #     # 'obj_joints': np.hstack((obj_joints, np.reshape(obj_joints_visibilities, (obj_joints_visibilities.shape[0], 1)))),
    #     #   fa  'obj_joints': obj_joints,
    #     #     'obj_joints_visibilities': obj_joints_visibilities,
    #     #     'scale': scale,
    #     #     'image_filename': image_filename
    #     # }
    #     #
    #     # if not metadata_flag:
    #     #     return image, labelmaps
    #     # else:
    #     #     return image, labelmaps, metadata
    #
    #     image_filename = annotation['img_paths']
    #     filename_search = re.search('(\d\d).jpg$', image_filename, re.IGNORECASE)
    #     image_last_digits = filename_search.group(1)
    #
    #
    #     image = scipy.misc.imread(os.path.join(self.images_dir, '_' + str(image_last_digits), image_filename))
    #
    #
    #     # get center
    #     center = np.array(annotation['objpos'])
    #     joints = np.array(annotation['joint_self'])
    #     scale = annotation['scale_provided']
    #
    #     # Adjust center/scale slightly to avoid cropping limbs
    #     if center[0] != -1:
    #         center[1] = center[1] + 15 * scale
    #         scale = scale * 1.25
    #
    #     # filp
    #     if flip_flag and random.choice([0, 1]):
    #         image, joints, center = self.flip(image, joints, center)
    #
    #     # scale
    #     if scale_flag:
    #         scale = scale * np.random.uniform(0.8, 1.2)
    #
    #     # rotate image
    #     if rotation_flag and random.choice([0, 1]):
    #         rot = np.random.randint(-1 * 30, 30)
    #     else:
    #         rot = 0
    #
    #     cropimg = their_preprocessing.crop(image, center, scale, self.input_shape, rot)
    #     cropimg = their_preprocessing.normalize(cropimg, self.color_mean)
    #
    #     # transform keypoints
    #     transformedKps = their_preprocessing.transform_kp(joints, center, scale, self.output_shape, rot)
    #     gtmap = their_preprocessing.generate_gtmap(transformedKps, self.sigma, self.output_shape)
    #
    #     # meta info
    #     metainfo = {'center': center, 'scale': scale,
    #                 'pts': joints, 'obj_joints': transformedKps, 'name': image_filename}
    #
    #     return cropimg, gtmap, metainfo
    #
    # def normalize(self, original_image):
    #     original_image = np.true_divide(original_image, 255.0)
    #
    #     for i in range(original_image.shape[-1]):
    #         original_image[:, :, i] -= self.color_mean[i]
    #     return original_image
    #
    # def flip(self, original_image, obj_center, obj_joints):
    #     flipped_joints = np.copy(obj_joints)
    #
    #     im_height, im_width, im_channels = original_image.shape
    #
    #     flipped_image = cv2.flip(original_image, flipCode=1)  # COMMENT mirrors image x coordinates
    #     flipped_joints[:, 0] = im_width - obj_joints[:, 0]  # COMMENT mirrors joints x coordinates
    #
    #     for joint_pair in self.joint_pairs:
    #         temp = np.copy(flipped_joints[joint_pair[0], :])
    #         flipped_joints[joint_pair[0], :] = flipped_joints[joint_pair[1], :]
    #         flipped_joints[joint_pair[1], :] = temp
    #
    #     flipped_center = np.copy(obj_center)
    #     flipped_center[0] = im_width - obj_center[0]
    #
    #     return flipped_image, flipped_center, flipped_joints
    #
    # def generate_labelmaps(self, obj_joints, obj_joints_visibilities, sigma):
    #     labelmaps = np.zeros(shape=(self.output_shape[0], self.output_shape[1], obj_joints.shape[0]))
    #
    #     for i in range(len(obj_joints)):
    #         if obj_joints[i, 0] > 0 and obj_joints[i, 1] > 0 and obj_joints_visibilities[i] > 0.0:
    #             labelmaps[:, :, i] = preprocessing.generate_labelmap(labelmaps[:, :, i], obj_joints[i], sigma)
    #
    #     return labelmaps
