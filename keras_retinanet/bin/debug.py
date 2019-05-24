#!/usr/bin/env python

"""
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from scipy import stats

import argparse
import random
import os
import sys
import cv2

# Allow relative imports when being executed as script.
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    import keras_retinanet.bin  # noqa: F401
    __package__ = "keras_retinanet.bin"

# Change these to absolute imports if you copy this script outside the keras_retinanet package.
from ..preprocessing.pascal_voc import PascalVocGenerator
from ..preprocessing.csv_generator import CSVGenerator
from ..preprocessing.kitti import KittiGenerator
from ..preprocessing.open_images import OpenImagesGenerator
from ..utils.keras_version import check_keras_version
from ..utils.transform import random_transform_generator
from ..utils.visualization import draw_annotations, draw_boxes
from ..utils.anchors import anchors_for_shape, compute_gt_annotations, compute_gt_annotations_for_visualisation
from ..utils.config import read_config_file, parse_anchor_parameters


def create_generator(args):
    """ Create the data generators.

    Args:
        args: parseargs arguments object.
    """
    # create random transform generator for augmenting training data
    transform_generator = random_transform_generator(
        min_rotation=-0.1,
        max_rotation=0.1,
        min_translation=(-0.1, -0.1),
        max_translation=(0.1, 0.1),
        min_shear=-0.1,
        max_shear=0.1,
        min_scaling=(0.9, 0.9),
        max_scaling=(1.1, 1.1),
        flip_x_chance=0.5,
        flip_y_chance=0.5,
    )

    if args.dataset_type == 'coco':
        # import here to prevent unnecessary dependency on cocoapi
        from ..preprocessing.coco import CocoGenerator

        generator = CocoGenerator(
            args.coco_path,
            args.coco_set,
            transform_generator=transform_generator,
            image_min_side=args.image_min_side,
            image_max_side=args.image_max_side,
            config=args.config
        )
    elif args.dataset_type == 'pascal':
        generator = PascalVocGenerator(
            args.pascal_path,
            args.pascal_set,
            transform_generator=transform_generator,
            image_min_side=args.image_min_side,
            image_max_side=args.image_max_side,
            config=args.config
        )
    elif args.dataset_type == 'csv':
        generator = CSVGenerator(
            args.annotations,
            args.classes,
            transform_generator=transform_generator,
            image_min_side=args.image_min_side,
            image_max_side=args.image_max_side,
            config=args.config
        )
    elif args.dataset_type == 'oid':
        generator = OpenImagesGenerator(
            args.main_dir,
            subset=args.subset,
            version=args.version,
            labels_filter=args.labels_filter,
            parent_label=args.parent_label,
            annotation_cache_dir=args.annotation_cache_dir,
            transform_generator=transform_generator,
            image_min_side=args.image_min_side,
            image_max_side=args.image_max_side,
            config=args.config
        )
    elif args.dataset_type == 'kitti':
        generator = KittiGenerator(
            args.kitti_path,
            subset=args.subset,
            transform_generator=transform_generator,
            image_min_side=args.image_min_side,
            image_max_side=args.image_max_side,
            config=args.config
        )
    else:
        raise ValueError('Invalid data type received: {}'.format(args.dataset_type))

    return generator


def parse_args(args):
    """ Parse the arguments.
    """
    parser     = argparse.ArgumentParser(description='Debug script for a RetinaNet network.')
    subparsers = parser.add_subparsers(help='Arguments for specific dataset types.', dest='dataset_type')
    subparsers.required = True

    coco_parser = subparsers.add_parser('coco')
    coco_parser.add_argument('coco_path',  help='Path to dataset directory (ie. /tmp/COCO).')
    coco_parser.add_argument('--coco-set', help='Name of the set to show (defaults to val2017).', default='val2017')

    pascal_parser = subparsers.add_parser('pascal')
    pascal_parser.add_argument('pascal_path', help='Path to dataset directory (ie. /tmp/VOCdevkit).')
    pascal_parser.add_argument('--pascal-set',  help='Name of the set to show (defaults to test).', default='test')

    kitti_parser = subparsers.add_parser('kitti')
    kitti_parser.add_argument('kitti_path', help='Path to dataset directory (ie. /tmp/kitti).')
    kitti_parser.add_argument('subset', help='Argument for loading a subset from train/val.')

    def csv_list(string):
        return string.split(',')

    oid_parser = subparsers.add_parser('oid')
    oid_parser.add_argument('main_dir', help='Path to dataset directory.')
    oid_parser.add_argument('subset', help='Argument for loading a subset from train/validation/test.')
    oid_parser.add_argument('--version',  help='The current dataset version is v4.', default='v4')
    oid_parser.add_argument('--labels-filter',  help='A list of labels to filter.', type=csv_list, default=None)
    oid_parser.add_argument('--annotation-cache-dir', help='Path to store annotation cache.', default='.')
    oid_parser.add_argument('--parent-label', help='Use the hierarchy children of this label.', default=None)

    csv_parser = subparsers.add_parser('csv')
    csv_parser.add_argument('annotations', help='Path to CSV file containing annotations for evaluation.')
    csv_parser.add_argument('classes',     help='Path to a CSV file containing class label mapping.')

    parser.add_argument('-l', '--loop', help='Loop forever, even if the dataset is exhausted.', action='store_true')
    parser.add_argument('--no-resize', help='Disable image resizing.', dest='resize', action='store_false')
    parser.add_argument('--anchors', help='Show positive anchors on the image.', action='store_true')
    parser.add_argument('--annotations', help='Show annotations on the image. Green annotations have anchors, red annotations don\'t and therefore don\'t contribute to training.', action='store_true')
    parser.add_argument('--random-transform', help='Randomly transform image and annotations.', action='store_true')
    parser.add_argument('--image-min-side', help='Rescale the image so the smallest side is min_side.', type=int, default=800)
    parser.add_argument('--image-max-side', help='Rescale the image if the largest side is larger than max_side.', type=int, default=1333)
    parser.add_argument('--positive_overlap_iou', help='IoU overlap or positive anchors (all anchors with overlap > positive_overlap are positive)', type=float, default=0.5)
    parser.add_argument('--negative_overlap_iou', help='IoU overlap for negative anchors (all anchors with overlap < negative_overlap are negative).', type=float, default=0.4)
    parser.add_argument('--config', help='Path to a configuration parameters .ini file.')
    parser.add_argument('--stats', help='Calculates and prints stats about the anchor coverage across the dataset', action='store_true')

    return parser.parse_args(args)

def calculate_stats(generator, args, anchor_params):
    """ Calculates stats for anchor coverage over given dataset.
        Output stats include:
        - Average number of positive & negative anchors per image
        - Max/min number of positive anchors in dataset
        - Proportion of positive to negative anchors across dataset
    """
    
    annotations_count = []
    missed_annotations_count = []
    positive_anchors_count = []
    negative_anchors_count = []

    num_images = generator.size()
    image_scale = None
    image_shape = None

    print("\n")
    for i in range(num_images):
        print("Processing {}/{} ".format(i, num_images), end="\r")

        annotations = generator.load_annotations(i)
        
        # Skip if there is no annotation label
        if len(annotations['labels']) == 0:
            continue

        # Resize the image and annotations
        # Save the relevant image properties (scale and shape) once and reuse - as we know that all images will have same properties
        # Saving these properties significantly speeds up the process
        if args.resize:
            if(image_scale is None):
                image = generator.load_image(i)
                image, image_scale = generator.resize_image(image)
                image_shape = image.shape

            annotations['bboxes'] *= image_scale
        else:
            if(image_shape is None):
                image = generator.load_image(i)
                image_shape = image.shape

        anchors = anchors_for_shape(image_shape, anchor_params=anchor_params)
        positive_indices, _, _, max_indices = compute_gt_annotations_for_visualisation(anchors, annotations['bboxes'], negative_overlap=args.negative_overlap_iou, positive_overlap=args.positive_overlap_iou)
        
        num_annotations = annotations['bboxes'].shape[0]
        missed_annotations = num_annotations - len(set(max_indices[positive_indices]))
        num_positive_anchors = annotations['bboxes'][max_indices[positive_indices], :].shape[0]

        annotations_count.append(num_annotations)
        missed_annotations_count.append(missed_annotations)
        positive_anchors_count.append(num_positive_anchors)
        negative_anchors_count.append(anchors.shape[0] - num_positive_anchors)

    prop = sum(positive_anchors_count) / sum(negative_anchors_count)    
    missed_annotations_stats = stats.describe(missed_annotations_count)
    positive_anchors_stats = stats.describe(positive_anchors_count)
    negative_anchors_stats = stats.describe(negative_anchors_count)

    print("##############################")
    print(f"\nResults for parameters:\nPositive IoU: {args.positive_overlap_iou}\nNegative IoU: {args.negative_overlap_iou}")
    print(f"\nAnchor parameters: \nsizes: {anchor_params.sizes}\nstrides: {anchor_params.strides}\nratios: {anchor_params.ratios}\nscales: {anchor_params.scales}")
    print("\n-------")
    print(f"\nTotal annotations: {sum(annotations_count)}")
    print(f"\nMissed annotations: \nMin, Max: {missed_annotations_stats.minmax} \nMean: {missed_annotations_stats.mean:.3f}")
    print(f"\nProportion of pos/neg anchors: {prop:.5f}")
    print(f"\nPositive anchors: \nMin, Max: {positive_anchors_stats.minmax}\nMean: {positive_anchors_stats.mean:.3f}")
    print(f"\nNegative anchors: \nMin, Max: {negative_anchors_stats.minmax}\nMean: {negative_anchors_stats.mean:.3f}")

    print("\n")

def run(generator, args, anchor_params):
    """ Main loop.

    Args
        generator: The generator to debug.
        args: parseargs args object.
    """
    # display images, one at a time
    for i in range(generator.size()):
        # load the data
        image       = generator.load_image(i)
        annotations = generator.load_annotations(i)
        if len(annotations['labels']) > 0 :
            # apply random transformations
            if args.random_transform:
                image, annotations = generator.random_transform_group_entry(image, annotations)
                print("Applying random_transforms")

            # resize the image and annotations
            if args.resize:
                image, image_scale = generator.resize_image(image)
                annotations['bboxes'] *= image_scale

            anchors = anchors_for_shape(image.shape, anchor_params=anchor_params)
            positive_indices, _, some_overlap_indices, max_indices = compute_gt_annotations_for_visualisation(anchors, annotations['bboxes'], negative_overlap=args.negative_overlap_iou, positive_overlap=args.positive_overlap_iou)

            # Find the annotations that only have 'some' overlap (i.e. not enough to be a 'positive' index)
            red_boxes_indices_set = set(max_indices[some_overlap_indices]).difference(set(max_indices[positive_indices]))
            red_boxes_indices_list = list(red_boxes_indices_set)

            # Find all overlapping anchors for these annotations
            if(len(red_boxes_indices_list) > 0):
                _, red_ignore_indices, _, red_max_indices = compute_gt_annotations_for_visualisation(anchors, annotations['bboxes'][red_boxes_indices_list, :], negative_overlap=0.01, positive_overlap=args.positive_overlap_iou)
                
                # Draw boxes of anchors that were close but not quite enough for the annotation
                # As there are so many anchors, only show 10% of them
                close_anchors = anchors[red_ignore_indices]
                random_close_anchors_indices = random.choices(range(0, len(close_anchors)), k = int(0.1 * len(close_anchors)))
                draw_boxes(image, close_anchors[random_close_anchors_indices, :], (0, 255, 255), thickness=1)

            # draw anchors on the image
            if args.anchors:
                draw_boxes(image, anchors[positive_indices], (255, 255, 0), thickness=1)

            # draw annotations on the image
            if args.annotations:
                # draw annotations in red
                draw_annotations(image, annotations, color=(0, 0, 255), label_to_name=generator.label_to_name)

                # Draw regressed anchors in green to override most red annotations
                # Result is that annotations without anchors are red, with anchors are green
                draw_boxes(image, annotations['bboxes'][max_indices[positive_indices], :], (0, 255, 0), thickness=1)
                num_positive_anchors = annotations['bboxes'][max_indices[positive_indices], :].shape[0]
                num_negative_anchors = anchors.shape[0] - num_positive_anchors
                print(f"Annotations for image: \t\t{annotations['bboxes'].shape[0]}")
                print(f"Positive anchors for image: \t{num_positive_anchors}")
                print(f"Negative anchors for image: \t{num_negative_anchors}")
                print(f"Proportion of pos/neg anchors: \t{num_positive_anchors / (num_negative_anchors):.3f}\n")

        cv2.imshow('Image', image)

        while True:
            key = cv2.waitKey()
            if key == ord('q'):
                return False

            elif key == ord('n'):
                break

    return True


def main(args=None):
    # parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    # make sure keras is the minimum required version
    check_keras_version()

    # create the generator
    generator = create_generator(args)

    # optionally load config parameters
    if args.config:
        args.config = read_config_file(args.config)

    # optionally load anchor parameters
    anchor_params = None
    if args.config and 'anchor_parameters' in args.config:
        anchor_params = parse_anchor_parameters(args.config)

    if args.loop:
        cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
        while run(generator, args, anchor_params=anchor_params):
            pass
    else:
        if(args.stats):
            calculate_stats(generator, args, anchor_params=anchor_params)
        else:
            cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
            run(generator, args, anchor_params=anchor_params)


if __name__ == '__main__':
    main()
