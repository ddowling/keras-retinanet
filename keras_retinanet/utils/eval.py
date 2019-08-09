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

from .anchors import compute_overlap
from .visualization import draw_detections, draw_annotations

import keras
import numpy as np
import os

import matplotlib.pyplot as plt
import pyclipper
import cv2
import progressbar
assert(callable(progressbar.progressbar)), "Using wrong progressbar module, install 'progressbar2' instead."


def _compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.

    Code originally from https://github.com/rbgirshick/py-faster-rcnn.

    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def _get_detections(generator, model, score_threshold=0.05, max_detections=100, save_path=None):
    """ Get the detections from the model using the generator.

    The result is a list of lists such that the size is:
        all_detections[num_images][num_classes] = detections[num_detections, 4 + num_classes]

    # Arguments
        generator       : The generator used to run images through the model.
        model           : The model to run on the images.
        score_threshold : The score confidence threshold to use.
        max_detections  : The maximum number of detections to use per image.
        save_path       : The path to save the images with visualized detections to.
    # Returns
        A list of lists containing the detections for each image in the generator.
    """
    all_detections = [[None for i in range(generator.num_classes()) if generator.has_label(i)] for j in range(generator.size())]

    for i in progressbar.progressbar(range(generator.size()), prefix='Running network: '):
        raw_image    = generator.load_image(i)
        image        = generator.preprocess_image(raw_image.copy())
        image, scale = generator.resize_image(image)

        if keras.backend.image_data_format() == 'channels_first':
            image = image.transpose((2, 0, 1))

        # run network
        boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))[:3]

        # correct boxes for image scale
        boxes /= scale

        # select indices which have a score above the threshold
        indices = np.where(scores[0, :] > score_threshold)[0]

        # select those scores
        scores = scores[0][indices]

        # find the order with which to sort the scores
        scores_sort = np.argsort(-scores)[:max_detections]

        # select detections
        image_boxes      = boxes[0, indices[scores_sort], :]
        image_scores     = scores[scores_sort]
        image_labels     = labels[0, indices[scores_sort]]
        image_detections = np.concatenate([image_boxes, np.expand_dims(image_scores, axis=1), np.expand_dims(image_labels, axis=1)], axis=1)

        if save_path is not None:
            draw_annotations(raw_image, generator.load_annotations(i), label_to_name=generator.label_to_name)
            draw_detections(raw_image, image_boxes, image_scores, image_labels, label_to_name=generator.label_to_name, score_threshold=score_threshold)

            cv2.imwrite(os.path.join(save_path, '{}.png'.format(i)), raw_image)

        # copy detections to all_detections
        for label in range(generator.num_classes()):
            if not generator.has_label(label):
                continue

            all_detections[i][label] = image_detections[image_detections[:, -1] == label, :-1]

    return all_detections


def _get_annotations(generator):
    """ Get the ground truth annotations from the generator.

    The result is a list of lists such that the size is:
        all_detections[num_images][num_classes] = annotations[num_detections, 5]

    # Arguments
        generator : The generator used to retrieve ground truth annotations.
    # Returns
        A list of lists containing the annotations for each image in the generator.
    """
    all_annotations = [[None for i in range(generator.num_classes())] for j in range(generator.size())]

    for i in progressbar.progressbar(range(generator.size()), prefix='Parsing annotations: '):
        # load the annotations
        annotations = generator.load_annotations(i)

        # copy detections to all_annotations
        for label in range(generator.num_classes()):
            if not generator.has_label(label):
                continue

            all_annotations[i][label] = annotations['bboxes'][annotations['labels'] == label, :].copy()

    return all_annotations


def evaluate(
    generator,
    model,
    conductor_eval,
    epoch=None,
    iou_threshold=0.5,
    score_threshold=0.05,
    max_detections=100,
    save_path=None,
    plot_interval=5
):
    """ Evaluate a given dataset using a given model.

    # Arguments
        generator       : The generator that represents the dataset to evaluate.
        model           : The model to evaluate.
        conductor_eval  : Boolean that specifies if conductor-detection specific mAP calculation should be used for evaluation
        epoch           : The current epoch
        iou_threshold   : The threshold used to consider when a detection is positive or negative.
        score_threshold : The score confidence threshold to use for detections.
        max_detections  : The maximum number of detections to use per image.
        save_path       : The path to save images with visualized detections to.
        plot_interval   : Inteval at which the Precision-Recall curve should be plotted & saved
    # Returns
        A dict mapping class names to mAP scores.
    """
    # gather all detections and annotations
    all_detections     = _get_detections(generator, model, score_threshold=score_threshold, max_detections=max_detections, save_path=save_path)
    all_annotations    = _get_annotations(generator)
    average_precisions = {}
    recall_precision_tuples = []

    # process detections and annotations
    for label in range(generator.num_classes()):
        if not generator.has_label(label):
            continue

        false_positives = np.zeros((0,))
        true_positives  = np.zeros((0,))
        scores          = np.zeros((0,))
        num_annotations = 0.0

        for i in range(generator.size()):
            detections           = all_detections[i][label]
            annotations          = all_annotations[i][label]
            num_annotations     += annotations.shape[0]
            detected_annotations = []

            for d in detections:
                eval_fn = conductor_evaluate if conductor_eval else normal_evaluate
                tp, fp, score, assigned_annotation = eval_fn(d, annotations, detected_annotations, iou_threshold)
                
                # Append values if they are valid
                if(fp is not None): false_positives = np.append(false_positives, fp)
                if(tp is not None): true_positives  = np.append(true_positives, tp)
                if(score is not None): scores = np.append(scores, d[4])
                if(assigned_annotation is not None): detected_annotations.append(assigned_annotation)

        # no annotations -> AP for this class is 0 (is this correct?)
        if num_annotations == 0:
            average_precisions[label] = 0, 0
            continue

        # sort by score
        indices         = np.argsort(-scores)
        false_positives = false_positives[indices]
        true_positives  = true_positives[indices]

        # compute false positives and true positives
        false_positives = np.cumsum(false_positives)
        true_positives  = np.cumsum(true_positives)

        # compute recall and precision
        recall    = true_positives / num_annotations
        precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

        # Save values for plotting
        recall_precision_tuples.append((recall, precision, generator.label_to_name(label)))

        # compute average precision
        average_precision  = _compute_ap(recall, precision)
        average_precisions[label] = average_precision, num_annotations

    # Create precision recall curve if an epoch is specified
    if(epoch is not None):
        fig = None
        if(epoch % plot_interval == 0):
            fig = create_curve(recall_precision_tuples, "Precision-Recall Curve")

        return average_precisions, fig

    else:
        return average_precisions

def normal_evaluate(detection, annotations, detected_annotations, iou_threshold):
    """ 
    """
    true_positive, false_positive, assigned_annotation = None, None, None
    score = detection[4]

    # If there are no ground-truth annotations, the network detection is a false positive
    if annotations.shape[0] == 0:
        false_positive = 1
        true_positive = 0
        return (true_positive, false_positive, score, assigned_annotation)

    overlaps = compute_overlap(np.expand_dims(detection, axis=0), annotations)
    assigned_annotation = np.argmax(overlaps, axis=1)
    max_overlap = overlaps[0, assigned_annotation]

    # True positive
    if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
        false_positive = 0
        true_positive  = 1

    # False positive
    else:
        false_positive = 1
        true_positive  = 0
        assigned_annotation = None

    return (true_positive, false_positive, score, assigned_annotation)

###    
### Custom Evaluation for Conductors
###

def conductor_evaluate(detection, annotations, detected_annotations, iou_threshold):
    true_positive, false_positive, assigned_annotation, score = None, None, None, None
    score_index = 4

    # If there are no ground-truth annotations, the network detection is a false positive
    if annotations.shape[0] == 0:
        false_positive = 1
        true_positive = 0
        return (true_positive, false_positive, detection[score_index], assigned_annotation)

    overlaps = compute_overlap(np.expand_dims(detection, axis=0), annotations)
    assigned_annotation = np.argmax(overlaps, axis=1)
    initial_overlap = overlaps[0, assigned_annotation]

    # Get neighbouring annotations (if they exist)
    left_annot = None if (assigned_annotation - 1) < 0 else annotations[assigned_annotation - 1][0]
    right_annot = None if (assigned_annotation + 1) >= annotations.shape[0] else annotations[assigned_annotation + 1][0]

    max_overlap = compute_conductor_overlap(detected_box=detection[:4], main_annot=annotations[assigned_annotation][0], left_annot=left_annot, right_annot=right_annot)
    max_overlap = initial_overlap if max_overlap is None else np.array([max_overlap], dtype=np.float32)

    # True positive
    if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
        false_positive = 0
        true_positive  = 1
        score = detection[4]

    # False positive
    elif max_overlap < iou_threshold:
        false_positive = 1
        true_positive  = 0
        score = detection[4]
        assigned_annotation = None

    # Duplicate detection - ignore this case
    elif assigned_annotation in detected_annotations:
        false_positive = None
        true_positive  = None
        score = detection = None
        assigned_annotation = None

    return (true_positive, false_positive, score, assigned_annotation)

def compute_conductor_overlap(detected_box, main_annot, left_annot, right_annot):
    """ Computes conductor overlap by taking into account the neighbouring annotations. The default RetinaNet implementation uses the
        largest overlapping annoation to calculate IoU, but this doesn't make sense for our case. As we split the long and thin Conductors
        arbitrarily (to make more 'squarish' objects), there is no 'correct' ground-truth annotation. Visual explanation:

        Diagram keys:
        - Single thin line      = Conductor
        - Double border box     = Largest overlapping ground-truth annotation (as determined by `compute_overlap`)
        - Single border box     = Network's guess at where object is
        - Dashed border boxe    = Neighbouring ground-truth annotations
        - The '#' area          = Correctly identified area                
        
        RetinaNet would compute this:

                  ┌──────────────────┐                     
        ┌ ─ ─ ─ ─ ┼ ─ ─ ─ ╔══════════│═══════╗ ─ ─ ─ ─ ─ ┐ 
                  │      │║##########│       ║│            
       ─┼─────────│───────╬##########│───────╬───────────┼─
                  │      │║##########│       ║│            
        └ ─ ─ ─ ─ ┼ ─ ─ ─ ╚══════════│═══════╝ ─ ─ ─ ─ ─ ┘ 
                  └──────────────────┘                          

        But the correct solution is to blend neighbouring annotations to match the network's prediction, and then compute the IoU
        (thick-border box is the blended annotation):

                  ┌──────────────────┐                     
        ┌ ─ ─ ─ ─ ┏━━━━━━━━━━━━━━━━━━┓═══════╗┌ ─ ─ ─ ─ ─  
                  ┃##################┃       ║           │ 
       ─┼─────────┃##################┃───────╬┼────────────
                  ┃##################┃       ║           │ 
        └ ─ ─ ─ ─ ┗━━━━━━━━━━━━━━━━━━┛═══════╝└ ─ ─ ─ ─ ─  
                  └──────────────────┘                     
        
        Note: Above ASCII representations are a bit simplified as it's hard to draw rotated rectangles and polygons.

        # Arguments
            detected_box   : Network's prediction (ndarray)
            main_annot      : Ground-truth annotation with largest overlap against network's detection (ndarray)
            left_annot      : Annotation to the left of `main_annot` (ndarray), or None if it doesn't exist 
            right_annot     : Annotation to the right of `main_annot` (ndarray), or None if it doesn't exist 

        # Returns
            The computed overlap (IoU) between `detected_box` and given annotations
    """

    assert main_annot.shape[0] == detected_box.shape[0], "annotation shape '{}' vs detected box shape '{}'".format(main_annot.shape[0], detected_box.shape[0])
    assert main_annot.shape[0] == 4

    pc = pyclipper.Pyclipper()
    ext_main_annot = main_annot

    # Easy way to index coordinates
    x1, _, x2, _ = (0,1,2,3)

    ## Shrink the edges (both left and right) of the `main` annotation if they extend past the detection.
    # Visual example for left edge:
    #
    #    ┌────────────┐       ┌────────────┐
    # ╔═══════════╗   │       ╔════════╗   │
    # ║  │        ║   │   ->  ║        ║   │
    # ╚═══════════╝   │       ╚════════╝   │
    #    └────────────┘       └────────────┘
    #

    # Left edge
    if(ext_main_annot[x1] < detected_box[x1]):
        ext_main_annot[x1] = detected_box[x1]

    # Right edge
    if(ext_main_annot[x2] > detected_box[x2]):
        ext_main_annot[x2] = detected_box[x2]

    pc.AddPath(box_to_path(ext_main_annot), pyclipper.PT_CLIP, True)

    ## Now we look at extending the edges of the `main` annotation by using it's neighbours (see diagram in docstring above)
    # The minimum length threshold to check against is 1. This is to guard against situations where we end up with 0 length boxes, as PyClipper
    # truncates values, so a length of 0.8 = 0.0

    # Check if left edge of detected box extends past the `main` annotation
    if(not(left_annot is None) and ((main_annot[x1] - detected_box[x1]) > 1)):
        ext_left_annot = left_annot

        # Only (likely) require a portion of the neighbouring annotation, so shrink the edges
        ext_left_annot[x1] = detected_box[x1]
        ext_left_annot[x2] = main_annot[x1]
        pc.AddPath(box_to_path(ext_left_annot), pyclipper.PT_CLIP, True)
    
    # Check the right edge
    if(not(right_annot is None) and ((detected_box[x2] - main_annot[x2]) > 1)):
        ext_right_annot = right_annot
        ext_right_annot[x1] = main_annot[x2]
        ext_right_annot[x2] = detected_box[x2]
        pc.AddPath(box_to_path(ext_right_annot), pyclipper.PT_CLIP, True)

    # Add the detected box as the subject and clip
    pc.AddPath(box_to_path(detected_box), pyclipper.PT_SUBJECT, True)
    intersection_poly = pc.Execute(pyclipper.CT_INTERSECTION, pyclipper.PFT_EVENODD, pyclipper.PFT_EVENODD)
    union_poly = pc.Execute(pyclipper.CT_UNION, pyclipper.PFT_EVENODD, pyclipper.PFT_EVENODD)

    # If there is no intersection, return 0.0
    if(len(intersection_poly) == 0):
        return 0.0

    intersection_area = pyclipper.Area(intersection_poly[0])
    union_area = pyclipper.Area(union_poly[0])

    return intersection_area / union_area

def box_to_path(box):
    """Convert an array of coordinates [x1, y1, x2, y2] to tuple that PyClipper accepts: (((x1, y1), (x2, y1), (x2, y2), (x1, y2)))"""
    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    return ((x1, y1), (x2, y1), (x2, y2), (x1, y2))

def create_curve(recall_precision_tuples, title):
    
    # Create a 500x500 pixel plot
    fig = plt.figure(figsize=(5, 5), dpi=100)
    ax = fig.add_subplot(111)

    # Add a line for each set of precision-recall values
    # Replace lw=2 with '-ok' for plotting points + line
    for recall, precision, label in recall_precision_tuples:
        _, = ax.plot(recall, precision, lw=1, label=label)

    # Set up the x and y axis
    ax.set_title(title)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim([0.0,1.1])
    ax.set_ylim([0.0,1.1])

    # Custom ticks
    ax.xaxis.set_ticks(np.arange(0.0, 1.1, 0.1))
    ax.yaxis.set_ticks(np.arange(0.0, 1.1, 0.1))

    plt.legend(loc='upper right')
    return fig