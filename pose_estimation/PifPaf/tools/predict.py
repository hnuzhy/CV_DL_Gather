"""Predict poses for given images."""

import argparse
import glob
import json
import logging
import os

import numpy as np
import PIL
import torch

from PIL import Image

from . import show
from . import transforms

import sys
sys.path.append(os.path.abspath('../'))

from network import nets
import decoder

LOG = logging.getLogger(__name__)

def cli():
    parser = argparse.ArgumentParser(
        # prog='python3 -m openpifpaf.predict -q --disable-cuda False',
        prog='python3 -m openpifpaf.predict -q',
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    nets.cli(parser)
    # decoder.cli(parser, force_complete_pose=False, instance_threshold=0.1, seed_threshold=0.5)
    decoder.cli(parser, force_complete_pose=False, instance_threshold=0.2, seed_threshold=0.5)
    parser.add_argument('images', nargs='*',
                        help='input images')
    parser.add_argument('--glob',
                        help='glob expression for input images (for many images)')
    parser.add_argument('-o', '--output-directory',
                        help=('Output directory. When using this option, make '
                              'sure input images have distinct file names.'))
    parser.add_argument('--show', default=False, action='store_true',
                        help='show image of output overlay')
    parser.add_argument('--output-types', nargs='+', default=['skeleton', 'json', 'keypoints'],  # 'skeleton', 'json', 'keypoints'
                        help='what to output: skeleton, keypoints, json')
    parser.add_argument('--batch-size', default=1, type=int,
                        help='processing batch size')
    parser.add_argument('--input-scale', default=0.8, type=float,
                        help='resize the input image shape to do faster inference')
    parser.add_argument('--long-edge', default=None, type=int,
                        help='apply preprocessing to batch images')
    parser.add_argument('--loader-workers', default=2, type=int,
                        help='number of workers for data loading')
    parser.add_argument('--disable-cuda', action='store_true',
                        help='disable CUDA')
    parser.add_argument('--figure-width', default=10.0, type=float,
                        help='figure width')
    parser.add_argument('--dpi-factor', default=1.0, type=float,
                        help='increase dpi of output image by this factor')
    group = parser.add_argument_group('logging')
    group.add_argument('-q', '--quiet', default=False, action='store_true',
                       help='only show warning messages or above')
    group.add_argument('--debug', default=False, action='store_true',
                       help='print debug messages')
    args = parser.parse_args()

    log_level = logging.INFO
    if args.quiet:
        log_level = logging.WARNING
    if args.debug:
        log_level = logging.DEBUG
    logging.basicConfig(level=log_level)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)

    # glob
    if args.glob:
        args.images += glob.glob(args.glob)
    # if not args.images:
        # raise Exception("no image files given")

    # add args.device
    args.device = torch.device('cpu')
    args.pin_memory = False
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
        args.pin_memory = True

    return args


def bbox_from_keypoints(kps):
    m = kps[:, 2] > 0
    if not np.any(m):
        return [0, 0, 0, 0]

    x, y = np.min(kps[:, 0][m]), np.min(kps[:, 1][m])
    w, h = np.max(kps[:, 0][m]) - x, np.max(kps[:, 1][m]) - y
    return [x, y, w, h]


def inference(model_path=None,imgs_path=None, use_gpu=False):
    args = cli()
    
    if model_path is not None:
        args.checkpoint = model_path
        
    if imgs_path is not None:
        args.images = imgs_path
    if not args.images:
        raise Exception("no image files given")
    
    if use_gpu:
        args.device = torch.device('cuda')
        args.pin_memory = True

    # load model
    model_cpu, _ = nets.factory_from_args(args)
    model = model_cpu.to(args.device)
    processor = decoder.factory_from_args(args, model)

    # data
    preprocess = None
    if args.long_edge:
        preprocess = transforms.Compose([
            transforms.NormalizeAnnotations(),
            transforms.RescaleAbsolute(args.long_edge),
            transforms.CenterPad(args.long_edge),
            transforms.EVAL_TRANSFORM,
        ])
    data = transforms.ImageList(args.images, args.input_scale, preprocess=preprocess)
    data_loader = torch.utils.data.DataLoader(
        data, batch_size=args.batch_size, shuffle=False,
        pin_memory=args.pin_memory, num_workers=args.loader_workers,
        collate_fn=transforms.collate_images_anns_meta)

    json_dict_list = []
    for batch_i, (image_tensors_batch, _, meta_batch) in enumerate(data_loader):
        image_tensors_batch_gpu = image_tensors_batch.to(args.device, non_blocking=True)
        fields_batch = processor.fields(image_tensors_batch_gpu)
        pred_batch = processor.annotations_batch(fields_batch, debug_images=image_tensors_batch)

        # unbatch
        for pred, meta in zip(pred_batch, meta_batch):
            if args.output_directory is None:
                output_path = meta['file_name']
            else:
                file_name = os.path.basename(meta['file_name'])
                output_path = os.path.join(args.output_directory, file_name)
            LOG.info('batch %d: %s to %s', batch_i, meta['file_name'], output_path)

            # load the original image if necessary
            cpu_image = None
            if args.debug or \
               'keypoints' in args.output_types or \
               'skeleton' in args.output_types:
                with open(meta['file_name'], 'rb') as f:
                    cpu_image = PIL.Image.open(f).convert('RGB')

            processor.set_cpu_image(cpu_image, None)
            if preprocess is not None:
                pred = preprocess.annotations_inverse(pred, meta)

            if 'json' in args.output_types:
                with open(output_path + '.pifpaf.json', 'w') as f:
                    json.dump([
                        {
                            'keypoints': np.around(ann.data, 1).reshape(-1).tolist(),
                            'bbox': np.around(bbox_from_keypoints(ann.data), 1).tolist(),
                            'score': round(ann.score(), 3),
                        }
                        for ann in pred
                    ], f)

            if 'keypoints' in args.output_types:
                # visualizers
                keypoint_painter = show.KeypointPainter(show_box=True, linewidth=1, markersize=1)
                with show.image_canvas(cpu_image,
                                       output_path + '.keypoints.png',
                                       show=args.show,
                                       fig_width=args.figure_width,
                                       dpi_factor=args.dpi_factor) as ax:
                    keypoint_painter.annotations(ax, pred)

            if 'skeleton' in args.output_types:
                skeleton_painter = show.KeypointPainter(color_connections=True,
                    show_box=False, markersize=1, linewidth=6)
                with show.image_canvas(cpu_image,
                                       output_path + '.skeleton.png',
                                       show=args.show,
                                       fig_width=args.figure_width,
                                       dpi_factor=args.dpi_factor) as ax:
                    skeleton_painter.annotations(ax, pred)
                    
            json_dict = [{'keypoints': np.around(ann.data, 1).reshape(-1).tolist(),
                'bbox': np.around(bbox_from_keypoints(ann.data), 1).tolist(),
                'score': round(ann.score(), 3)} for ann in pred]
            # print(json_dict)
            json_dict_list.append(json_dict)
            
    return json_dict_list


if __name__ == '__main__':
    inference()
