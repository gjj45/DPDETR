# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

try:
    from collections.abc import Sequence
except Exception:
    from collections import Sequence

from numbers import Number, Integral

import cv2
import numpy as np
import math
import copy

from .operators import register_op, BaseOperator
from ppdet.modeling.rbox_utils import poly2rbox_le135_np, poly2rbox_oc_np, rbox2poly_np
from ppdet.utils.logger import setup_logger
logger = setup_logger(__name__)


@register_op
class RRotate(BaseOperator):
    """ Rotate Image, Polygon, Box

    Args:
        scale (float): rotate scale
        angle (float): rotate angle
        fill_value (int, tuple): fill color
        auto_bound (bool): whether auto bound or not
    """

    def __init__(self, scale=1.0, angle=0., fill_value=0., auto_bound=True):
        super(RRotate, self).__init__()
        self.scale = scale
        self.angle = angle
        self.fill_value = fill_value
        self.auto_bound = auto_bound

    def get_rotated_matrix(self, angle, scale, h, w):
        center = ((w - 1) * 0.5, (h - 1) * 0.5)
        matrix = cv2.getRotationMatrix2D(center, -angle, scale)
        # calculate the new size
        cos = np.abs(matrix[0, 0])
        sin = np.abs(matrix[0, 1])
        new_w = h * sin + w * cos
        new_h = h * cos + w * sin
        # calculate offset
        n_w = int(np.round(new_w))
        n_h = int(np.round(new_h))
        if self.auto_bound:
            ratio = min(w / n_w, h / n_h)
            matrix = cv2.getRotationMatrix2D(center, -angle, ratio)
        else:
            matrix[0, 2] += (new_w - w) * 0.5
            matrix[1, 2] += (new_h - h) * 0.5
            w = n_w
            h = n_h
        return matrix, h, w

    def get_rect_from_pts(self, pts, h, w):
        """ get minimum rectangle of points
        """
        assert pts.shape[-1] % 2 == 0, 'the dim of input [pts] is not correct'
        min_x, min_y = np.min(pts[:, 0::2], axis=1), np.min(pts[:, 1::2],
                                                            axis=1)
        max_x, max_y = np.max(pts[:, 0::2], axis=1), np.max(pts[:, 1::2],
                                                            axis=1)
        min_x, min_y = np.clip(min_x, 0, w), np.clip(min_y, 0, h)
        max_x, max_y = np.clip(max_x, 0, w), np.clip(max_y, 0, h)
        boxes = np.stack([min_x, min_y, max_x, max_y], axis=-1)
        return boxes

    def apply_image(self, image, matrix, h, w):
        return cv2.warpAffine(
            image, matrix, (w, h), borderValue=self.fill_value)

    def apply_pts(self, pts, matrix, h, w):
        assert pts.shape[-1] % 2 == 0, 'the dim of input [pts] is not correct'
        # n is number of samples and m is two times the number of points due to (x, y)
        _, m = pts.shape
        # transpose points
        pts_ = pts.reshape(-1, 2).T
        # pad 1 to convert the points to homogeneous coordinates
        padding = np.ones((1, pts_.shape[1]), pts.dtype)
        rotated_pts = np.matmul(matrix, np.concatenate((pts_, padding), axis=0))
        return rotated_pts[:2, :].T.reshape(-1, m)

    def apply(self, sample, context=None):
        image = sample['image']
        h, w = image.shape[:2]
        matrix, h, w = self.get_rotated_matrix(self.angle, self.scale, h, w)
        sample['image'] = self.apply_image(image, matrix, h, w)
        polys = sample['gt_poly']
        # TODO: segment or keypoint to be processed 
        if len(polys) > 0:
            pts = self.apply_pts(polys, matrix, h, w)
            sample['gt_poly'] = pts
            sample['gt_bbox'] = self.get_rect_from_pts(pts, h, w)

        return sample

@register_op
class Multi_RRotate(BaseOperator):
    """ Rotate Image, Polygon, Box

    Args:
        scale (float): rotate scale
        angle (float): rotate angle
        fill_value (int, tuple): fill color
        auto_bound (bool): whether auto bound or not
    """

    def __init__(self, scale=1.0, angle=0., fill_value=0., auto_bound=True):
        super(Multi_RRotate, self).__init__()
        self.scale = scale
        self.angle = angle
        self.fill_value = fill_value
        self.auto_bound = auto_bound

    def get_rotated_matrix(self, angle, scale, h, w):
        center = ((w - 1) * 0.5, (h - 1) * 0.5)
        matrix = cv2.getRotationMatrix2D(center, -angle, scale)
        # calculate the new size
        cos = np.abs(matrix[0, 0])
        sin = np.abs(matrix[0, 1])
        new_w = h * sin + w * cos
        new_h = h * cos + w * sin
        # calculate offset
        n_w = int(np.round(new_w))
        n_h = int(np.round(new_h))
        if self.auto_bound:
            ratio = min(w / n_w, h / n_h)
            matrix = cv2.getRotationMatrix2D(center, -angle, ratio)
        else:
            matrix[0, 2] += (new_w - w) * 0.5
            matrix[1, 2] += (new_h - h) * 0.5
            w = n_w
            h = n_h
        return matrix, h, w

    def get_rect_from_pts(self, pts, h, w):
        """ get minimum rectangle of points
        """
        assert pts.shape[-1] % 2 == 0, 'the dim of input [pts] is not correct'
        min_x, min_y = np.min(pts[:, 0::2], axis=1), np.min(pts[:, 1::2],
                                                            axis=1)
        max_x, max_y = np.max(pts[:, 0::2], axis=1), np.max(pts[:, 1::2],
                                                            axis=1)
        min_x, min_y = np.clip(min_x, 0, w), np.clip(min_y, 0, h)
        max_x, max_y = np.clip(max_x, 0, w), np.clip(max_y, 0, h)
        boxes = np.stack([min_x, min_y, max_x, max_y], axis=-1)
        return boxes

    def apply_image(self, image, matrix, h, w):
        return cv2.warpAffine(
            image, matrix, (w, h), borderValue=self.fill_value)

    def apply_pts(self, pts, matrix, h, w):
        assert pts.shape[-1] % 2 == 0, 'the dim of input [pts] is not correct'
        # n is number of samples and m is two times the number of points due to (x, y)
        _, m = pts.shape
        # transpose points
        pts_ = pts.reshape(-1, 2).T
        # pad 1 to convert the points to homogeneous coordinates
        padding = np.ones((1, pts_.shape[1]), pts.dtype)
        rotated_pts = np.matmul(matrix, np.concatenate((pts_, padding), axis=0))
        return rotated_pts[:2, :].T.reshape(-1, m)

    def apply(self, sample, context=None):
        vis_image = sample['vis_image']
        ir_image = sample['ir_image']
        h, w = vis_image.shape[:2]
        matrix, h, w = self.get_rotated_matrix(self.angle, self.scale, h, w)
        sample['vis_image'] = self.apply_image(vis_image, matrix, h, w)
        sample['ir_image'] = self.apply_image(ir_image, matrix, h, w)
        polys = sample['gt_poly']
        # TODO: segment or keypoint to be processed
        if len(polys) > 0:
            pts = self.apply_pts(polys, matrix, h, w)
            sample['gt_poly'] = pts
            sample['gt_bbox'] = self.get_rect_from_pts(pts, h, w)

        return sample


@register_op
class Multi_RRotate_Paired(BaseOperator):
    """ Rotate Image, Polygon, Box

    Args:
        scale (float): rotate scale
        angle (float): rotate angle
        fill_value (int, tuple): fill color
        auto_bound (bool): whether auto bound or not
    """

    def __init__(self, scale=1.0, angle=0., fill_value=0., auto_bound=True):
        super(Multi_RRotate_Paired, self).__init__()
        self.scale = scale
        self.angle = angle
        self.fill_value = fill_value
        self.auto_bound = auto_bound

    def get_rotated_matrix(self, angle, scale, h, w):
        center = ((w - 1) * 0.5, (h - 1) * 0.5)
        matrix = cv2.getRotationMatrix2D(center, -angle, scale)
        # calculate the new size
        cos = np.abs(matrix[0, 0])
        sin = np.abs(matrix[0, 1])
        new_w = h * sin + w * cos
        new_h = h * cos + w * sin
        # calculate offset
        n_w = int(np.round(new_w))
        n_h = int(np.round(new_h))
        if self.auto_bound:
            ratio = min(w / n_w, h / n_h)
            matrix = cv2.getRotationMatrix2D(center, -angle, ratio)
        else:
            matrix[0, 2] += (new_w - w) * 0.5
            matrix[1, 2] += (new_h - h) * 0.5
            w = n_w
            h = n_h
        return matrix, h, w

    def get_rect_from_pts(self, pts, h, w):
        """ get minimum rectangle of points
        """
        assert pts.shape[-1] % 2 == 0, 'the dim of input [pts] is not correct'
        min_x, min_y = np.min(pts[:, 0::2], axis=1), np.min(pts[:, 1::2],
                                                            axis=1)
        max_x, max_y = np.max(pts[:, 0::2], axis=1), np.max(pts[:, 1::2],
                                                            axis=1)
        min_x, min_y = np.clip(min_x, 0, w), np.clip(min_y, 0, h)
        max_x, max_y = np.clip(max_x, 0, w), np.clip(max_y, 0, h)
        boxes = np.stack([min_x, min_y, max_x, max_y], axis=-1)
        return boxes

    def apply_image(self, image, matrix, h, w):
        return cv2.warpAffine(
            image, matrix, (w, h), borderValue=self.fill_value)

    def apply_pts(self, pts, matrix, h, w):
        assert pts.shape[-1] % 2 == 0, 'the dim of input [pts] is not correct'
        # n is number of samples and m is two times the number of points due to (x, y)
        _, m = pts.shape
        # transpose points
        pts_ = pts.reshape(-1, 2).T
        # pad 1 to convert the points to homogeneous coordinates
        padding = np.ones((1, pts_.shape[1]), pts.dtype)
        rotated_pts = np.matmul(matrix, np.concatenate((pts_, padding), axis=0))
        return rotated_pts[:2, :].T.reshape(-1, m)

    def apply(self, sample, context=None):
        vis_image = sample['vis_image']
        ir_image = sample['ir_image']
        h, w = vis_image.shape[:2]
        matrix, h, w = self.get_rotated_matrix(self.angle, self.scale, h, w)
        sample['vis_image'] = self.apply_image(vis_image, matrix, h, w)
        sample['ir_image'] = self.apply_image(ir_image, matrix, h, w)
        polys_vis = sample['gt_poly_vis']
        polys_ir = sample['gt_poly_ir']
        # TODO: segment or keypoint to be processed
        if len(polys_vis) > 0:
            pts_vis = self.apply_pts(polys_vis, matrix, h, w)
            pts_ir = self.apply_pts(polys_ir, matrix, h, w)
            sample['gt_poly_vis'] = pts_vis
            sample['gt_poly_ir'] = pts_ir
            sample['gt_bbox_vis'] = self.get_rect_from_pts(pts_vis, h, w)
            sample['gt_bbox_ir'] = self.get_rect_from_pts(pts_ir, h, w)

        return sample

@register_op
class RandomRRotate(BaseOperator):
    """ Random Rotate Image
    Args:
        scale (float, tuple, list): rotate scale
        scale_mode (str): mode of scale, [range, value, None]
        angle (float, tuple, list): rotate angle
        angle_mode (str): mode of angle, [range, value, None]
        fill_value (float, tuple, list): fill value
        rotate_prob (float): probability of rotation
        auto_bound (bool): whether auto bound or not
    """

    def __init__(self,
                 scale=1.0,
                 scale_mode=None,
                 angle=0.,
                 angle_mode=None,
                 fill_value=0.,
                 rotate_prob=1.0,
                 auto_bound=True):
        super(RandomRRotate, self).__init__()
        self.scale = scale
        self.scale_mode = scale_mode
        self.angle = angle
        self.angle_mode = angle_mode
        self.fill_value = fill_value
        self.rotate_prob = rotate_prob
        self.auto_bound = auto_bound

    def get_angle(self, angle, angle_mode):
        assert not angle_mode or angle_mode in [
            'range', 'value'
        ], 'angle mode should be in [range, value, None]'
        if not angle_mode:
            return angle
        elif angle_mode == 'range':
            low, high = angle
            return np.random.rand() * (high - low) + low
        elif angle_mode == 'value':
            return np.random.choice(angle)

    def get_scale(self, scale, scale_mode):
        assert not scale_mode or scale_mode in [
            'range', 'value'
        ], 'scale mode should be in [range, value, None]'
        if not scale_mode:
            return scale
        elif scale_mode == 'range':
            low, high = scale
            return np.random.rand() * (high - low) + low
        elif scale_mode == 'value':
            return np.random.choice(scale)

    def apply(self, sample, context=None):
        if np.random.rand() > self.rotate_prob:
            return sample

        angle = self.get_angle(self.angle, self.angle_mode)
        scale = self.get_scale(self.scale, self.scale_mode)
        rotator = RRotate(scale, angle, self.fill_value, self.auto_bound)
        return rotator(sample)


@register_op
class Multi_RandomRRotate(BaseOperator):
    """ Random Rotate Image
    Args:
        scale (float, tuple, list): rotate scale
        scale_mode (str): mode of scale, [range, value, None]
        angle (float, tuple, list): rotate angle
        angle_mode (str): mode of angle, [range, value, None]
        fill_value (float, tuple, list): fill value
        rotate_prob (float): probability of rotation
        auto_bound (bool): whether auto bound or not
    """

    def __init__(self,
                 scale=1.0,
                 scale_mode=None,
                 angle=0.,
                 angle_mode=None,
                 fill_value=0.,
                 rotate_prob=1.0,
                 auto_bound=True):
        super(Multi_RandomRRotate, self).__init__()
        self.scale = scale
        self.scale_mode = scale_mode
        self.angle = angle
        self.angle_mode = angle_mode
        self.fill_value = fill_value
        self.rotate_prob = rotate_prob
        self.auto_bound = auto_bound

    def get_angle(self, angle, angle_mode):
        assert not angle_mode or angle_mode in [
            'range', 'value'
        ], 'angle mode should be in [range, value, None]'
        if not angle_mode:
            return angle
        elif angle_mode == 'range':
            low, high = angle
            return np.random.rand() * (high - low) + low
        elif angle_mode == 'value':
            return np.random.choice(angle)

    def get_scale(self, scale, scale_mode):
        assert not scale_mode or scale_mode in [
            'range', 'value'
        ], 'scale mode should be in [range, value, None]'
        if not scale_mode:
            return scale
        elif scale_mode == 'range':
            low, high = scale
            return np.random.rand() * (high - low) + low
        elif scale_mode == 'value':
            return np.random.choice(scale)

    def apply(self, sample, context=None):
        if np.random.rand() > self.rotate_prob:
            return sample

        angle = self.get_angle(self.angle, self.angle_mode)
        scale = self.get_scale(self.scale, self.scale_mode)
        rotator = Multi_RRotate(scale, angle, self.fill_value, self.auto_bound)
        return rotator(sample)


@register_op
class Multi_RandomRRotate_Paired(BaseOperator):
    """ Random Rotate Image
    Args:
        scale (float, tuple, list): rotate scale
        scale_mode (str): mode of scale, [range, value, None]
        angle (float, tuple, list): rotate angle
        angle_mode (str): mode of angle, [range, value, None]
        fill_value (float, tuple, list): fill value
        rotate_prob (float): probability of rotation
        auto_bound (bool): whether auto bound or not
    """

    def __init__(self,
                 scale=1.0,
                 scale_mode=None,
                 angle=0.,
                 angle_mode=None,
                 fill_value=0.,
                 rotate_prob=1.0,
                 auto_bound=True):
        super(Multi_RandomRRotate_Paired, self).__init__()
        self.scale = scale
        self.scale_mode = scale_mode
        self.angle = angle
        self.angle_mode = angle_mode
        self.fill_value = fill_value
        self.rotate_prob = rotate_prob
        self.auto_bound = auto_bound

    def get_angle(self, angle, angle_mode):
        assert not angle_mode or angle_mode in [
            'range', 'value'
        ], 'angle mode should be in [range, value, None]'
        if not angle_mode:
            return angle
        elif angle_mode == 'range':
            low, high = angle
            return np.random.rand() * (high - low) + low
        elif angle_mode == 'value':
            return np.random.choice(angle)

    def get_scale(self, scale, scale_mode):
        assert not scale_mode or scale_mode in [
            'range', 'value'
        ], 'scale mode should be in [range, value, None]'
        if not scale_mode:
            return scale
        elif scale_mode == 'range':
            low, high = scale
            return np.random.rand() * (high - low) + low
        elif scale_mode == 'value':
            return np.random.choice(scale)

    def apply(self, sample, context=None):
        if np.random.rand() > self.rotate_prob:
            return sample

        angle = self.get_angle(self.angle, self.angle_mode)
        scale = self.get_scale(self.scale, self.scale_mode)
        rotator = Multi_RRotate_Paired(scale, angle, self.fill_value, self.auto_bound)
        return rotator(sample)

@register_op
class Poly2RBox(BaseOperator):
    """ Polygon to Rotated Box, using new OpenCV definition since 4.5.1

    Args:
        filter_threshold (int, float): threshold to filter annotations
        filter_mode (str): filter mode, ['area', 'edge']
        rbox_type (str): rbox type, ['le135', 'oc']

    """

    def __init__(self, filter_threshold=4, filter_mode=None, rbox_type='le135'):
        super(Poly2RBox, self).__init__()
        self.filter_fn = lambda size: self.filter(size, filter_threshold, filter_mode)
        self.rbox_fn = poly2rbox_le135_np if rbox_type == 'le135' else poly2rbox_oc_np

    def filter(self, size, threshold, mode):
        if mode == 'area':
            if size[0] * size[1] < threshold:
                return True
        elif mode == 'edge':
            if min(size) < threshold:
                return True
        return False

    def get_rbox(self, polys):
        valid_ids, rboxes, bboxes = [], [], []
        for i, poly in enumerate(polys):
            cx, cy, w, h, angle = self.rbox_fn(poly)
            if self.filter_fn((w, h)):
                continue
            rboxes.append(np.array([cx, cy, w, h, angle], dtype=np.float32))
            valid_ids.append(i)
            xmin, ymin = min(poly[0::2]), min(poly[1::2])
            xmax, ymax = max(poly[0::2]), max(poly[1::2])
            bboxes.append(np.array([xmin, ymin, xmax, ymax], dtype=np.float32))

        if len(valid_ids) == 0:
            rboxes = np.zeros((0, 5), dtype=np.float32)
            bboxes = np.zeros((0, 4), dtype=np.float32)
        else:
            rboxes = np.stack(rboxes)
            bboxes = np.stack(bboxes)

        return rboxes, bboxes, valid_ids

    def apply(self, sample, context=None):
        rboxes, bboxes, valid_ids = self.get_rbox(sample['gt_poly'])
        sample['gt_rbox'] = rboxes
        sample['gt_bbox'] = bboxes
        for k in ['gt_class', 'gt_score', 'gt_poly', 'is_crowd', 'difficult']:
            if k in sample:
                sample[k] = sample[k][valid_ids]

        return sample


@register_op
class Poly2RBox_Paired(BaseOperator):
    """ Polygon to Rotated Box, using new OpenCV definition since 4.5.1

    Args:
        filter_threshold (int, float): threshold to filter annotations
        filter_mode (str): filter mode, ['area', 'edge']
        rbox_type (str): rbox type, ['le135', 'oc']

    """

    def __init__(self, filter_threshold=4, filter_mode=None, rbox_type='le135'):
        super(Poly2RBox_Paired, self).__init__()
        self.filter_fn = lambda size: self.filter(size, filter_threshold, filter_mode)
        self.rbox_fn = poly2rbox_le135_np if rbox_type == 'le135' else poly2rbox_oc_np

    def filter(self, size, threshold, mode):
        if mode == 'area':
            if size[0] * size[1] < threshold:
                return True
        elif mode == 'edge':
            if min(size) < threshold:
                return True
        return False

    def get_rbox(self, polys_vis, polys_ir):
        valid_ids, rboxes_vis, bboxes_vis, rboxes_ir, bboxes_ir = [], [], [], [], []
        for i, (poly_vis, poly_ir) in enumerate(zip(polys_vis, polys_ir)):
            cx_vis, cy_vis, w_vis, h_vis, angle_vis = self.rbox_fn(poly_vis)
            cx_ir, cy_ir, w_ir, h_ir, angle_ir = self.rbox_fn(poly_ir)
            if self.filter_fn((w_vis, h_vis)) or self.filter_fn((w_ir, h_ir)):
                continue
            rboxes_vis.append(np.array([cx_vis, cy_vis, w_vis, h_vis, angle_vis], dtype=np.float32))
            rboxes_ir.append(np.array([cx_ir, cy_ir, w_ir, h_ir, angle_ir], dtype=np.float32))
            valid_ids.append(i)
            xmin_vis, ymin_vis = min(poly_vis[0::2]), min(poly_vis[1::2])
            xmax_vis, ymax_vis = max(poly_vis[0::2]), max(poly_vis[1::2])
            bboxes_vis.append(np.array([xmin_vis, ymin_vis, xmax_vis, ymax_vis], dtype=np.float32))

            xmin_ir, ymin_ir = min(poly_ir[0::2]), min(poly_ir[1::2])
            xmax_ir, ymax_ir = max(poly_ir[0::2]), max(poly_ir[1::2])
            bboxes_ir.append(np.array([xmin_ir, ymin_ir, xmax_ir, ymax_ir], dtype=np.float32))

        if len(valid_ids) == 0:
            rboxes_vis = np.zeros((0, 5), dtype=np.float32)
            bboxes_vis = np.zeros((0, 4), dtype=np.float32)
            rboxes_ir = np.zeros((0, 5), dtype=np.float32)
            bboxes_ir = np.zeros((0, 4), dtype=np.float32)
        else:
            rboxes_vis = np.stack(rboxes_vis)
            bboxes_vis = np.stack(bboxes_vis)
            rboxes_ir = np.stack(rboxes_ir)
            bboxes_ir = np.stack(bboxes_ir)

        return rboxes_vis, bboxes_vis, rboxes_ir, bboxes_ir, valid_ids

    def apply(self, sample, context=None):
        rboxes_vis, bboxes_vis, rboxes_ir, bboxes_ir, valid_ids = self.get_rbox(sample['gt_poly_vis'], sample['gt_poly_ir'])
        sample['gt_rbox_vis'] = rboxes_vis
        sample['gt_bbox_vis'] = bboxes_vis
        sample['gt_rbox_ir'] = rboxes_ir
        sample['gt_bbox_ir'] = bboxes_ir
        for k in ['gt_class', 'gt_score', 'gt_poly_vis', 'gt_poly_ir', 'is_crowd', 'difficult']:
            if k in sample:
                sample[k] = sample[k][valid_ids]

        return sample

@register_op
class Poly2Array(BaseOperator):
    """ convert gt_poly to np.array for rotated bboxes
    """

    def __init__(self):
        super(Poly2Array, self).__init__()

    def apply(self, sample, context=None):
        if 'gt_poly' in sample:
            sample['gt_poly'] = np.array(
                sample['gt_poly'], dtype=np.float32).reshape((-1, 8))

        return sample


@register_op
class Poly2Array_Paired(BaseOperator):
    """ convert gt_poly to np.array for rotated bboxes
    """

    def __init__(self):
        super(Poly2Array_Paired, self).__init__()

    def apply(self, sample, context=None):
        if 'gt_poly_vis' in sample:
            sample['gt_poly_vis'] = np.array(
                sample['gt_poly_vis'], dtype=np.float32).reshape((-1, 8))
        if 'gt_poly_ir' in sample:
            sample['gt_poly_ir'] = np.array(
                sample['gt_poly_ir'], dtype=np.float32).reshape((-1, 8))
        return sample

@register_op
class RResize(BaseOperator):
    def __init__(self, target_size, keep_ratio, interp=cv2.INTER_LINEAR):
        """
        Resize image to target size. if keep_ratio is True, 
        resize the image's long side to the maximum of target_size
        if keep_ratio is False, resize the image to target size(h, w)
        Args:
            target_size (int|list): image target size
            keep_ratio (bool): whether keep_ratio or not, default true
            interp (int): the interpolation method
        """
        super(RResize, self).__init__()
        self.keep_ratio = keep_ratio
        self.interp = interp
        if not isinstance(target_size, (Integral, Sequence)):
            raise TypeError(
                "Type of target_size is invalid. Must be Integer or List or Tuple, now is {}".
                format(type(target_size)))
        if isinstance(target_size, Integral):
            target_size = [target_size, target_size]
        self.target_size = target_size

    def apply_image(self, image, scale):
        im_scale_x, im_scale_y = scale

        return cv2.resize(
            image,
            None,
            None,
            fx=im_scale_x,
            fy=im_scale_y,
            interpolation=self.interp)

    def apply_pts(self, pts, scale, size):
        im_scale_x, im_scale_y = scale
        resize_w, resize_h = size
        pts[:, 0::2] *= im_scale_x
        pts[:, 1::2] *= im_scale_y
        pts[:, 0::2] = np.clip(pts[:, 0::2], 0, resize_w)
        pts[:, 1::2] = np.clip(pts[:, 1::2], 0, resize_h)
        return pts

    def apply(self, sample, context=None):
        """ Resize the image numpy.
        """
        im = sample['image']
        if not isinstance(im, np.ndarray):
            raise TypeError("{}: image type is not numpy.".format(self))
        if len(im.shape) != 3:
            raise ImageError('{}: image is not 3-dimensional.'.format(self))

        # apply image
        im_shape = im.shape
        if self.keep_ratio:

            im_size_min = np.min(im_shape[0:2])
            im_size_max = np.max(im_shape[0:2])

            target_size_min = np.min(self.target_size)
            target_size_max = np.max(self.target_size)

            im_scale = min(target_size_min / im_size_min,
                           target_size_max / im_size_max)

            resize_h = im_scale * float(im_shape[0])
            resize_w = im_scale * float(im_shape[1])

            im_scale_x = im_scale
            im_scale_y = im_scale
        else:
            resize_h, resize_w = self.target_size
            im_scale_y = resize_h / im_shape[0]
            im_scale_x = resize_w / im_shape[1]

        im = self.apply_image(sample['image'], [im_scale_x, im_scale_y])
        sample['image'] = im.astype(np.float32)
        sample['im_shape'] = np.asarray([resize_h, resize_w], dtype=np.float32)
        if 'scale_factor' in sample:
            scale_factor = sample['scale_factor']
            sample['scale_factor'] = np.asarray(
                [scale_factor[0] * im_scale_y, scale_factor[1] * im_scale_x],
                dtype=np.float32)
        else:
            sample['scale_factor'] = np.asarray(
                [im_scale_y, im_scale_x], dtype=np.float32)

        # apply bbox
        if 'gt_bbox' in sample and len(sample['gt_bbox']) > 0:
            sample['gt_bbox'] = self.apply_pts(sample['gt_bbox'],
                                               [im_scale_x, im_scale_y],
                                               [resize_w, resize_h])

        # apply polygon
        if 'gt_poly' in sample and len(sample['gt_poly']) > 0:
            sample['gt_poly'] = self.apply_pts(sample['gt_poly'],
                                               [im_scale_x, im_scale_y],
                                               [resize_w, resize_h])

        return sample


@register_op
class Multi_RResize(BaseOperator):
    def __init__(self, target_size, keep_ratio, interp=cv2.INTER_LINEAR):
        """
        Resize image to target size. if keep_ratio is True,
        resize the image's long side to the maximum of target_size
        if keep_ratio is False, resize the image to target size(h, w)
        Args:
            target_size (int|list): image target size
            keep_ratio (bool): whether keep_ratio or not, default true
            interp (int): the interpolation method
        """
        super(Multi_RResize, self).__init__()
        self.keep_ratio = keep_ratio
        self.interp = interp
        if not isinstance(target_size, (Integral, Sequence)):
            raise TypeError(
                "Type of target_size is invalid. Must be Integer or List or Tuple, now is {}".
                format(type(target_size)))
        if isinstance(target_size, Integral):
            target_size = [target_size, target_size]
        self.target_size = target_size

    def apply_image(self, image, scale):
        im_scale_x, im_scale_y = scale

        return cv2.resize(
            image,
            None,
            None,
            fx=im_scale_x,
            fy=im_scale_y,
            interpolation=self.interp)

    def apply_pts(self, pts, scale, size):
        im_scale_x, im_scale_y = scale
        resize_w, resize_h = size
        pts[:, 0::2] *= im_scale_x
        pts[:, 1::2] *= im_scale_y
        pts[:, 0::2] = np.clip(pts[:, 0::2], 0, resize_w)
        pts[:, 1::2] = np.clip(pts[:, 1::2], 0, resize_h)
        return pts

    def apply(self, sample, context=None):
        """ Resize the image numpy.
        """
        vis_im = sample['vis_image']
        ir_im = sample['ir_image']
        if not isinstance(vis_im, np.ndarray):
            raise TypeError("{}: image type is not numpy.".format(self))
        if len(vis_im.shape) != 3:
            raise ImageError('{}: image is not 3-dimensional.'.format(self))
        if len(ir_im.shape) != 3:
            raise ImageError('{}: image is not 3-dimensional.'.format(self))

        # apply image
        im_shape = vis_im.shape
        if self.keep_ratio:

            im_size_min = np.min(im_shape[0:2])
            im_size_max = np.max(im_shape[0:2])

            target_size_min = np.min(self.target_size)
            target_size_max = np.max(self.target_size)

            im_scale = min(target_size_min / im_size_min,
                           target_size_max / im_size_max)

            resize_h = im_scale * float(im_shape[0])
            resize_w = im_scale * float(im_shape[1])

            im_scale_x = im_scale
            im_scale_y = im_scale
        else:
            resize_h, resize_w = self.target_size
            im_scale_y = resize_h / im_shape[0]
            im_scale_x = resize_w / im_shape[1]

        vis_im = self.apply_image(sample['vis_image'], [im_scale_x, im_scale_y])
        ir_im = self.apply_image(sample['ir_image'], [im_scale_x, im_scale_y])
        sample['vis_image'] = vis_im.astype(np.float32)
        sample['ir_image'] = ir_im.astype(np.float32)
        sample['im_shape'] = np.asarray([resize_h, resize_w], dtype=np.float32)
        if 'scale_factor' in sample:
            scale_factor = sample['scale_factor']
            sample['scale_factor'] = np.asarray(
                [scale_factor[0] * im_scale_y, scale_factor[1] * im_scale_x],
                dtype=np.float32)
        else:
            sample['scale_factor'] = np.asarray(
                [im_scale_y, im_scale_x], dtype=np.float32)

        # apply bbox
        if 'gt_bbox' in sample and len(sample['gt_bbox']) > 0:
            sample['gt_bbox'] = self.apply_pts(sample['gt_bbox'],
                                               [im_scale_x, im_scale_y],
                                               [resize_w, resize_h])

        # apply polygon
        if 'gt_poly' in sample and len(sample['gt_poly']) > 0:
            sample['gt_poly'] = self.apply_pts(sample['gt_poly'],
                                               [im_scale_x, im_scale_y],
                                               [resize_w, resize_h])

        return sample


@register_op
class Multi_RResize_Paired(BaseOperator):
    def __init__(self, target_size, keep_ratio, interp=cv2.INTER_LINEAR):
        """
        Resize image to target size. if keep_ratio is True,
        resize the image's long side to the maximum of target_size
        if keep_ratio is False, resize the image to target size(h, w)
        Args:
            target_size (int|list): image target size
            keep_ratio (bool): whether keep_ratio or not, default true
            interp (int): the interpolation method
        """
        super(Multi_RResize_Paired, self).__init__()
        self.keep_ratio = keep_ratio
        self.interp = interp
        if not isinstance(target_size, (Integral, Sequence)):
            raise TypeError(
                "Type of target_size is invalid. Must be Integer or List or Tuple, now is {}".
                format(type(target_size)))
        if isinstance(target_size, Integral):
            target_size = [target_size, target_size]
        self.target_size = target_size

    def apply_image(self, image, scale):
        im_scale_x, im_scale_y = scale

        return cv2.resize(
            image,
            None,
            None,
            fx=im_scale_x,
            fy=im_scale_y,
            interpolation=self.interp)

    def apply_pts(self, pts, scale, size):
        im_scale_x, im_scale_y = scale
        resize_w, resize_h = size
        pts[:, 0::2] *= im_scale_x
        pts[:, 1::2] *= im_scale_y
        pts[:, 0::2] = np.clip(pts[:, 0::2], 0, resize_w)
        pts[:, 1::2] = np.clip(pts[:, 1::2], 0, resize_h)
        return pts

    def apply(self, sample, context=None):
        """ Resize the image numpy.
        """
        vis_im = sample['vis_image']
        ir_im = sample['ir_image']
        # cv2.imwrite('/data1/guojunjie1/python_project/MS-PaddleDetection-develop-240117/output/vis.jpg',vis_im)
        # cv2.imwrite('/data1/guojunjie1/python_project/MS-PaddleDetection-develop-240117/output/ir.jpg', ir_im)
        if not isinstance(vis_im, np.ndarray):
            raise TypeError("{}: image type is not numpy.".format(self))
        if len(vis_im.shape) != 3:
            raise ImageError('{}: image is not 3-dimensional.'.format(self))
        if len(ir_im.shape) != 3:
            raise ImageError('{}: image is not 3-dimensional.'.format(self))

        # apply image
        im_shape = vis_im.shape
        if self.keep_ratio:

            im_size_min = np.min(im_shape[0:2])
            im_size_max = np.max(im_shape[0:2])

            target_size_min = np.min(self.target_size)
            target_size_max = np.max(self.target_size)

            im_scale = min(target_size_min / im_size_min,
                           target_size_max / im_size_max)

            resize_h = im_scale * float(im_shape[0])
            resize_w = im_scale * float(im_shape[1])

            im_scale_x = im_scale
            im_scale_y = im_scale
        else:
            resize_h, resize_w = self.target_size
            im_scale_y = resize_h / im_shape[0]
            im_scale_x = resize_w / im_shape[1]

        vis_im = self.apply_image(sample['vis_image'], [im_scale_x, im_scale_y])
        ir_im = self.apply_image(sample['ir_image'], [im_scale_x, im_scale_y])
        sample['vis_image'] = vis_im.astype(np.float32)
        sample['ir_image'] = ir_im.astype(np.float32)
        sample['im_shape'] = np.asarray([resize_h, resize_w], dtype=np.float32)
        if 'scale_factor' in sample:
            scale_factor = sample['scale_factor']
            sample['scale_factor'] = np.asarray(
                [scale_factor[0] * im_scale_y, scale_factor[1] * im_scale_x],
                dtype=np.float32)
        else:
            sample['scale_factor'] = np.asarray(
                [im_scale_y, im_scale_x], dtype=np.float32)

        # apply bbox
        if 'gt_bbox_vis' in sample and len(sample['gt_bbox_vis']) > 0:
            sample['gt_bbox_vis'] = self.apply_pts(sample['gt_bbox_vis'],
                                               [im_scale_x, im_scale_y],
                                               [resize_w, resize_h])
            sample['gt_bbox_ir'] = self.apply_pts(sample['gt_bbox_ir'],
                                                   [im_scale_x, im_scale_y],
                                                   [resize_w, resize_h])

        # apply polygon
        if 'gt_poly_vis' in sample and len(sample['gt_poly_vis']) > 0:
            sample['gt_poly_vis'] = self.apply_pts(sample['gt_poly_vis'],
                                               [im_scale_x, im_scale_y],
                                               [resize_w, resize_h])
            sample['gt_poly_ir'] = self.apply_pts(sample['gt_poly_ir'],
                                                   [im_scale_x, im_scale_y],
                                                   [resize_w, resize_h])
        return sample




@register_op
class RandomRFlip(BaseOperator):
    def __init__(self, prob=0.5):
        """
        Args:
            prob (float): the probability of flipping image
        """
        super(RandomRFlip, self).__init__()
        self.prob = prob
        if not (isinstance(self.prob, float)):
            raise TypeError("{}: input type is invalid.".format(self))

    def apply_image(self, image):
        return image[:, ::-1, :]

    def apply_pts(self, pts, width):
        oldx = pts[:, 0::2].copy()
        pts[:, 0::2] = width - oldx - 1
        return pts

    def apply(self, sample, context=None):
        """Filp the image and bounding box.
        Operators:
            1. Flip the image numpy.
            2. Transform the bboxes' x coordinates.
              (Must judge whether the coordinates are normalized!)
            3. Transform the segmentations' x coordinates.
              (Must judge whether the coordinates are normalized!)
        Output:
            sample: the image, bounding box and segmentation part
                    in sample are flipped.
        """
        if np.random.uniform(0, 1) < self.prob:
            im = sample['image']
            height, width = im.shape[:2]
            im = self.apply_image(im)
            if 'gt_bbox' in sample and len(sample['gt_bbox']) > 0:
                sample['gt_bbox'] = self.apply_pts(sample['gt_bbox'], width)
            if 'gt_poly' in sample and len(sample['gt_poly']) > 0:
                sample['gt_poly'] = self.apply_pts(sample['gt_poly'], width)

            sample['flipped'] = True
            sample['image'] = im
        return sample

@register_op
class Multi_RandomRFlip(BaseOperator):
    def __init__(self, prob=0.5):
        """
        Args:
            prob (float): the probability of flipping image
        """
        super(Multi_RandomRFlip, self).__init__()
        self.prob = prob
        if not (isinstance(self.prob, float)):
            raise TypeError("{}: input type is invalid.".format(self))

    def apply_image(self, image):
        return image[:, ::-1, :]

    def apply_pts(self, pts, width):
        oldx = pts[:, 0::2].copy()
        pts[:, 0::2] = width - oldx - 1
        return pts

    def apply(self, sample, context=None):
        """Filp the image and bounding box.
        Operators:
            1. Flip the image numpy.
            2. Transform the bboxes' x coordinates.
              (Must judge whether the coordinates are normalized!)
            3. Transform the segmentations' x coordinates.
              (Must judge whether the coordinates are normalized!)
        Output:
            sample: the image, bounding box and segmentation part
                    in sample are flipped.
        """
        if np.random.uniform(0, 1) < self.prob:
            im_vis = sample['vis_image']
            im_ir = sample['ir_image']
            height, width = im_vis.shape[:2]
            im_vis = self.apply_image(im_vis)
            im_ir = self.apply_image(im_ir)
            if 'gt_bbox' in sample and len(sample['gt_bbox']) > 0:
                sample['gt_bbox'] = self.apply_pts(sample['gt_bbox'], width)
            if 'gt_poly' in sample and len(sample['gt_poly']) > 0:
                sample['gt_poly'] = self.apply_pts(sample['gt_poly'], width)

            sample['flipped'] = True
            sample['vis_image'] = im_vis
            sample['ir_image'] = im_ir
        return sample

@register_op
class Multi_RandomRFlip_Paired(BaseOperator):
    def __init__(self, prob=0.5):
        """
        Args:
            prob (float): the probability of flipping image
        """
        super(Multi_RandomRFlip_Paired, self).__init__()
        self.prob = prob
        if not (isinstance(self.prob, float)):
            raise TypeError("{}: input type is invalid.".format(self))

    def apply_image(self, image):
        return image[:, ::-1, :]

    def apply_pts(self, pts, width):
        oldx = pts[:, 0::2].copy()
        pts[:, 0::2] = width - oldx - 1
        return pts

    def apply(self, sample, context=None):
        """Filp the image and bounding box.
        Operators:
            1. Flip the image numpy.
            2. Transform the bboxes' x coordinates.
              (Must judge whether the coordinates are normalized!)
            3. Transform the segmentations' x coordinates.
              (Must judge whether the coordinates are normalized!)
        Output:
            sample: the image, bounding box and segmentation part
                    in sample are flipped.
        """
        if np.random.uniform(0, 1) < self.prob:
            im_vis = sample['vis_image']
            im_ir = sample['ir_image']
            height, width = im_vis.shape[:2]
            im_vis = self.apply_image(im_vis)
            im_ir = self.apply_image(im_ir)
            if 'gt_bbox_vis' in sample and len(sample['gt_bbox_vis']) > 0:
                sample['gt_bbox_vis'] = self.apply_pts(sample['gt_bbox_vis'], width)
                sample['gt_bbox_ir'] = self.apply_pts(sample['gt_bbox_ir'], width)
            if 'gt_poly_vis' in sample and len(sample['gt_poly_vis']) > 0:
                sample['gt_poly_vis'] = self.apply_pts(sample['gt_poly_vis'], width)
                sample['gt_poly_ir'] = self.apply_pts(sample['gt_poly_ir'], width)

            sample['flipped'] = True
            sample['vis_image'] = im_vis
            sample['ir_image'] = im_ir
        return sample

@register_op
class Multi_Shift_Paired(BaseOperator):
    def __init__(self, prob=0.3, max_shift = 13):
        """
        Args:
            prob (float): the probability of flipping image
        """
        super(Multi_Shift_Paired, self).__init__()
        self.prob = prob
        self.max_shift = max_shift
        if not (isinstance(self.prob, float)):
            raise TypeError("{}: input type is invalid.".format(self))

    def apply_image_top2down(self, image, pix):
        top_pix_rows = image[:pix,:,:]
        image = np.concatenate((image[pix:,:,:], top_pix_rows), axis=0)
        return image

    def apply_image_down2top(self, image, pix):
        down_pix_rows = image[-pix:,:,:]
        image = np.concatenate((down_pix_rows,image[:-pix,:,:]), axis=0)
        return image

    def apply_image_left2right(self, image, pix):
        left_pix_cols = image[:,:pix,:]
        image = np.concatenate((image[:,pix:],left_pix_cols), axis=1)
        return image

    def apply_image_right2left(self, image, pix):
        right_pix_cols = image[:,-pix:,:]
        image = np.concatenate((right_pix_cols,image[:,:-pix,:]), axis=1)
        return image

    def apply_pts_top2down(self, pts, pix):
        oldy = pts[:, 1::2].copy()
        pts[:, 1::2] = oldy - pix
        return pts

    def apply_pts_down2top(self, pts, pix):
        oldy = pts[:, 1::2].copy()
        pts[:, 1::2] = oldy + pix
        return pts

    def apply_pts_left2right(self, pts, pix):
        oldx = pts[:, 0::2].copy()
        pts[:, 0::2] = oldx - pix
        return pts

    def apply_pts_right2left(self, pts, pix):
        oldx = pts[:, 0::2].copy()
        pts[:, 0::2] = oldx + pix
        return pts

    def apply(self, sample, context=None):
        """Shift the image and bounding box.
        Operators:
            1. shift the image numpy for 0~13 pix
            2. Transform the bboxes' x coordinates.
              (Must judge whether the coordinates are normalized!)
            3. Transform the segmentations' x coordinates.
              (Must judge whether the coordinates are normalized!)
        Output:
            sample: the image, bounding box and segmentation part
                    in sample are flipped.
        """
        if np.random.uniform(0, 1) < self.prob:
            #select shift modality
            if np.random.uniform(0, 1) < 0.5:
                flag = 'vis'
                im = sample['vis_image']
                gt_bbox = sample['gt_bbox_vis']
                gt_poly = sample['gt_poly_vis']
            else:
                flag = 'ir'
                im = sample['ir_image']
                gt_bbox = sample['gt_bbox_ir']
                gt_poly = sample['gt_poly_ir']
            height, width = im.shape[:2]

            decision = np.random.randint(0,7)
            if decision == 0: #top->down
                pix = np.random.randint(1, self.max_shift)
                # apply img
                im = self.apply_image_top2down(im,pix)
                # apply bbox
                gt_bbox = self.apply_pts_top2down(gt_bbox,pix)
                gt_poly = self.apply_pts_top2down(gt_poly,pix)
            elif decision == 1: #down->top
                pix = np.random.randint(1, self.max_shift)
                # apply img
                im = self.apply_image_down2top(im, pix)
                # apply bbox
                gt_bbox = self.apply_pts_down2top(gt_bbox, pix)
                gt_poly = self.apply_pts_down2top(gt_poly, pix)
            elif decision == 2: #left->right
                pix = np.random.randint(1, self.max_shift)
                # apply img
                im = self.apply_image_left2right(im, pix)
                # apply bbox
                gt_bbox = self.apply_pts_left2right(gt_bbox, pix)
                gt_poly = self.apply_pts_left2right(gt_poly, pix)
            elif decision == 3: #right->left
                pix = np.random.randint(1, self.max_shift)
                # apply img
                im = self.apply_image_right2left(im, pix)
                # apply bbox
                gt_bbox = self.apply_pts_right2left(gt_bbox, pix)
                gt_poly = self.apply_pts_right2left(gt_poly, pix)
            elif decision == 4:  #left top
                pix = np.random.randint(1, self.max_shift)
                # apply img
                im = self.apply_image_top2down(im, pix)
                # apply bbox
                gt_bbox = self.apply_pts_top2down(gt_bbox, pix)
                gt_poly = self.apply_pts_top2down(gt_poly, pix)

                pix = np.random.randint(1, self.max_shift)
                # apply img
                im = self.apply_image_left2right(im, pix)
                # apply bbox
                gt_bbox = self.apply_pts_left2right(gt_bbox, pix)
                gt_poly = self.apply_pts_left2right(gt_poly, pix)
            elif decision == 5:  #right top
                pix = np.random.randint(1, self.max_shift)
                # apply img
                im = self.apply_image_top2down(im, pix)
                # apply bbox
                gt_bbox = self.apply_pts_top2down(gt_bbox, pix)
                gt_poly = self.apply_pts_top2down(gt_poly, pix)

                pix = np.random.randint(1, self.max_shift)
                # apply img
                im = self.apply_image_right2left(im, pix)
                # apply bbox
                gt_bbox = self.apply_pts_right2left(gt_bbox, pix)
                gt_poly = self.apply_pts_right2left(gt_poly, pix)
            elif decision == 6:  #left down
                pix = np.random.randint(1, self.max_shift)
                # apply img
                im = self.apply_image_down2top(im, pix)
                # apply bbox
                gt_bbox = self.apply_pts_down2top(gt_bbox, pix)
                gt_poly = self.apply_pts_down2top(gt_poly, pix)

                pix = np.random.randint(1, self.max_shift)
                # apply img
                im = self.apply_image_left2right(im, pix)
                # apply bbox
                gt_bbox = self.apply_pts_left2right(gt_bbox, pix)
                gt_poly = self.apply_pts_left2right(gt_poly, pix)
            elif decision == 7:  #right down
                pix = np.random.randint(1, self.max_shift)
                # apply img
                im = self.apply_image_down2top(im, pix)
                # apply bbox
                gt_bbox = self.apply_pts_down2top(gt_bbox, pix)
                gt_poly = self.apply_pts_down2top(gt_poly, pix)

                pix = np.random.randint(1, self.max_shift)
                # apply img
                im = self.apply_image_right2left(im, pix)
                # apply bbox
                gt_bbox = self.apply_pts_right2left(gt_bbox, pix)
                gt_poly = self.apply_pts_right2left(gt_poly, pix)

            # cv2.drawContours(im, gt_poly, )
            # cv2.imwrite('/data1/guojunjie1/python_project/MS-PaddleDetection-develop-240117/im.jpg',im)

            if flag == 'vis':
                sample['vis_image'] = im
                sample['gt_bbox_vis'] = gt_bbox
                sample['gt_poly_vis'] = gt_poly
            elif flag == 'ir':
                sample['ir_image'] = im
                sample['gt_bbox_ir'] = gt_bbox
                sample['gt_poly_ir'] = gt_poly
        return sample

@register_op
class VisibleRBox(BaseOperator):
    """
    In debug mode, visualize images according to `gt_box`.
    (Currently only supported when not cropping and flipping image.)
    """

    def __init__(self, output_dir='debug'):
        super(VisibleRBox, self).__init__()
        self.output_dir = output_dir
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)

    def apply(self, sample, context=None):
        image = Image.fromarray(sample['image'].astype(np.uint8))
        out_file_name = '{:012d}.jpg'.format(sample['im_id'][0])
        width = sample['w']
        height = sample['h']
        # gt_poly = sample['gt_rbox']
        gt_poly = sample['gt_poly']
        gt_class = sample['gt_class']
        draw = ImageDraw.Draw(image)
        for i in range(gt_poly.shape[0]):
            x1, y1, x2, y2, x3, y3, x4, y4 = gt_poly[i]
            draw.line(
                [(x1, y1), (x2, y2), (x3, y3), (x4, y4), (x1, y1)],
                width=2,
                fill='green')
            # draw label
            xmin = min(x1, x2, x3, x4)
            ymin = min(y1, y2, y3, y4)
            text = str(gt_class[i][0])
            tw, th = draw.textsize(text)
            draw.rectangle(
                [(xmin + 1, ymin - th), (xmin + tw + 1, ymin)], fill='green')
            draw.text((xmin + 1, ymin - th), text, fill=(255, 255, 255))

        if 'gt_keypoint' in sample.keys():
            gt_keypoint = sample['gt_keypoint']
            if self.is_normalized:
                for i in range(gt_keypoint.shape[1]):
                    if i % 2:
                        gt_keypoint[:, i] = gt_keypoint[:, i] * height
                    else:
                        gt_keypoint[:, i] = gt_keypoint[:, i] * width
            for i in range(gt_keypoint.shape[0]):
                keypoint = gt_keypoint[i]
                for j in range(int(keypoint.shape[0] / 2)):
                    x1 = round(keypoint[2 * j]).astype(np.int32)
                    y1 = round(keypoint[2 * j + 1]).astype(np.int32)
                    draw.ellipse(
                        (x1, y1, x1 + 5, y1 + 5), fill='green', outline='green')
        save_path = os.path.join(self.output_dir, out_file_name)
        image.save(save_path, quality=95)
        return sample


@register_op
class Rbox2Poly(BaseOperator):
    """
    Convert rbbox format to poly format.
    """

    def __init__(self):
        super(Rbox2Poly, self).__init__()

    def apply(self, sample, context=None):
        assert 'gt_rbox' in sample
        assert sample['gt_rbox'].shape[1] == 5
        rboxes = sample['gt_rbox']
        polys = rbox2poly_np(rboxes)
        sample['gt_poly'] = polys
        xmin, ymin = polys[:, 0::2].min(1), polys[:, 1::2].min(1)
        xmax, ymax = polys[:, 0::2].max(1), polys[:, 1::2].max(1)
        sample['gt_bbox'] = np.stack([xmin, ymin, xmin, ymin], axis=1)
        return sample
