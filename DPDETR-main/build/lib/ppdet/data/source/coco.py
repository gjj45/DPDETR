# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved. 
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

import os
import copy
try:
    from collections.abc import Sequence
except Exception:
    from collections import Sequence
import numpy as np
from ppdet.core.workspace import register, serializable
from .dataset import DetDataset,Multi_DetDataset,Multi_DetDataset_Paired

from ppdet.utils.logger import setup_logger
logger = setup_logger(__name__)

__all__ = ['COCODataSet', 'SlicedCOCODataSet', 'SemiCOCODataSet', 'Mutisprctral_COCODataSet','Mutisprctral_COCODataSet_Paired']



@register
@serializable
class Mutisprctral_COCODataSet_Paired(Multi_DetDataset_Paired):
    """
    Load dataset with COCO format.

    Args:
        dataset_dir (str): root directory for dataset.
        vis_image_dir (str): directory for  vis_images.
        ir_image_dir (str): directory for  ir_images.
        anno_path (str): coco annotation file path.
        data_fields (list): key name of data dictionary, at least have 'image'.
        sample_num (int): number of samples to load, -1 means all.
        load_crowd (bool): whether to load crowded ground-truth.
            False as default
        allow_empty (bool): whether to load empty entry. False as default
        empty_ratio (float): the ratio of empty record number to total
            record's, if empty_ratio is out of [0. ,1.), do not sample the
            records and use all the empty entries. 1. as default
        repeat (int): repeat times for dataset, use in benchmark.
    """

    def __init__(self,
                 dataset_dir=None,
                 vis_image_dir=None,
                 ir_image_dir=None,
                 anno_path_vis=None,
                 anno_path_ir=None,
                 data_fields=['image'],
                 sample_num=-1,
                 load_crowd=False,
                 allow_empty=False,
                 empty_ratio=1.,
                 repeat=1):
        super(Mutisprctral_COCODataSet_Paired, self).__init__(
            dataset_dir,
            vis_image_dir,
            ir_image_dir,
            anno_path_vis,
            anno_path_ir,
            data_fields,
            sample_num,
            repeat=repeat)
        self.load_image_only = False
        self.load_semantic = False
        self.load_crowd = load_crowd
        self.allow_empty = allow_empty
        self.empty_ratio = empty_ratio

    def _sample_empty(self, records, num):
        # if empty_ratio is out of [0. ,1.), do not sample the records
        if self.empty_ratio < 0. or self.empty_ratio >= 1.:
            return records
        import random
        sample_num = min(
            int(num * self.empty_ratio / (1 - self.empty_ratio)), len(records))
        records = random.sample(records, sample_num)
        return records

    def parse_dataset(self):
        anno_path_vis = os.path.join(self.dataset_dir, self.anno_path_vis)
        anno_path_ir = os.path.join(self.dataset_dir, self.anno_path_ir)
        vis_image_dir = os.path.join(self.dataset_dir, self.vis_image_dir)
        ir_image_dir = os.path.join(self.dataset_dir, self.ir_image_dir)

        assert anno_path_vis.endswith('.json'), \
            'invalid coco annotation file: ' + anno_path_vis
        assert anno_path_ir.endswith('.json'), \
            'invalid coco annotation file: ' + anno_path_ir

        from pycocotools.coco import COCO
        coco_vis = COCO(anno_path_vis)
        coco_ir = COCO(anno_path_ir)
        img_ids = coco_ir.getImgIds()
        img_ids.sort()
        cat_ids = coco_ir.getCatIds()
        records = []
        empty_records = []
        ct = 0


        self.catid2clsid = dict({catid: i for i, catid in enumerate(cat_ids)})
        self.cname2cid = dict({
            coco_ir.loadCats(catid)[0]['name']: clsid
            for catid, clsid in self.catid2clsid.items()
        })

        if 'annotations' not in coco_vis.dataset:
            self.load_image_only = True
            logger.warning('Annotation file: {} does not contains ground truth '
                           'and load image information only.'.format(anno_path_vis))
        if 'annotations' not in coco_ir.dataset:
            self.load_image_only = True
            logger.warning('Annotation file: {} does not contains ground truth '
                           'and load image information only.'.format(anno_path_ir))

        for img_id in img_ids:
            img_anno = coco_ir.loadImgs([img_id])[0]

            im_fname = img_anno['file_name']
            im_w = float(img_anno['width'])
            im_h = float(img_anno['height'])

            vis_im_path = os.path.join(vis_image_dir,
                                   im_fname) if vis_image_dir else im_fname
            ir_im_path = os.path.join(ir_image_dir,
                                       im_fname) if ir_image_dir else im_fname
            is_empty = False
            if not os.path.exists(vis_im_path):
                logger.warning('Illegal vis_image file: {}, and it will be '
                               'ignored'.format(vis_im_path))
                continue
            if not os.path.exists(ir_im_path):
                logger.warning('Illegal ir_image file: {}, and it will be '
                               'ignored'.format(ir_im_path))
                continue

            if im_w < 0 or im_h < 0:
                logger.warning('Illegal width: {} or height: {} in annotation, '
                               'and im_id: {} will be ignored'.format(
                                   im_w, im_h, img_id))
                continue

            coco_rec = {
                'vis_im_file': vis_im_path,
                'ir_im_file': ir_im_path,
                'im_id': np.array([img_id]),
                'h': im_h,
                'w': im_w,
            } if 'image' in self.data_fields else {}

            if not self.load_image_only:
                ins_anno_ids_vis = coco_vis.getAnnIds(
                    imgIds=[img_id], iscrowd=None if self.load_crowd else False)
                instances_vis = coco_vis.loadAnns(ins_anno_ids_vis)

                ins_anno_ids_ir = coco_ir.getAnnIds(
                    imgIds=[img_id], iscrowd=None if self.load_crowd else False)
                instances_ir = coco_ir.loadAnns(ins_anno_ids_ir)

                bboxes_vis = []
                bboxes_ir = []
                is_rbox_anno = False
                for inst in instances_vis:
                    # check gt bbox
                    if inst.get('ignore', False):
                        print(vis_im_path)
                        continue
                    if 'bbox' not in inst.keys():
                        print(vis_im_path)
                        continue
                    else:
                        if not any(np.array(inst['bbox'])):
                            print(vis_im_path)
                            continue

                    x1, y1, box_w, box_h = inst['bbox']
                    x2 = x1 + box_w
                    y2 = y1 + box_h
                    eps = 1e-5
                    if inst['area'] > 0 and x2 - x1 > eps and y2 - y1 > eps:
                        inst['clean_bbox'] = [
                            round(float(x), 3) for x in [x1, y1, x2, y2]
                        ]
                        bboxes_vis.append(inst)
                    else:
                        logger.warning(
                            'Found an invalid bbox in annotations: im_id: {}, '
                            'area: {} x1: {}, y1: {}, x2: {}, y2: {}.'.format(
                                img_id, float(inst['area']), x1, y1, x2, y2))


                for inst in instances_ir:
                    # check gt bbox
                    if inst.get('ignore', False):
                        print(ir_im_path)
                        continue
                    if 'bbox' not in inst.keys():
                        print(ir_im_path)
                        continue
                    else:
                        if not any(np.array(inst['bbox'])):
                            print(ir_im_path)
                            continue

                    x1, y1, box_w, box_h = inst['bbox']
                    x2 = x1 + box_w
                    y2 = y1 + box_h
                    eps = 1e-5
                    if inst['area'] > 0 and x2 - x1 > eps and y2 - y1 > eps:
                        inst['clean_bbox'] = [
                            round(float(x), 3) for x in [x1, y1, x2, y2]
                        ]
                        bboxes_ir.append(inst)
                    else:
                        logger.warning(
                            'Found an invalid bbox in annotations: im_id: {}, '
                            'area: {} x1: {}, y1: {}, x2: {}, y2: {}.'.format(
                                img_id, float(inst['area']), x1, y1, x2, y2))

                num_bbox = len(bboxes_ir)
                if num_bbox <= 0 and not self.allow_empty:
                    print(ir_im_path)
                    continue
                elif num_bbox <= 0:
                    is_empty = True

                gt_bbox_vis = np.zeros((num_bbox, 4), dtype=np.float32)
                gt_bbox_ir = np.zeros((num_bbox, 4), dtype=np.float32)
                gt_class = np.zeros((num_bbox, 1), dtype=np.int32)
                is_crowd = np.zeros((num_bbox, 1), dtype=np.int32)
                gt_poly_vis = [None] * num_bbox
                gt_poly_ir = [None] * num_bbox
                gt_track_id = -np.ones((num_bbox, 1), dtype=np.int32)

                has_segmentation = False
                has_track_id = False
                for i, (box_vis, box_ir) in enumerate(zip(bboxes_vis, bboxes_ir)):
                    catid = box_ir['category_id']
                    gt_class[i][0] = self.catid2clsid[catid]
                    gt_bbox_vis[i, :] = box_vis['clean_bbox']
                    gt_bbox_ir[i, :] = box_ir['clean_bbox']
                    is_crowd[i][0] = box_vis['iscrowd']
                    # check RLE format
                    if 'segmentation' in box_ir and box_ir['iscrowd'] == 1:
                        gt_poly_vis[i] = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
                        gt_poly_ir[i] = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
                    elif 'segmentation' in box_ir and box_ir['segmentation']:
                        if not np.array(
                                box_ir['segmentation'],
                                dtype=object).size > 0 and not self.allow_empty:
                            bboxes_ir.pop(i)
                            bboxes_vis.pop(i)
                            gt_poly_vis.pop(i)
                            gt_poly_ir.pop(i)
                            np.delete(is_crowd, i)
                            np.delete(gt_class, i)
                            np.delete(gt_bbox_ir, i)
                            np.delete(gt_bbox_vis, i)
                        else:
                            gt_poly_vis[i] = box_vis['segmentation']
                            gt_poly_ir[i] = box_ir['segmentation']
                        has_segmentation = True

                    if 'track_id' in box_ir:
                        gt_track_id[i][0] = box_ir['track_id']
                        has_track_id = True

                if has_segmentation and not any(
                        gt_poly_vis) and not self.allow_empty:
                    print(vis_im_path)
                    continue

                gt_rec = {
                    'is_crowd': is_crowd,
                    'gt_class': gt_class,
                    'gt_bbox_vis': gt_bbox_vis,
                    'gt_bbox_ir': gt_bbox_ir,
                    'gt_poly_vis': gt_poly_vis,
                    'gt_poly_ir': gt_poly_ir,
                }
                if has_track_id:
                    gt_rec.update({'gt_track_id': gt_track_id})

                for k, v in gt_rec.items():
                    if k in self.data_fields:
                        coco_rec[k] = v

                # TODO: remove load_semantic
                if self.load_semantic and 'semantic' in self.data_fields:
                    seg_path = os.path.join(self.dataset_dir, 'stuffthingmaps',
                                            'train2017', im_fname[:-3] + 'png')
                    coco_rec.update({'semantic': seg_path})

            logger.debug('Load vis_file: {},Load ir_file: {}, im_id: {}, h: {}, w: {}.'.format(
                vis_im_path,ir_im_path, img_id, im_h, im_w))
            if is_empty:
                empty_records.append(coco_rec)
            else:
                records.append(coco_rec)
            ct += 1
            if self.sample_num > 0 and ct >= self.sample_num:
                break
        assert ct > 0, 'not found any coco record in %s' % (anno_path_ir)
        logger.info('Load [{} samples valid, {} samples invalid] in file {}.'.
                    format(ct, len(img_ids) - ct, anno_path_ir))
        if self.allow_empty and len(empty_records) > 0:
            empty_records = self._sample_empty(empty_records, len(records))
            records += empty_records
        self.roidbs = records



@register
@serializable
class Mutisprctral_COCODataSet(Multi_DetDataset):
    """
    Load dataset with COCO format.

    Args:
        dataset_dir (str): root directory for dataset.
        vis_image_dir (str): directory for  vis_images.
        ir_image_dir (str): directory for  ir_images.
        anno_path (str): coco annotation file path.
        data_fields (list): key name of data dictionary, at least have 'image'.
        sample_num (int): number of samples to load, -1 means all.
        load_crowd (bool): whether to load crowded ground-truth.
            False as default
        allow_empty (bool): whether to load empty entry. False as default
        empty_ratio (float): the ratio of empty record number to total
            record's, if empty_ratio is out of [0. ,1.), do not sample the
            records and use all the empty entries. 1. as default
        repeat (int): repeat times for dataset, use in benchmark.
    """

    def __init__(self,
                 dataset_dir=None,
                 vis_image_dir=None,
                 ir_image_dir=None,
                 anno_path=None,
                 data_fields=['image'],
                 sample_num=-1,
                 load_crowd=False,
                 allow_empty=False,
                 empty_ratio=1.,
                 repeat=1):
        super(Mutisprctral_COCODataSet, self).__init__(
            dataset_dir,
            vis_image_dir,
            ir_image_dir,
            anno_path,
            data_fields,
            sample_num,
            repeat=repeat)
        self.load_image_only = False
        self.load_semantic = False
        self.load_crowd = load_crowd
        self.allow_empty = allow_empty
        self.empty_ratio = empty_ratio

    def _sample_empty(self, records, num):
        # if empty_ratio is out of [0. ,1.), do not sample the records
        if self.empty_ratio < 0. or self.empty_ratio >= 1.:
            return records
        import random
        sample_num = min(
            int(num * self.empty_ratio / (1 - self.empty_ratio)), len(records))
        records = random.sample(records, sample_num)
        return records

    def parse_dataset(self):
        anno_path = os.path.join(self.dataset_dir, self.anno_path)
        vis_image_dir = os.path.join(self.dataset_dir, self.vis_image_dir)
        ir_image_dir = os.path.join(self.dataset_dir, self.ir_image_dir)

        assert anno_path.endswith('.json'), \
            'invalid coco annotation file: ' + anno_path
        from pycocotools.coco import COCO
        coco = COCO(anno_path)
        img_ids = coco.getImgIds()
        img_ids.sort()
        cat_ids = coco.getCatIds()
        records = []
        empty_records = []
        ct = 0


        self.catid2clsid = dict({catid: i for i, catid in enumerate(cat_ids)})
        self.cname2cid = dict({
            coco.loadCats(catid)[0]['name']: clsid
            for catid, clsid in self.catid2clsid.items()
        })

        if 'annotations' not in coco.dataset:
            self.load_image_only = True
            logger.warning('Annotation file: {} does not contains ground truth '
                           'and load image information only.'.format(anno_path))

        for img_id in img_ids:
            img_anno = coco.loadImgs([img_id])[0]
            im_fname = img_anno['file_name']
            im_w = float(img_anno['width'])
            im_h = float(img_anno['height'])

            vis_im_path = os.path.join(vis_image_dir,
                                   im_fname) if vis_image_dir else im_fname
            ir_im_path = os.path.join(ir_image_dir,
                                       im_fname) if ir_image_dir else im_fname
            is_empty = False
            if not os.path.exists(vis_im_path):
                logger.warning('Illegal vis_image file: {}, and it will be '
                               'ignored'.format(vis_im_path))
                continue
            if not os.path.exists(ir_im_path):
                logger.warning('Illegal ir_image file: {}, and it will be '
                               'ignored'.format(ir_im_path))
                continue

            if im_w < 0 or im_h < 0:
                logger.warning('Illegal width: {} or height: {} in annotation, '
                               'and im_id: {} will be ignored'.format(
                                   im_w, im_h, img_id))
                continue

            coco_rec = {
                'vis_im_file': vis_im_path,
                'ir_im_file': ir_im_path,
                'im_id': np.array([img_id]),
                'h': im_h,
                'w': im_w,
            } if 'image' in self.data_fields else {}

            if not self.load_image_only:
                ins_anno_ids = coco.getAnnIds(
                    imgIds=[img_id], iscrowd=None if self.load_crowd else False)
                instances = coco.loadAnns(ins_anno_ids)

                bboxes = []
                is_rbox_anno = False
                for inst in instances:
                    # check gt bbox
                    if inst.get('ignore', False):
                        print(vis_im_path)
                        continue
                    if 'bbox' not in inst.keys():
                        print(vis_im_path)
                        continue
                    else:
                        if not any(np.array(inst['bbox'])):
                            print(vis_im_path)
                            continue

                    x1, y1, box_w, box_h = inst['bbox']
                    x2 = x1 + box_w
                    y2 = y1 + box_h
                    eps = 1e-5
                    if inst['area'] > 0 and x2 - x1 > eps and y2 - y1 > eps:
                        inst['clean_bbox'] = [
                            round(float(x), 3) for x in [x1, y1, x2, y2]
                        ]
                        bboxes.append(inst)
                    else:
                        logger.warning(
                            'Found an invalid bbox in annotations: im_id: {}, '
                            'area: {} x1: {}, y1: {}, x2: {}, y2: {}.'.format(
                                img_id, float(inst['area']), x1, y1, x2, y2))

                num_bbox = len(bboxes)
                if num_bbox <= 0 and not self.allow_empty:
                    print(vis_im_path)
                    continue
                elif num_bbox <= 0:
                    is_empty = True

                gt_bbox = np.zeros((num_bbox, 4), dtype=np.float32)
                gt_class = np.zeros((num_bbox, 1), dtype=np.int32)
                is_crowd = np.zeros((num_bbox, 1), dtype=np.int32)
                gt_poly = [None] * num_bbox
                gt_track_id = -np.ones((num_bbox, 1), dtype=np.int32)

                has_segmentation = False
                has_track_id = False
                for i, box in enumerate(bboxes):
                    catid = box['category_id']
                    gt_class[i][0] = self.catid2clsid[catid]
                    gt_bbox[i, :] = box['clean_bbox']
                    is_crowd[i][0] = box['iscrowd']
                    # check RLE format
                    if 'segmentation' in box and box['iscrowd'] == 1:
                        gt_poly[i] = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
                    elif 'segmentation' in box and box['segmentation']:
                        if not np.array(
                                box['segmentation'],
                                dtype=object).size > 0 and not self.allow_empty:
                            bboxes.pop(i)
                            gt_poly.pop(i)
                            np.delete(is_crowd, i)
                            np.delete(gt_class, i)
                            np.delete(gt_bbox, i)
                        else:
                            gt_poly[i] = box['segmentation']
                        has_segmentation = True

                    if 'track_id' in box:
                        gt_track_id[i][0] = box['track_id']
                        has_track_id = True

                if has_segmentation and not any(
                        gt_poly) and not self.allow_empty:
                    print(vis_im_path)
                    continue

                gt_rec = {
                    'is_crowd': is_crowd,
                    'gt_class': gt_class,
                    'gt_bbox': gt_bbox,
                    'gt_poly': gt_poly,
                }
                if has_track_id:
                    gt_rec.update({'gt_track_id': gt_track_id})

                for k, v in gt_rec.items():
                    if k in self.data_fields:
                        coco_rec[k] = v

                # TODO: remove load_semantic
                if self.load_semantic and 'semantic' in self.data_fields:
                    seg_path = os.path.join(self.dataset_dir, 'stuffthingmaps',
                                            'train2017', im_fname[:-3] + 'png')
                    coco_rec.update({'semantic': seg_path})

            logger.debug('Load vis_file: {},Load ir_file: {}, im_id: {}, h: {}, w: {}.'.format(
                vis_im_path,ir_im_path, img_id, im_h, im_w))
            if is_empty:
                empty_records.append(coco_rec)
            else:
                records.append(coco_rec)
            ct += 1
            if self.sample_num > 0 and ct >= self.sample_num:
                break
        assert ct > 0, 'not found any coco record in %s' % (anno_path)
        logger.info('Load [{} samples valid, {} samples invalid] in file {}.'.
                    format(ct, len(img_ids) - ct, anno_path))
        if self.allow_empty and len(empty_records) > 0:
            empty_records = self._sample_empty(empty_records, len(records))
            records += empty_records
        self.roidbs = records


@register
@serializable
class COCODataSet(DetDataset):
    """
    Load dataset with COCO format.

    Args:
        dataset_dir (str): root directory for dataset.
        image_dir (str): directory for images.
        anno_path (str): coco annotation file path.
        data_fields (list): key name of data dictionary, at least have 'image'.
        sample_num (int): number of samples to load, -1 means all.
        load_crowd (bool): whether to load crowded ground-truth. 
            False as default
        allow_empty (bool): whether to load empty entry. False as default
        empty_ratio (float): the ratio of empty record number to total 
            record's, if empty_ratio is out of [0. ,1.), do not sample the 
            records and use all the empty entries. 1. as default
        repeat (int): repeat times for dataset, use in benchmark.
    """

    def __init__(self,
                 dataset_dir=None,
                 image_dir=None,
                 anno_path=None,
                 data_fields=['image'],
                 sample_num=-1,
                 load_crowd=False,
                 allow_empty=False,
                 empty_ratio=1.,
                 repeat=1):
        super(COCODataSet, self).__init__(
            dataset_dir,
            image_dir,
            anno_path,
            data_fields,
            sample_num,
            repeat=repeat)
        self.load_image_only = False
        self.load_semantic = False
        self.load_crowd = load_crowd
        self.allow_empty = allow_empty
        self.empty_ratio = empty_ratio

    def _sample_empty(self, records, num):
        # if empty_ratio is out of [0. ,1.), do not sample the records
        if self.empty_ratio < 0. or self.empty_ratio >= 1.:
            return records
        import random
        sample_num = min(
            int(num * self.empty_ratio / (1 - self.empty_ratio)), len(records))
        records = random.sample(records, sample_num)
        return records

    def parse_dataset(self):
        anno_path = os.path.join(self.dataset_dir, self.anno_path)
        image_dir = os.path.join(self.dataset_dir, self.image_dir)

        assert anno_path.endswith('.json'), \
            'invalid coco annotation file: ' + anno_path
        from pycocotools.coco import COCO
        coco = COCO(anno_path)
        img_ids = coco.getImgIds()
        img_ids.sort()
        cat_ids = coco.getCatIds()
        records = []
        empty_records = []
        ct = 0

        self.catid2clsid = dict({catid: i for i, catid in enumerate(cat_ids)})
        self.cname2cid = dict({
            coco.loadCats(catid)[0]['name']: clsid
            for catid, clsid in self.catid2clsid.items()
        })

        if 'annotations' not in coco.dataset:
            self.load_image_only = True
            logger.warning('Annotation file: {} does not contains ground truth '
                           'and load image information only.'.format(anno_path))

        for img_id in img_ids:
            img_anno = coco.loadImgs([img_id])[0]
            im_fname = img_anno['file_name']
            im_w = float(img_anno['width'])
            im_h = float(img_anno['height'])

            im_path = os.path.join(image_dir,
                                   im_fname) if image_dir else im_fname
            is_empty = False
            if not os.path.exists(im_path):
                logger.warning('Illegal image file: {}, and it will be '
                               'ignored'.format(im_path))
                continue

            if im_w < 0 or im_h < 0:
                logger.warning('Illegal width: {} or height: {} in annotation, '
                               'and im_id: {} will be ignored'.format(
                                   im_w, im_h, img_id))
                continue

            coco_rec = {
                'im_file': im_path,
                'im_id': np.array([img_id]),
                'h': im_h,
                'w': im_w,
            } if 'image' in self.data_fields else {}

            if not self.load_image_only:
                ins_anno_ids = coco.getAnnIds(
                    imgIds=[img_id], iscrowd=None if self.load_crowd else False)
                instances = coco.loadAnns(ins_anno_ids)

                bboxes = []
                is_rbox_anno = False
                for inst in instances:
                    # check gt bbox
                    if inst.get('ignore', False):
                        continue
                    if 'bbox' not in inst.keys():
                        continue
                    else:
                        if not any(np.array(inst['bbox'])):
                            continue

                    x1, y1, box_w, box_h = inst['bbox']
                    x2 = x1 + box_w
                    y2 = y1 + box_h
                    eps = 1e-5
                    if inst['area'] > 0 and x2 - x1 > eps and y2 - y1 > eps:
                        inst['clean_bbox'] = [
                            round(float(x), 3) for x in [x1, y1, x2, y2]
                        ]
                        bboxes.append(inst)
                    else:
                        logger.warning(
                            'Found an invalid bbox in annotations: im_id: {}, '
                            'area: {} x1: {}, y1: {}, x2: {}, y2: {}.'.format(
                                img_id, float(inst['area']), x1, y1, x2, y2))

                num_bbox = len(bboxes)
                if num_bbox <= 0 and not self.allow_empty:
                    continue
                elif num_bbox <= 0:
                    is_empty = True

                gt_bbox = np.zeros((num_bbox, 4), dtype=np.float32)
                gt_class = np.zeros((num_bbox, 1), dtype=np.int32)
                is_crowd = np.zeros((num_bbox, 1), dtype=np.int32)
                gt_poly = [None] * num_bbox
                gt_track_id = -np.ones((num_bbox, 1), dtype=np.int32)

                has_segmentation = False
                has_track_id = False
                for i, box in enumerate(bboxes):
                    catid = box['category_id']
                    gt_class[i][0] = self.catid2clsid[catid]
                    gt_bbox[i, :] = box['clean_bbox']
                    is_crowd[i][0] = box['iscrowd']
                    # check RLE format 
                    if 'segmentation' in box and box['iscrowd'] == 1:
                        gt_poly[i] = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
                    elif 'segmentation' in box and box['segmentation']:
                        if not np.array(
                                box['segmentation'],
                                dtype=object).size > 0 and not self.allow_empty:
                            bboxes.pop(i)
                            gt_poly.pop(i)
                            np.delete(is_crowd, i)
                            np.delete(gt_class, i)
                            np.delete(gt_bbox, i)
                        else:
                            gt_poly[i] = box['segmentation']
                        has_segmentation = True

                    if 'track_id' in box:
                        gt_track_id[i][0] = box['track_id']
                        has_track_id = True

                if has_segmentation and not any(
                        gt_poly) and not self.allow_empty:
                    continue

                gt_rec = {
                    'is_crowd': is_crowd,
                    'gt_class': gt_class,
                    'gt_bbox': gt_bbox,
                    'gt_poly': gt_poly,
                }
                if has_track_id:
                    gt_rec.update({'gt_track_id': gt_track_id})

                for k, v in gt_rec.items():
                    if k in self.data_fields:
                        coco_rec[k] = v

                # TODO: remove load_semantic
                if self.load_semantic and 'semantic' in self.data_fields:
                    seg_path = os.path.join(self.dataset_dir, 'stuffthingmaps',
                                            'train2017', im_fname[:-3] + 'png')
                    coco_rec.update({'semantic': seg_path})

            logger.debug('Load file: {}, im_id: {}, h: {}, w: {}.'.format(
                im_path, img_id, im_h, im_w))
            if is_empty:
                empty_records.append(coco_rec)
            else:
                records.append(coco_rec)
            ct += 1
            if self.sample_num > 0 and ct >= self.sample_num:
                break
        assert ct > 0, 'not found any coco record in %s' % (anno_path)
        logger.info('Load [{} samples valid, {} samples invalid] in file {}.'.
                    format(ct, len(img_ids) - ct, anno_path))
        if self.allow_empty and len(empty_records) > 0:
            empty_records = self._sample_empty(empty_records, len(records))
            records += empty_records
        self.roidbs = records


@register
@serializable
class SlicedCOCODataSet(COCODataSet):
    """Sliced COCODataSet"""

    def __init__(
            self,
            dataset_dir=None,
            image_dir=None,
            anno_path=None,
            data_fields=['image'],
            sample_num=-1,
            load_crowd=False,
            allow_empty=False,
            empty_ratio=1.,
            repeat=1,
            sliced_size=[640, 640],
            overlap_ratio=[0.25, 0.25], ):
        super(SlicedCOCODataSet, self).__init__(
            dataset_dir=dataset_dir,
            image_dir=image_dir,
            anno_path=anno_path,
            data_fields=data_fields,
            sample_num=sample_num,
            load_crowd=load_crowd,
            allow_empty=allow_empty,
            empty_ratio=empty_ratio,
            repeat=repeat, )
        self.sliced_size = sliced_size
        self.overlap_ratio = overlap_ratio

    def parse_dataset(self):
        anno_path = os.path.join(self.dataset_dir, self.anno_path)
        image_dir = os.path.join(self.dataset_dir, self.image_dir)

        assert anno_path.endswith('.json'), \
            'invalid coco annotation file: ' + anno_path
        from pycocotools.coco import COCO
        coco = COCO(anno_path)
        img_ids = coco.getImgIds()
        img_ids.sort()
        cat_ids = coco.getCatIds()
        records = []
        empty_records = []
        ct = 0
        ct_sub = 0

        self.catid2clsid = dict({catid: i for i, catid in enumerate(cat_ids)})
        self.cname2cid = dict({
            coco.loadCats(catid)[0]['name']: clsid
            for catid, clsid in self.catid2clsid.items()
        })

        if 'annotations' not in coco.dataset:
            self.load_image_only = True
            logger.warning('Annotation file: {} does not contains ground truth '
                           'and load image information only.'.format(anno_path))
        try:
            import sahi
            from sahi.slicing import slice_image
        except Exception as e:
            logger.error(
                'sahi not found, plaese install sahi. '
                'for example: `pip install sahi`, see https://github.com/obss/sahi.'
            )
            raise e

        sub_img_ids = 0
        for img_id in img_ids:
            img_anno = coco.loadImgs([img_id])[0]
            im_fname = img_anno['file_name']
            im_w = float(img_anno['width'])
            im_h = float(img_anno['height'])

            im_path = os.path.join(image_dir,
                                   im_fname) if image_dir else im_fname
            is_empty = False
            if not os.path.exists(im_path):
                logger.warning('Illegal image file: {}, and it will be '
                               'ignored'.format(im_path))
                continue

            if im_w < 0 or im_h < 0:
                logger.warning('Illegal width: {} or height: {} in annotation, '
                               'and im_id: {} will be ignored'.format(
                                   im_w, im_h, img_id))
                continue

            slice_image_result = sahi.slicing.slice_image(
                image=im_path,
                slice_height=self.sliced_size[0],
                slice_width=self.sliced_size[1],
                overlap_height_ratio=self.overlap_ratio[0],
                overlap_width_ratio=self.overlap_ratio[1])

            sub_img_num = len(slice_image_result)
            for _ind in range(sub_img_num):
                im = slice_image_result.images[_ind]
                coco_rec = {
                    'image': im,
                    'im_id': np.array([sub_img_ids + _ind]),
                    'h': im.shape[0],
                    'w': im.shape[1],
                    'ori_im_id': np.array([img_id]),
                    'st_pix': np.array(
                        slice_image_result.starting_pixels[_ind],
                        dtype=np.float32),
                    'is_last': 1 if _ind == sub_img_num - 1 else 0,
                } if 'image' in self.data_fields else {}
                records.append(coco_rec)
            ct_sub += sub_img_num
            ct += 1
            if self.sample_num > 0 and ct >= self.sample_num:
                break
        assert ct > 0, 'not found any coco record in %s' % (anno_path)
        logger.info('{} samples and slice to {} sub_samples in file {}'.format(
            ct, ct_sub, anno_path))
        if self.allow_empty and len(empty_records) > 0:
            empty_records = self._sample_empty(empty_records, len(records))
            records += empty_records
        self.roidbs = records


@register
@serializable
class SemiCOCODataSet(COCODataSet):
    """Semi-COCODataSet used for supervised and unsupervised dataSet"""

    def __init__(self,
                 dataset_dir=None,
                 image_dir=None,
                 anno_path=None,
                 data_fields=['image'],
                 sample_num=-1,
                 load_crowd=False,
                 allow_empty=False,
                 empty_ratio=1.,
                 repeat=1,
                 supervised=True):
        super(SemiCOCODataSet, self).__init__(
            dataset_dir, image_dir, anno_path, data_fields, sample_num,
            load_crowd, allow_empty, empty_ratio, repeat)
        self.supervised = supervised
        self.length = -1  # defalut -1 means all

    def parse_dataset(self):
        anno_path = os.path.join(self.dataset_dir, self.anno_path)
        image_dir = os.path.join(self.dataset_dir, self.image_dir)

        assert anno_path.endswith('.json'), \
            'invalid coco annotation file: ' + anno_path
        from pycocotools.coco import COCO
        coco = COCO(anno_path)
        img_ids = coco.getImgIds()
        img_ids.sort()
        cat_ids = coco.getCatIds()
        records = []
        empty_records = []
        ct = 0

        self.catid2clsid = dict({catid: i for i, catid in enumerate(cat_ids)})
        self.cname2cid = dict({
            coco.loadCats(catid)[0]['name']: clsid
            for catid, clsid in self.catid2clsid.items()
        })

        if 'annotations' not in coco.dataset or self.supervised == False:
            self.load_image_only = True
            logger.warning('Annotation file: {} does not contains ground truth '
                           'and load image information only.'.format(anno_path))

        for img_id in img_ids:
            img_anno = coco.loadImgs([img_id])[0]
            im_fname = img_anno['file_name']
            im_w = float(img_anno['width'])
            im_h = float(img_anno['height'])

            im_path = os.path.join(image_dir,
                                   im_fname) if image_dir else im_fname
            is_empty = False
            if not os.path.exists(im_path):
                logger.warning('Illegal image file: {}, and it will be '
                               'ignored'.format(im_path))
                continue

            if im_w < 0 or im_h < 0:
                logger.warning('Illegal width: {} or height: {} in annotation, '
                               'and im_id: {} will be ignored'.format(
                                   im_w, im_h, img_id))
                continue

            coco_rec = {
                'im_file': im_path,
                'im_id': np.array([img_id]),
                'h': im_h,
                'w': im_w,
            } if 'image' in self.data_fields else {}

            if not self.load_image_only:
                ins_anno_ids = coco.getAnnIds(
                    imgIds=[img_id], iscrowd=None if self.load_crowd else False)
                instances = coco.loadAnns(ins_anno_ids)

                bboxes = []
                is_rbox_anno = False
                for inst in instances:
                    # check gt bbox
                    if inst.get('ignore', False):
                        continue
                    if 'bbox' not in inst.keys():
                        continue
                    else:
                        if not any(np.array(inst['bbox'])):
                            continue

                    x1, y1, box_w, box_h = inst['bbox']
                    x2 = x1 + box_w
                    y2 = y1 + box_h
                    eps = 1e-5
                    if inst['area'] > 0 and x2 - x1 > eps and y2 - y1 > eps:
                        inst['clean_bbox'] = [
                            round(float(x), 3) for x in [x1, y1, x2, y2]
                        ]
                        bboxes.append(inst)
                    else:
                        logger.warning(
                            'Found an invalid bbox in annotations: im_id: {}, '
                            'area: {} x1: {}, y1: {}, x2: {}, y2: {}.'.format(
                                img_id, float(inst['area']), x1, y1, x2, y2))

                num_bbox = len(bboxes)
                if num_bbox <= 0 and not self.allow_empty:
                    continue
                elif num_bbox <= 0:
                    is_empty = True

                gt_bbox = np.zeros((num_bbox, 4), dtype=np.float32)
                gt_class = np.zeros((num_bbox, 1), dtype=np.int32)
                is_crowd = np.zeros((num_bbox, 1), dtype=np.int32)
                gt_poly = [None] * num_bbox

                has_segmentation = False
                for i, box in enumerate(bboxes):
                    catid = box['category_id']
                    gt_class[i][0] = self.catid2clsid[catid]
                    gt_bbox[i, :] = box['clean_bbox']
                    is_crowd[i][0] = box['iscrowd']
                    # check RLE format 
                    if 'segmentation' in box and box['iscrowd'] == 1:
                        gt_poly[i] = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
                    elif 'segmentation' in box and box['segmentation']:
                        if not np.array(box['segmentation']
                                        ).size > 0 and not self.allow_empty:
                            bboxes.pop(i)
                            gt_poly.pop(i)
                            np.delete(is_crowd, i)
                            np.delete(gt_class, i)
                            np.delete(gt_bbox, i)
                        else:
                            gt_poly[i] = box['segmentation']
                        has_segmentation = True

                if has_segmentation and not any(
                        gt_poly) and not self.allow_empty:
                    continue

                gt_rec = {
                    'is_crowd': is_crowd,
                    'gt_class': gt_class,
                    'gt_bbox': gt_bbox,
                    'gt_poly': gt_poly,
                }

                for k, v in gt_rec.items():
                    if k in self.data_fields:
                        coco_rec[k] = v

                # TODO: remove load_semantic
                if self.load_semantic and 'semantic' in self.data_fields:
                    seg_path = os.path.join(self.dataset_dir, 'stuffthingmaps',
                                            'train2017', im_fname[:-3] + 'png')
                    coco_rec.update({'semantic': seg_path})

            logger.debug('Load file: {}, im_id: {}, h: {}, w: {}.'.format(
                im_path, img_id, im_h, im_w))
            if is_empty:
                empty_records.append(coco_rec)
            else:
                records.append(coco_rec)
            ct += 1
            if self.sample_num > 0 and ct >= self.sample_num:
                break
        assert ct > 0, 'not found any coco record in %s' % (anno_path)
        logger.info('Load [{} samples valid, {} samples invalid] in file {}.'.
                    format(ct, len(img_ids) - ct, anno_path))
        if self.allow_empty and len(empty_records) > 0:
            empty_records = self._sample_empty(empty_records, len(records))
            records += empty_records
        self.roidbs = records

        if self.supervised:
            logger.info(f'Use {len(self.roidbs)} sup_samples data as LABELED')
        else:
            if self.length > 0:  # unsup length will be decide by sup length
                all_roidbs = self.roidbs.copy()
                selected_idxs = [
                    np.random.choice(len(all_roidbs))
                    for _ in range(self.length)
                ]
                self.roidbs = [all_roidbs[i] for i in selected_idxs]
            logger.info(
                f'Use {len(self.roidbs)} unsup_samples data as UNLABELED')

    def __getitem__(self, idx):
        n = len(self.roidbs)
        if self.repeat > 1:
            idx %= n
        # data batch
        roidb = copy.deepcopy(self.roidbs[idx])
        if self.mixup_epoch == 0 or self._epoch < self.mixup_epoch:
            idx = np.random.randint(n)
            roidb = [roidb, copy.deepcopy(self.roidbs[idx])]
        elif self.cutmix_epoch == 0 or self._epoch < self.cutmix_epoch:
            idx = np.random.randint(n)
            roidb = [roidb, copy.deepcopy(self.roidbs[idx])]
        elif self.mosaic_epoch == 0 or self._epoch < self.mosaic_epoch:
            roidb = [roidb, ] + [
                copy.deepcopy(self.roidbs[np.random.randint(n)])
                for _ in range(4)
            ]
        if isinstance(roidb, Sequence):
            for r in roidb:
                r['curr_iter'] = self._curr_iter
        else:
            roidb['curr_iter'] = self._curr_iter
        self._curr_iter += 1

        return self.transform(roidb)
