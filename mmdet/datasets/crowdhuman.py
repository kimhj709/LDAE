import copy
import json
import random

import numpy as np

from mmdet.datasets import DATASETS, CustomDataset

from .crowdhumantools import Database


@DATASETS.register_module()
class CrowdHumanDataset(CustomDataset):
    def __init__(self, *args, debug=False, **kwargs):
        self.debug = debug
        super(CrowdHumanDataset, self).__init__(*args, **kwargs)

    def load_annotations(self, json_file):

        with open(json_file, 'r') as file:
            gt_records = file.readlines()
        dataset_dicts = []

        ann_keys = ['tag', 'hbox', 'vbox', 'head_attr', 'extra']
        for i, anno_str in enumerate(gt_records):
            if i > 10 and self.debug:
                break

            anno_dict = json.loads(anno_str)
            record = {}
            record['file_name'] = '{}.jpg'.format(anno_dict['ID'])
            record['image_id'] = anno_dict['ID']
            anns = dict()
            bbox_list = []
            label_list = []
            for anno in anno_dict['gtboxes']:

                obj = {key: anno[key] for key in ann_keys if key in anno}
                if obj['tag'] == 'mask':
                    continue
                x, y, w, h = anno['fbox']
                bbox_list.append([x, y, x + w, y + h])
                label_list.append(0)
            if len(bbox_list):
                anns['bboxes'] = np.array(bbox_list, dtype=np.float32)
                anns['labels'] = np.array(label_list, dtype=np.int64)
            else:
                anns['bboxes'] = np.zeros((0, 4), dtype=np.float32)
                anns['labels'] = np.array([], dtype=np.int64)
            record['anns'] = anns
            # import mmcv
            # mmcv.imshow_bboxes(record["file_name"], anns["bboxes"])
            dataset_dicts.append(record)

        return dataset_dicts

    def _set_group_flag(self):
        self.flag = np.zeros(len(self), dtype=np.uint8)

    def _filter_imgs(self, min_size=16):
        valid_inds = []
        for i, record in enumerate(self.data_infos):
            if not len(record['anns']['bboxes']):
                continue
            valid_inds.append(i)
        return valid_inds

    def prepare_train_img(self, idx):

        img_info = dict()
        ann_info = dict(bboxes=None,
                        labels=None,
                        bboxes_ignore=None,
                        masks=None,
                        seg_map=None)
        img_info['filename'] = copy.deepcopy(self.data_infos[idx]['file_name'])

        ann_info['bboxes'] = copy.deepcopy(
            self.data_infos[idx]['anns']['bboxes'])
        ann_info['labels'] = copy.deepcopy(
            self.data_infos[idx]['anns']['labels'])
        ann_info['bboxes_ignore'] = np.zeros((0, 4), dtype=np.float32)
        results = dict(img_info=img_info, ann_info=ann_info)
        self.pre_pipeline(results)
        return self.pipeline(results)

    def prepare_test_img(self, idx):
        img_info = dict()
        img_info['filename'] = copy.deepcopy(self.data_infos[idx]['file_name'])
        results = dict(img_info=img_info)
        self.pre_pipeline(results)
        return self.pipeline(results)

    def evaluate(self,
                 results,
                 metric='mAP',
                 logger=None,
                 proposal_nums=(100, 300, 1000),
                 iou_thr=0.5,
                 scale_ranges=None):
        f = open(__file__.replace('crowdhuman.py', 'id_hw_val.json'))
        ID_hw = json.load(f)
        f.close()
        filename = str(random.random()).split('.')[-1]
        with open(f'./{filename}_crowd_eval.json', 'w') as f:
            for i, single_results in enumerate(results):
                dump_dict = dict()
                dump_dict['ID'] = self.data_infos[i]['image_id']
                dump_dict['height'] = ID_hw[dump_dict['ID']][0]
                dump_dict['width'] = ID_hw[dump_dict['ID']][1]
                dtboxes = []
                bboxes = single_results[0].tolist()
                for bbox_id, single_bbox in enumerate(bboxes):
                    temp_dict = dict()
                    x1, y1, x2, y2, score = single_bbox
                    temp_dict['box'] = [x1, y1, x2 - x1, y2 - y1]
                    temp_dict['score'] = single_bbox[-1]
                    temp_dict['tag'] = 1
                    dtboxes.append(temp_dict)
                dump_dict['dtboxes'] = dtboxes
                f.write(json.dumps(dump_dict) + '\n')

        database = Database(self.ann_file, f'./{filename}_crowd_eval.json',
                            'box', None, 0)
        database.compare()
        AP, recall, _ = database.eval_AP()
        mMR, _ = database.eval_MR()
        import os
        os.popen(f'rm {filename}_crowd_eval.json')
        return dict(bbox_mAP=round(AP*100, 1), mMR=round(mMR*100, 1), recall=round(recall*100, 1))

    def __repr__(self):
        """Print the number of instance number."""
        pass

    def __len__(self):
        return len(self.data_infos)