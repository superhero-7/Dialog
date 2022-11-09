# coding=utf-8

from pathlib import Path
import argparse
import json

import cv2
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

from detectron2_given_target_box_maxnms import extract, DIM

from pycocotools.coco import COCO


class RefCOCODataset(Dataset):
    def __init__(self, refcoco_dir, refcoco_images_dir, bad_sent_path, coco_dir, split='val',dataset='refcoco',
                 dataset_split='unc', ofa_extract=False):

        self.image_dir = refcoco_images_dir
        self.ofa_extract = ofa_extract

        # coco_train_annFile = coco_dir.joinpath('annotations/instances_train2014.json')
        # self.coco = COCO(coco_train_annFile)

        assert split in ['train', 'val', 'test', 'testA', 'testB']

        workspace_dir = Path(__file__).resolve().parent.parent
        refcoco_util_dir = workspace_dir.joinpath('VL-T5', 'src')
        import sys
        sys.path.append(str(refcoco_util_dir))
        from refcoco_utils import REFER
        #self.refer = REFER('refcocog', 'umd')
        self.refer = REFER(dataset, dataset_split)

        if self.ofa_extract:
            with open(bad_sent_path, 'r') as f:
                self.bad_sent = json.load(f)

            self.ref_ids = []
            self.refid2bbox = {}
            for item in self.bad_sent:
                ref_id = item['ref_id']
                self.ref_ids.append(ref_id)
                self.refid2bbox[ref_id] = item['bbox']
        else:
            self.ref_ids = self.refer.getRefIds(split=split)


    def __len__(self):
        return len(self.ref_ids)

    def __getitem__(self, idx):

        # ref_id 是int 类型的...
        ref_id = self.ref_ids[idx]

        ref = self.refer.Refs[ref_id]
        category_id = ref['category_id']
        image_id = ref['image_id']
        fn_ann = ref['file_name']
        # COCO_train2014_000000419645_398406.jpg
        # COCO_train2014_000000419645.jpg
        suffix = fn_ann.split('.')[-1]
        image_fn = '_'.join(fn_ann.split('_')[:-1]) + '.' + suffix
        image_path = self.image_dir.joinpath(image_fn)


        assert Path(image_path).exists(), image_path

        img = cv2.imread(str(image_path))

        H, W, C = img.shape

        if self.ofa_extract:
            x1, y1, x2, y2 = self.refid2bbox[ref_id]
        else:
            det = self.refer.getRefBox(ref_id)
        #det = self.id2dets[ref_id]
        # cat_names = [det['category_name'] for det in dets]
            x, y, w, h = det[:4]
            x1, y1, x2, y2 = x, y, x + w, y + h

        boxes = []
        # (x1, y1, x2, y2)

        if x2>W:
            x2 = W
        if y2>H:
            y2 = H

        # x1, y1, x2, y2 = region[:4]

        assert x2 <= W, (ref_id, det, x2, W)
        assert y2 <= H, (ref_id, det, y2, H)

        box = [x1, y1, x2, y2]
        boxes.append(box)

        boxes = np.array(boxes)

        return {
            'ref_id': str(ref_id),
            'img_id': str(image_id),
            'img_fn': image_fn,
            'img': img,
            'boxes': boxes,
            'category_id': category_id,
            # 'captions': cat_names
        }

def collate_fn(batch):
    img_ids = []
    imgs = []

    boxes = []
    ref_ids = []
    # 另外一个脚本: refcocog_mattner.py 用了caption,是不是不用caption也可以？
    captions = []
    category_ids = []

    for i, entry in enumerate(batch):

        ref_ids.append(entry['ref_id'])
        img_ids.append(entry['img_id'])
        imgs.append(entry['img'])
        boxes.append(entry['boxes'])
        category_ids.append(entry['category_id'])
        # captions.append(entry['captions'])

    batch_out = {}
    batch_out['img_ids'] = img_ids
    batch_out['imgs'] = imgs

    batch_out['boxes'] = boxes
    batch_out['ref_ids'] = ref_ids
    batch_out['category_id'] = category_ids

    # batch_out['captions'] = captions

    return batch_out


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', default=1, type=int, help='batch_size')
    parser.add_argument('--refcocoroot', type=str, default='/raid_sda/yfl/datasets/RefCOCO')
    parser.add_argument('--cocoroot', type=str, default='/raid_sda/yfl/datasets/')
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val', 'test', 'testA', 'testB'])
    parser.add_argument('--dataset', type=str, default='refcoco+')
    parser.add_argument('--dataset_split', type=str, default='unc')

    args = parser.parse_args()

    refcoco_dir = Path(args.refcocoroot).resolve()
    refcocog_dir = refcoco_dir.joinpath(args.dataset)
    coco_dir = Path(args.cocoroot).resolve()
    refcoco_images_dir = coco_dir.joinpath('train2014')
    dataset_name = args.dataset
    bad_sent_path = '/raid_sda/yfl/codebase/VL-T5-REG/VL-T5/src/REG_mmi_refcoco+_vlt5_bad_sent_threshold_0.5_with_bbox_val.json'

    out_dir = refcocog_dir.joinpath('features')
    if not out_dir.exists():
        out_dir.mkdir()

    dataset = RefCOCODataset(refcoco_dir, refcoco_images_dir, bad_sent_path, coco_dir, args.split, dataset=args.dataset,
                             dataset_split=args.dataset_split, ofa_extract=True)
    print('# Images:', len(dataset))

    dataloader = DataLoader(dataset, batch_size=args.batchsize,
                            shuffle=False, collate_fn=collate_fn, num_workers=4)

    output_fname = out_dir.joinpath(f'{args.split}_ofa_target.h5')
    print('features will be saved at', output_fname)

    # DIM是2048
    desc = f'{dataset_name}_given_boxes_({DIM})'

    extract(output_fname, dataloader, desc)
