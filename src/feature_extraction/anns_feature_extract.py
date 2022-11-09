# coding=utf-8

from pathlib import Path
import argparse
import json

import cv2
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

from detectron2_given_target_box_ann import extract, DIM

from pycocotools.coco import COCO


class RefCOCODataset(Dataset):
    def __init__(self, refcoco_dir, refcoco_images_dir, coco_dir, split='val',dataset='refcoco',
                 dataset_split='unc'):

        self.image_dir = refcoco_images_dir

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

        self.ref_ids = self.refer.getRefIds(split=split)
        self.img_ids = []

        for ref_id in self.ref_ids:
            ref = self.refer.Refs[ref_id]
            img_id = ref['image_id']
            if img_id not in self.img_ids:
                self.img_ids.append(img_id)

        self.ann_ids = []
        for img_id in self.img_ids:
            for ann in self.refer.imgToAnns[img_id]:
                self.ann_ids.append(ann['id'])

    def __len__(self):
        return len(self.ann_ids)

    def __getitem__(self, idx):

        # ref_id 是int 类型的...
        ann_id = self.ann_ids[idx]

        ann = self.refer.Anns[ann_id]
        category_id = ann['category_id']
        image_id = ann['image_id']
        # COCO_train2014_000000419645_398406.jpg
        # COCO_train2014_000000419645.jpg
        l = len(str(image_id))
        pad_l = 12-l
        image_fn = 'COCO_train2014_' + '0'*pad_l + str(image_id) + '.jpg'
        image_path = self.image_dir.joinpath(image_fn)


        assert Path(image_path).exists(), image_path

        img = cv2.imread(str(image_path))

        H, W, C = img.shape

        det = ann['bbox']
        #det = self.id2dets[ref_id]
        # cat_names = [det['category_name'] for det in dets]

        boxes = []
        # (x1, y1, x2, y2)
        x, y, w, h = det[:4]
        x1, y1, x2, y2 = x, y, x+w, y+h

        if x2>W:
            x2 = W
        if y2>H:
            y2 = H

        # x1, y1, x2, y2 = region[:4]

        assert x2 <= W, (ann_id, det, x2, W)
        assert y2 <= H, (ann_id, det, y2, H)

        box = [x1, y1, x2, y2]
        boxes.append(box)

        boxes = np.array(boxes)

        return {
            'ann_id': str(ann_id),
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
    ann_ids = []
    # 另外一个脚本: refcocog_mattner.py 用了caption,是不是不用caption也可以？
    captions = []
    category_ids = []

    for i, entry in enumerate(batch):

        ann_ids.append(entry['ann_id'])
        img_ids.append(entry['img_id'])
        imgs.append(entry['img'])
        boxes.append(entry['boxes'])
        category_ids.append(entry['category_id'])
        # captions.append(entry['captions'])

    batch_out = {}
    batch_out['img_ids'] = img_ids
    batch_out['imgs'] = imgs

    batch_out['boxes'] = boxes
    batch_out['ann_ids'] = ann_ids
    batch_out['category_id'] = category_ids

    # batch_out['captions'] = captions

    return batch_out


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', default=1, type=int, help='batch_size')
    parser.add_argument('--refcocoroot', type=str, default='/workspace/yfl/codebase/VL-T5/datasets/RefCOCO')
    parser.add_argument('--cocoroot', type=str, default='/workspace/yfl/datasets/')
    parser.add_argument('--split', type=str, default='testB', choices=['train', 'val', 'test', 'testA', 'testB'])
    parser.add_argument('--dataset', type=str, default='refcoco')
    parser.add_argument('--dataset_split', type=str, default='unc')

    args = parser.parse_args()

    refcoco_dir = Path(args.refcocoroot).resolve()
    refcocog_dir = refcoco_dir.joinpath(args.dataset)
    coco_dir = Path(args.cocoroot).resolve()
    refcoco_images_dir = coco_dir.joinpath('train2014')
    dataset_name = args.dataset

    out_dir = refcocog_dir.joinpath('features')
    if not out_dir.exists():
        out_dir.mkdir()

    dataset = RefCOCODataset(refcoco_dir, refcoco_images_dir, coco_dir, args.split, dataset=args.dataset,
                             dataset_split=args.dataset_split)
    print('# Images:', len(dataset))

    dataloader = DataLoader(dataset, batch_size=args.batchsize,
                            shuffle=False, collate_fn=collate_fn, num_workers=4)

    output_fname = out_dir.joinpath(f'{args.split}_ann.h5')
    print('features will be saved at', output_fname)

    # DIM是2048
    desc = f'{dataset_name}_given_boxes_ann_extract_({DIM})'

    extract(output_fname, dataloader, desc)
