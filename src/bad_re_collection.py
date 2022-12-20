from param import parse_args
from reg import Trainer
from reg_data import get_loader, RefCOCOGenerationFineTuneDataset
import json
from refcoco_utils import REFER
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import os
from reg import Critic

# 这个函数要把中间的一些变量都拉成输入，尽量写得以后要用方便一点；
def bad_re_collection(dataset='refcoco+', split='testB', save_path=None, ckpt_path=None):

    args = parse_args()
    args.gpu = 0
    args.train = 'val'
    args.num_beams = 5
    args.batch_size = 24
    args.dataset = dataset
    split_map = {'refcoco+': 'unc',
                 'refcoco': 'unc',
                 'refcocog': 'umd'}
    args.dataset_split = split_map[args.dataset]
    args.load = ckpt_path
    args.use_rec = True
    args.experiment_name = task
    args.mode = 'val'
    args.workers = 8
    args.test_threshold = 0.5

    # args.rl_training = True
    # args.dialog_training = True
    # args.dialog_round = 2
    # args.zero_shot_test = True
    # args.last_round = True



    verbose = True

    refer = REFER(args.dataset, args.dataset_split, verbose=verbose)

    # args mode设置为'val', 不是 'train' 所以每个ref_id只会有一个与之对应的sent
    dataset = RefCOCOGenerationFineTuneDataset(
        refer=refer,
        split=split,
        rank=args.gpu,
        verbose=verbose,
        args=args,
        mode=args.mode)

    # if args.distributed and args.mode == 'train':
    #     sampler = DistributedSampler(dataset)
    # else:
    sampler = None

    # if args.mode == 'train':
    #     # shuffle与sampler是不能共存的
    #     loader = DataLoader(
    #         dataset, batch_size=args.batch_size, shuffle=(sampler is None),
    #         num_workers=args.workers, pin_memory=True, sampler=sampler,
    #         collate_fn=dataset.collate_fn)
    # else:
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
        sampler=sampler,
        collate_fn=dataset.collate_fn,
        drop_last=False)


    trainer = Trainer(args, train=False)

    gen_kwargs = {}
    gen_kwargs["num_beams"] = args.num_beams
    gen_kwargs["max_length"] = args.gen_max_length
    gen_kwargs["num_return_sequences"] = 5
    total_bad_sents = []
    total_bad_ids = []
    total_bad_boxes = []
    total_good_sents = []
    total_good_ids = []
    total_good_boxes = []

    for i, batch in enumerate(tqdm(loader, ncols=120, desc="Interaction")):

        result = trainer.model.test_step_for_bad_re_collection(batch, trainer.critic, threshold=args.test_threshold, **gen_kwargs)
        ref_ids = [item for item in batch['ref_ids'] for i in range(5)]
        sents = result['sents']
        masks = result['masks']
        boxes = result['boxes']

        filter_sents = []
        filter_ref_ids = []
        filter_boxes = []
        good_sents = []
        good_sents_ref_ids = []
        good_boxes = []
        for idx, mask in enumerate(masks):
            if not mask:
                filter_sents.append(sents[idx])
                filter_ref_ids.append(ref_ids[idx])
                filter_boxes.append(boxes[idx])
            else:
                good_sents.append(sents[idx])
                good_sents_ref_ids.append(ref_ids[idx])
                good_boxes.append(boxes[idx])

        total_bad_sents += filter_sents
        total_bad_ids += filter_ref_ids
        total_bad_boxes += filter_boxes

        total_good_sents += good_sents
        total_good_ids += good_sents_ref_ids
        total_good_boxes += good_boxes

    assert len(total_bad_sents) == len(total_bad_ids), "Something is wrong!"

    # ref_ids_2_bad_sents = {}
    bad_case = []
    for ref_id, bad_sent, bad_box in zip(total_bad_ids, total_bad_sents, total_bad_boxes):
        # ref_ids_2_bad_sents[ref_id] = bad_sent
        bad_case.append(
            {
                'ref_id': ref_id,
                'sent': bad_sent,
                'bbox': bad_box,
            }
        )

    # ref_ids_2_good_sents = {}
    good_case = []
    for ref_id, good_sent, good_box in zip(total_good_ids, total_good_sents, total_good_boxes):
        # ref_ids_2_good_sents[ref_id] = good_sent
        good_case.append(
            {
                'ref_id': ref_id,
                'sent': good_sent,
                'bbox': good_box,
            }
        )

    # dir = './REG_refcoco+_bad_res.json'
    # os.makedirs(dir, exist_ok=True)
    # with open(save_path, 'w') as f:
    #     json.dump(ref_ids_2_bad_sents, f)
    # os.makedirs(save_path, exist_ok=True)
    dir = save_path.rsplit('/',1)[0]
    os.makedirs(dir, exist_ok=True)

    with open(save_path, 'w') as f:
        json.dump(bad_case, f)

    good_save_path = save_path.replace("bad", "good")
    with open(good_save_path, 'w') as f:
        json.dump(good_case, f)


class OfaRecTester:

    def __init__(self, args=None, verbose=False):

        self.args=args
        self.critic = Critic(args)
        self.refer = REFER(args.dataset, args.dataset_split, verbose=verbose)

        self.dataset = RefCOCOGenerationFineTuneDataset(
        refer=self.refer,
        split=args.split,
        # raw_dataset=_dset,
        rank=args.gpu,
        verbose=verbose,
        args=args,
        mode=args.mode)

        self.loader = DataLoader(
        self.dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
        sampler=None,
        collate_fn=self.dataset.collate_fn,
        drop_last=False)

        print('Length of dataloader is {}'.format(len(self.loader)))


    def test_on_gt(self):

        res = 0
        total_len = len(self.loader)
        for i, batch in enumerate(tqdm(self.loader, ncols=120, desc="Interaction")):

            sample_dict = {}
            sample_dict['image_ids'] = batch['image_ids']
            sample_dict['refBoxes'] = batch['refBoxes']
            sample_dict['sents'] = batch['sents']

            sample_rewards, sample_rewards_mask, ofa_results = self.critic.compute_score(sample_dict)

            for idx, mask in enumerate(sample_rewards_mask):
                if mask:
                    res += 1

        return res/(total_len*self.args.batch_size)


if __name__ == '__main__':

    # task = "REG_mmi"
    import argparse
    parser_outter = argparse.ArgumentParser()
    parser_outter.add_argument('--dataset', type=str, default="refcoco")
    args_outter = parser_outter.parse_args()
    dataset = args_outter.dataset
    
    if dataset == 'refcoco':
        ckpt_path = '/sharefs/baai-mrnd/yfl/codebase/Dialog/snap/refcoco/ddl_reg_baseline/5e-05/BEST'
    elif dataset == 'refcoco+':
        ckpt_path = '/sharefs/baai-mrnd/yfl/codebase/Dialog/snap/refcoco+/ddl_reg_baseline/5e-05/BEST'
    elif dataset == 'refcocog':
        ckpt_path = '/sharefs/baai-mrnd/yfl/codebase/Dialog/snap/refcocog/ddl_reg_baseline/5e-05/BEST'
    else:
        print("Fool! You input a wrong dataset name!")

    task = 'ddl_vlt5_reg_baseline'
    # split = "testA"
    # for split in splits:
    #     for dataset in datasets:
    #         test(dataset=dataset, split=split, task=task, epoch="BEST", save=True)
    for split in ['train']:
        if split == 'train':
            save_path = './new_generate_sent_set' + '/' + task + '/' + dataset + '/' + task + '_' + dataset + '_bad_sent_threshold_0.5_with_bbox.json'
        else:
            save_path = './new_generate_sent_set' + '/' + task + '/' + dataset + '/' + task + '_' + dataset + '_bad_sent_threshold_0.5_with_bbox_' + split +'.json'
        bad_re_collection(dataset=dataset, split=split, save_path=save_path, ckpt_path=ckpt_path)

    # args = parse_args()
    # args.gpu = 0
    # args.train = 'val'
    # args.num_beams = 5
    # args.batch_size = 32
    # args.mode = 'val'
    # args.workers = 8
    # verbose = (args.gpu == 0)
    #
    # args.dataset = 'refcocog'
    # split_map = {'refcoco+': 'unc',
    #              'refcoco': 'unc',
    #              'refcocog': 'umd'}
    # args.dataset_split = split_map[args.dataset]
    # args.split = 'train'
    #
    #
    # tester = OfaRecTester(args=args, verbose=verbose)
    # acc = tester.test_on_gt()
    # print("{}: REC acc on GT is {}".format(args.dataset, acc))




