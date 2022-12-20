from param import parse_args
from multitask_reg import Trainer
from multitask_reg_data import get_loader
from reg_data import RefCOCOGenerationFineTuneDataset
import json
from refcoco_utils import REFER
from copy import deepcopy

def test(dataset='refcoco+', split='testB', task='REG', epoch=0, lr=None, save_name=None):

    args = parse_args()
    args.gpu = 0
    args.train = 'val'
    args.num_beams = 5
    args.batch_size = 1
    args.dataset = dataset
    split_map = {'refcoco+': 'unc',
                 'refcoco': 'unc',
                 'refcocog': 'umd'}
    args.dataset_split = split_map[args.dataset]
    if lr:
        args.load = '/sharefs/baai-mrnd/yfl/codebase/Dialog/snap/'+args.dataset+'/' + task + '/' + lr + '/' + str(epoch)
    else:
        args.load = '/sharefs/baai-mrnd/yfl/codebase/Dialog/snap/'+args.dataset+'/' + task + '/' + str(epoch)
    args.rl_training = False
    args.use_rec = True
    args.experiment_name = '2022.11.09'
    args.dialog_training = True
    args.dialog_round = 5
    args.zero_shot_test = True
    args.last_round = True
    args.use_detector = True
    # args.refine = False
    args.test_threshold = 0.5
    args.dialog_sp_training = False
    # args.refine_load = '/raid_sda/yfl/codebase/VL-T5-REG/VL-T5/snap/' + args.dataset + '/' + \
    #                    'vlt5_ofa_mmi_dialog_sp_training_threshold_0.5_use_region_feature' + '/' + '5e-05' + '/' + "LAST"
    # args.bad_res_path = './REG_mmi_refcocog_vlt5_bad_sent_threshold_0.5_with_bbox.json'
    args.mode = 'val'
    args.distributed = False
    # print("===============")
    # print("test threshold is {}".format(args.test_threshold))
    # print("===============")


    # val_loader = get_loader(
    #     args,
    #     split=split, mode='val', batch_size=args.batch_size,
    #     distributed=args.distributed, gpu=args.gpu,
    #     workers=args.num_workers,
    #     topk=args.train_topk,
    # )
    refer = REFER(args.dataset, args.dataset_split, verbose=True)
    reg_dataset = RefCOCOGenerationFineTuneDataset(
        refer=refer,
        split=split,
        # raw_dataset=_dset,
        rank=args.gpu,
        topk=args.train_topk,
        verbose=True,
        args=args,
        mode='val',
        task='reg',
    )
    val_loader = get_loader(
        dataset=reg_dataset,
        split=split,
        mode='val',
        task='reg',
        batch_size=args.batch_size,
        workers=args.num_workers,
        distributed=args.distributed,
    )

    # val_loader = get_loader(
    #     args,
    #     refer=refer,
    #     split=split, mode='val', batch_size=args.batch_size,
    #     distributed=False, gpu=args.gpu,
    #     workers=4,
    #     topk=args.valid_topk,
    # )


    args_train = deepcopy(args)
    args_train.dialog_sp_training = True
    trainer = Trainer(args_train, train=False)

    # data = json.dumps(results)
    # with open('refcoco+_testB', 'w') as f:
    #     f.write(data)
    #
    # print(results)

    Score, results = trainer.evaluate(val_loader)

    # print(len(Score['CIDErs']))
    # if save:
    #     i = 0
    #     for item in results:
    #         item['cider'] = Score['CIDErs'][i]
    #         item['meteor'] = Score['METEORs'][i]
    #         i = i+1
    #
    #     data = json.dumps(results)
    #     if mmi:
    #         with open('result/'+args.dataset+'_'+split+'_mmi.json', 'w') as f:
    #             f.write(data)
    #     else:
    #         with open('result/'+args.dataset+'_'+split+'.json', 'w') as f:
    #             f.write(data)


if __name__ == '__main__':
    task = "vlt5_ofa_dialog_sp_training_one_model_with_new_badsents_plus_feature"
    test(dataset='refcoco+', split='testB', task=task, lr='5e-06', epoch="17")

    # task = 'vlt5_reg_new'
    # test(dataset='refcoco+', split='testB', task=task, lr='0.0003', epoch="LAST")
