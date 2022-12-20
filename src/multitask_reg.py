from fairseq import utils,tasks
from fairseq import checkpoint_utils
import os
import os.path as osp
import sys

if osp.join('/sharefs/baai-mrnd/yfl/codebase', 'OFA') not in sys.path:
    sys.path.insert(0, osp.join('/sharefs/baai-mrnd/yfl/codebase', 'OFA'))
# if osp.join('/home/yfl/', 'OFA') not in sys.path:
#     sys.path.insert(0, osp.join('/home/yfl/', 'OFA'))

from utils.eval_utils import eval_step
from tasks.mm_tasks.refcoco import RefcocoTask
from models.ofa import OFAModel
from data.mm_data.refcoco_dataset import collate
# Register refcoco task
tasks.register_task('refcoco', RefcocoTask)
from PIL import Image

import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import os.path as osp
import sys
import collections
from pathlib import Path
from packaging import version

import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import logging
import shutil
from pprint import pprint
import json

from param import parse_args

from reg_data import RefCOCOGenerationFineTuneDataset, DialogREGDataset
from vlt5_utils import load_state_dict, LossMeter, set_global_logging_level, count_parameters
import dist_utils
import wandb
from pprint import pformat
from trainer_base import TrainerBase2
from refcoco_utils import REFER
# from memory_profiler import profile

# 一直觉得这种写法看起来就不是很优雅~
if osp.join('/sharefs/baai-mrnd/yfl/codebase/Dialog/src', 'refer2/evaluation') not in sys.path:
    sys.path.insert(0, osp.join('/sharefs/baai-mrnd/yfl/codebase/Dialog/src', 'refer2/evaluation'))
# if osp.join('/home/yfl/catr/pyutils', 'refer2', 'evaluation') not in sys.path:
#     sys.path.insert(0, osp.join('/home/yfl/catr/pyutils', 'refer2', 'evaluation'))
from refEvaluation import RefEvaluation
import gc

from copy import deepcopy
from multitask_reg_data import MultiTaskLoader,get_loader

# from error1_data import Error_Task1_Dataset
# from error2_data import Error_Task2_Dataset

if osp.join('/sharefs/baai-mrnd/yfl/codebase/Dialog/src/feature_extraction') not in sys.path:
    sys.path.insert(0, osp.join('/sharefs/baai-mrnd/yfl/codebase/Dialog/src/feature_extraction'))

from detectron2_given_target_box_maxnms import doit, build_model
import datetime

set_global_logging_level(logging.ERROR, ["transformers"])

proj_dir = Path(__file__).resolve().parent.parent

_use_native_amp = False
_use_apex = False

# Check if Pytorch version >= 1.6 to switch between Native AMP and Apex
if version.parse(torch.__version__) < version.parse("1.6"):
    from transormers.file_utils import is_apex_available

    if is_apex_available():
        from apex import amp
    _use_apex = True
else:
    _use_native_amp = True
    from torch.cuda.amp import autocast

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class Critic():

    def __init__(self, args):

        self.args = args

        # turn on cuda if GPU is available
        self.use_cuda = torch.cuda.is_available()
        # use fp16 only when GPU is available
        self.use_fp16 = False

        # Load pretrained ckpt & config
        overrides = {"bpe_dir": "/sharefs/baai-mrnd/yfl/codebase/OFA/utils/BPE"}
        # overrides = {"bpe_dir": "/home/yfl/OFA/utils/BPE"}
        if self.args.dataset in ['refcoco', 'refcocog']:
            ofa_checkepoint_path = '/sharefs/baai-mrnd/yfl/codebase/OFA/checkpoints/' + \
                               self.args.dataset+'_base_best.pt'
            # ofa_checkepoint_path = '/home/yfl/OFA/checkpoints/' + \
            #                    self.args.dataset+'_base_best.pt'
        elif self.args.dataset =='refcoco+':
            ofa_checkepoint_path = '/sharefs/baai-mrnd/yfl/codebase/OFA/checkpoints/' + \
                                   'refcocoplus' + '_base_best.pt'
            # ofa_checkepoint_path = '/home/yfl/OFA/checkpoints/' + \
            #                        'refcocoplus' + '_base_best.pt'

        print("Load OFA model from:{}".format(ofa_checkepoint_path))
        self.models, self.cfg, self.task = checkpoint_utils.load_model_ensemble_and_task(
            utils.split_paths(ofa_checkepoint_path),
            arg_overrides=overrides
        )

        self.cfg.common.seed = 7
        self.cfg.generation.beam = 5
        self.cfg.generation.min_len = 4
        self.cfg.generation.max_len_a = 0
        self.cfg.generation.max_len_b = 4
        self.cfg.generation.no_repeat_ngram_size = 3
        self.patch_image_size = self.cfg.task.patch_image_size

        # Fix seed for stochastic decoding
        if self.cfg.common.seed is not None and not self.cfg.generation.no_seed_provided:
            np.random.seed(self.cfg.common.seed)
            utils.set_torch_seed(self.cfg.common.seed)

        # Move models to GPU 这里到到后面分布式可能需要指定move到哪个GPU上
        for model in self.models:
            model.eval()
            if self.use_fp16:
                model.half()
            if self.use_cuda and not self.cfg.distributed_training.pipeline_model_parallel:
                model.cuda()
            model.prepare_for_inference_(self.cfg)

        # Initialize generator
        self.generator = self.task.build_generator(self.models, self.cfg.generation)

        # Image transform
        from torchvision import transforms

        self.patch_resize_transform = transforms.Compose([
            lambda image: image.convert("RGB"),
            transforms.Resize((self.cfg.task.patch_image_size, self.cfg.task.patch_image_size),
                              interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            # transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        # Text preprocess
        self.bos_item = torch.LongTensor([self.task.src_dict.bos()])
        self.eos_item = torch.LongTensor([self.task.src_dict.eos()])
        self.pad_idx = self.task.src_dict.pad()
        self.eos_idx = self.task.src_dict.eos()  # 2

    def encode_text(self, text, length=None, append_bos=False, append_eos=False):
        s = self.task.tgt_dict.encode_line(
            line=self.task.bpe.encode(text.lower()),
            add_if_not_exist=False,
            append_eos=False
        ).long()
        if length is not None:
            s = s[:length]
        if append_bos:
            s = torch.cat([self.bos_item, s])
        if append_eos:
            s = torch.cat([s, self.eos_item])
        return s

    # Construct input for refcoco task

    def construct_sample(self, image: Image, text: str, img_id: int, region_coords):
        w, h = image.size
        w_resize_ratio = torch.tensor(self.patch_image_size / w)
        h_resize_ratio = torch.tensor(self.patch_image_size / h)
        patch_image = self.patch_resize_transform(image)
        patch_mask = torch.tensor([True])
        source = self.encode_text(' which region does the text " {} " describe?'.format(text), append_bos=True,
                                  append_eos=True).unsqueeze(0)
        sample = {
            "id": str(img_id),
            "source": source[0],
            "patch_image": patch_image,
            "patch_mask": patch_mask,
            "w_resize_ratio": w_resize_ratio,
            "h_resize_ratio": h_resize_ratio,
            "region_coord": torch.tensor(region_coords),
        }
        return sample

    # Function to turn FP32 to FP16
    def apply_half(self, t):
        if t.dtype is torch.float32:
            return t.to(dtype=torch.half)
        return t

    def compute_score(self, dict, ofa_test=False, threshold=0.5):
        image_ids = dict['image_ids']
        refBoxes = dict['refBoxes']
        sents = dict['sents']

        assert len(image_ids) == len(refBoxes) == len(sents), 'Length does not match!'

        samples = []

        for image_id, refBox, sent in zip(image_ids, refBoxes, sents):
            img_path = '/sharefs/baai-mrnd/yfl/database/train2014/train2014/COCO_train2014_' + str(image_id).zfill(12) + '.jpg'
            # img_path = '/home/yfl/datasets/train2014/COCO_train2014_' + str(image_id).zfill(12) + '.jpg'
            image = Image.open(img_path)
            sample = self.construct_sample(image, sent, image_id, refBox)
            samples.append(deepcopy(sample))
        ofa_batch = collate(samples, self.pad_idx, self.eos_idx)
        ofa_batch = utils.move_to_cuda(ofa_batch) if self.use_cuda else ofa_batch
        ofa_batch = utils.apply_to_sample(self.apply_half, ofa_batch) if self.use_fp16 else ofa_batch
        with torch.no_grad():
            if ofa_test:
                result, scores = eval_step(self.task, self.generator, self.models, ofa_batch, ofa_test=ofa_test, threshold=threshold)
            else:
                result, scores, scores_mask = eval_step(self.task, self.generator, self.models, ofa_batch, ofa_test=ofa_test, threshold=threshold)

        if ofa_test:
            return scores, [], result
        else:
            return scores, scores_mask, result


class Trainer(TrainerBase2):
    # @profile
    def __init__(self, args, train_loader=None, val_loader=None, test_loaderA=None, test_loaderB=None, train=True, refer=None, task_name_list=[]):
        super().__init__(
            args,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loaderA=test_loaderA,
            test_loaderB=test_loaderB,
            train=train)

        self.wandb_initialized = False

        from multitask_reg_model import VLT5REG, VLBartREG

        if refer != None:
            self.refer = refer
        else:
            self.refer = REFER(args.dataset, args.dataset_split, verbose=True)

        model_kwargs = {}
        if 't5' in args.backbone:
            self.model_class = VLT5REG
        elif 'bart' in args.backbone:
            self.model_class = VLBartREG

        self.config = self.create_config()
        self.tokenizer = self.create_tokenizer()
        if 'bart' in self.args.tokenizer:
            num_added_toks = 0
            if self.config.use_vis_order_embedding:
                additional_special_tokens = [f'<extra_id_{i}>' for i in range(100 - 1, -1, -1)] + \
                                            [f'<vis_extra_id_{i}>' for i in range(100 - 1, -1, -1)]
                special_tokens_dict = {'additional_special_tokens': additional_special_tokens}
                num_added_toks = self.tokenizer.add_special_tokens(special_tokens_dict)

                self.config.default_obj_order_ids = self.tokenizer.convert_tokens_to_ids(
                    [f'<vis_extra_id_{i}>' for i in range(100)])

        self.model = self.create_model(self.model_class, self.config, **model_kwargs)
        if self.args.refine and not (self.args.mode == 'train'):
            self.refine_model = self.create_model(self.model_class, self.config, **model_kwargs)


        if self.verbose:
            print("The total parameter required calculate "
              "gradient is:{}".format(count_parameters(self.model)))

        if 't5' in self.args.tokenizer:
            self.model.resize_token_embeddings(self.tokenizer.vocab_size)
            if self.args.refine and not (self.args.mode == 'train'):
                self.refine_model.resize_token_embeddings(self.tokenizer.vocab_size)
        elif 'bart' in self.args.tokenizer:
            self.model.resize_token_embeddings(self.model.model.shared.num_embeddings + num_added_toks)
            if self.args.refine and not (self.args.mode == 'train'):
                self.refine_model.resize_token_embeddings(self.model.model.shared.num_embeddings + num_added_toks)
        self.model.tokenizer = self.tokenizer
        if self.args.refine and not (self.args.mode == 'train'):
            self.refine_model.tokenizer = self.tokenizer

        # Load Checkpoint
        self.start_epoch = None
        if args.load is not None:
            ckpt_path = args.load + '.pth'
            self.load_checkpoint(self.model, ckpt_path)

        if self.args.refine_load is not None and not (self.args.mode == 'train'):
            refine_ckpt_path = self.args.refine_load + '.pth'
            self.load_checkpoint(self.refine_model, refine_ckpt_path)

        # This suppose not to be use in refine model i guess.
        if self.args.from_scratch:
            self.init_weights()

        # GPU Options
        print(f'Model Launching at GPU {self.args.gpu}')
        if self.verbose:
            from time import time
            start = time()
        self.model = self.model.to(args.gpu)
        if self.args.refine and not (self.args.mode == 'train'):
            self.refine_model = self.refine_model.to(args.gpu)

        # Optimizer
        if train:
            self.optim, self.lr_scheduler = self.create_optimizer_and_scheduler()

            # _use_apex 设置为false，所以其实是没有用过apex的
            if self.args.fp16 and _use_native_amp:
                self.scaler = torch.cuda.amp.GradScaler()
            elif _use_apex:
                self.model, self.optim = amp.initialize(
                    self.model, self.optim, opt_level='O1', verbosity=self.verbose)

        if args.multiGPU:
            if args.distributed:
                self.model = DDP(self.model, device_ids=[args.gpu],
                                 find_unused_parameters=True
                                 )
                if self.args.refine and not (self.args.mode == 'train'):
                    self.refine_model = DDP(self.refine_model, device_ids=[args.gpu],
                                            find_unused_parameters=True)
        if self.verbose:
            print(f'It took {time() - start:.1f}s')

        if self.args.use_rec:
            self.critic = Critic(self.args)

        if self.args.use_detector:
            self.detector = build_model()

        if self.args.hyperparameter_search:
            self.args.output = self.args.output + '/' + str(self.args.lr)

        self.task_name_list = task_name_list

    # @profile
    def train(self):
        if self.verbose:
            # for task_name in self.task_name_list:
            #     exec("{} = LossMeter()".format(task_name+"_loss_meter"))
            reg_loss_meter = LossMeter()
            dialog_reg_loss_meter = LossMeter()
            task2_loss_meter = LossMeter()
            task1_loss_meter = LossMeter()
            best_valid = 0.
            best_epoch = 0

            if not self.wandb_initialized and not self.args.debug:

                if 't5' in self.args.backbone:
                    # 这个地方project_name可以设置为实验的名称
                    project_name = "REG_" + args.experiment_name
                elif 'bart' in self.args.backbone:
                    project_name = "VLBart_REG"

                wandb.init(project=project_name)
                wandb.run.name = self.args.run_name
                wandb.config.update(self.args)
                # wandb.watch(self.model)

                # /workspace/yfl/codebase/VL-T5/VL-T5/src
                src_dir = Path(__file__).resolve().parent
                base_path = str(src_dir.parent)
                src_dir = str(src_dir)
                wandb.save(os.path.join(src_dir + "/*.py"), base_path=base_path)

                self.wandb_initialized = True

        if self.args.distributed:
            dist.barrier()

        global_step = 0
        epochs = self.args.epochs

        for epoch in range(epochs):

            if self.start_epoch is not None:
                epoch += self.start_epoch
            self.model.train()
            if self.args.distributed:
                self.train_loader.set_epoch(epoch)
            if self.verbose:
                pbar = tqdm(total=len(self.train_loader), ncols=185)

            # epoch_results = {
            #     'reg_loss': 0.,
            #     'dialog_reg_loss': 0.,
            #     'task2_loss': 0.,
            #     'task1_loss': 0.,
            # }
            # task_counter = {
            #     'reg': 0,
            #     'dialog_reg': 0,
            #     'task2': 0,
            #     'task1': 0,
            # }

            epoch_results = {}
            task_counter = {}
            for task_name in self.task_name_list:
                epoch_results[task_name+'_loss'] = 0.
                task_counter[task_name] = 0

            if self.verbose:
                print(task_counter)

            for step_i, batch in enumerate(self.train_loader):
                task = batch['task']
                task_counter[task] += 1

                if self.args.fp16 and _use_native_amp:
                    with autocast():
                        if self.args.distributed:
                            results = self.model.module.train_step(batch, self.args.use_mmi,
                                                            epoch=epoch, lama=self.args.lama, margin=self.args.margin)
                        else:
                            results = self.model.train_step(batch, self.args.use_mmi,
                                                            epoch=epoch, lama=self.args.lama, margin=self.args.margin)
                else:
                    if self.args.distributed:
                        results = self.model.module.train_step(batch, self.args.use_mmi, epoch=epoch)
                    else:
                        results = self.model.train_step(batch, self.args.use_mmi, epoch=epoch)

                loss = results['loss']

                if self.args.fp16 and _use_native_amp:
                    self.scaler.scale(loss).backward()
                elif self.args.fp16 and _use_apex:
                    with amp.scale_loss(loss, self.optim) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                loss = loss.detach()

                # Update Parameters
                if self.args.clip_grad_norm > 0:
                    if self.args.fp16 and _use_native_amp:
                        self.scaler.unscale_(self.optim)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.args.clip_grad_norm)
                    elif self.args.fp16 and _use_apex:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(
                            self.optim), self.args.clip_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.args.clip_grad_norm)

                update = True
                if self.args.gradient_accumulation_steps > 1:
                    if step_i == 0:
                        update = False
                    elif step_i % self.args.gradient_accumulation_steps == 0 or step_i == len(self.train_loader) - 1:
                        update = True
                    else:
                        update = False

                if update:
                    if self.args.fp16 and _use_native_amp:
                        self.scaler.step(self.optim)
                        self.scaler.update()
                    else:
                        self.optim.step()

                    if self.lr_scheduler:
                        self.lr_scheduler.step()
                    # self.model.zero_grad()
                    for param in self.model.parameters():
                        param.grad = None
                    global_step += 1

                # for k, v in results.items():
                #     if k in epoch_results:
                #         epoch_results[k] += v.item()

                if self.lr_scheduler:
                    if version.parse(torch.__version__) >= version.parse("1.4"):
                        lr = self.lr_scheduler.get_last_lr()[0]
                    else:
                        lr = self.lr_scheduler.get_lr()[0]
                else:
                    try:
                        lr = self.optim.get_lr()[0]
                    except AttributeError:
                        lr = self.args.lr

                if self.verbose:
                    if task == 'reg':
                        reg_loss_meter.update(loss.item())
                        epoch_results['reg_loss'] += loss.item()
                    if task == 'dialog_reg':
                        dialog_reg_loss_meter.update(loss.item())
                        epoch_results['dialog_reg_loss'] += loss.item()
                    if task == 'task2':
                        task2_loss_meter.update(loss.item())
                        epoch_results['task2_loss'] += loss.item()
                    if task == 'task1':
                        task1_loss_meter.update(loss.item())
                        epoch_results['task1_loss'] += loss.item()
                    # for task_name in self.task_name_list:
                    #     # exec(
                    #     # """
                    #     # if task == {}:
                    #     #     {}_loss_meter.update(loss.item())
                    #     #     epoch_results[{}_loss] += loss.item()
                    #     # """.format(task_name, task_name, task_name)
                    #     # )
                    #     if task == task_name:
                    #         exec("{}_loss_meter.update(loss.item())".format(task_name))
                    #         epoch_results[task_name+'_loss'] += loss.item()

                    desc_str = f'Epoch {epoch} | LR {lr:.6f} | Steps {global_step}'
                    if len(reg_loss_meter) > 0:
                        desc_str += f' | Reg Loss {reg_loss_meter.val:4f}'
                    if len(dialog_reg_loss_meter) > 0:
                        desc_str += f' | Dialog Loss {dialog_reg_loss_meter.val:4f}'
                    if len(task2_loss_meter) > 0:
                        desc_str += f' | Task2 Loss {task2_loss_meter.val:4f}'
                    if len(task1_loss_meter) > 0:
                        desc_str += f' | Task1 Loss {task1_loss_meter.val:4f}'
                    # for task_name in self.task_name_list:
                    #     # exec(
                    #     # """
                    #     # if len({}_loss_meter) > 0:
                    #     #     desc_str += f' | {} Loss {{}_loss_meter.val:4f}'
                    #     # """.format(task_name, task_name, task_name)
                    #     # )
                    #     if exec("len({}_loss_meter) > 0".format(task_name)):
                    #         desc_str += ' | {} Loss {:4f}'.format(task_name, exec('{_loss_meter.val}'.format(task_name)))

                    pbar.set_description(desc_str)
                    pbar.update(1)

                # for debug need
                if self.args.debug:
                    if global_step == 15:
                        break

            if self.args.distributed:
                dist.barrier()

            if self.verbose:
                pbar.close()

            if self.verbose:
                wandb_log_dict = {}
                for task_name in self.task_name_list:
                    if task_counter[task_name]:
                        wandb_log_dict['Train/' + task_name + '_Loss'] = epoch_results[task_name + '_loss'] / task_counter[task_name]
                        # wandb_log_dict['Train/Dialog_Loss'] = epoch_results['dialog_reg_loss'] / task_counter['dialog_reg']
                        # wandb_log_dict['Train/Task2_Loss'] = epoch_results['task2_loss'] / task_counter['task2']
                    # wandb_log_dict['Train/Task1_Loss'] = epoch_results['task1_loss'] / task_counter['task1']
                # for task_name in self.task_name_list:
                #     exec(
                #         """
                #         wandb_log_dict['Train/{}_Loss'] = epoch_results['{}_loss'] / task_counter['{}']
                #         """.format(task_name, task_name, task_name)
                #     )
            
            if not self.args.debug and not self.args.no_evaluate and epoch%5==0:
                valid_results, valid_pred = self.evaluate(self.val_loader)
                if self.verbose:
                    valid_score = valid_results['CIDEr']
                    if valid_score > best_valid or epoch == 0:
                        best_valid = valid_score
                        best_epoch = epoch
                        self.save("BEST")
                    log_str = ''
                    valid_results_for_pprint = {'CIDEr': valid_results['CIDEr'],
                                                'METEOR': valid_results['METEOR']}
                    log_str += pformat(valid_results_for_pprint)
                    log_str += "\nEpoch %d: Valid CIDEr %0.4f" % (epoch, valid_score)
                    log_str += "\nEpoch %d: Best CIDEr %0.4f\n" % (best_epoch, best_valid)
                    print(log_str)
                    for score_name, score in valid_results.items():
                        if not (type(score) is np.ndarray):
                            wandb_log_dict[f'Valid/{score_name}'] = score

                    wandb_log_dict[f'Valid/best_epoch'] = best_epoch
                    wandb_log_dict['Train/epoch'] = epoch
                if self.args.dataset == 'refcocog':
                    test_results_during_train, _ = self.evaluate(self.test_loaderA)
                    if self.verbose:
                        for score_name, score in test_results_during_train.items():
                            if not (type(score) is np.ndarray):
                                wandb_log_dict[f'Train/Test_{score_name}'] = score
                else:
                    test_results_during_trainA, _ = self.evaluate(self.test_loaderA)
                    test_results_during_trainB, _ = self.evaluate(self.test_loaderB)
                    if self.verbose:
                        for score_name, score in test_results_during_trainA.items():
                            if not (type(score) is np.ndarray):
                                wandb_log_dict[f'Train/TestA_{score_name}'] = score

                        for score_name, score in test_results_during_trainB.items():
                            if not (type(score) is np.ndarray):
                                wandb_log_dict[f'Train/TestB_{score_name}'] = score

            if self.verbose:
                wandb.log(wandb_log_dict)
                self.save(str(epoch))

            if self.args.distributed:
                dist.barrier()

        # 这里只会保存主显卡的参数权重
        if self.verbose:
            self.save("LAST")

        # Test Set
        if not self.args.debug:
            wandb_log_dict = {}
            best_path = os.path.join(self.args.output, '19')
            self.load(best_path)

        # if self.verbose:
        #     wandb.save(best_path, base_path=self.args.output)
        #     print(f'\nUploaded checkpoint {best_epoch}', best_path)
        # 如果训练过程中每一个都epoch都evaluate过一次，那么最后也就没有必要进行下面的步骤了；
            if self.args.dataset == 'refcocog':
                test_results, test_pred = self.evaluate(self.test_loaderA, save_name='best_epoch.json')
                val_results, val_pred = self.evaluate(self.val_loader, save_name='best_epoch.json')
                if self.verbose:
                    # 这个print还不如人家 wandb 设置得好
                    # log_str = 'Test set results\n'
                    # test_results_for_pprint = {'CIDEr': test_results['CIDEr'],
                    #                            'METEOR': test_results['METEOR']}
                    # log_str += pformat(test_results_for_pprint)
                    # print(log_str)
                    for score_name, score in test_results.items():
                        if not (type(score) is np.ndarray):
                            wandb_log_dict[f'LAST/Test_{score_name}'] = score

                    for score_name, score in val_results.items():
                        if not (type(score) is np.ndarray):
                            wandb_log_dict[f'LAST/Val_{score_name}'] = score
            else:
                test_resultsA, pred_testA = self.evaluate(self.test_loaderA, save_name='best_epoch.json')
                test_resultsB, pred_testB = self.evaluate(self.test_loaderB, save_name='best_epoch.json')
                if self.verbose:
                    log_str = 'Test set results\n'
                    test_results_for_pprintA = {'CIDEr': test_resultsA['CIDEr'],
                                               'METEOR': test_resultsA['METEOR']}
                    test_results_for_pprintB = {'CIDEr': test_resultsB['CIDEr'],
                                                'METEOR': test_resultsB['METEOR']}
                    log_str += pformat(test_results_for_pprintA)
                    log_str += pformat(test_results_for_pprintB)

                    print(log_str)

                    for score_name, score in test_resultsA.items():
                        if not (type(score) is np.ndarray):
                            wandb_log_dict[f'LAST/TestA_{score_name}'] = score

                    for score_name, score in test_resultsB.items():
                        if not (type(score) is np.ndarray):
                            wandb_log_dict[f'LAST/TestB_{score_name}'] = score
        if self.verbose:
            wandb.log(wandb_log_dict)

        if self.args.distributed:
            dist.barrier()

    def predict(self, loader):
        """
        Predict the answers to questions in a data split.
        :param eval_tuple: The data tuple to be evaluated.
        :param dump: The path of saved file to dump results.
        :return: A dict of question_id to answer.
        """
        self.model.eval()
        if self.args.refine:
            self.refine_model.eval()

        with torch.no_grad():

            predictions = []
            # targets = []

            gen_kwargs = {}
            gen_kwargs['num_beams'] = self.args.num_beams
            # gen_kwargs['num_beams'] = 5
            gen_kwargs['max_length'] = self.args.gen_max_length

            for i, batch in enumerate(tqdm(loader, ncols=120, desc="Prediction", disable=not self.verbose)):

                if self.args.distributed:
                    if self.args.zero_shot_test:
                        if self.args.refine:
                            results = self.model.module.new_test_step(
                                batch,
                                self.critic,
                                self.refine_model,
                                self.args.dialog_round,
                                self.args.last_round,
                                self.args.test_threshold,
                                self.detector,
                                **gen_kwargs)
                        else:
                            results = self.model.module.new_task2_test_step(
                                batch,
                                self.critic,
                                None,
                                self.args.dialog_round,
                                self.args.last_round,
                                self.args.test_threshold,
                                self.detector,
                                **gen_kwargs)
                    else:
                        results = self.model.module.test_step(
                            batch,
                            self.args.dialog_training,
                            self.critic,
                            self.args.dialog_round,
                            self.args.last_round,
                            **gen_kwargs)
                else:
                    if self.args.zero_shot_test:
                        if self.args.refine:
                            results = self.model.new_task2_test_step(
                                batch,
                                self.critic,
                                self.refine_model,
                                self.args.dialog_round,
                                self.args.last_round,
                                self.args.test_threshold,
                                self.detector,
                                **gen_kwargs)
                        else:
                            results = self.model.new_task2_test_step(
                                batch,
                                self.critic,
                                None,
                                self.args.dialog_round,
                                self.args.last_round,
                                self.args.test_threshold,
                                self.detector,
                                **gen_kwargs)
                    else:
                        results = self.model.test_step(
                            batch,
                            self.args.dialog_training,
                            self.critic,
                            self.args.dialog_round,
                            self.args.last_round,
                            **gen_kwargs)

                predictions.extend(results)

            # 需要把分到各张卡的predict结果收集起来
            if self.args.distributed:
                dist.barrier()

                dist_results = dist_utils.all_gather(predictions)
                predictions = []
                for result in dist_results:
                    predictions.extend(result)

            return predictions

    def evaluate(self, loader, save_name=None):

        """
        evaluate 输入： loader
        输出： 第一、二轮的cider,meteor,ofa acc值， 并可以决定是否要保存结果；
        """
        preds = self.predict(loader)

        generate_res = {}

        for pred in preds:
            generate_res[pred['ref_id']] = pred['sent']

        # In distributed mode, the distributed sampler will complement the batch size to the
        # same in every sing gpu, when the drop_last is default False. So there may be some redundant.
        if not self.args.distributed:
            assert len(preds) == len(generate_res), 'The length does not match, prediction is{}, and generate_res is{}'.format(len(preds),
                                                                                                      len(generate_res))
        ofa_acc_sum = torch.FloatTensor([0]).cuda()
        ofa_acc_cnt = torch.FloatTensor([0]).cuda()
        ofa_score_recoder = {}
        with torch.no_grad():

            for i, batch in enumerate(tqdm(loader, ncols=120, desc="ofa_evaluate", disable=not self.verbose)):
                sample_dict = {}
                sample_dict['image_ids'] = batch['image_ids']  # ids is a list of int
                sample_dict['refBoxes'] = batch['refBoxes']
                ref_ids = batch['ref_ids']

                sample_sents = []
                for ref_id in ref_ids:
                    sent = generate_res[ref_id]
                    sample_sents.append(sent)

                sample_dict['sents'] = sample_sents  # a list of sent

                if self.args.use_rec:
                    scores, _, result = self.critic.compute_score(sample_dict, ofa_test=True, threshold=0.5)
                    ofa_acc_sum += sum(scores) if scores is not None else 0
                    ofa_acc_cnt += len(scores) if scores is not None else 0

                for score, ref_id in zip(scores, ref_ids):
                    ofa_score_recoder[ref_id] = score.item()

            if self.args.distributed:
                dist.barrier()

                ofa_acc_sum_for_dist = dist_utils.all_gather(ofa_acc_sum.item())
                ofa_acc_cnt_for_dist = dist_utils.all_gather(ofa_acc_cnt.item())
                ofa_score_recoder_for_dist = dist_utils.all_gather(ofa_score_recoder)
                ofa_score_recoder_gather = {}
                for dict in ofa_score_recoder_for_dist:
                    ofa_score_recoder_gather.update(dict)
                ofa_acc_sum_gather = sum(ofa_acc_sum_for_dist)
                ofa_acc_cnt_gather = sum(ofa_acc_cnt_for_dist)
            else:
                ofa_acc_sum_gather = ofa_acc_sum.item()
                ofa_acc_cnt_gather = ofa_acc_cnt.item()
                ofa_score_recoder_gather = ofa_score_recoder

        result = {}
        if self.verbose:
            print('# predictions:', len(preds))
            evaluator = RefEvaluation(self.refer, preds)
            CIDEr_sc, CIDEr_scs, METEOR_sc, METEOR_scs = evaluator.evaluate()
            result['CIDEr'] = CIDEr_sc
            result['CIDErs'] = CIDEr_scs
            result['METEOR'] = METEOR_sc
            result['METEORs'] = METEOR_scs
            if self.args.use_rec:
                result['OFA_Acc'] = ofa_acc_sum_gather / ofa_acc_cnt_gather

            print("CIDEr score:{}".format(result['CIDEr']))
            print("METEOR score:{}".format(result['METEOR']))
            if self.args.use_rec:
                print("OFA_Acc_Score:{}".format(ofa_acc_sum.item() / ofa_acc_cnt.item()))
                print("OFA_Acc_Score_Sum:{}".format(ofa_acc_sum.item()))
                print("OFA_Acc_Score_Cnt:{}".format(ofa_acc_cnt.item()))

            # 这块先暂时这样吧，不知道要用什么来决定是否要测多轮的；
            if self.args.dialog_sp_training:
                first_round_pred = []
                cnt = 0
                OFA_Acc_first_round = 0
                for i, item in enumerate(preds):
                    first_round_pred.append(
                        {
                            'ref_id': item['ref_id'],
                            'sent': item['dialog_generate_sent'][0],
                        }
                    )
                    # 这里保存下来的ofa iou值需要稍微转换一下才是acc值
                    if item['dialog_generate_sent_ofa_iou'][0] >= 0.5:
                        OFA_Acc_first_round += 1
                    cnt += 1
                result['OFA_Acc_first_round'] = OFA_Acc_first_round/cnt
                evaluator_first_round = RefEvaluation(self.refer, first_round_pred)
                CIDEr_sc_first_round, CIDEr_scs_first_round, METEOR_sc_first_round, METEOR_scs_first_round = evaluator_first_round.evaluate()
                result['CIDEr_first_round'] = CIDEr_sc_first_round
                result['CIDErs_first_round'] = CIDEr_scs_first_round
                result['METEOR_first_round'] = METEOR_sc_first_round
                result['METEORs_first_round'] = METEOR_scs_first_round
                print("OFA_Acc_first_round_score:{}".format(result['OFA_Acc_first_round']))
                print("CIDEr_first_round_score:{}".format(result['CIDEr_first_round']))
                print("METEOR_first_round_score:{}".format(result['METEOR_first_round']))
            if save_name is not None:
                for i, item in enumerate(preds):
                    item['cider'] = result['CIDErs'][i]
                    item['meteor'] = result['METEORs'][i]
                    item['last_round_ofa_iou'] = ofa_score_recoder_gather[item['ref_id']]

                    if self.args.dialog_sp_training:
                        item['cider_first_round'] = result['CIDErs_first_round'][i]
                        item['meteor_first_round'] = result['METEORs_first_round'][i]
                data = json.dumps(preds)
                dir = 'generate_result/' + str(self.args.dataset) + '/' + str(self.args.experiment_name) + '/' + str(loader.split_name) + '/'
                os.makedirs(dir, exist_ok=True)
                print("Evaluation Result save at {}!!!!".format(dir+save_name))
                with open(dir+save_name, 'w') as f:
                    f.write(data)

        return result, preds


def main_worker(gpu, args):
    # GPU is assigned
    args.gpu = gpu
    args.rank = gpu
    print(f'Process Launching at GPU {gpu}')

    if args.distributed:
        torch.cuda.set_device(args.gpu)
        dist.init_process_group(backend='nccl', timeout=datetime.timedelta(seconds=5400))
    if gpu == 0:
        verbose = True
    else:
        verbose = False
    refer = REFER(args.dataset, args.dataset_split, verbose=verbose)

    dialog_dataset = DialogREGDataset(
        refer=refer,
        split=args.train,
        # raw_dataset=_dset,
        rank=gpu,
        topk=args.train_topk,
        verbose=verbose,
        args=args,
        mode='train',
        task='dialog_reg',
    )

    dialog_train_loader = get_loader(
        dataset=dialog_dataset,
        split=args.train,
        mode='train',
        task='dialog_reg',
        batch_size=args.batch_size,
        workers=args.num_workers,
        distributed=args.distributed,
    )

    reg_args = deepcopy(args)
    reg_args.dialog_sp_training = False
    reg_dataset = RefCOCOGenerationFineTuneDataset(
        refer=refer,
        split=reg_args.train,
        # raw_dataset=_dset,
        rank=gpu,
        topk=reg_args.train_topk,
        verbose=verbose,
        args=reg_args,
        mode='train',
        task='reg',
    )
    reg_train_loader = get_loader(
        dataset=reg_dataset,
        split=args.train,
        mode='train',
        task='reg',
        batch_size=args.batch_size,
        workers=args.num_workers,
        distributed=args.distributed,
    )

    # task1_dataset = Error_Task1_Dataset(
    #     refer=refer,
    #     split=reg_args.train,
    #     # raw_dataset=_dset,
    #     rank=gpu,
    #     topk=reg_args.train_topk,
    #     verbose=verbose,
    #     args=reg_args,
    #     mode='train',
    #     task='task1',
    # )
    # task1_train_loader = get_loader(
    #     dataset=task1_dataset,
    #     split=args.train,
    #     mode='train',
    #     task='task1',
    #     batch_size=reg_args.batch_size,
    #     workers=reg_args.num_workers,
    #     distributed=reg_args.distributed,
    # )

    # task2_dataset = Error_Task2_Dataset(
    #     refer=refer,
    #     split=reg_args.train,
    #     # raw_dataset=_dset,
    #     rank=gpu,
    #     topk=reg_args.train_topk,
    #     verbose=verbose,
    #     args=reg_args,
    #     mode='train',
    #     task='task2',
    # )
    # task2_train_loader = get_loader(
    #     dataset=task2_dataset,
    #     split=args.train,
    #     mode='train',
    #     task='task2',
    #     batch_size=reg_args.batch_size,
    #     workers=reg_args.num_workers,
    #     distributed=reg_args.distributed,
    # )

    train_loaders = []
    train_loaders.append(reg_train_loader)
    train_loaders.append(dialog_train_loader)
    # train_loaders.append(task1_train_loader)
    # train_loaders.append(task2_train_loader)

    print(f'Building train loader at GPU {gpu}')
    train_loader = MultiTaskLoader(train_loaders, verbose=(gpu == 0))

    if args.valid_batch_size is not None:
        valid_batch_size = args.valid_batch_size
    else:
        valid_batch_size = args.batch_size
    print(f'Building val loader at GPU {gpu}')

    val_dataset = RefCOCOGenerationFineTuneDataset(
        refer=refer,
        split=args.valid,
        # raw_dataset=_dset,
        rank=args.gpu,
        topk=args.valid_topk,
        verbose=verbose,
        args=args,
        mode='val',
        task='reg',
    )

    val_loader = get_loader(
        dataset=val_dataset,
        split=args.valid,
        mode='val',
        task='reg',
        batch_size=valid_batch_size,
        workers=args.num_workers,
        distributed=args.distributed,
    )

    print('# len val loader:', len(val_loader))

    print(f'Building test loader at GPU {gpu}')
    if args.dataset == 'refcocog':
        test_dataset = RefCOCOGenerationFineTuneDataset(
            refer=refer,
            split=args.test,
            # raw_dataset=_dset,
            rank=args.gpu,
            topk=args.valid_topk,
            verbose=verbose,
            args=args,
            mode='val',
            task='reg',
        )

        test_loader = get_loader(
            dataset=test_dataset,
            split=args.test,
            mode='val',
            task='reg',
            batch_size=valid_batch_size,
            workers=args.num_workers,
            distributed=args.distributed,
        )
        task_name_list = ['reg', 'dialog_reg', 'task2']
        trainer = Trainer(args, train_loader, val_loader, test_loader, train=True, refer=refer, task_name_list=task_name_list)
    else:
        testA_dataset = RefCOCOGenerationFineTuneDataset(
            refer=refer,
            split='testA',
            # raw_dataset=_dset,
            rank=args.gpu,
            topk=args.valid_topk,
            verbose=verbose,
            args=args,
            mode='val',
            task='reg',
        )

        test_loaderA = get_loader(
            dataset=testA_dataset,
            split='testA',
            mode='val',
            task='reg',
            batch_size=valid_batch_size,
            workers=args.num_workers,
            distributed=args.distributed,
        )

        testB_dataset = RefCOCOGenerationFineTuneDataset(
            refer=refer,
            split='testB',
            # raw_dataset=_dset,
            rank=args.gpu,
            topk=args.valid_topk,
            verbose=verbose,
            args=args,
            mode='val',
            task='reg',
        )

        test_loaderB = get_loader(
            dataset=testB_dataset,
            split='testB',
            mode='val',
            task='reg',
            batch_size=valid_batch_size,
            workers=args.num_workers,
            distributed=args.distributed,
        )

        # task_name_list = ['reg', 'dialog_reg', 'task1', 'task2']
        # task_name_list = ['reg', 'dialog_reg', 'task1']
        task_name_list = ['reg', 'dialog_reg']
        trainer = Trainer(args, train_loader, val_loader, test_loaderA, test_loaderB, train=True, refer=refer, task_name_list=task_name_list)

    # trainer = Trainer(args, train_loader, train=True)
    trainer.train()


if __name__ == "__main__":

    cudnn.benchmark = True
    args = parse_args()

    # refcoco_dir = dataset_dir.joinpath('RefCOCO')
    refcoco_dir = Path(args.refcoco_dir)
    refcocog_feature_dir = refcoco_dir.joinpath(args.dataset)
    refcocog_feature_dir = refcocog_feature_dir.joinpath('features')

    ngpus_per_node = torch.cuda.device_count()
    args.world_size = ngpus_per_node
    if args.local_rank in [0, -1]:
        print(args)

        comments = []
        if args.load is not None:
            ckpt_str = "_".join(args.load.split('/')[-3:])
            comments.append(ckpt_str)
        if args.comment != '':
            comments.append(args.comment)
        comment = '_'.join(comments)

        current_time = datetime.datetime.now().strftime('%b%d_%H-%M')

        # run_name其实可以自己设置一下
        run_name = f'{current_time}_GPU{args.world_size}_{args.dataset}_{args.lr}'
        if len(comments) > 0:
            run_name += f'_{comment}'

        # run_name设置在project下面的每个小实验的项目名字
        args.run_name = run_name

    if args.distributed:
        main_worker(args.local_rank, args)
    else:
        main_worker(0, args)



