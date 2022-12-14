import torch
import torch.nn as nn
import copy
# from memory_profiler import profile
from undecorated import undecorated
from types import MethodType
from torch.nn import NLLLoss, CrossEntropyLoss
import os.path as osp
import sys
if osp.join('/sharefs/baai-mrnd/yfl/codebase/Dialog/src', 'refer2', 'evaluation') not in sys.path:
    sys.path.insert(0, osp.join('/sharefs/baai-mrnd/yfl/codebase/Dialog/src', 'refer2', 'evaluation'))
from cider.cider import Cider
from tokenizer.ptbtokenizer import PTBTokenizer
from transformers import LogitsProcessorList, TopKLogitsWarper, TemperatureLogitsWarper


# if osp.join('/raid_sda/yfl/codebase/VL-T5-REG/feature_extraction') not in sys.path:
#     sys.path.insert(0, osp.join('/raid_sda/yfl/codebase/VL-T5-REG/feature_extraction'))

# from detectron2_given_target_box_maxnms import doit, build_model
import cv2
# cv2.setNumThreads(0)

from modeling_t5 import VLT5
import numpy as np
from copy import deepcopy

class VLT5REG(VLT5):
    def __init__(self, config):
        super().__init__(config)

    # @profile
    def train_step(self, batch, use_mmi=False, epoch=None, lama=1, margin=0.5, use_negative_text_training=False):

        device = next(self.parameters()).device
        if use_mmi:
            vis_feats = torch.squeeze(batch['vis_feats'][:, 0].to(device))
            vis_pos = torch.squeeze(batch['boxes'][:, 0].to(device))

            neg_vis_feats = torch.squeeze(batch['vis_feats'][:, 1].to(device))
            neg_vis_pos = torch.squeeze(batch['boxes'][:, 1].to(device))

            input_ids = batch['input_ids'][:].to(device)

            lm_labels = batch["target_ids"].to(device)

            reduce_loss = True
            output = self(
                input_ids=input_ids,
                vis_inputs=(vis_feats, vis_pos),
                labels=lm_labels,
                reduce_loss=reduce_loss
            )

            neg_output = self(
                input_ids=input_ids,
                vis_inputs=(neg_vis_feats, neg_vis_pos),
                labels=lm_labels,
                reduce_loss=reduce_loss
            )

            lm_mask = lm_labels != -100
            B, L = lm_labels.size()

            pos_loss = output['loss']
            neg_loss = neg_output['loss']

            # ?????????????????????????????????????????????...
            # if epoch % 10 == 0:
            #     margin /= 2
            loss = pos_loss + lama * (max(0, margin + pos_loss - neg_loss))

            result = {
                'loss': loss
            }
            return result
        elif use_negative_text_training:
            vis_feats = batch['vis_feats'].to(device)
            input_ids = batch['input_ids'].to(device)
            vis_pos = batch['boxes'].to(device)

            lm_labels = batch["target_ids"].to(device)
            negative_labels = batch["negative_sent_ids"].to(device)

            reduce_loss = True
            output = self(
                input_ids=input_ids,
                vis_inputs=(vis_feats, vis_pos),
                labels=lm_labels,
                reduce_loss=reduce_loss
            )

            # ?????????????????????????????????????????????????????????
            neg_output = self(
                input_ids=input_ids,
                vis_inputs=(vis_feats, vis_pos),
                labels=negative_labels,
                reduce_loss=reduce_loss
            )

            lm_mask = lm_labels != -100
            B, L = lm_labels.size()

            pos_loss = output['loss']
            neg_loss = neg_output['loss']

            # ?????????????????????????????????????????????...
            # if epoch % 10 == 0:
            #     margin /= 2
            loss = pos_loss + lama * (max(0, margin + pos_loss - neg_loss))
            # import pdb
            # pdb.set_trace()
            # loss = torch.mean(pos_loss + lama * (torch.clamp(margin + pos_loss - neg_loss, min=0.0)))

            result = {
                'loss': loss
            }
            return result
        else:
            vis_feats = batch['vis_feats'].to(device)
            input_ids = batch['input_ids'].to(device)
            vis_pos = batch['boxes'].to(device)

            lm_labels = batch["target_ids"].to(device)

            reduce_loss = True
            output = self(
                input_ids=input_ids,
                vis_inputs=(vis_feats, vis_pos),
                labels=lm_labels,
                reduce_loss=reduce_loss
            )

            lm_mask = lm_labels != -100
            B, L = lm_labels.size()

            loss = output['loss']

            result = {
                'loss': loss
            }
            return result

    def rec_rl_train_step(self, batch, rewarder, use_combine=False, lamda=0.5, combine_with_celoss=False):

        reslut = {}
        criterion = CrossEntropyLoss(reduction='none', ignore_index=0)
        rewarder = rewarder
        device = next(self.parameters()).device
        vis_feats = batch['vis_feats'].to(device)
        input_ids = batch['input_ids'].to(device)
        vis_pos = batch['boxes'].to(device)
        ref_ids = batch['ref_ids']
        target_sents = batch['target_texts']  # list:batch_size
        bs = len(target_sents)

        generate_with_grad = undecorated(self.generate)
        self.generate_with_grad = MethodType(generate_with_grad, self)

        sample_output = self.generate_with_grad(
            input_ids=input_ids,
            vis_inputs=(vis_feats, vis_pos),
            output_scores=True,
            return_dict_in_generate=True,
            do_sample=True,
            max_length=20,
        )
        sample_sents = self.tokenizer.batch_decode(sample_output.sequences, skip_special_tokens=True)
        # print('output_sents:', len(output_sents))
        # original scores: tuple: (tensor_matrix1, ..., tensor_matrixT) tensor_matrix_i: batch_size*vocab_size
        scores = torch.stack(sample_output.scores, dim=0).permute(1, 0, 2)  # batch_size*sentence_len*vocabulary
        scores = scores.reshape(-1, scores.size(-1))
        target = sample_output.sequences[:, 1:].reshape(-1)
        # index = target != 0
        # print(scores[list(range(len(scores))), target[index]])

        loss = criterion(scores,
                         target,
                         )
        loss = loss.view(bs, -1)
        loss = torch.mean(loss, dim=1)

        sample_dict = {}
        sample_dict['image_ids'] = batch['image_ids']  # ids is a list of int
        sample_dict['refBoxes'] = batch['refBoxes']
        sample_dict['sents'] = sample_sents  # a list of sent
        # rewarder should return a tensor in the shape of bacthsize
        sample_rewards, sample_rewards_mask = rewarder.compute_score(sample_dict)
        # sample_rewards = torch.from_numpy(sample_rewards).to(device)

        greedy_output = self.generate(
            input_ids=input_ids,
            vis_inputs=(vis_feats, vis_pos),
            do_sample=False,
            max_length=20,
        )
        greedy_sents = self.tokenizer.batch_decode(greedy_output, skip_special_tokens=True)

        greedy_dict = {}
        greedy_dict['image_ids'] = batch['image_ids']
        greedy_dict['refBoxes'] = batch['refBoxes']
        greedy_dict['sents'] = greedy_sents
        reward_baseline, reward_baseline_mask = rewarder.compute_score(greedy_dict)

        # try to put greedy_dict and sample_dict together
        # sadly,this will out of memory
        # dict = {}
        # dict['image_ids'] = batch['image_ids'] + batch['image_ids']
        # dict['refBoxes'] = batch['refBoxes'] + batch['refBoxes']
        # dict['sents'] = sample_sents + greedy_sents
        # rewards, masks = rewarder.compute_score(dict)
        # sample_rewards = rewards[:bs]
        # reward_baseline = rewards[bs:]
        # sample_rewards_mask = masks[:bs]
        # reward_baseline_mask = masks[bs:]

        # The code below maybe need may be not, I think keep it will be better
        # reward_baseline = torch.from_numpy(greedy_rewards).to(device)
        # print(output_rewards.size(), reward_baseline.size())
        # ?????????mask??????false??????????????? ious ??????????????????0.5????????????mask???
        # final_reward_mask = sample_rewards_mask | reward_baseline_mask
        # reward = torch.clamp((sample_rewards-reward_baseline)*final_reward_mask, min=0.0)
        # reward = torch.exp(torch.clamp(torch.tensor(1.5)*(sample_rewards-reward_baseline)*final_reward_mask, min=0.0))-torch.tensor(1.0)
        reward = torch.clamp((sample_rewards-reward_baseline), min=0.0)
        # reward = torch.exp(torch.clamp(torch.tensor(2.0)*(sample_rewards - reward_baseline), min=0.0)) - torch.tensor(1.0)
        # reward = sample_rewards-reward_baseline

        # reward = torch.clamp((sample_rewards-torch.tensor(0.5)), min=0.0)
        # reward = sample_rewards*sample_rewards_mask
        # reward = sample_rewards-torch.tensor(0.5)

        if use_combine:
            cider = Cider()
            sample_sents_dict = {}
            for ref_id, output_sent in zip(list(range(len(sample_sents))), sample_sents):
                sample_sents_dict[str(ref_id)] = [output_sent]
            target_sents_dict = {}
            for ref_id, target_sent in zip(list(range(len(target_sents))), target_sents):
                target_sents_dict[str(ref_id)] = [target_sent]
            tokenizer = PTBTokenizer()
            sample_sents_dict = tokenizer.tokenize(sample_sents_dict)
            target_sents_dict = tokenizer.tokenize(target_sents_dict)

            sample_cider_reward, sample_cider_rewards = cider.compute_score(target_sents_dict, sample_sents_dict)
            sample_cider_rewards = torch.from_numpy(sample_cider_rewards).to(device)

            greedy_sents_dict = {}
            for idx, greedy_sent in zip(list(range(len(greedy_sents))), greedy_sents):
                greedy_sents_dict[str(idx)] = [greedy_sent]

            greedy_cider_reward, greedy_cider_rewards = cider.compute_score(target_sents_dict, greedy_sents_dict)
            reward_cider_baseline = torch.from_numpy(greedy_cider_rewards).to(device)
            reward = lamda*reward + (1-lamda)*torch.clamp((sample_cider_rewards-reward_cider_baseline), min=0.0)
            reslut['sample_cider_reward'] = sample_cider_rewards.mean()

        loss = reward*loss
        loss = loss.mean()

        if combine_with_celoss:
            lm_labels = batch["target_ids"].to(device)
            reduce_loss = True
            output = self(
                input_ids=input_ids,
                vis_inputs=(vis_feats, vis_pos),
                labels=lm_labels,
                reduce_loss=reduce_loss
            )
            cross_entropy_loss = output['loss']

            loss += 0.1*cross_entropy_loss
        reslut['loss'] = loss
        # reslut['reward_baseline'] = reward_baseline.mean()
        reslut['sample_reward'] = sample_rewards.mean()
        reslut['reward'] = reward.mean()

        return reslut

    def dialog_train_step(self, batch, rewarder, dialog_round=1):

        reslut = {}
        criterion = CrossEntropyLoss(reduction='none', ignore_index=0)
        rewarder = rewarder
        device = next(self.parameters()).device
        vis_feats = batch['vis_feats'].to(device)
        input_ids = batch['input_ids'].to(device)
        vis_pos = batch['boxes'].to(device)
        ref_ids = batch['ref_ids']
        target_sents = batch['target_texts']  # list:batch_size
        bs = len(target_sents)

        generate_with_grad = undecorated(self.generate)
        self.generate_with_grad = MethodType(generate_with_grad, self)
        dialog_loss = []
        dialog_reward = []

        for i in range(dialog_round):

            sample_output = self.generate_with_grad(
                input_ids=input_ids,
                vis_inputs=(vis_feats, vis_pos),
                output_scores=True,
                return_dict_in_generate=True,
                do_sample=True,
                max_length=20,
            )
            sample_sents = self.tokenizer.batch_decode(sample_output.sequences, skip_special_tokens=True)
            # original scores: tuple: (tensor_matrix1, ..., tensor_matrixT) tensor_matrix_i: batch_size*vocab_size
            scores = torch.stack(sample_output.scores, dim=0).permute(1, 0, 2)  # batch_size*sentence_len*vocabulary
            scores = scores.reshape(-1, scores.size(-1))  # (batch_size*sentence_len, vocabulary)
            target = sample_output.sequences[:, 1:].reshape(-1)  # (batch_size*sentence_len,)

            # here loss is a vector which length is batch_size*sentence_len
            loss = criterion(scores,
                             target,
                             )
            loss = loss.view(bs, -1)
            loss = torch.mean(loss, dim=1)  # (batch_size, 1)
            dialog_loss.append(loss)

            sample_dict = {}
            sample_dict['image_ids'] = batch['image_ids']  # ids is a list of int
            sample_dict['refBoxes'] = batch['refBoxes']
            sample_dict['sents'] = sample_sents  # a list of sent
            # rewarder should return a tensor in the shape of bacthsize
            # sample_rewards: (batch_size, 1)
            sample_rewards, sample_rewards_mask = rewarder.compute_score(sample_dict)
            dialog_reward.append(sample_rewards)

            # update input ids
            # TODO how to cat the "located" and "unlocated" special token
            # ??????????????????batch??????????????????????????????????????????????????????????????????"unlocated"???????????????
            unlocated_ids = self.tokenizer.encode("unlocated")
            unlocated_ids = [unlocated_ids for _ in range(bs)]
            unlocated_ids = torch.LongTensor(unlocated_ids)
            unlocated_ids = unlocated_ids[:, :-1].to(device)
            input_ids = torch.cat((input_ids, unlocated_ids, sample_output.sequences[:, 1:]), 1)

        for i in range(dialog_round):
            if i == 0:
                final_loss = dialog_loss[i]
                final_reward = dialog_reward[i]
            else:
                final_loss += dialog_loss[i]
                final_reward += 2*dialog_reward[i] - dialog_reward[i-1]

        final_loss *= final_reward
        final_loss = final_loss.mean()
        reslut['loss'] = final_loss
        reslut['sample_reward'] = final_reward.mean()
        reslut['reward'] = final_reward.mean()

        return reslut

    def rl_train_step(self, batch):

        reslut = {}
        criterion = CrossEntropyLoss(reduction='none', ignore_index=0)
        rewarder = Cider()
        device = next(self.parameters()).device
        vis_feats = batch['vis_feats'].to(device)
        input_ids = batch['input_ids'].to(device)
        vis_pos = batch['boxes'].to(device)
        ref_ids = batch['ref_ids']
        target_sents = batch['target_texts']  # list:batch_size
        bs = len(target_sents)


        generate_with_grad = undecorated(self.generate)
        self.generate_with_grad = MethodType(generate_with_grad, self)

        output = self.generate_with_grad(
            input_ids=input_ids,
            vis_inputs=(vis_feats, vis_pos),
            output_scores=True,
            return_dict_in_generate=True,
            do_sample=True,
            max_length=20,
        )
        output_sents = self.tokenizer.batch_decode(output.sequences, skip_special_tokens=True)
        # print('output_sents:', len(output_sents))
        # original scores: tuple: (tensor_matrix1, ..., tensor_matrixT) tensor_matrix_i: batch_size*vocab_size
        scores = torch.stack(output.scores, dim=0).permute(1, 0, 2)  # batch_size*sentence_len*vocabulary
        # probs = torch.nn.functional.softmax(scores, dim=-1)
        scores = scores.reshape(-1, scores.size(-1))  # (batch_size*sentence_len, vocabulary)
        target = output.sequences[:, 1:].reshape(-1)  # (batch_size*sentence_len)
        # index = target != 0
        # print(scores[list(range(len(scores))), target[index]])

        # here loss is a vector which length is batch_size*sentence_len
        loss = criterion(scores,
                         target,
                         )
        loss = loss.view(bs, -1)  # (batch_size, sentence_len)
        loss = torch.mean(loss, dim=1)

        output_sents_dict = {}
        for ref_id, output_sent in zip(list(range(len(output_sents))), output_sents):
            output_sents_dict[str(ref_id)] = [output_sent]
        # print('output_sents_dict:', len(output_sents_dict))
        target_sents_dict = {}
        for ref_id, target_sent in zip(list(range(len(target_sents))), target_sents):
            target_sents_dict[str(ref_id)] = [target_sent]
        # print('target_sent_dict', len(target_sents_dict))
        # It seems change nothing bt PTBTokenizer,emmmmmm??????????????????
        tokenizer = PTBTokenizer()
        output_sents_dict = tokenizer.tokenize(output_sents_dict)
        target_sents_dict = tokenizer.tokenize(target_sents_dict)
        # print('output_sents_dict after tokenize', len(output_sents_dict))
        # print('target_sents_dict after tokenize', len(target_sents_dict))

        output_reward, output_rewards = rewarder.compute_score(target_sents_dict, output_sents_dict)
        output_rewards = torch.from_numpy(output_rewards).to(device)
        # print('output_rewards:', len(output_rewards))

        # logits_warper = LogitsProcessorList([])

        greedy_output = self.generate(
            input_ids=input_ids,
            vis_inputs=(vis_feats, vis_pos),
            do_sample=False,
            max_length=20,
        )
        greedy_sents = self.tokenizer.batch_decode(greedy_output, skip_special_tokens=True)

        greedy_sents_dict = {}
        for idx, greedy_sent in zip(list(range(len(greedy_sents))), greedy_sents):
            greedy_sents_dict[str(idx)] = [greedy_sent]

        greedy_reward, greedy_rewards = rewarder.compute_score(target_sents_dict, greedy_sents_dict)
        reward_baseline = torch.from_numpy(greedy_rewards).to(device)
        # print(output_rewards.size(), reward_baseline.size())
        # I think here maybe need a little change, get the every reward bigger than zero
        # reward = torch.clamp((output_rewards-reward_baseline), min=0.0)
        reward = output_rewards - reward_baseline
        loss = reward*loss
        loss = loss.mean()
        reslut['loss'] = loss
        reslut['sample_reward'] = output_rewards.mean()
        reslut['reward_baseline'] = reward_baseline.mean()
        reslut['reward'] = reward.mean()

        return reslut

    def rl_train_step2(self, batch):

        reslut = {}
        criterion = CrossEntropyLoss(reduction='none')
        rewarder = Cider()
        device = next(self.parameters()).device
        vis_feats = batch['vis_feats'].to(device)
        input_ids = batch['input_ids'].to(device)
        vis_pos = batch['boxes'].to(device)
        ref_ids = batch['ref_ids']
        target_sents = batch['target_texts']  # list:batch_size


        generate_with_grad = undecorated(self.generate)
        self.generate_with_grad = MethodType(generate_with_grad, self)

        output = self.generate_with_grad(
            input_ids=input_ids,
            vis_inputs=(vis_feats, vis_pos),
            output_scores=True,
            return_dict_in_generate=True
        )
        output_sents = self.tokenizer.batch_decode(output.sequences, skip_special_tokens=True)
        # print('output_sents:', len(output_sents))
        scores = torch.stack(output.scores, dim=0).permute(1, 0, 2)
        loss = criterion(scores.reshape(-1, scores.size(-1)),
                         output.sequences[:, 1:].reshape(-1),
                         )
        loss = loss.view(len(target_sents), -1)
        loss = torch.mean(loss, dim=1)

        output_sents_dict = {}
        for ref_id, output_sent in zip(list(range(len(output_sents))), output_sents):
            output_sents_dict[str(ref_id)] = [output_sent]
        # print('output_sents_dict:', len(output_sents_dict))
        target_sents_dict = {}
        for ref_id, target_sent in zip(list(range(len(target_sents))), target_sents):
            target_sents_dict[str(ref_id)] = [target_sent]
        # print('target_sent_dict', len(target_sents_dict))
        # It seems change nothing bt PTBTokenizer,emmmmmm??????????????????
        tokenizer = PTBTokenizer()
        output_sents_dict = tokenizer.tokenize(output_sents_dict)
        target_sents_dict = tokenizer.tokenize(target_sents_dict)
        # print('output_sents_dict after tokenize', len(output_sents_dict))
        # print('target_sents_dict after tokenize', len(target_sents_dict))

        output_reward, output_rewards = rewarder.compute_score(target_sents_dict, output_sents_dict)
        output_rewards = torch.from_numpy(output_rewards).to(device)
        # print('output_rewards:', len(output_rewards))

        beam_output = self.generate(
            input_ids=input_ids,
            vis_inputs=(vis_feats, vis_pos),
            num_beams=5,
            num_return_sequences=5,
            max_length=20,
        )
        beam_sents = self.tokenizer.batch_decode(beam_output, skip_special_tokens=True)
        beam_target_sents = []
        for target_sent in target_sents:
            beam_target_sents += [target_sent]*5

        beam_sents_dict = {}
        for idx, beam_sent in zip(list(range(len(beam_sents))), beam_sents):
            beam_sents_dict[str(idx)] = [beam_sent]

        beam_target_sents_dict = {}
        for idx, beam_target_sent in zip(list(range(len(beam_target_sents))), beam_target_sents):
            beam_target_sents_dict[str(idx)] = [beam_target_sent]
        beam_reward, beam_rewards = rewarder.compute_score(beam_target_sents_dict, beam_sents_dict)
        beam_rewards = torch.from_numpy(beam_rewards).to(device)
        beam_rewards = beam_rewards.view(-1, 5)
        reward_baseline = torch.mean(beam_rewards, dim=1)
        # print(output_rewards.size(), reward_baseline.size())
        # I think here maybe need a little change, get the every reward bigger than zero
        loss = torch.maximum((output_rewards-reward_baseline), torch.tensor(0))*loss
        loss = loss.mean()
        reslut['loss'] = loss

        return reslut

    # @profile(precision=4,stream=open('memory_profiler.log','w+'))
    def test_step(self, batch, dialog_training=False, dialog_round=1, last_round=True,**kwargs):
        device = next(self.parameters()).device
        vis_feats = batch['vis_feats'].to(device)
        input_ids = batch['input_ids'].to(device)
        vis_pos = batch['boxes'].to(device)
        ref_ids = deepcopy(batch['ref_ids'])
        bs = len(ref_ids)

        if dialog_training:
            dialog_generate_sents = []
            for i in range(dialog_round):
                output = self.generate(
                    input_ids=input_ids,
                    vis_inputs=(vis_feats, vis_pos),
                    **kwargs
                )
                output_sents = self.tokenizer.batch_decode(output, skip_special_tokens=True) # bs*sentence_len
                dialog_generate_sents.append(output_sents)
                # update input ids
                # TODO how to cat the "located" and "unlocated" special token
                # ??????????????????batch??????????????????????????????????????????????????????????????????"unlocated"???????????????
                unlocated_ids = self.tokenizer.encode("unlocated")
                unlocated_ids = [unlocated_ids for _ in range(bs)]
                unlocated_ids = torch.LongTensor(unlocated_ids)
                unlocated_ids = unlocated_ids[:, :-1].to(device)
                input_ids = torch.cat((input_ids, unlocated_ids, output[:, 1:]), 1)
            if last_round:
                generated_sents = output_sents
            else:
                generated_sents = []
                for i in range(bs):
                    text = ''
                    for j in range(dialog_round):
                        text += ' ' + dialog_generate_sents[j][i]
                    generated_sents.append(text)
        else:
            # generate ????????????num_beams, ??????num_return_sequence(default=1), so here return only 1 sentence for 1 ref_id???
            output = self.generate(
                input_ids=input_ids,
                vis_inputs=(vis_feats, vis_pos),
                **kwargs
            )

            # this is a list type, length equal to batch size,
            # e.g.['A giraffe standing in the shade of a tree.','A giraffe standing in the middle of two other giraffes.', ...]
            generated_sents = self.tokenizer.batch_decode(output, skip_special_tokens=True)

        result = []
        for i, sent in enumerate(generated_sents):

            result.append(
                {
                    'ref_id': ref_ids[i],
                    'sent': deepcopy(sent),
                }
            )

        return result


    def test_step_for_bad_re_collection(self, batch, rewarder, threshold=0.5, **kwargs):
        device = next(self.parameters()).device
        vis_feats = batch['vis_feats'].to(device)
        input_ids = batch['input_ids'].to(device)
        vis_pos = batch['boxes'].to(device)
        ref_ids = batch['ref_ids']
        bs = len(ref_ids)

        # generate ????????????num_beams, ??????num_return_sequence(default=1), so here return only 1 sentence for 1 ref_id???
        output = self.generate(
            input_ids=input_ids,
            vis_inputs=(vis_feats, vis_pos),
            **kwargs
        )

        # this is a list type, length equal to batch size,
        # e.g.['A giraffe standing in the shade of a tree.','A giraffe standing in the middle of two other giraffes.', ...]
        generated_sents = self.tokenizer.batch_decode(output, skip_special_tokens=True)
        sample_dict = {}
        sample_dict['image_ids'] = [item for item in batch['image_ids'] for i in range(5)]
        sample_dict['refBoxes'] = [item for item in batch['refBoxes'] for i in range(5)]
        sample_dict['sents'] = generated_sents

        sample_rewards, sample_rewards_mask, ofa_results = rewarder.compute_score(sample_dict, threshold=threshold)

        ofa_bboxes = []
        for ofa_result in ofa_results:
            ofa_bboxes.append(ofa_result['box'])

        result = {
            'sents': generated_sents,
            'masks': sample_rewards_mask,
            'boxes': ofa_bboxes,
        }

        return result

    def zero_shot_test_step(self, batch, rewarder, refine_model=None, dialog_round=1, last_round=False, threshold=0.5, detector=None, **kwargs):

        device = next(self.parameters()).device
        vis_feats = batch['vis_feats'].to(device)
        input_ids = batch['input_ids'].to(device)
        vis_pos = batch['boxes'].to(device)
        ref_ids = batch['ref_ids']
        bs = len(ref_ids)  # ????????????1?????????

        prefix = "caption region:"
        visual_token_38 = "<vis_extra_id_38>"
        visual_token = "<vis_extra_id_37>"

        dialog_generatae_sents = [['']*dialog_round for _ in range(bs)]  # size: bs*num_dialog_round
        dialog_generatae_sents_ofa_ious = [[-1]*dialog_round for _ in range(bs)]
        for dialog_round_idx in range(dialog_round):
            if refine_model==None:
                output = self.generate(
                    input_ids=input_ids,
                    vis_inputs=(vis_feats, vis_pos),
                    **kwargs
                )
            else:
                if dialog_round_idx == 0:
                    output = self.generate(
                        input_ids=input_ids,
                        vis_inputs=(vis_feats, vis_pos),
                        **kwargs
                    )
                else:
                    output = refine_model.generate(
                        input_ids=input_ids,
                        vis_inputs=(new_vis_feats, new_vis_pos),
                        **kwargs
                    )

            output_sents = self.tokenizer.batch_decode(output, skip_special_tokens=True)  # bs*sentence_len
            for bs_idx, output_sent in enumerate(output_sents):
                dialog_generatae_sents[bs_idx][dialog_round_idx] = output_sent
            sample_dict = {}
            sample_dict['image_ids'] = batch['image_ids']  # ids is a list of int
            sample_dict['refBoxes'] = batch['refBoxes']
            sample_dict['sents'] = output_sents  # a list of sent
            # rewarder should return a tensor in the shape of bacthsize
            # sample_rewards: (batch_size, 1)
            sample_rewards, sample_rewards_mask, det_result = rewarder.compute_score(sample_dict)

            for bs_idx, sample_reward in enumerate(sample_rewards):
                dialog_generatae_sents_ofa_ious[bs_idx][dialog_round_idx] = sample_reward.item()
            # IOU surpass 0.5, the we think it located the target object.
            if sample_rewards[0] >= threshold:
                break
            # update input ids
            ofa_box = [det_result[0]['box']]
            img_path = '/raid_sda/yfl/datasets/train2014/COCO_train2014_' + str(batch['image_ids'][0]).zfill(12) + '.jpg'
            img = cv2.imread(img_path)
            instances, ofa_feature = doit(img, np.array(ofa_box), detector)
            # ofa_feature = torch.from_numpy(ofa_feature)
            new_vis_feats = torch.cat((vis_feats[0], ofa_feature), axis=0)
            new_vis_feats = new_vis_feats.unsqueeze(0)
            ofa_box = torch.tensor(ofa_box).to(device)
            new_vis_pos = torch.cat((vis_pos[0], ofa_box), axis=0)
            new_vis_pos = new_vis_pos.unsqueeze(0)

            input_text = f'{prefix} {visual_token_38} {output_sents[0]} {prefix} {visual_token}'
            input_ids = self.tokenizer.encode(input_text)
            input_ids = torch.LongTensor(input_ids).to(device)
            input_ids = input_ids.unsqueeze(0)

            # unlocated_ids = self.tokenizer.encode("unlocated")
            # unlocated_ids = [unlocated_ids for _ in range(bs)]
            # unlocated_ids = torch.LongTensor(unlocated_ids)
            # unlocated_ids = unlocated_ids[:, :-1].to(device)
            # input_ids = torch.cat((input_ids, output[:, 1:], unlocated_ids), 1)

        # last_round ???????????????????????????????????????????????????????????????
        if last_round:
            generated_sents = output_sents

        result = []
        for bs_idx, sent in enumerate(generated_sents):
            result.append(
                {
                    'ref_id': ref_ids[bs_idx],
                    'sent': sent,
                    'dialog_generate_sent': dialog_generatae_sents[bs_idx],
                    'dialog_generate_sent_ofa_iou': dialog_generatae_sents_ofa_ious[bs_idx],
                }
            )

        return result


    def multitask_test_step(self, batch, rewarder, refine_model=None, dialog_round=1, last_round=False, threshold=0.5, detector=None, **kwargs):
        '''
        ??????????????????????????????????????????????????????task1???task2???????????????predict??????OFA??????????????????
        '''


        device = next(self.parameters()).device
        vis_feats = batch['vis_feats'].to(device)
        input_ids = batch['input_ids'].to(device)
        vis_pos = batch['boxes'].to(device)
        ref_ids = batch['ref_ids']
        bs = len(ref_ids)  # ????????????1?????????

        prefix = "caption region:"
        visual_token_38 = "<vis_extra_id_38>"
        visual_token = "<vis_extra_id_37>"

        dialog_generatae_sents = [['']*dialog_round for _ in range(bs)]  # size: bs*num_dialog_round
        dialog_generatae_sents_ofa_ious = [[-1]*dialog_round for _ in range(bs)]
        for dialog_round_idx in range(dialog_round):
            if refine_model==None:
                if dialog_round_idx==0:
                    output = self.generate(
                        input_ids=input_ids,
                        vis_inputs=(vis_feats, vis_pos),
                        **kwargs
                    )
                else:
                    task1_output = self.generate(
                        input_ids=task1_input_ids,
                        vis_inputs=(vis_feats, vis_pos),
                        **kwargs
                    )

                    task2_output = self.generate(
                        input_ids=task2_input_ids,
                        vis_inputs=(task2_vis_feats, task2_vis_pos),
                        **kwargs
                    )

            else:
                if dialog_round_idx == 0:
                    output = self.generate(
                        input_ids=input_ids,
                        vis_inputs=(vis_feats, vis_pos),
                        **kwargs
                    )
                else:
                    task1_output = refine_model.generate(
                        input_ids=task1_input_ids,
                        vis_inputs=(vis_feats, vis_pos),
                        **kwargs
                    )

                    task2_output = refine_model.generate(
                        input_ids=task2_input_ids,
                        vis_inputs=(task2_vis_feats, task2_vis_pos),
                        **kwargs
                    )
            if dialog_round_idx == 0:
                output_sents = self.tokenizer.batch_decode(output, skip_special_tokens=True)  # bs*sentence_len
                for bs_idx, output_sent in enumerate(output_sents):
                    dialog_generatae_sents[bs_idx][dialog_round_idx] = output_sent
                sample_dict = {}
                sample_dict['image_ids'] = batch['image_ids']  # ids is a list of int
                sample_dict['refBoxes'] = batch['refBoxes']
                sample_dict['sents'] = output_sents  # a list of sent
                # rewarder should return a tensor in the shape of bacthsize
                # sample_rewards: (batch_size, 1)
                sample_rewards, sample_rewards_mask, det_result = rewarder.compute_score(sample_dict)

                for bs_idx, sample_reward in enumerate(sample_rewards):
                    dialog_generatae_sents_ofa_ious[bs_idx][dialog_round_idx] = sample_reward.item()
                # IOU surpass 0.5, the we think it located the target object.
                if sample_rewards[0] >= threshold:
                    break
                # update input ids
                ofa_box = [det_result[0]['box']]
                img_path = '/raid_sda/yfl/datasets/train2014/COCO_train2014_' + str(batch['image_ids'][0]).zfill(12) + '.jpg'
                img = cv2.imread(img_path)
                instances, ofa_feature = doit(img, np.array(ofa_box), detector)
                # ofa_feature = torch.from_numpy(ofa_feature)
                task2_vis_feats = torch.cat((vis_feats[0], ofa_feature), axis=0)
                task2_vis_feats = task2_vis_feats.unsqueeze(0)
                ofa_box = torch.tensor(ofa_box).to(device)
                task2_vis_pos = torch.cat((vis_pos[0], ofa_box), axis=0)
                task2_vis_pos = task2_vis_pos.unsqueeze(0)

                task2_input_text = f'{prefix} {visual_token_38} {output_sent[0]} {prefix} {visual_token}'
                task2_input_ids = self.tokenizer.encode(task2_input_text)
                task2_input_ids = torch.LongTensor(task2_input_ids).to(device)
                task2_input_ids = task2_input_ids.unsqueeze(0)

                task1_input_text = f'{prefix} {visual_token} {output_sents[0]} '
                task1_input_ids = self.tokenizer.encode(task1_input_text)
                task1_input_ids = torch.LongTensor(task1_input_ids).to(device)
                task1_input_ids = task1_input_ids.unsqueeze(0)
            else:
                task1_output_sents = self.tokenizer.batch_decode(task1_output, skip_special_tokens=True)  # bs*sentence_len
                task2_output_sents = self.tokenizer.batch_decode(task2_output, skip_special_tokens=True)  # bs*sentence_len

                sample_dict = {}
                sample_dict['image_ids'] = batch['image_ids']  # ids is a list of int
                sample_dict['refBoxes'] = batch['refBoxes']

                sample_dict['sents'] = task1_output_sents  # a list of sent
                # rewarder should return a tensor in the shape of bacthsize
                # sample_rewards: (batch_size, 1)
                task1_sample_rewards, task1_sample_rewards_mask, task1_det_result = rewarder.compute_score(sample_dict)

                sample_dict['sents'] = task2_output_sents
                task2_sample_rewards, task2_sample_rewards_mask, task2_det_result = rewarder.compute_score(sample_dict)

                if task1_sample_rewards[0] >= task2_sample_rewards[0]:
                    for bs_idx, task1_output_sent in enumerate(task1_output_sents):
                        dialog_generatae_sents[bs_idx][dialog_round_idx] = task1_output_sent
                    for bs_idx, task1_sample_reward in enumerate(task1_sample_rewards):
                        dialog_generatae_sents_ofa_ious[bs_idx][dialog_round_idx] = task1_sample_reward.item()
                    output_sents = task1_output_sents
                else:
                    for bs_idx, task2_output_sent in enumerate(task2_output_sents):
                        dialog_generatae_sents[bs_idx][dialog_round_idx] = task2_output_sent
                    for bs_idx, task2_sample_reward in enumerate(task2_sample_rewards):
                        dialog_generatae_sents_ofa_ious[bs_idx][dialog_round_idx] = task2_sample_reward.item()
                    output_sents = task2_output_sents

                # for bs_idx, task1_output_sent in enumerate(task1_output_sents):
                #     dialog_generatae_sents[bs_idx][dialog_round_idx] = task1_output_sent
                # for bs_idx, task1_sample_reward in enumerate(task1_sample_rewards):
                #     dialog_generatae_sents_ofa_ious[bs_idx][dialog_round_idx] = task1_sample_reward.item()
                # output_sents = task1_output_sents
                # IOU surpass 0.5, the we think it located the target object.

            # unlocated_ids = self.tokenizer.encode("unlocated")
            # unlocated_ids = [unlocated_ids for _ in range(bs)]
            # unlocated_ids = torch.LongTensor(unlocated_ids)
            # unlocated_ids = unlocated_ids[:, :-1].to(device)
            # input_ids = torch.cat((input_ids, output[:, 1:], unlocated_ids), 1)

        # last_round ???????????????????????????????????????????????????????????????
        if last_round:
            generated_sents = output_sents

        result = []
        for bs_idx, sent in enumerate(generated_sents):
            result.append(
                {
                    'ref_id': ref_ids[bs_idx],
                    'sent': sent,
                    'dialog_generate_sent': dialog_generatae_sents[bs_idx],
                    'dialog_generate_sent_ofa_iou': dialog_generatae_sents_ofa_ious[bs_idx],
                }
            )

        return result

    def task1_test_step(self, batch, rewarder, refine_model=None, dialog_round=1, last_round=False, threshold=0.5, detector=None, **kwargs):
        '''
        ??????????????????????????????????????????????????????task1???task2???????????????predict??????OFA??????????????????
        '''


        device = next(self.parameters()).device
        vis_feats = batch['vis_feats'].to(device)
        input_ids = batch['input_ids'].to(device)
        vis_pos = batch['boxes'].to(device)
        ref_ids = batch['ref_ids']
        bs = len(ref_ids)  # ????????????1?????????

        prefix = "caption region:"
        visual_token_38 = "<vis_extra_id_38>"
        visual_token = "<vis_extra_id_37>"

        dialog_generatae_sents = [['']*dialog_round for _ in range(bs)]  # size: bs*num_dialog_round
        dialog_generatae_sents_ofa_ious = [[-1]*dialog_round for _ in range(bs)]
        for dialog_round_idx in range(dialog_round):
            if refine_model==None:
                if dialog_round_idx==0:
                    output = self.generate(
                        input_ids=input_ids,
                        vis_inputs=(vis_feats, vis_pos),
                        **kwargs
                    )
                else:
                    task1_output = self.generate(
                        input_ids=task1_input_ids,
                        vis_inputs=(vis_feats, vis_pos),
                        **kwargs
                    )

                    # task2_output = self.generate(
                    #     input_ids=task2_input_ids,
                    #     vis_inputs=(task2_vis_feats, task2_vis_pos),
                    #     **kwargs
                    # )

            else:
                if dialog_round_idx == 0:
                    output = self.generate(
                        input_ids=input_ids,
                        vis_inputs=(vis_feats, vis_pos),
                        **kwargs
                    )
                else:
                    task1_output = refine_model.generate(
                        input_ids=task1_input_ids,
                        vis_inputs=(vis_feats, vis_pos),
                        **kwargs
                    )

                    # task2_output = refine_model.generate(
                    #     input_ids=task2_input_ids,
                    #     vis_inputs=(task2_vis_feats, task2_vis_pos),
                    #     **kwargs
                    # )
            if dialog_round_idx == 0:
                output_sents = self.tokenizer.batch_decode(output, skip_special_tokens=True)  # bs*sentence_len
                for bs_idx, output_sent in enumerate(output_sents):
                    dialog_generatae_sents[bs_idx][dialog_round_idx] = output_sent
                sample_dict = {}
                sample_dict['image_ids'] = batch['image_ids']  # ids is a list of int
                sample_dict['refBoxes'] = batch['refBoxes']
                sample_dict['sents'] = output_sents  # a list of sent
                # rewarder should return a tensor in the shape of bacthsize
                # sample_rewards: (batch_size, 1)
                sample_rewards, sample_rewards_mask, det_result = rewarder.compute_score(sample_dict)

                for bs_idx, sample_reward in enumerate(sample_rewards):
                    dialog_generatae_sents_ofa_ious[bs_idx][dialog_round_idx] = sample_reward.item()
                # IOU surpass 0.5, the we think it located the target object.
                if sample_rewards[0] >= threshold:
                    break
                # update input ids
                # ofa_box = [det_result[0]['box']]
                # img_path = '/raid_sda/yfl/datasets/train2014/COCO_train2014_' + str(batch['image_ids'][0]).zfill(12) + '.jpg'
                # img = cv2.imread(img_path)
                # instances, ofa_feature = doit(img, np.array(ofa_box), detector)
                # # ofa_feature = torch.from_numpy(ofa_feature)
                # task2_vis_feats = torch.cat((vis_feats[0], ofa_feature), axis=0)
                # task2_vis_feats = task2_vis_feats.unsqueeze(0)
                # ofa_box = torch.tensor(ofa_box).to(device)
                # task2_vis_pos = torch.cat((vis_pos[0], ofa_box), axis=0)
                # task2_vis_pos = task2_vis_pos.unsqueeze(0)
                #
                # task2_input_text = f'{prefix} {visual_token_38} {output_sent[0]} {prefix} {visual_token}'
                # task2_input_ids = self.tokenizer.encode(task2_input_text)
                # task2_input_ids = torch.LongTensor(task2_input_ids).to(device)
                # task2_input_ids = task2_input_ids.unsqueeze(0)

                task1_input_text = f'{prefix} {visual_token} {output_sents[0]} '
                task1_input_ids = self.tokenizer.encode(task1_input_text)
                task1_input_ids = torch.LongTensor(task1_input_ids).to(device)
                task1_input_ids = task1_input_ids.unsqueeze(0)
            else:
                task1_output_sents = self.tokenizer.batch_decode(task1_output, skip_special_tokens=True)  # bs*sentence_len
                # task2_output_sents = self.tokenizer.batch_decode(task2_output, skip_special_tokens=True)  # bs*sentence_len

                sample_dict = {}
                sample_dict['image_ids'] = batch['image_ids']  # ids is a list of int
                sample_dict['refBoxes'] = batch['refBoxes']

                sample_dict['sents'] = task1_output_sents  # a list of sent
                # rewarder should return a tensor in the shape of bacthsize
                # sample_rewards: (batch_size, 1)
                task1_sample_rewards, task1_sample_rewards_mask, task1_det_result = rewarder.compute_score(sample_dict)

                # sample_dict['sents'] = task2_output_sents
                # task2_sample_rewards, task2_sample_rewards_mask, task2_det_result = rewarder.compute_score(sample_dict)

                # if task1_sample_rewards[0] >= task2_sample_rewards[0]:
                #     for bs_idx, task1_output_sent in enumerate(task1_output_sents):
                #         dialog_generatae_sents[bs_idx][dialog_round_idx] = task1_output_sent
                #     for bs_idx, task1_sample_reward in enumerate(task1_sample_rewards):
                #         dialog_generatae_sents_ofa_ious[bs_idx][dialog_round_idx] = task1_sample_reward.item()
                #     output_sents = task1_output_sents
                # else:
                #     for bs_idx, task2_output_sent in enumerate(task2_output_sents):
                #         dialog_generatae_sents[bs_idx][dialog_round_idx] = task2_output_sent
                #     for bs_idx, task2_sample_reward in enumerate(task2_sample_rewards):
                #         dialog_generatae_sents_ofa_ious[bs_idx][dialog_round_idx] = task2_sample_reward.item()
                #     output_sents = task2_output_sents

                for bs_idx, task1_output_sent in enumerate(task1_output_sents):
                    dialog_generatae_sents[bs_idx][dialog_round_idx] = task1_output_sent
                for bs_idx, task1_sample_reward in enumerate(task1_sample_rewards):
                    dialog_generatae_sents_ofa_ious[bs_idx][dialog_round_idx] = task1_sample_reward.item()
                output_sents = task1_output_sents
                # IOU surpass 0.5, the we think it located the target object.

            # unlocated_ids = self.tokenizer.encode("unlocated")
            # unlocated_ids = [unlocated_ids for _ in range(bs)]
            # unlocated_ids = torch.LongTensor(unlocated_ids)
            # unlocated_ids = unlocated_ids[:, :-1].to(device)
            # input_ids = torch.cat((input_ids, output[:, 1:], unlocated_ids), 1)

        # last_round ???????????????????????????????????????????????????????????????
        if last_round:
            generated_sents = output_sents

        result = []
        for bs_idx, sent in enumerate(generated_sents):
            result.append(
                {
                    'ref_id': ref_ids[bs_idx],
                    'sent': sent,
                    'dialog_generate_sent': dialog_generatae_sents[bs_idx],
                    'dialog_generate_sent_ofa_iou': dialog_generatae_sents_ofa_ious[bs_idx],
                }
            )

        return result


    def task2_test_step(self, batch, rewarder, refine_model=None, dialog_round=1, last_round=False, threshold=0.5, detector=None, **kwargs):
        '''
        ??????????????????????????????????????????????????????task1???task2???????????????predict??????OFA??????????????????
        '''


        device = next(self.parameters()).device
        vis_feats = batch['vis_feats'].to(device)
        input_ids = batch['input_ids'].to(device)
        vis_pos = batch['boxes'].to(device)
        ref_ids = batch['ref_ids']
        bs = len(ref_ids)  # ????????????1?????????

        prefix = "caption region:"
        visual_token_38 = "<vis_extra_id_38>"
        visual_token = "<vis_extra_id_37>"

        dialog_generatae_sents = [['']*dialog_round for _ in range(bs)]  # size: bs*num_dialog_round
        dialog_generatae_sents_ofa_ious = [[-1]*dialog_round for _ in range(bs)]
        for dialog_round_idx in range(dialog_round):
            if refine_model==None:
                if dialog_round_idx==0:
                    output = self.generate(
                        input_ids=input_ids,
                        vis_inputs=(vis_feats, vis_pos),
                        **kwargs
                    )
                else:
                    # task1_output = self.generate(
                    #     input_ids=task1_input_ids,
                    #     vis_inputs=(vis_feats, vis_pos),
                    #     **kwargs
                    # )

                    task2_output = self.generate(
                        input_ids=task2_input_ids,
                        vis_inputs=(task2_vis_feats, task2_vis_pos),
                        **kwargs
                    )

            else:
                if dialog_round_idx == 0:
                    output = self.generate(
                        input_ids=input_ids,
                        vis_inputs=(vis_feats, vis_pos),
                        **kwargs
                    )
                else:
                    # task1_output = refine_model.generate(
                    #     input_ids=task1_input_ids,
                    #     vis_inputs=(vis_feats, vis_pos),
                    #     **kwargs
                    # )

                    task2_output = refine_model.generate(
                        input_ids=task2_input_ids,
                        vis_inputs=(task2_vis_feats, task2_vis_pos),
                        **kwargs
                    )
            if dialog_round_idx == 0:
                output_sents = self.tokenizer.batch_decode(output, skip_special_tokens=True)  # bs*sentence_len
                for bs_idx, output_sent in enumerate(output_sents):
                    dialog_generatae_sents[bs_idx][dialog_round_idx] = output_sent
                sample_dict = {}
                sample_dict['image_ids'] = batch['image_ids']  # ids is a list of int
                sample_dict['refBoxes'] = batch['refBoxes']
                sample_dict['sents'] = output_sents  # a list of sent
                # rewarder should return a tensor in the shape of bacthsize
                # sample_rewards: (batch_size, 1)
                sample_rewards, sample_rewards_mask, det_result = rewarder.compute_score(sample_dict)

                for bs_idx, sample_reward in enumerate(sample_rewards):
                    dialog_generatae_sents_ofa_ious[bs_idx][dialog_round_idx] = sample_reward.item()
                # IOU surpass 0.5, the we think it located the target object.
                if sample_rewards[0] >= threshold:
                    break
                # update input ids
                ofa_box = [det_result[0]['box']]
                img_path = '/raid_sda/yfl/datasets/train2014/COCO_train2014_' + str(batch['image_ids'][0]).zfill(12) + '.jpg'
                img = cv2.imread(img_path)
                instances, ofa_feature = doit(img, np.array(ofa_box), detector)
                # ofa_feature = torch.from_numpy(ofa_feature)
                task2_vis_feats = torch.cat((vis_feats[0], ofa_feature), axis=0)
                task2_vis_feats = task2_vis_feats.unsqueeze(0)
                ofa_box = torch.tensor(ofa_box).to(device)
                task2_vis_pos = torch.cat((vis_pos[0], ofa_box), axis=0)
                task2_vis_pos = task2_vis_pos.unsqueeze(0)

                task2_input_text = f'{prefix} {visual_token_38} {output_sent[0]} {prefix} {visual_token}'
                task2_input_ids = self.tokenizer.encode(task2_input_text)
                task2_input_ids = torch.LongTensor(task2_input_ids).to(device)
                task2_input_ids = task2_input_ids.unsqueeze(0)

                # task1_input_text = f'{prefix} {visual_token} {output_sents[0]} '
                # task1_input_ids = self.tokenizer.encode(task1_input_text)
                # task1_input_ids = torch.LongTensor(task1_input_ids).to(device)
                # task1_input_ids = task1_input_ids.unsqueeze(0)
            else:
                # task1_output_sents = self.tokenizer.batch_decode(task1_output, skip_special_tokens=True)  # bs*sentence_len
                task2_output_sents = self.tokenizer.batch_decode(task2_output, skip_special_tokens=True)  # bs*sentence_len

                sample_dict = {}
                sample_dict['image_ids'] = batch['image_ids']  # ids is a list of int
                sample_dict['refBoxes'] = batch['refBoxes']

                # sample_dict['sents'] = task1_output_sents  # a list of sent
                # rewarder should return a tensor in the shape of bacthsize
                # sample_rewards: (batch_size, 1)
                # task1_sample_rewards, task1_sample_rewards_mask, task1_det_result = rewarder.compute_score(sample_dict)

                sample_dict['sents'] = task2_output_sents
                task2_sample_rewards, task2_sample_rewards_mask, task2_det_result = rewarder.compute_score(sample_dict)

                # if task1_sample_rewards[0] >= task2_sample_rewards[0]:
                #     for bs_idx, task1_output_sent in enumerate(task1_output_sents):
                #         dialog_generatae_sents[bs_idx][dialog_round_idx] = task1_output_sent
                #     for bs_idx, task1_sample_reward in enumerate(task1_sample_rewards):
                #         dialog_generatae_sents_ofa_ious[bs_idx][dialog_round_idx] = task1_sample_reward.item()
                #     output_sents = task1_output_sents
                # else:
                #     for bs_idx, task2_output_sent in enumerate(task2_output_sents):
                #         dialog_generatae_sents[bs_idx][dialog_round_idx] = task2_output_sent
                #     for bs_idx, task2_sample_reward in enumerate(task2_sample_rewards):
                #         dialog_generatae_sents_ofa_ious[bs_idx][dialog_round_idx] = task2_sample_reward.item()
                #     output_sents = task2_output_sents

                for bs_idx, task1_output_sent in enumerate(task2_output_sents):
                    dialog_generatae_sents[bs_idx][dialog_round_idx] = task1_output_sent
                for bs_idx, task1_sample_reward in enumerate(task2_sample_rewards):
                    dialog_generatae_sents_ofa_ious[bs_idx][dialog_round_idx] = task1_sample_reward.item()
                output_sents = task2_output_sents
                # IOU surpass 0.5, the we think it located the target object.

            # unlocated_ids = self.tokenizer.encode("unlocated")
            # unlocated_ids = [unlocated_ids for _ in range(bs)]
            # unlocated_ids = torch.LongTensor(unlocated_ids)
            # unlocated_ids = unlocated_ids[:, :-1].to(device)
            # input_ids = torch.cat((input_ids, output[:, 1:], unlocated_ids), 1)

        # last_round ???????????????????????????????????????????????????????????????
        if last_round:
            generated_sents = output_sents

        result = []
        for bs_idx, sent in enumerate(generated_sents):
            result.append(
                {
                    'ref_id': ref_ids[bs_idx],
                    'sent': sent,
                    'dialog_generate_sent': dialog_generatae_sents[bs_idx],
                    'dialog_generate_sent_ofa_iou': dialog_generatae_sents_ofa_ious[bs_idx],
                }
            )

        return result

from modeling_bart import VLBart
class VLBartREG(VLBart):
    def __init__(self, config):
        super().__init__(config)

    def train_step(self, batch):
        device = next(self.parameters()).device
        vis_feats = batch['vis_feats'].to(device)
        input_ids = batch['input_ids'].to(device)
        vis_pos = batch['boxes'].to(device)


        lm_labels = batch["target_ids"].to(device)

        reduce_loss = True
        output = self(
            input_ids=input_ids,
            vis_inputs=(vis_feats, vis_pos),
            labels=lm_labels,
            reduce_loss=reduce_loss
        )

        lm_mask = lm_labels != -100
        B, L = lm_labels.size()

        loss = output['loss']

        result = {
            'loss': loss
        }
        return result

    def test_step(self, batch, **kwargs):
        device = next(self.parameters()).device
        vis_feats = batch['vis_feats'].to(device)
        input_ids = batch['input_ids'].to(device)
        vis_pos = batch['boxes'].to(device)

        output = self.generate(
            input_ids=input_ids,
            vis_inputs=(vis_feats, vis_pos),
            **kwargs
        )

        generated_sents = self.tokenizer.batch_decode(output, skip_special_tokens=True)

        result = {}
        result['pred'] = generated_sents

        return result