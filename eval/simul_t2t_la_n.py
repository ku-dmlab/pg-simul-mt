# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import gc
from collections import deque
import torch
from fairseq import checkpoint_utils, tasks
from fairseq.sequence_generator import SequenceGenerator
from examples.srd.modules.simul_seq_generator import SimulSequenceGenerator

try:
    from simuleval import READ_ACTION, WRITE_ACTION
    from simuleval.agents import TextAgent
except ImportError:
    print("Please install simuleval 'pip install simuleval'")


BOS_PREFIX = "\u2581"
BOW_PREFIX='@@'

class SimulTransTextAgent(TextAgent):
    def __init__(self, args):
        self.gpu = getattr(args, "gpu", False)
        self.max_len = args.max_len
        
        self.wait_k = args.wait_k
        self.la_n = args.la_n
        self.beam_size = args.beam_size
        self.load_model_vocab(args) # model, vocab, generator
        
    def load_model_vocab(self, args):
        filename = args.model_path
        if not os.path.exists(filename):
            raise IOError("Model file not found: {}".format(filename))

        state = checkpoint_utils.load_checkpoint_to_cpu(filename)

        task_args = state["cfg"]["task"]
        task_args.data = args.data_bin

        task = tasks.setup_task(task_args)

        # build model for ensemble
        state["cfg"]["model"].load_pretrained_encoder_from = None
        state["cfg"]["model"].load_pretrained_decoder_from = None

        self.model = task.build_model(state["cfg"]["model"])
        self.model.load_state_dict(state["model"], strict=True)
        self.model.eval()
        self.model.share_memory()

        if self.gpu:
            self.model.cuda()

        # Set dictionary
        self.tgt_dict = task.target_dictionary
        self.src_dict = task.source_dictionary
        self.eos_token = self.tgt_dict.eos_word
        self.eos_index = self.tgt_dict.eos_index

        # set Generator
        # self.generator = SequenceGenerator(models=[self.model], 
        #                                     tgt_dict=self.tgt_dict,
        #                                     beam_size=self.beam_size,
        #                                     max_len=self.max_len)

        self.generator = SimulSequenceGenerator(
                                [self.model], self.tgt_dict,
                                beam_size=self.beam_size,
                                max_len=self.max_len)

    def initialize_states(self, states):
        states.hypos = []
        states.prefix = deque()
        states.new_tok = False
        states.last_agree = 0

    def to_device(self, tensor):
        if self.gpu:
            return tensor.cuda()
        else:
            return tensor.cpu()

    @staticmethod
    def add_args(parser):
        # fmt: off
        parser.add_argument('--model-path', type=str, required=True,
                            help='path to your pretrained model.')
        parser.add_argument("--data-bin", type=str, required=True,
                            help="Path of data binary")
        parser.add_argument("--max-len", type=int, default=100,
                            help="Max length of translation") # 200 memory error
        parser.add_argument("--wait_k", type=int, default=5)
        parser.add_argument("--la_n", type=int, default=2, choices=range(2, 10000))
        parser.add_argument("--beam_size", type=int, default=1) # 8 memory error
        # max_len beam_size
        # 200 8 X
        # 100 6 X
        # fmt: on
        return parser

    def segment_to_units(self, segment, states):
        return [segment]

    def update_states_read(self, states): # after READ
        if len(states.source.value) - states.last_agree >= self.wait_k:
            states.new_tok = True

    def units_to_segment(self, units, states): # after WRITE, remove @@
        if None in units.value:
            units.value.remove(None)

        token = units.value[-1]
        if BOS_PREFIX == token:
            return None

        if token.endswith(BOW_PREFIX):
            return None
        else:
            segment = []
            for tok in units:
                if self.eos_token == tok:
                    segment += [self.eos_token]
                else:    
                    segment += [tok.replace(BOW_PREFIX, "")]
                string_to_return = ["".join(segment)]
            for _ in range(len(segment)):
                units.value.popleft()
                         
            return string_to_return

# ------------------------------------------------------------------------------------ #

    def _get_sample(self, states):
        src_indices = [
            self.src_dict.index(x)
            for x in states.units.source.value
        ]
        
        # 1
        # if states.finish_read():
        #     src_indices += [self.eos_index]
        # 2
        src_indices += [self.eos_index]
        
        src_indices = self.to_device(
            torch.LongTensor(src_indices).unsqueeze(0)
        )
        src_lengths = self.to_device(
            torch.LongTensor([src_indices.size(1)])
        )
        tgt_indices = self.to_device(
            torch.LongTensor([self.tgt_dict.index(x)
                            for x in states.units.target.value 
                            if x is not None]).unsqueeze(0))

        sample = {"net_input": {'src_tokens': src_indices, 
                                'src_lengths': src_lengths}}
        return sample, tgt_indices

    def _get_prefix(self, states):
        states.new_tok = False
        
        sample, tgt_indices = self._get_sample(states)

        if self.beam_size > 1:
            hypos = self.generator.generate(
                sample=sample,
                prefix_tokens=tgt_indices,
                bos_token=self.eos_index,
                # max_len= 50 + tgt_indices.size(1)
            )[0]
        else:
            hypos = self.generator.no_beam_generate(
                sample=sample,
                tgt_indices=tgt_indices,
                eos=self.eos_index,
                max_len=self.max_len
            )
        
        top_hypo = hypos[0]['tokens']
        if tgt_indices.size(1) > 0:
            states.hypos.append(torch.concat((tgt_indices.squeeze(0), top_hypo)))
        else:
            states.hypos.append(top_hypo)
        # if top_hypo.size(0) > 0:

        if states.finish_read():
            for t in states.hypos[-1][states.last_agree:]:
                states.prefix.append(t.item())

        if len(states.hypos) >= self.la_n:
            comp_hyps =[h[states.last_agree:] for h in states.hypos[-self.la_n:]]
            for hyps in zip(*comp_hyps):
                t = hyps[0]
                if all(o == t for o in hyps) and t != self.eos_index:
                    states.prefix.append(t.item())
                else:
                    break
        
        torch.cuda.empty_cache()
        if states.prefix:
            return True
        else:
            return False

    # def _get_prefix(self, states):
    #     states.new_tok = False

    #     gc.collect()
    #     torch.cuda.empty_cache()

    #     sample, tgt_indices = self._get_sample(states)
    #     hypos = self.generator._generate(
    #         sample=sample,
    #         prefix_tokens=tgt_indices,
    #         bos_token=self.eos_index,
    #     )
        
    #     # 1
    #     # top_hypo = hypos[0][0]['tokens']
    #     # 2
    #     top_hypo = hypos[0][0]['tokens'] if states.finish_read() else hypos[0][0]['tokens'][:-1]
    #     states.hypos.append(top_hypo)

    #     if states.finish_read():
    #         for t in states.hypos[-1][states.last_agree:]:
    #             states.prefix.append(t.item())

    #     if len(states.hypos) >= self.la_n:
    #         comp_hyps =[h[states.last_agree:] for h in states.hypos[-self.la_n:]]
    #         for hyps in zip(*comp_hyps):
    #             t = hyps[0]
    #             if all(o == t for o in hyps) and t != self.eos_index:
    #                 states.prefix.append(t.item())
    #             else:
    #                 break
        
    #     if states.prefix:
    #         return True
    #     else:
    #         return False

    def policy(self, states):
        if len(states.prefix) > 0:
            return WRITE_ACTION
        if states.new_tok:
            if self._get_prefix(states):
                return WRITE_ACTION
        if states.finish_read():
            self._get_prefix(states)
            return WRITE_ACTION
        return READ_ACTION

    def predict(self, states):
        if len(states.prefix) == 0:
            return self.eos_token
        
        index = states.prefix.popleft()
        if index != self.tgt_dict.eos_index:
            token = self.tgt_dict.string([int(index)])
        else:
            token = self.tgt_dict.eos_word
        states.last_agree += 1

        return token