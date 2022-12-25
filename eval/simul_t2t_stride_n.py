# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import gc
from collections import deque
import torch
from fairseq import checkpoint_utils, tasks
# from fairseq.sequence_generator import SequenceGenerator
from examples.srd.modules.simul_translation import SimulSequenceGenerator

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
        self.stride_n = args.stride_n
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
                            help="Max length of translation")
        parser.add_argument("--wait_k", type=int, default=5)
        parser.add_argument("--stride_n", type=int, default=2, choices=range(1, 10000))
        parser.add_argument("--beam_size", type=int, default=1)
        return parser

    def segment_to_units(self, segment, states):
        return [segment]

    def update_states_read(self, states): # after READ
        stride = len(states.source.value) - self.wait_k - len(states.target.value)
        if stride == self.stride_n:
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

        if states.finish_read():
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
            
            top_hypo = hypos[0]['tokens']
        else:
            enc_states = self.model.encoder(sample["net_input"]["src_tokens"],
                                            sample["net_input"]["src_lengths"])
            tmp_hyps = []
            for _ in range(self.stride_n):
                tgt_indices = self.to_device(
                    torch.LongTensor(
                        [self.eos_index]
                        + [self.tgt_dict.index(x)
                            for x in states.units.target.value
                            if x is not None]
                        + [t for t in tmp_hyps if tmp_hyps]
                    ).unsqueeze(0)
                )
                dec_states, _ = self.model.decoder.forward(
                                        prev_output_tokens=tgt_indices,
                                        encoder_out=enc_states)
                lprobs = self.model.get_normalized_probs(
                    [dec_states[:, -1:]], log_probs=True
                )

                index = lprobs.argmax(dim=-1)[0, 0].item()
                tmp_hyps.append(index)
            top_hypo = torch.LongTensor(tmp_hyps)

        torch.cuda.empty_cache()
        for tok in top_hypo[:self.stride_n]:
            states.prefix.append(tok.item())

    def policy(self, states):
        if len(states.prefix) > 0:
            return WRITE_ACTION
        if states.new_tok or states.finish_read():
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