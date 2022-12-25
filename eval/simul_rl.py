# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import gc
from collections import deque
import torch
from fairseq import checkpoint_utils, tasks

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
        self.trans_model_path = args.trans_model_path
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
        
        self.trans_model = task.load_translation_model(self.trans_model_path)
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
        parser.add_argument("--max-len", type=int, default=200,
                            help="Max length of translation")
        parser.add_argument("--trans-model-path", type=str)
        return parser

    def segment_to_units(self, segment, states):
        return [segment]

    def update_model_encoder(self, states):
        if len(states.units.source) == 0:
            return

        src_indices = [
            self.dict['src'].index(x)
            for x in states.units.source.value
        ]

        if states.finish_read():
            # Append the eos index when the prediction is over
            src_indices += [self.dict["tgt"].eos_index]

        src_indices = self.to_device(
            torch.LongTensor(src_indices).unsqueeze(0)
        )
        src_lengths = self.to_device(
            torch.LongTensor([src_indices.size(1)])
        )

        states.encoder_states = self.model.encoder(src_indices, src_lengths)

        torch.cuda.empty_cache()

    def update_states_read(self, states):
        # Happens after a read action.
        self.update_model_encoder(states)
        
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
        
    def get_action(self, states):
        tgt_indices = self.to_device(
            torch.LongTensor(
                [self.model.decoder.dictionary.eos()]
                + [
                    self.dict['tgt'].index(x)
                    for x in states.units.target.value
                    if x is not None
                ]
            ).unsqueeze(0)
        )
        
        dec_states, extra = self.trans_model.decoder.forward(
            prev_output_tokens=tgt_indices,
            encoder_out=states.encoder_states,
        )

        lprobs = self.trans_model.get_normalized_probs(
            [dec_states[:, -1:]], log_probs=True
        )

        index = lprobs.argmax(dim=-1)[0, 0].item()
        
        torch.cuda.empty_cache()
         
        action = self.model.get_action(dec_states)
        
        return action
        

    def policy(self, states):
        if len(states.prefix) == 0:
            return WRITE_ACTION

        action = self.get_action(states)
        if action == 1:
            return READ_ACTION
        if action == 0 or states.finish_read(): 
            return WRITE_ACTION

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