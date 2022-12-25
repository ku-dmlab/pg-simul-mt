# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os

from fairseq import checkpoint_utils, tasks
import torch

try:
    from simuleval import READ_ACTION, WRITE_ACTION, DEFAULT_EOS
    from simuleval.agents import TextAgent
except ImportError:
    print("Please install simuleval 'pip install simuleval'")


BOS_PREFIX = "\u2581"
BOW_PREFIX='@@'

class SimulTransTextAgent(TextAgent):
    """
    Simultaneous Translation
    Text agent for Japanese
    """
    def __init__(self, args):
        self.gpu = getattr(args, "gpu", False)
        self.max_len = args.max_len
        self.load_model_vocab(args)
        self.eos = DEFAULT_EOS
        
        self.wait_k = args.wait_k
        self.tmp = []
        
    def initialize_states(self, states):
        states.incremental_states = dict()
        states.incremental_states["online"] = dict()

    def to_device(self, tensor):
        if self.gpu:
            return tensor.cuda()
        else:
            return tensor.cpu()

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
        self.dict = {}
        self.dict["tgt"] = task.target_dictionary
        self.dict["src"] = task.source_dictionary

        # self.tokenizer = MosesTokenizer(self.model.args.source_lang)
        # self.detokenizer = MosesDetokenizer(self.model.args.target_lang)


    @staticmethod
    def add_args(parser):
        # fmt: off
        parser.add_argument('--model-path', type=str, required=True,
                            help='path to your pretrained model.')
        parser.add_argument("--data-bin", type=str, required=True,
                            help="Path of data binary")
        parser.add_argument("--max-len", type=int, default=5000,
                            help="Max length of translation")
        parser.add_argument("--wait_k", type=int, default=10000)
        # fmt: on
        return parser

    def segment_to_units(self, segment, states):
        # Split a full word (segment) into subwords (units)
        # return self.tokenizer(segment)
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

    # def units_to_segment(self, units, states):
    #     token = units.value.pop()

    #     if (
    #         token == self.dict["tgt"].eos_word
    #         or len(states.segments.target) > self.max_len
    #     ):
    #         return DEFAULT_EOS

    #     if BOS_PREFIX == token:
    #         return None
    #     if token[0] == BOS_PREFIX:
    #         return token[1:]
    #     else:
    #         return token


    def units_to_segment(self, units, states):
        if None in units.value:
            units.value.remove(None)

        token = units.value[-1]

        if BOS_PREFIX == token:
            return None

        if token.endswith(BOW_PREFIX):
            return None
        else:
            segment = []
            for index in units:
                if self.dict["tgt"].eos_word == index:
                    segment += [DEFAULT_EOS]
                else:    
                    segment += [index.replace(BOW_PREFIX,"")]
                
                string_to_return = ["".join(segment)]

            for j in range(len(segment)):
                units.value.popleft()
                         
            return string_to_return

    def policy(self, states):
        if not getattr(states, "encoder_states", None):
            return READ_ACTION

        torch.cuda.empty_cache()

        if len(states.source) - len(states.target) < self.wait_k and not states.finish_read():
            return READ_ACTION
        else:
            return WRITE_ACTION

    def predict(self, states):
        # encode previous predicted target tokens
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
    
        x, outputs = self.model.decoder.forward(
            prev_output_tokens=tgt_indices,
            encoder_out=states.encoder_states,
            # incremental_state=states.incremental_states,
        )

        states.decoder_out = x
        # Predict target token from decoder states
        decoder_states = states.decoder_out

        lprobs = self.model.get_normalized_probs(
            [decoder_states[:, -1:]], log_probs=True
        )

        index = lprobs.argmax(dim=-1)[0, 0].item()

        if index != self.dict['tgt'].eos_index:
            token = self.dict['tgt'].string([index])
        else:
            token = self.dict['tgt'].eos_word

        return token