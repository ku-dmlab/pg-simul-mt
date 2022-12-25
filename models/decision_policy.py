from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq.models import register_model, BaseFairseqModel
from fairseq.models import register_model_architecture
from collections import OrderedDict


@register_model("decision_policy")
class DecisionPolicy(BaseFairseqModel):
    def __init__(self, embed_dim, output_size, max_len, vocab_size, beam_steps, beam_size, simul_trans):
        super().__init__()
        self.max_len = max_len
        self.vocab_size = vocab_size

        self.beam_steps = beam_steps
        self.beam_steps = beam_size
        self.simul_trans = simul_trans

        self.input_size = vocab_size
        size = self.input_size // 2

        modules = OrderedDict()
        modules["linear"] = nn.Linear(self.input_size, size)
        modules["relu"] = nn.ReLU()
        
        for i in range(3):
            modules[f"linear_{i}"] = nn.Linear(size, size)
            modules[f"relu_{i}"] = nn.ReLU()
        modules["linear_outupt"] = nn.Linear(size, output_size)

        self.layers = nn.Sequential(modules)
        
        # self.action_mixing = nn.Linear(self.input_size + self.observation_dim, self.input_size)
        
    @staticmethod
    def add_args(parser):
        parser.add_argument("--beam-decoding", type=bool, default=False)
        parser.add_argument("--max-len", type=int)

    @classmethod
    def build_model(cls, args, task):
        max_len = args.max_len + 2 # for eos
        beam_steps = args.beam_steps
        beam_size = args.beam_size
        output_size = 1
        simul_trans = task.simul_trans
        embed_size = task.simul_trans.embed_dim()
        vocab_size = len(task.target_dictionary)

        return cls(embed_size, output_size, max_len, vocab_size, beam_steps, beam_size, simul_trans)

    def get_action(self, input):
        # Reshape Input as N X (max_len X D)
        '''
            max_len x batch x D
            -> batch x max_len x D
        '''
        # input = F.pad(input, (0,0,0, (self.max_len-input.shape[1]) ) )
        # assert x.shape[1] == self.input_size * self.max_len
        # assert input.shape[0] == x.shape[0]
        x = self.layers(input)

        # Make Distributions
        dist = torch.distributions.Bernoulli(logits=x)
        actions = dist.sample()
        
        return dist, actions

    def forward(self, sample):
        """
        sample
            "src_tokens": B x s_T
            "src_lengths": B
            "prev_ouput_tokens": B x t_T
            + actions, prefix_tokens, dec_states
        """
        bsz = sample['src_tokens'].size(0)
        device = sample['src_tokens'].device

        while True:
            sample, dec_states, detokens = self.simul_trans.get_decoding(sample)
            
            input = dec_states.reshape(dec_states.shape[1], dec_states.shape[0], -1)
            cur_action = self.get_action(input)[1].type(torch.LongTensor).to(device)
            # cur_action = torch.LongTensor([[1]]*bsz).to(device) # test
            
            sample, finish = self.simul_trans.make_transitions(sample, cur_action, detokens)

            # sample["actions"] = torch.concat((sample["actions"], cur_action), 1) # B x timestep
            sample["dec_states"].append(dec_states.tolist()) # timestep x (t_T, B,embed)
            
            if finish:
                result = self.simul_trans.get_episode_result(sample)
                # 1 sentence, 1 episode,
                # num of timesteps data : actions, dec_states
                # 1 prefix_tokens(simul hypo)
                break

        return result # (actions, dec_states, sample["prefix_tokens"])

@register_model_architecture(model_name="decision_policy", arch_name="decision_policy")
def decision_policy(args):
    args.max_len = getattr(args, "max_len", 200)
    args.beam_steps = getattr(args, "beam_steps", 10)
    args.beam_size = getattr(args, "beam_size", 8)