import math
import torch
from copy import deepcopy
from examples.srd.modules.simul_seq_generator import SimulSequenceGenerator
from fairseq.sequence_generator import SequenceGenerator

class SimulTranslation():
    def __init__(self, trans_model, tgt_dict, max_len):
        self.trans_model = trans_model
        self.max_len = max_len+2
        # self.simul_generator = SimulSequenceGenerator([trans_model], tgt_dict)
        # self.generator = SequenceGenerator([trans_model], tgt_dict, max_len=max_len)

        self.bos = tgt_dict.bos_index
        self.pad = tgt_dict.pad_index
        self.eos = tgt_dict.eos_index
        self.unk = tgt_dict.unk_index
        
    def embed_dim(self):
        return self.trans_model.decoder.embed_dim
    
    def data_check(self, sample):
        mask = (sample["src_tokens"] == 1)
        if all(mask.flatten()):
            tmp = sample["src_tokens"][mask]
            sample["src_tokens"] = tmp
            sample["src_lengths"][mask] -= 1
            
        return sample

    def init_sample(self, sample):
        net_input = self.data_check(sample["net_input"])
        self.device = device = net_input["src_tokens"].device
        bsz, s_t = net_input["src_tokens"].size()
        s_t += 1

        eos_tensor = torch.LongTensor([self.eos]).repeat(bsz).unsqueeze(-1).to(device)
        net_input["src_tokens"] = torch.concat((eos_tensor, net_input["src_tokens"]), -1).to(device)

        net_input["actions"] = torch.LongTensor([[]] * bsz).to(device)
        net_input["dist"] = torch.LongTensor([[]] * bsz).to(device)
        net_input["stream_src"] = torch.full((bsz, s_t), self.pad).long().to(device)
        net_input["stream_src"][:, 0] = self.eos
        
        net_input["prefix_tokens"] = torch.full((bsz, self.max_len), self.pad).long().to(device)
        net_input["prefix_tokens"][:, 0] = self.eos

        net_input["dec_states"] = []
        net_input["valid_actions"] = [] # remove action seq after finish read
        
        net_input["src_idx"] = torch.full((bsz,1), 0).long().to(device)
        net_input["tgt_idx"] = torch.full((bsz,1), 0).long().to(device)
        
        net_input["read_finish"] = torch.BoolTensor(bsz).fill_(False).to(device)
        net_input["decode_finish"] = torch.BoolTensor(bsz).fill_(False).to(device)
        
        net_input["delays"] = []

        init_action = torch.ones((bsz,1)).long().to(device) # first all read for 1 seg
        
        net_input = self.make_transitions(net_input, init_action)
        sample["net_input"] = net_input[0]

        return sample

    def make_transitions(self, sample, cur_action, detokens=None): # 0 write, 1 read
        src_idx = sample["src_idx"]
        tgt_idx = deepcopy(sample["tgt_idx"])

        # finish read -> action write (0)
        cur_action[sample["read_finish"]] = 0
        # read, update time step
        next_idx = cur_action + src_idx
        cur_action[sample["decode_finish"]] = 2
        sample["actions"] = torch.concat((sample["actions"], cur_action), 1) # B x timestep

        if detokens is not None:
            # write and not finish read not finish decoding, update dotoken
            write_mask = torch.logical_and((cur_action == 0).squeeze(), ~sample["decode_finish"])
            
            sample["tgt_idx"][write_mask] = tgt_idx[write_mask] + 1
            sample["delays"].append(sample["src_idx"].squeeze().tolist())
            for i, idx in enumerate(sample["tgt_idx"]):
                if write_mask[i]:
                    sample["prefix_tokens"][i, idx] = detokens.squeeze()[i]
            # sample["prefix_tokens"][write_mask, sample["tgt_idx"][write_mask]] = detokens.squeeze()[write_mask]
        
        stream_src = torch.gather(sample["src_tokens"], 1, next_idx)
        
        sample["stream_src"] = sample["stream_src"].scatter_(index=next_idx, dim=1, src=stream_src)
        sample["src_idx"] = next_idx
        
        read_finish = (sample["stream_src"] == sample["src_tokens"]).all(dim=-1) #
        sample["read_finish"] = read_finish = read_finish.squeeze()

        valid_actions = ~torch.logical_or(read_finish, sample["decode_finish"])
        sample["valid_actions"].append(valid_actions.tolist())

        tgt_idx = deepcopy(sample["tgt_idx"])
        tgt_idx[sample["tgt_idx"] == 0] = 1

        decode_finish = torch.gather(sample["prefix_tokens"], 1, tgt_idx) == self.eos
        decode_finish = torch.logical_or(decode_finish, (tgt_idx == self.max_len-1))
        # decode_finish = torch.logical_and(decode_finish.squeeze(), read_finish)
        sample["decode_finish"] = decode_finish.squeeze()

        if all(decode_finish):
            return sample, True

        return sample, False

    @torch.no_grad()
    def _get_dec_states(self, sample):
        # no beam
        with torch.no_grad():
            # encoder_states = self.trans_model.encoder(
            #                                 sample["stream_src"][:, 1:], 
            #                                 sample["src_idx"].squeeze()
            #                             )
            encoder_states = self.trans_model.encoder(
                                            sample["stream_src"], 
                                            sample["src_idx"].squeeze() + 1
                                        )
            
            dec_states, extra = self.trans_model.decoder.forward(
                prev_output_tokens=sample["prefix_tokens"],
                encoder_out=encoder_states,
            ) # B x max_len(prefix) x detokens
            
            attn = extra["attn"][0] # B x max_len x s_T
            inners = extra["inner_states"][-1] # max_len x B x embed_dim
            # inners = 

            tgt_idx = sample['tgt_idx'].squeeze()
            add_idx = torch.arange(tgt_idx.size(0)) * self.max_len
            tgt_idx = tgt_idx + add_idx.to(self.device)
            
            b, l, e = dec_states.size()
            last_states = dec_states.view(b*l, e)[tgt_idx]

            lprobs = self.trans_model.get_normalized_probs(
                [last_states.unsqueeze(1)], log_probs=True
            ) # lprobs : B x s_T x target_vocab

            # lprobs = self.trans_model.get_normalized_probs(
            #     [dec_states[:, -1:]], log_probs=True
            # ) # lprobs : B x s_T x target_vocab

            # lprobs[lprobs != lprobs] = torch.tensor(-math.inf).to(lprobs)

            indexs = lprobs.argmax(dim=-1).tolist() # B, 1

        torch.cuda.empty_cache()
        
        return inners, indexs

    def get_decoding(self, sample):
        dec_states, detokens = self._get_dec_states(sample)
        detokens = torch.LongTensor(detokens).to(self.device)

        return sample, dec_states, detokens

    def get_episode_result(self, sample):
        """
        sample
            origin data : src_tokens, src_lengths, prev_output_tokens
            made transitions : actions, strea_src, prefix_tokens, 
                                dec_states, valid_actions, dist
                                src_idx, tgt_idx, 
                                read_finish, decode_finsih
            Transitons = Timesteps = N
            
            - model_actions : (Tensor) B x N
            # - dist : probs, logits B x N
            - dec_states : (List) N x (max_len+2 x B x embed_dim)
            - prefix_tokens : (Tensor) B x max_len+2
            - valid_actions : (List) N+1 x B
        """

        valid_mask = torch.BoolTensor(sample["valid_actions"][1:]) # remove initial action
        valid_mask = valid_mask.permute(1,0) # B x N
        valid_idxs = torch.sum(valid_mask, 1).tolist() # B
        
        origin_dec = torch.FloatTensor(sample["dec_states"])
        # B, N, max_len, embed_dim
        # origin_dec = origin_dec.permute(2, 0, 1, 3).contiguous()
        
        # N, B, tokens -> B, N, tokens
        origin_dec = origin_dec.permute(1, 0, 2).contiguous()

        model_actions, dec_states = [], [] # each batch has diffrent timestep N
        for i, batch_valid_idx in enumerate(valid_idxs):
            batch_ac = sample["actions"][i, 1:batch_valid_idx].tolist()
            model_actions.append(torch.LongTensor(batch_ac).to(self.device))

            # batch_st = origin_dec[i, 1:batch_valid_idx, :, :]
            batch_st = origin_dec[i, 1:batch_valid_idx, :]
            dec_states.append(batch_st.to(self.device))
            
        delays = torch.LongTensor(sample["delays"]).permute(1,0).contiguous().to(self.device)

        return (model_actions, sample["actions"], dec_states, sample["prefix_tokens"], delays)