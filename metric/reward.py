import torch
import sacrebleu
import numpy as np
from examples.srd.metric import latency

def remove_prefix(hypos, tgt_dict):
    BPE_SEP = '@@'
    puncs = ['.', ',', '?', '!']
    eos_word = tgt_dict.eos_word
    pad_word = tgt_dict.pad_word

    hypos = hypos.replace(pad_word, '')

    result, tmp = [], ''
    for token in hypos.split():
        if token.endswith(BPE_SEP):
            tmp += token.replace(BPE_SEP, '')
        else:
            if token == eos_word:
                result.append(tmp)
                break
            if tmp and token not in puncs:
                result.append(tmp + token)
                tmp = ''
            elif tmp and token in puncs:
                result.extend([tmp] + [token])
                tmp = ''
            else:
                result.append(token)
    result = ' '.join(result)
    return result

def get_reward(hypos, refs, tgt_dict, with_latency, delays, actions, src_len, tgt_len):
    hypos = hypos[:, 1:].cpu().detach().numpy() # except <eos>
    refs = refs.cpu().numpy()
    actions = actions[:, 1:] # except first READ for first src token

    bleu_rewards, check = [], []
    latency_rewards = []
    for hypo, ref, delay, action, s, t in zip(hypos, refs, delays, actions, src_len, tgt_len):
        sent_hypo = remove_prefix(tgt_dict.string(hypo), tgt_dict)
        sent_ref = remove_prefix(tgt_dict.string(ref), tgt_dict)
        
        sent_bleu = sacrebleu.sentence_bleu([sent_hypo], [sent_ref])
        bleu_rewards.append(sent_bleu.score)
        
        check.append(f'H : {sent_hypo}')
        check.append(f'R : {sent_ref}')

        if len(np.where(ref==1)[0]) != 0 :
            ref_len = torch.tensor([np.where(ref==1)[0][0]]).to(s.device)-1 # except <eos>
        else :
            ref_len = torch.tensor([len(ref)]).to(s.device)-1
        if len(np.where(hypo==1)[0]) != 0 :
            t = torch.tensor([np.where(hypo==1)[0][0]]).to(t.device)-1
        
        if with_latency:
            delay = delay[(action == 0)]
            al = latency.length_adaptive_average_lagging(delay.unsqueeze(0), 
                                                         s.unsqueeze(0), 
                                                         t.unsqueeze(0),
                                                         ref_lens=ref_len)
            # al = latency.AverageLagging(delay.unsqueeze(0), s.unsqueeze(0), t.unsqueeze(0))
            if al.item() < 0 : 
                al = torch.tensor([30.0],device=al.device) 
            latency_rewards.append(al.item())
            
    return bleu_rewards, check, latency_rewards
