# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import math
from dataclasses import dataclass, field
import torch
from omegaconf import II
import numpy as np
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass

from examples.srd.metric import reward

## TODO :
## Change it for RL
@dataclass
class PolicyGradientCriterionConfig(FairseqDataclass):
    label_smoothing: float = field(
        default=0.0,
        metadata={"help": "epsilon for label smoothing, 0 means no label smoothing"},
    )
    report_accuracy: bool = field(
        default=False,
        metadata={"help": "report accuracy metric"},
    )
    ignore_prefix_size: int = field(
        default=0,
        metadata={"help": "Ignore first N tokens"},
    )
    sentence_avg: bool = II("optimization.sentence_avg")
@register_criterion(
    "policy_gradient", dataclass=PolicyGradientCriterionConfig
)


class PolicyGradientCriterion(FairseqCriterion):
    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,
        ignore_prefix_size=0,
        report_accuracy=False,
    ):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing
        self.ignore_prefix_size = ignore_prefix_size
        self.report_accuracy = report_accuracy
        self.gamma = 0.99
        
    def get_discounted_rewards(self, rewards, actions) :
        discounted_rewards = []
        for batch_actions, batch_rewards in zip(actions, rewards):
            tmp_rewards = []
            running_add = 0.
            for t in range(len(batch_actions)):
                running_add = batch_rewards * (self.gamma ** t)
                tmp_rewards.insert(0, running_add)
            discounted_rewards.append(torch.FloatTensor(tmp_rewards).to(actions[0].device))

        return discounted_rewards

    def forward(self, model, sample, reduce=True):
        # Decoder OutPut : B x N x max_len+2 x embed_dim
        # Action Seqeunce : B X N

        outputs = model(sample["net_input"])
        actions, whole_actions, states, hypos, delays = outputs
        
        src_len = sample["net_input"]["src_lengths"].unsqueeze(1).float()
        tgt_len = torch.FloatTensor([len(t) for t  in sample["target"]]).unsqueeze(1).to(src_len)
        
        bleu_rewards, check, latency_rewards = reward.get_reward(hypos=hypos,
                                                                 refs=sample["target"],
                                                                 tgt_dict=self.task.tgt_dict,
                                                                 with_latency=self.task.use_latency,
                                                                 delays=delays,
                                                                 actions=whole_actions,
                                                                 src_len=src_len,
                                                                 tgt_len=tgt_len,
                                                                 )

        # normalize
        logging_bleu = bleu_rewards
        logging_al = latency_rewards
        bleu_rewards = (bleu_rewards - np.mean(bleu_rewards)) / (np.std(bleu_rewards) + 1e-5)
        
        if len(latency_rewards) > 0:
            latency_rewards = (latency_rewards - np.mean(latency_rewards)) / (np.std(latency_rewards) + 1e-5)
            rewards = []
            for bleu, latency in zip(bleu_rewards, latency_rewards):
                rewards.append(self.task.latency_weight * latency + bleu)
            # rewards = self.task.latency_weight * latency_reward + bleu_reward
        else :
            rewards = bleu_rewards
        
        # normalized_rewards = torch.from_numpy((rewards - np.mean(rewards)) / np.std(rewards)).to(model.device).squeeze()
        discounted_rewards = self.get_discounted_rewards(rewards, actions)
        # discounted_rewards = self.get_discounted_rewards(normalized_rewards, actions)
        # discounted_rewards = self.get_discounted_rewards(rewards, actions) # [Tensor, Tensor..]
        
        dist, _ = model.get_action(torch.concat(states, dim=0))
        all_actions = torch.concat(actions).float()
        all_rewards = torch.concat(discounted_rewards)
        loss = -(dist.log_prob(all_actions) * all_rewards).mean()

        # For logging
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
            "avg_bleu" : torch.mean(torch.Tensor(logging_bleu)).item(),
            "avg_al": torch.mean(torch.Tensor(logging_al)).item(),
            "read_ratio": (torch.sum(all_actions).item() / all_actions.size(0)) * 100,
            #"average_reward" : reward.mean().data
        }
        return loss, sample_size, logging_output

    ## TODO :
    ## Modify for RL
    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        # nll_loss_sum = sum(log.get("average_bleu", 0) for log in logging_outputs)
        # ntokens = sum(log.get("average_reward", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        # total = utils.item(sum(log.get("total", 0) for log in logging_outputs))
        # if total > 0:
        #     metrics.log_scalar("total", total)
        #     n_correct = utils.item(
        #         sum(log.get("n_correct", 0) for log in logging_outputs)
        #     )
        #     metrics.log_scalar("n_correct", n_correct)
        #     metrics.log_derived(
        #         "accuracy",
        #         lambda meters: round(
        #             meters["n_correct"].sum * 100.0 / meters["total"].sum, 3
        #         )
        #         if meters["total"].sum > 0
        #         else float("nan"),
        #     )
        
    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True