# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os

import torch
import numpy as np

from fairseq import metrics, utils, checkpoint_utils, tasks
from fairseq.optim.amp_optimizer import AMPOptimizer

from fairseq.data import LanguagePairDataset
from fairseq.tasks import register_task, FairseqTask
from fairseq.tasks.translation import load_langpair_dataset

from examples.srd.modules.simul_translation import SimulTranslation
from examples.srd.data.epoch_shuffle import EpochShuffleDataset

EVAL_BLEU_ORDER = 4

logger = logging.getLogger(__name__)

# ---------------------------------------------------!@ task
# setup_task -> init -> build_model -> build_criterion -> load_dataset -> trainer -> train()

@register_task("rw_decision")
class RWDecisionTask(FairseqTask):

    @classmethod
    def add_args(cls, parser):
        parser.add_argument(
            "--translation-model", type=str, help="Translation Model ckpt Root",
        )
        parser.add_argument("--beam_steps", type=int)
        parser.add_argument("--beam_size", type=int)
        parser.add_argument("--use-latency-reward", type=bool, default=True)
        parser.add_argument("--latency-weight", type=float, default=-5.0)

    def __init__(self, args):
        super().__init__(args)
        self.beam_steps = args.beam_steps
        self.beam_size = args.beam_size
        self.max_len = args.max_len

        self.use_latency = args.use_latency_reward
        self.latency_weight = args.latency_weight

        self.load_translation_model(args.translation_model)

    @classmethod
    def setup_task(cls, args, **kwargs):
        # from datetime import datetime
        # file_name = datetime.now().isoformat()
        # args.tensorboard_logdir += f'/{file_name}'
        return cls(args)

    def load_translation_model(self, model_path):
        if not os.path.exists(model_path):
            raise IOError("Model file not found: {}".format(model_path))

        state = checkpoint_utils.load_checkpoint_to_cpu(model_path)

        task_args = state["cfg"]["task"]
        translation_task = tasks.setup_task(task_args)

        # build model for ensemble
        state["cfg"]["model"].load_pretrained_encoder_from = None
        state["cfg"]["model"].load_pretrained_decoder_from = None

        translation_model = translation_task.build_model(state["cfg"]["model"])
        translation_model.load_state_dict(state["model"], strict=True)
        translation_model.eval()
        translation_model.share_memory()
        translation_model.cuda()

        logger.info(f"[TRANSLATION_MODEL] : {translation_model.args.arch}")

        # Set dictionary
        tgt_dict = translation_task.target_dictionary
        src_dict = translation_task.source_dictionary

        # set Generator
        # self.simul_generator = SimulSequenceGenerator([translation_model], tgt_dict)

        self.simul_trans = SimulTranslation(translation_model, tgt_dict, self.max_len)
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.trans_args = task_args
        
        return translation_model


    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = utils.split_paths(self.trans_args.data)
        assert len(paths) > 0
        if split != self.trans_args.train_subset:
            # if not training data set, use the first shard for valid and test
            paths = paths[:1]
        data_path = paths[(epoch - 1) % len(paths)]

        # infer langcode
        src, tgt = self.trans_args.source_lang, self.trans_args.target_lang
        
        langpair_data = load_langpair_dataset(
            data_path,
            split,
            src,
            self.src_dict,
            tgt,
            self.tgt_dict,
            combine=combine,
            dataset_impl=self.trans_args.dataset_impl,
            upsample_primary=self.trans_args.upsample_primary,
            # left_pad_source=self.trans_args.left_pad_source,
            left_pad_source=False,
            left_pad_target=self.trans_args.left_pad_target,
            max_source_positions=self.trans_args.max_source_positions,
            max_target_positions=self.trans_args.max_target_positions,
            load_alignments=self.trans_args.load_alignments,
            truncate_source=self.trans_args.truncate_source,
            num_buckets=self.trans_args.num_batch_buckets,
            # num_buckets=1000,
            shuffle=(split != "test"),
            pad_to_multiple=self.trans_args.required_seq_len_multiple,
        )
        # self.datasets[split] = langpair_data
        self.datasets[split] = EpochShuffleDataset(langpair_data, num_samples=40000, seed=42)

    def build_dataset_for_inference(self, src_tokens, src_lengths, constraints=None):
        return LanguagePairDataset(
            src_tokens,
            src_lengths,
            self.source_dictionary,
            tgt_dict=self.target_dictionary,
            constraints=constraints,
        )

    def build_model(self, args, from_checkpoint=False):
        model = super().build_model(args, from_checkpoint)
        # if self.args.eval_bleu:
        #     detok_args = json.loads(self.args.eval_bleu_detok_args)
        #     self.tokenizer = encoders.build_tokenizer(
        #         Namespace(tokenizer=self.args.eval_bleu_detok, **detok_args)
        #     )

        #     gen_args = json.loads(self.args.eval_bleu_args)
        #     self.sequence_generator = self.build_generator(
        #         [model], Namespace(**gen_args)
        #     )
        return model
        
    def train_step(
        self, sample, model, criterion, optimizer, update_num, ignore_grad=False
    ):
        model.train()
        model.set_num_updates(update_num)
        sample = self.simul_trans.init_sample(sample)
        with torch.autograd.profiler.record_function("forward"):
            with torch.cuda.amp.autocast(enabled=(isinstance(optimizer, AMPOptimizer))):
                loss, sample_size, logging_output = criterion(model, sample, update_num)
        if ignore_grad:
            loss *= 0
        with torch.autograd.profiler.record_function("backward"):
            optimizer.backward(loss)
        return loss, sample_size, logging_output
    
    
    def valid_step(self, sample, model, criterion):
        loss, sample_size, logging_output = super().valid_step(sample, model, criterion)
        if self.args.eval_bleu:
            bleu = self._inference_with_bleu(self.sequence_generator, sample, model)
            logging_output["_bleu_sys_len"] = bleu.sys_len
            logging_output["_bleu_ref_len"] = bleu.ref_len
            # we split counts into separate entries so that they can be
            # summed efficiently across workers using fast-stat-sync
            assert len(bleu.counts) == EVAL_BLEU_ORDER
            for i in range(EVAL_BLEU_ORDER):
                logging_output["_bleu_counts_" + str(i)] = bleu.counts[i]
                logging_output["_bleu_totals_" + str(i)] = bleu.totals[i]
        return loss, sample_size, logging_output

    
    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)
        # avg_rewards = []
        # for log in logging_outputs:
        #     avg_rewards.append(log["avg_reward"].item())
        # metrics.log_scalar("avg_bleu", np.array(avg_rewards))
        
        avg_bleu = logging_outputs[0]["avg_bleu"].item() / 4
        avg_al = logging_outputs[0]["avg_al"].item() / 4
        read_ratio = logging_outputs[0]["read_ratio"].item() / 4
        metrics.log_scalar("avg_bleu", round(avg_bleu, 2))
        metrics.log_scalar("avg_al", round(avg_al, 2))
        metrics.log_scalar("read_ratio", round(read_ratio, 2))

        # if self.args.eval_bleu:

        #     def sum_logs(key):
        #         import torch

        #         result = sum(log.get(key, 0) for log in logging_outputs)
        #         if torch.is_tensor(result):
        #             result = result.cpu()
        #         return result

        #     counts, totals = [], []
        #     for i in range(EVAL_BLEU_ORDER):
        #         counts.append(sum_logs("_bleu_counts_" + str(i)))
        #         totals.append(sum_logs("_bleu_totals_" + str(i)))

        #     if max(totals) > 0:
        #         # log counts as numpy arrays -- log_scalar will sum them correctly
        #         metrics.log_scalar("_bleu_counts", np.array(counts))
        #         metrics.log_scalar("_bleu_totals", np.array(totals))
        #         metrics.log_scalar("_bleu_sys_len", sum_logs("_bleu_sys_len"))
        #         metrics.log_scalar("_bleu_ref_len", sum_logs("_bleu_ref_len"))

        #         def compute_bleu(meters):
        #             import inspect

        #             try:
        #                 from sacrebleu.metrics import BLEU

        #                 comp_bleu = BLEU.compute_bleu
        #             except ImportError:
        #                 # compatibility API for sacrebleu 1.x
        #                 import sacrebleu

        #                 comp_bleu = sacrebleu.compute_bleu

        #             fn_sig = inspect.getfullargspec(comp_bleu)[0]
        #             if "smooth_method" in fn_sig:
        #                 smooth = {"smooth_method": "exp"}
        #             else:
        #                 smooth = {"smooth": "exp"}
        #             bleu = comp_bleu(
        #                 correct=meters["_bleu_counts"].sum,
        #                 total=meters["_bleu_totals"].sum,
        #                 sys_len=int(meters["_bleu_sys_len"].sum),
        #                 ref_len=int(meters["_bleu_ref_len"].sum),
        #                 **smooth,
        #             )
        #             return round(bleu.score, 2)

        #         metrics.log_derived("bleu", compute_bleu)

# ---------------------------------------------------------------------------!
    def max_positions(self):
        """Return the max sentence length allowed by the task."""
        return (self.trans_args.max_source_positions, self.trans_args.max_target_positions)

    @property
    def source_dictionary(self):
        """Return the source :class:`~fairseq.data.Dictionary`."""
        return self.src_dict

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary`."""
        return self.tgt_dict

    def _inference_with_bleu(self, generator, sample, model):
        import sacrebleu

        def decode(toks, escape_unk=False):
            s = self.tgt_dict.string(
                toks.int().cpu(),
                self.cfg.eval_bleu_remove_bpe,
                unk_string=("UNKNOWNTOKENINREF" if escape_unk else "UNKNOWNTOKENINHYP"),
            )
            if self.tokenizer:
                s = self.tokenizer.decode(s)
            return s

        gen_out = self.inference_step(generator, [model], sample, prefix_tokens=None)
        hyps, refs = [], []
        for i in range(len(gen_out)):
            hyps.append(decode(gen_out[i][0]["tokens"]))
            refs.append(
                decode(
                    utils.strip_pad(sample["target"][i], self.tgt_dict.pad()),
                    escape_unk=True,  # don't count <unk> as matches to the hypo
                )
            )
        if self.cfg.eval_bleu_print_samples:
            logger.info("example hypothesis: " + hyps[0])
            logger.info("example reference: " + refs[0])
        if self.cfg.eval_tokenized_bleu:
            return sacrebleu.corpus_bleu(hyps, [refs], tokenize="none")
        else:
            return sacrebleu.corpus_bleu(hyps, [refs])