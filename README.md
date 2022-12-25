## SR-KU Mid-term report repository

Base MT model, Training Dataset, Trained Agent model
[Download link](https://www.notion.so/Back-Data-29ce49f76aab483eb3e4890f220247fa)

### Dataset
- Train Dataset : IWSLT14(en-de), MuST-C v2.0 (en-de)
- Test Dataset : MuST-C tst-COMMON v2.0 (en-de)

### Base MT model based on fairseq

```
fairseq-train \
    data-bin/train_data.en-de \
    --arch transformer_iwslt_de_en --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 4096

fairseq-generate data-bin/train_data.en-de \
    --path checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe
```

### Simuleval

```
simuleval \
    --source test-mustc-tst.en --target test-mustc-tst.de \
    --data-bin data-bin/train_data.en-de \
    --eval-latency-unit word \
    --agent /fairseq/examples/srd/eval/simul_t2t_la_n.py \ 
    # --agent /fairseq/examples/srd/eval/simul_t2t_stride_n.py \ 
    # --agent /fairseq/examples/srd/eval/simul_t2t_waitk.py \ 
    --model-path checkpoint_best.pt \
    --max-len 200 --output output \
    --wait_k 5 --la_n 2 --beam_size 8 \
    # --wait_k 5 --stride_n 2 --beam_size 8 \
    # --wait_k 5 \
    --scores --gpu
```




