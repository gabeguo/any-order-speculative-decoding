# Reviving Any-Subset Autoregressive Models via Principled Parallel Sampling and Speculative Decoding

## Requirements

Python 3.10.16
```pip install -r requirements.txt```

TODO: verify this

You need at least one GPU with 16GB, such as an A4000.

The commands also generally work with sbatch, if you're using slurm.

## How to Run

You'll need to change the directories, batch sizes, etc. 

### Getting Models

#### Train Yourself

An A4000 (16GB) should support a batch size of 6, A5000 (24GB) should support a batch size of 8, A6000 (48GB) should support a batch size of 16. Gradient accumulation and distributed training are your friends.

In the .sh script, set the dataset to "bigcode/starcoderdata" or "openwebtext".

```
bash _run_training.sh
```

#### Download Them

Training takes a long time. We're nice, so you can freely download our models here: [https://huggingface.co/therealgabeguo/AO-ARM-generative/tree/main](https://huggingface.co/therealgabeguo/AO-ARM-generative/tree/main)

### Evaluations

#### Speculative Decoding

This will take samples from WikiText, blank out 95% of them, and complete them. It compares the *Any-Subset Speculative Decoding (ASSD)* to the sequential sampling algorithm. Given sufficiently large sample size, the distribution (as measured by perplexity and entropy) should be the same, while ASSD should provide ~10% decrease in NFEs and wall-clock time. All text outputs and metrics are saved.

```
bash _run_speculative_decoding.sh
```

#### Infilling Tasks

TODO
