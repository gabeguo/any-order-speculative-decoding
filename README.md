# Reviving Any-Subset Autoregressive Models via Principled Parallel Sampling and Speculative Decoding

## Requirements

Python 3.10.16
```pip install -r requirements.txt```

TODO: verify this

You need at least one GPU with 16GB, such as an A4000.

The commands also generally work with sbatch, if you're using slurm.

## How to Run

### Getting Models

#### Train Yourself

TODO

#### Download Them

Download here: [https://huggingface.co/therealgabeguo/AO-ARM-generative/tree/main](https://huggingface.co/therealgabeguo/AO-ARM-generative/tree/main)

### Evaluations

#### Speculative Decoding

This will take samples from WikiText, blank out 95% of them, and complete them. It compares the *Any-Subset Speculative Decoding (ASSD)* to the sequential sampling algorithm. Given sufficiently large sample size, the distribution (as measured by perplexity and entropy) should be the same, while ASSD should provide ~10% decrease in NFEs and wall-clock time. All text outputs and metrics are saved.

```
bash _run_speculative_decoding.sh
```

#### Infilling Tasks

TODO
