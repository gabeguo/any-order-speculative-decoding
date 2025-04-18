# Reviving Any-Subset Autoregressive Models via Principled Parallel Sampling and Speculative Decoding

**By [Gabe Guo](https://gabeguo.github.io/) and [Stefano Ermon](https://cs.stanford.edu/~ermon/).**

*In arbitrary-order language models, it is an open question how to sample tokens in parallel from the correct joint distribution. With [discrete diffusion models](https://arxiv.org/abs/2310.16834), the more tokens they generate in parallel, the less their predicted distributions adhere to the originally learned data distribution, as they rely on a conditional independence assumption that only works with infinitesimally small timesteps. We find that a different class of models, [**Any-Subset Autoregressive Models (AS-ARMs)**](https://github.com/AndyShih12/mac), holds the solution. As implied by the name, AS-ARMs can generate tokens in any order, and in parallel. Moreover, AS-ARMs support parallelized joint probability density estimation, allowing them to correct their own parallel-generated token distributions, via our **Any-Subset Speculative Decoding (ASSD)** algorithm. ASSD provably enables generation of tokens from the correct joint distribution, with the number of neural network calls upper bounded by the number of tokens predicted.
We empirically verify that ASSD speeds up language generation, without sacrificing quality. Furthermore, we provide a mathematically justified scheme for training AS-ARMs for generation, and show that AS-ARMs achieve **state-of-the-art performance among sub-200M parameter models** on infilling benchmark tasks, and nearly **match the performance of models 50X larger** on code generation. 
Our theoretical and empirical results indicate that the once-forgotten AS-ARMs are a promising direction of language modeling.*

<img src="asarm.gif" width=400>

## Requirements

Python 3.10.16
```pip install -r requirements.txt```

You also need to install [https://github.com/openai/human-eval-infilling](https://github.com/openai/human-eval-infilling)

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

#### *NLP*

For the human language infills, first download [ROCStories](https://github.com/HKUNLP/DiffuLLaMA/blob/main/evaluation/evaluation/cloze_test_val__spring2016.csv), and put it in ```eval_datasets``` folder. Then,

```
bash _run_nlp_infill_eval.sh
```

#### *Code*

For the code infills, run:

```
bash _run_code_infill_eval.sh
```

You will need to use the evaluation suite here to get metrics: [https://github.com/openai/human-eval-infilling](https://github.com/openai/human-eval-infilling)

## Citations

If you find the code and/or ideas in this repository helpful, please cite
```
@article{guo2025_asarm,
  title={Reviving Any-Subset Autoregressive Models via Principled Parallel Sampling and Speculative Decoding},
  author={Guo, Gabe and Ermon, Stefano},
  journal={arXiv preprint},
  year={2025}
}
```
