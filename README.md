
![SAE-RAVEL](./figures/title1.png)

# Official Repository for "Evaluating Open-Source Sparse Autoencoders on Disentangling Factual Knowledge in GPT-2 Small"

![](https://img.shields.io/badge/Code-Python3.11-red)
![](https://img.shields.io/badge/Code-Pytorch-green)
![](https://img.shields.io/badge/Code-MIT_License-blue)


by [Maheep Chaudhary](https://maheepchaudhary.github.io) and [Atticus Geiger](https://atticusg.github.io).

Access our paper on [ArXiv]().

## About

We evaluate different open-source Sparse Autoencoders for GPT-2 small by different organisations, specifically by [OpenAI](https://github.com/openai/sparse_autoencoder), [Apollo Research](https://github.com/ApolloResearch/e2e_sae), and [Joseph Bloom](https://huggingface.co/jbloom/GPT2-Small-SAEs-Reformatted) on the [RAVEL](https://github.com/explanare/ravel) dataset.
We compare them against neurons and [DAS](https://arxiv.org/abs/2303.02536) based on how much they are able to disentangle the concept in neurons or latent space.

## Result

The below graphs shows the performance:

![](figures/continent.png)
![](figures/country.png)

## Setup:

The code is divided 


## Citation:
If you find this repository useful in your research, please consider citing our paper:


```
@misc{chaudhary2024evaluatingopensourcesparseautoencodersongpt2small,
      title={Evaluating Open-Source Sparse Autoencoders on Disentangling Factual Knowledge in GPT-2 Small}, 
      author={Maheep Chaudhary and Atticus Geiger},
      year={2024},
      eprint={},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={}, 
}
```