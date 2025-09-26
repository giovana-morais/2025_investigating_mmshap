# 2025_measuring_mmshap
Source code for the paper "Investigating Modality Contribution in Audio LLMs for Music", currently under review.

## Getting Started
---

This project was developed with Python 3.10. We have two requirements files, one
for each models (they have dependencies conflict).

### Dependencies

The two models we use are available on GitHub and you can download them as
submodules:

```bash
git submodule add https://github.com/QwenLM/Qwen-Audio --name Qwen_Audio
git submodule add https://github.com/ncsrsadhana/MULLaMA
```

For MULLaMA, there are additional checkpoints that you need to download. Please
refer to its
[documentation](https://github.com/ncsrsadhana/MULLaMA?tab=readme-ov-file#mu-llama-demo)
.

### Installing
Install the module via `pip`
```
pip install -e .
```

### Executing

You can run the experiments either via `sh` or `slurm`. The scripts are in the
folder `scripts`. All the json files for the experiments are provided in the
`data` folder.
This repo should be self-contained, i.e. you should not need to download
any extra .csv file or things like this. Let me know if something is not
working!

---

## Cite

```
@misc{morais2025investigatingmodalitycontributionaudio,
	title={Investigating Modality Contribution in Audio LLMs for Music},
	author={Giovana Morais and Magdalena Fuentes},
	year={2025},
	eprint={2509.20641},
	archivePrefix={arXiv},
	primaryClass={cs.LG},
	url={https://arxiv.org/abs/2509.20641},
}
```
