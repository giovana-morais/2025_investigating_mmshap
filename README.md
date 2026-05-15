Source code for the paper "Investigating Modality Contribution in Audio LLMs for Music", currently under review. 
Interactive examples can be found in the [demo page](https://giovana-morais.github.io/2025_investigating_mmshap_demo/).

## Getting Started
---

This project was developed with Python 3.10. We have two requirements files, one
for each model (they have dependency conflicts).

### Dependencies

The two models we use are available on GitHub, and you can download them as
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
This repo should be self-contained, i.e., you should not need to download
any extra .csv file or things like this. Let me know if something is not
working!

---

## Cite

```
@inproceedings{morais2025investigatingmodalitycontributionaudio,
  author={Morais, Giovana and Fuentes, Magdalena},
  booktitle={ICASSP 2026 - 2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={Investigating Modality Contribution in Audio LLMs for Music}, 
  year={2026},
  volume={},
  number={},
  pages={3496-3500},
  doi={10.1109/ICASSP55912.2026.11463350}}
```
