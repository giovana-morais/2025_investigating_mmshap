general:
- [x] organize folder (what should be the right structure for my case?)
	- slurm jobs, bash scripts, python scripts, notebooks, figures and the
	  moedls we need to run
- [ ] create scripts to create fs, qo, and desc json files
- [ ] add requirements.txt for reproduction
- [ ] replace argparse for hydra
- [ ] add tests? (for what?)
- [ ] try to use library shap instead of version from mm-shap (what is the
  difference anyway?)

qwen audio:
- [x] (when possible) use qwenaudio functions instead of copying and pasting things
- [ ] rewrite `tokenizer.process_audio` to provide the actual waveform instead of the audio path
	- this way we avoid temporary files and the code will be much cleaner
	- should i edit this to receive multiple audios?

mmshap calculation:
- [x] separate basic functions in a `mmshap.py` module or something similar
- [ ] add tests to the  `compute_mm_score` function
