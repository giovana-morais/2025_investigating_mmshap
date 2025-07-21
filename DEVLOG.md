general:
- [x] organize folder (what should be the right structure for my case?)
	- slurm jobs, bash scripts, python scripts, notebooks, figures and the
	  models we need to run
- [ ] create scripts to create fs, qo, and desc json files
- [ ] add requirements.txt for reproduction
- [ ] replace argparse for hydra
- [ ] try to use library shap instead of version from mm-shap (what is the
  difference anyway?)
- [ ] create tests for data scripts

qwen audio:
- [x] (when possible) use qwenaudio functions instead of copying and pasting things
- [x] rewrite `tokenizer.process_audio` to provide the actual waveform instead of the audio path
	- this way we avoid temporary files and the code will be much cleaner
	- should i edit this to receive multiple audios? (probably not)
	- this was done by creating a custom tokenizer that inherits from
	  qwen tokenizer and add two new methods to it
- [x] add requirements for qwen
- [ ] refactor qwenaudio experiment to use process_audio_no_url instead of
  process_audio
- [ ] write some sanity check tests for this. we need to ensure that the
  processed audio is the masked one!

mmshap calculation:
- [x] separate basic functions in a `mmshap.py` module or something similar
- [ ] add tests to the  `compute_mm_score` function
