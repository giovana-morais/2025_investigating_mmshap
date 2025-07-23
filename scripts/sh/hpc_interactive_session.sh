# very lazy way of requesting an interactive session
#!/bin/bash
srun \
	-t2:00:00 -c 4 \
	--mem=64GB \
	--gres=gpu:1 \
	--account=pr_230_tandon_priority \
	--pty /bin/bash
	# --gres=gpu:rtx8000:1 \
