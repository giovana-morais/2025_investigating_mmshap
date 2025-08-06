#!/bin/sh -e

# change model name
mv src/models/MULLaMA/MU-LLaMA src/models/MULLaMA/MU_LLaMA

# make it a module so we can import it
touch src/models/MULLaMA/__init__.py
touch src/models/MULLaMA/MU_LLaMA/__init__.py
