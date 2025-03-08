#!/bin/bash

# if uv is not installed, install it, otherwise just say uv is installed
if ! command -v uv &> /dev/null; then
    echo "uv is not installed. Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh 
else
    echo "uv is installed"
fi

echo 'eval "$(uv generate-shell-completion bash)"' >> ~/.bashrc

uv venv --python 3.11
uv pip install -r requirements.txt