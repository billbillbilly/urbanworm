
### 1 install the package
The package `urban-worm` can be installed with `pip`:
```sh
pip install urban-worm
```

### 2 Inference with llama.cpp
To run more pre-quantized models with vision capabilities, please install pre-built version of llama.cpp:
``` sh
# Windows
winget install llama.cpp

# Mac and Linux
brew install llama.cpp
```
More information about the installation 
[here](https://github.com/ggml-org/llama.cpp/blob/master/docs/install.md)

More GGUF models can be found at the Hugging Face pages 
[here](https://huggingface.co/collections/ggml-org/multimodal-ggufs-68244e01ff1f39e5bebeeedc) and [here](https://huggingface.co/models?pipeline_tag=image-text-to-text&sort=trending&search=gguf)

### 3 Inference with Ollama client

Please make sure [Ollama](https://ollama.com/) is installed before using urban-worm if you plan to rely on Ollama

For Linux, users can also install ollama by running in the terminal:
```sh
curl -fsSL https://ollama.com/install.sh | sh
```
For MacOS, users can also install ollama using `brew`:
```sh
brew install ollama
```

To install `brew`, run in the terminal:
```sh
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

Windows users should directly install the [Ollama client](https://ollama.com/)

To install the development version from this repo:
``` sh
pip install -e git+https://github.com/billbillbilly/urbanworm.git#egg=urban-worm
```