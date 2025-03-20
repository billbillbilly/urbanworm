
## Step 1: install Ollama client
Please make sure [Ollama](https://ollama.com/) is installed before installing urban-worm

#### Linux
For Linux, users can also install ollama by running in the terminal:
```sh
curl -fsSL https://ollama.com/install.sh | sh
```

#### MAC
For MacOS, users can also install ollama using `brew`:
```sh
brew install ollama
```

To install `brew`, run in the terminal:
```sh
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```
After installing `brew`, you will see a following instruction:
```
==> Next steps:
- Run these commands in your terminal to add Homebrew to your PATH:
    echo >> /Users/yourusername/.bash_profile
    echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> /Users/yourusername/.bash_profile
    eval "$(/opt/homebrew/bin/brew shellenv)"
```

#### Windows
Windows users should directly install the [Ollama client](https://ollama.com/)

## Step 2: install GDAL first
For macOS, Linux, and Windows users, `gdal` may need to be installed at very begining using `conda`. 

#### install conda
Please download and install [Anaconda](https://www.anaconda.com/download/success) to use `conda`.

#### install GDAL
If the installation method above does not work, try to install with `conda`:
```sh
 conda install -c conda-forge gdal
```

Mac users may install `gdal` (if the installation method below does not work, try to install with conda):
```sh
 brew install gdal
```

#### Note
if you come across error like `conda command not found` when using `conda`, please refer following solutions to add Conda to the PATH:

- Linux: `export PATH=~/anaconda3/bin:$PATH` ([source](https://askubuntu.com/questions/1001865/conda-command-not-found))
- Mac: `export PATH="/home/username/miniconda/bin:$PATH"`. Please make sure to replace `/home/username/miniconda` with your actual path ([source](https://stackoverflow.com/questions/35246386/conda-command-not-found))
- Windows: Open Anaconda Prompt > Check Conda Installed Location: `where conda` > Open Advanced System Settings > Click on Environment Variables > Edit Path > Add New Path: 
``` 
 C:\Users\<username>\Anaconda3\Scripts
 C:\Users\<username>\Anaconda3
 C:\Users\<username>\Anaconda3\Library\bin
```
([source](https://stackoverflow.com/questions/44515769/conda-is-not-recognized-as-internal-or-external-command))

## Step 3: install urabn-worm from PyPi
The package urabnworm can be installed with `pip`:
```sh
pip install urban-worm 
```