Refer to https://huggingface.co/unsloth/DeepSeek-R1-GGUF

```bash

git clone https://github.com/ggerganov/llama.cpp

conda install -c conda-forge curl

export CURL_LIBRARY=/data/user_data/jiaruil5/miniconda3/envs/tinyzero/lib/libcurl.so
export CURL_INCLUDE_DIR=/data/user_data/jiaruil5/miniconda3/envs/tinyzero/include

cmake llama.cpp -B llama.cpp/build -DGGML_CUDA=ON -DLLAMA_CURL=ON -DBUILD_SHARED_LIBS=OFF -DCURL_INCLUDE_DIR=$CURL_INCLUDE_DIR -DCURL_LIBRARY=$CURL_LIBRARY

cmake --build llama.cpp/build --config Release -j --clean-first --target llama-quantize llama-cli llama-gguf-split

cp llama.cpp/build/bin/llama-* llama.cpp
```

Then run `model_download.py`.

```bash
./llama.cpp/build/bin/llama-cli \
    --model /compute/babel-4-33/jiaruil5/.cache/DeepSeek-R1-Distill-Llama-70B-Q8_0/DeepSeek-R1-Distill-Llama-70B-Q8_0-00001-of-00002.gguf \
    --cache-type-k q8_0 \
    --threads 16 \
    --prompt '<｜User｜>What is 1+1?<｜Assistant｜>' \
    -no-cnv

```


# Training

```bash
pip install "unsloth[cu121-torch240] @ git+https://github.com/unslothai/unsloth.git"


```