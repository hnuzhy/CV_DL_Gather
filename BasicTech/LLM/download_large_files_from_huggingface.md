# Download Large Files from HuggingFace

1) The official website: https://huggingface.co/
2) The mirror website: https://hf-mirror.com/
3) The referred blog: [如何快速下载huggingface模型——全方法总结](https://www.yunqiic.com/2024/01/04/%E5%A6%82%E4%BD%95%E5%BF%AB%E9%80%9F%E4%B8%8B%E8%BD%BDhuggingface%E6%A8%A1%E5%9E%8B-%E5%85%A8%E6%96%B9%E6%B3%95%E6%80%BB%E7%BB%93/)

* Way 1:
```
# Directly download them from the official repo.
https://huggingface.co/bigscience/bloom-560m 
```

* Way 2:
```bash
# for ubuntu using command lines
$ pip install -U huggingface_hub hf-transfer
$ export HF_ENDPOINT="https://hf-mirror.com"
$ export HF_HUB_ENABLE_HF_TRANSFER=1
$ huggingface-cli download --resume-download bigscience/bloom-560m --local-dir bloom-560m
```
