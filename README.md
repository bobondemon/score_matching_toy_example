# Score Matching Practicing in PyTorch

👏 Credit is made by this Repository [toy_gradlogp](https://github.com/Ending2015a/toy_gradlogp?ref=pythonawesome.com).
The author's blog is worth reading too, for example this article: [[Tuto] 夢魘の CUDA: 使用 Preconditioned Conjugate Gradient 輕鬆解決大型稀疏線性方程組](https://ending2015a.github.io/Ending2015a/52045/)

What I did is just modifying the data pipeline from tensorflow to pytorch's `dataloader` and re-factored the folder/config structure with `hydra` config tool.

> If your are interested in the theory behind Score Matching, maybe read the note I made:
[Estimation of Non-Normalized Statistical Models by Score Matching](https://bobondemon.github.io/2022/01/08/Estimation-of-Non-Normalized-Statistical-Models-by-Score-Matching/)

## How to run

Configuration is following `hydra` usage.

For the simplest case, you can modify `conf/train_config.yaml` then execute:

```bash
python train.py
```

## Results

Same as [toy_gradlogp](https://github.com/Ending2015a/toy_gradlogp?ref=pythonawesome.com)