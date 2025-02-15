# Transformer
A repo that implements the "Attention Is All You Need" paper from scratch using PyTorch 


note if you want to run the train.py and are using an M1 device run the following cmd first
```cmd
conda create python=3.9 -n transformer -y
pip install -r requirements.txt
python research/train.py
```


## Citation
```
@misc{vaswani2023attentionneed,
      title={Attention Is All You Need}, 
      author={Ashish Vaswani and Noam Shazeer and Niki Parmar and Jakob Uszkoreit and Llion Jones and Aidan N. Gomez and Lukasz Kaiser and Illia Polosukhin},
      year={2023},
      eprint={1706.03762},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/1706.03762}, 
}
```
