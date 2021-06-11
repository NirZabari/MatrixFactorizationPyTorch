# Matrix Factorization - PyTorch implementation
A minimal implementation of matrix factorization in PyTorch.
As item/user embeddings are not defined in the 
model (i.e not as nn.Embedding layer),
Modifications can be made for the implementation to scale up.

## Running
* install the conda environment from [environment.yml](environment.yml).
* activate the environment and run: 
```bash
python MF.py --lr 0.01 --latent_dimension 10 --batch_size 256 --num_epochs 30
```


## License
MIT