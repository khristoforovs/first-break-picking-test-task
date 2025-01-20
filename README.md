# First brake picking

## Implemented Algorithms

### STA-LTA

Implemented the variation of STA-LTA method called Energy-Ratio:

$$
er_i=\frac{\sum_{j=i}^{i+ne}x_j^2}{\sum_{j=i-ne}^{i}x_j^2} 
$$

Here $n$ is the window size in data-points and determined via the signal sampling rate 

### Neural Network Based Recognizer

Classical CNN-AE-like architecture:

```
Model(
  (enc): Encoder(
    (model): Sequential(
      (0): Conv2d(1, 32, kernel_size=(32, 8), stride=(4, 2))
      (1): ELU(alpha=1.0, inplace=True)
      (2): Conv2d(32, 64, kernel_size=(16, 4), stride=(4, 2))
      (3): ELU(alpha=1.0, inplace=True)
      (4): Conv2d(64, 64, kernel_size=(2, 2), stride=(2, 2))
      (5): ELU(alpha=1.0, inplace=True)
      (6): Conv2d(64, 128, kernel_size=(8, 2), stride=(2, 2))
      (7): ELU(alpha=1.0, inplace=True)
      (8): Flatten(start_dim=1, end_dim=-1)
    )
  )
  (dec): Decoder(
    (model): Sequential(
      (0): Unflatten(dim=1, unflattened_size=(128, 7, 2))
      (1): ConvTranspose2d(128, 64, kernel_size=(9, 3), stride=(2, 2))
      (2): ELU(alpha=1.0, inplace=True)
      (3): ConvTranspose2d(64, 64, kernel_size=(3, 2), stride=(2, 2))
      (4): ELU(alpha=1.0, inplace=True)
      (5): ConvTranspose2d(64, 32, kernel_size=(16, 4), stride=(4, 2))
      (6): ELU(alpha=1.0, inplace=True)
      (7): ConvTranspose2d(32, 1, kernel_size=(32, 8), stride=(4, 2))
      (8): Hardsigmoid()
    )
  )
)
```