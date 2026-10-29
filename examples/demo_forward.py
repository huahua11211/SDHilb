"""Initialize SDHilb and run a forward pass on an example input."""
from pathlib import Path
import sys
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.SDHilb import Model


def main():
    torch.manual_seed(2026)
    args = SimpleNamespace(
        seq_len=336,
        pred_len=96,
        enc_in=7,
        kernel_sizes="15_31_63_127",
        conv_trainable=True,
    )
    model = Model(args)
    model.eval()
    x = torch.randn(2, args.seq_len, args.enc_in)
    with torch.no_grad():
        prediction = model(x)
    print(f"Input shape:  {tuple(x.shape)}")
    print(f"Output shape: {tuple(prediction.shape)}")


if __name__ == "__main__":
    main()
