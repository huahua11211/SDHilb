# SDHilb

**SDHilb: Schrödinger Dynamics-Guided Neural Network with Adaptive Multi-Scale Hilbert Transform for Time Series Forecasting**

Accepted at **NeurIPS 2026**.

This repository provides the core implementation of SDHilb and a forward-pass example.

## Model

SDHilb constructs multi-scale Hilbert representations and combines real and quadrature components for time series forecasting. The Hilbert kernels use Hann-windowed initialization and support learning through backpropagation. Channel-wise scale weights are computed with Gumbel-Softmax.

The model accepts inputs of shape `[batch, input_length, channels]` and returns forecasts of shape `[batch, prediction_length, channels]`.

## Installation

The example was tested on CPU with Python 3.11.3, PyTorch 2.3.0, and NumPy 1.26.4.

```bash
python -m pip install -r requirements.txt
```

## Example

Run from the repository root:

```bash
python examples/demo_forward.py
```

Expected output:

```text
Input shape:  (2, 336, 7)
Output shape: (2, 96, 7)
```

## Configuration

Pass a configuration object to `Model`:

| Field | Description | Example |
| --- | --- | --- |
| `seq_len` | Input sequence length | `336` |
| `pred_len` | Forecast horizon | `96` |
| `enc_in` | Number of input channels | `7` |
| `kernel_sizes` | Underscore-separated filter widths | `"15_31_63_127"` |
| `conv_trainable` | Whether to optimize the Hilbert kernels; defaults to `True` | `True` |

## Files

- `models/SDHilb.py`: model implementation.
- `examples/demo_forward.py`: initialization and forward-pass example.
- `requirements.txt`: tested dependencies.

## License

See [LICENSE](LICENSE) for the Apache License 2.0 terms.
