# Model size comparison

Tensor bytes include persistent scales and zero points. File bytes include checkpoint overhead.
The inventory excludes runtime activations and temporary buffers.

| Model | Parameters | Weights (B) | Metadata (B) | Tensor total (B) | File (B) | Tensor ratio | File ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fp32 | 240,000 | 960,000 | 0 | 960,000 | 968,291 | 1.00× | 1.00× |
| dynamic | 240,000 | 281,472 | 7,120 | 288,592 | 301,041 | 3.33× | 3.22× |
| static | 240,000 | 281,472 | 7,192 | 288,664 | 304,445 | 3.33× | 3.18× |
