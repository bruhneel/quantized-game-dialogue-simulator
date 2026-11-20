# Generation speed comparison

Measured on Darwin 27.2.0 arm64, PyTorch 2.15.0, CPU with two threads.
Batch size one, context filled from the first step, 10 warm-up tokens, and median latency across 3 runs of 80 new tokens.

| Model | FP32 MAC/token | INT8 MAC/token | FLOP-equivalent/token | Simulated tok/s | Laptop tok/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| fp32 | 16,048,128 | 0 | 32,096,256 | 6.05 | 2319.41 |
| dynamic | 1,572,864 | 14,475,264 | 32,096,256 | 17.60 | 400.12 |
| static | 1,572,864 | 14,475,264 | 32,096,256 | 18.34 | 425.85 |

Device assumptions: 100M FP32 MAC/s, 400M INT8 MAC/s, 200 MB/s of weight bandwidth, and 80M activation elements/s. Compute, weight reads, and activation conversion times are added.
Dynamic quantization models two extra passes for range lookup. A full tensor footprint is charged per step.
Attention uses a dense MAC estimate, including masked positions. Two arithmetic operations per MAC are counted.
The simulator omits caching, cache reuse, kernel overhead and several elementwise ops.
On the laptop, both INT8 paths widen packed weights and activations to int32 for reference matmul.
