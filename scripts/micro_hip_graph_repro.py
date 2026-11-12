import argparse
import time

import torch


parser = argparse.ArgumentParser()
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--flat", action="store_true")
parser.add_argument("--float32", action="store_true")
parser.add_argument("--out-of-place", action="store_true")
args = parser.parse_args()
dtype = torch.float32 if args.float32 else torch.bfloat16

layers = [
    torch.nn.Conv2d(32, 32, 3, padding=1, groups=32, bias=False),
    torch.nn.BatchNorm2d(32),
    torch.nn.ReLU(inplace=not args.out_of_place),
    torch.nn.Conv2d(32, 16, 1, bias=False),
    torch.nn.BatchNorm2d(16),
]
last = torch.nn.Conv2d(16, 48, 1, bias=False)
layers.append(last if args.flat else torch.nn.Sequential(last))
model = torch.nn.Sequential(*layers).eval().to("cuda", dtype=dtype)
x = torch.randn(args.batch_size, 32, 112, 112, device="cuda", dtype=dtype)

with torch.inference_mode():
    for i in range(3):
        start = time.perf_counter()
        model(x)
        torch.cuda.synchronize()
        print(f"warmup {i}: {(time.perf_counter() - start) * 1000:.3f} ms", flush=True)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = model(x)
    torch.cuda.synchronize()
    print("capture complete", flush=True)

    start = time.perf_counter()
    graph.replay()
    torch.cuda.synchronize()
    print(f"replay: {(time.perf_counter() - start) * 1000:.3f} ms", flush=True)
