"""Measure unchanged app extractor in its own existing Python environment."""
import argparse
import time
from pathlib import Path

import numpy as np
from backend.core import describe_bytes
from .common import image_path, load_json, save_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--benchmark-samples',type=int,default=128)
    args = parser.parse_args()
    rows = load_json(args.run / 'split.json')['query']
    indices = np.linspace(0,len(rows)-1,min(args.benchmark_samples,len(rows)),dtype=int)
    describe_bytes(image_path(rows[0]).read_bytes())
    timings = []
    for index in indices:
        start = time.perf_counter_ns()
        describe_bytes(image_path(rows[index]).read_bytes())
        timings.append((time.perf_counter_ns()-start)/1e6)
    save_json(args.run / 'hsv_extraction_latency.json',{'samples':len(timings),'mean_ms':float(np.mean(timings)),
        'p95_ms':float(np.percentile(timings,95)),'includes':'image read + original app HSV/Sobel descriptor',
        'timing_indices':indices.tolist()})


if __name__ == '__main__': main()
