"""Official torchvision ResNet-18, eval/inference only, 512D after avgpool."""
import argparse
import gc
import hashlib
import os
import time
from pathlib import Path
from importlib.metadata import version

import numpy as np
from PIL import Image

from .common import DESCRIPTOR, ROOT, image_path, load_json, save_json, sha256


def build_extractor(requested='auto', threads=2):
    os.environ.setdefault('TORCH_HOME', str(ROOT / 'runtime/experiments/torch-cache'))
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import torch
    from torchvision.models import resnet18, ResNet18_Weights
    torch.manual_seed(0)
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    weights = ResNet18_Weights.IMAGENET1K_V1
    # Never fall back to weights=None: this experiment requires pretraining.
    network = resnet18(weights=weights, progress=True)
    assert network.fc.in_features == 512
    network.fc = torch.nn.Identity()
    network.requires_grad_(False)
    network.eval()
    preprocessing = weights.transforms()
    device = 'cpu'
    cuda_reason = None
    if requested != 'cpu':
        try:
            if not torch.cuda.is_available():
                raise RuntimeError('torch.cuda.is_available() is false')
            torch.ones((32,32),device='cuda').matmul(torch.ones((32,32),device='cuda'))
            torch.cuda.synchronize()
            device = 'cuda'
        except Exception as exc:
            if requested == 'cuda':
                raise RuntimeError('CUDA verification failed; use --device cpu.') from exc
            cuda_reason = str(exc)
    try:
        network.to(device)
        with torch.inference_mode():
            probe = network(torch.zeros(1,3,224,224,device=device))
            if tuple(probe.shape) != (1,512) or not torch.isfinite(probe).all():
                raise RuntimeError('Invalid 512D output from real convolution probe.')
        if device == 'cuda':
            torch.cuda.synchronize()
    except RuntimeError as exc:
        if requested != 'auto' or device == 'cpu':
            raise
        cuda_reason = str(exc)
        device = 'cpu'
        network.cpu()
        torch.cuda.empty_cache()
    checkpoint = Path(torch.hub.get_dir()) / 'checkpoints' / Path(weights.url).name
    checkpoint_hash = sha256(checkpoint)
    if not checkpoint_hash.startswith('f37072fd'):
        raise ValueError('Official pretrained checkpoint checksum is unexpected.')
    metadata = {'descriptor': DESCRIPTOR, 'weights': str(weights), 'weights_url': weights.url,
        'checkpoint_sha256': checkpoint_hash, 'device_requested': requested, 'device': device,
        'cuda_fallback_reason': cuda_reason, 'cuda_runtime': torch.version.cuda,
        'gpu': torch.cuda.get_device_name(0) if device == 'cuda' else None,
        'versions': {p: version(p) for p in ['torch','torchvision','Pillow','numpy']},
        'preprocessing': {'implementation': 'ResNet18_Weights.IMAGENET1K_V1.transforms()',
                          'repr': repr(preprocessing), 'input': 'PIL RGB', 'random_augmentation': False},
        'output': '512 float32 activations after global average pooling, fc=Identity; no extra normalization',
        'fine_tuning': False, 'amp': False, 'threads': threads}
    return network, preprocessing, device, metadata


def state_hash(network):
    digest = hashlib.sha256()
    for name,tensor in network.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def preprocess_row(row, preprocessing):
    path = image_path(row)
    with Image.open(path) as image:
        tensor = preprocessing(image.convert('RGB'))
    return tensor


def extract(run, requested, batch_size, threads, benchmark_samples):
    import torch
    run = Path(run)
    if (run / 'extraction.json').exists():
        raise ValueError('Extraction already exists; use a new run or validate it, never overwrite.')
    manifest = load_json(run / 'split.json')
    classes = load_json(run / 'classes.json')
    encoded = {name:i for i,name in enumerate(classes)}
    network, preprocessing, device, metadata = build_extractor(requested, threads)
    metadata['state_sha256_before'] = state_hash(network)
    metadata['batch_size'] = batch_size
    metadata['splits'] = {}
    started = time.perf_counter()
    for split in ['train','query']:
        rows = manifest[split]
        temp = run / f'{split}_resnet18.partial.npy'
        values = np.lib.format.open_memmap(temp,mode='w+',dtype=np.float32,shape=(len(rows),512))
        records = []
        split_start = time.perf_counter()
        for offset in range(0,len(rows),batch_size):
            group = rows[offset:offset+batch_size]
            batch = torch.stack([preprocess_row(row,preprocessing) for row in group])
            with torch.inference_mode():
                features = network(batch.to(device)).cpu().numpy().astype(np.float32,copy=False)
            if features.shape != (len(group),512) or not np.isfinite(features).all():
                raise ValueError('Invalid embeddings; no images will be silently dropped.')
            values[offset:offset+len(group)] = features
            for i,row in enumerate(group,offset):
                records.append({**row,'row':i,'label':encoded[row['category']], 'image_sha256':sha256(image_path(row))})
            if offset % (batch_size*25) == 0 or offset+len(group) == len(rows):
                print(f'{split}: {offset+len(group)}/{len(rows)} images, {time.perf_counter()-split_start:.1f}s, {device}',flush=True)
        values.flush(); del values
        final = run / f'{split}_resnet18.npy'
        temp.replace(final)
        np.save(run / f'{split}_labels.npy',np.array([encoded[row['category']] for row in rows],dtype=np.int64),allow_pickle=False)
        save_json(run / f'{split}_rows.json',records)
        metadata['splits'][split] = {'n':len(rows), 'dimensions':512,'seconds':time.perf_counter()-split_start,
                                    'vectors_sha256':sha256(final)}
    metadata['bulk_extraction_s'] = time.perf_counter()-started
    # Independent unit-query extraction latency, including image load/preprocess/H2D/D2H.
    timings = []
    indices = np.linspace(0,len(manifest['query'])-1,min(benchmark_samples,len(manifest['query'])),dtype=int)
    for index in indices:
        if device == 'cuda': torch.cuda.synchronize()
        start = time.perf_counter_ns()
        tensor = preprocess_row(manifest['query'][index],preprocessing).unsqueeze(0)
        with torch.inference_mode():
            network(tensor.to(device)).cpu().numpy()
        if device == 'cuda': torch.cuda.synchronize()
        timings.append((time.perf_counter_ns()-start)/1e6)
    metadata['unit_extraction_latency'] = {'samples':len(timings),'mean_ms':float(np.mean(timings)),
        'p95_ms':float(np.percentile(timings,95)), 'includes':'image read + official preprocessing + transfer + CNN + transfer back',
        'timing_indices':indices.tolist()}
    # Re-extract actual rows singly to detect batch/order/label mistakes.
    verification = []
    for split in ['train','query']:
        stored = np.load(run / f'{split}_resnet18.npy',mmap_mode='r')
        for index in sorted({0,len(stored)//2,len(stored)-1}):
            with torch.inference_mode():
                actual = network(preprocess_row(manifest[split][index],preprocessing).unsqueeze(0).to(device)).cpu().numpy()[0]
            difference = float(np.max(np.abs(stored[index]-actual)))
            if not np.allclose(stored[index],actual,rtol=2e-4,atol=2e-5):
                raise ValueError(f'Embedding does not correspond to its original image: {split}/{index}, error={difference}')
            verification.append({'split':split,'row':index,'max_absolute_error_batch_vs_single':difference})
    metadata['row_reextraction_checks'] = verification
    metadata['state_sha256_after'] = state_hash(network)
    if metadata['state_sha256_after'] != metadata['state_sha256_before']:
        raise ValueError('The frozen network state changed.')
    metadata['trainable_parameters'] = sum(p.numel() for p in network.parameters() if p.requires_grad)
    if device == 'cuda': metadata['peak_cuda_allocated_bytes'] = torch.cuda.max_memory_allocated()
    save_json(run / 'extraction.json',metadata)
    print(f'Embeddings complete; frozen state unchanged; extraction mean {np.mean(timings):.3f} ms/image.',flush=True)
    del network
    gc.collect()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    parser.add_argument('--batch-size',type=int,default=8)
    parser.add_argument('--threads',type=int,default=2)
    parser.add_argument('--benchmark-samples',type=int,default=128)
    args = parser.parse_args()
    if min(args.batch_size,args.threads,args.benchmark_samples) < 1: parser.error('Limits must be positive.')
    extract(args.run,args.device,args.batch_size,args.threads,args.benchmark_samples)


if __name__ == '__main__': main()
