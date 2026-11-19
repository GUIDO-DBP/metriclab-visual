"""Persistent extractor in its isolated torch environment; JSON lines only."""
import base64
import io
import json
import os
import sys

def main():
    try:
        import torch
        from PIL import Image
        from experiments.resnet18.extract import build_extractor
        requested=os.getenv('METRICLAB_RESNET_DEVICE','auto')
        if requested not in {'auto','cpu','cuda'}: raise ValueError('METRICLAB_RESNET_DEVICE debe ser auto, cpu o cuda.')
        network,preprocess,device,_=build_extractor(requested,2)
        print(json.dumps({'ready':True,'device':device}),flush=True)
    except Exception as exc:
        print(json.dumps({'ready':False,'error':str(exc)}),flush=True); return
    for line in sys.stdin:
        try:
            request=json.loads(line)
            if request.get('close'): return
            raw=base64.b64decode(request['image'],validate=True)
            with Image.open(io.BytesIO(raw)) as image:
                if image.format not in {'JPEG','PNG'} or image.width*image.height>24_000_000: raise ValueError('Formato o tamaño de imagen no permitido.')
                tensor=preprocess(image.convert('RGB')).unsqueeze(0)
            with torch.inference_mode(): vector=network(tensor.to(device)).cpu().numpy()[0].tolist()
            print(json.dumps({'vector':vector,'device':device},allow_nan=False),flush=True)
        except Exception as exc: print(json.dumps({'error':str(exc)}),flush=True)

if __name__=='__main__': main()
