"""One explicit CPU smoke check against a saved embedding, no training."""
import json
import os
import time
import numpy as np
from backend.core import ROOT,DATA
from backend.descriptors import ResNetRegistry

def main():
    os.environ['METRICLAB_RESNET_DEVICE']='cpu'
    manifest=json.loads((DATA/'manifest.json').read_text(encoding='utf-8'))
    manager=ResNetRegistry(manifest)
    try:
        start=time.perf_counter()
        vector,info=manager.extract((ROOT/manifest['query'][0]['path']).read_bytes())
        seconds=time.perf_counter()-start
        expected=np.load(ROOT/'runtime/experiments/resnet18/20261118T195917Z-42d3213e/query_resnet18.npy',mmap_mode='r')[0]
        np.testing.assert_allclose(vector,expected,rtol=2e-4,atol=2e-5)
        assert info['device']=='cpu'
        result={**info,'dimensions':len(vector),'cold_extraction_s':seconds,'max_absolute_difference':float(np.max(np.abs(vector-expected))),'saved_embedding_matches':True}
        (ROOT/'runtime/resnet_cpu_check.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        print(json.dumps(result))
    finally: manager.close()

if __name__=='__main__': main()
