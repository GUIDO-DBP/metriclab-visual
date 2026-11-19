"""Create a reversible descriptor registry, preserving all model artifacts."""
import argparse
from pathlib import Path
from experiments.resnet18.common import ROOT,load_json,save_json,sha256
from backend.descriptors import RESNET_ID

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--comparison',type=Path,required=True); parser.add_argument('--lightgbm',type=Path,required=True)
    args=parser.parse_args(); comparison=args.comparison.resolve(); tuning=args.lightgbm.resolve()
    catalog=load_json(comparison/'model_catalog_proposal.json'); result=load_json(tuning/'results.json')
    if not result['improved_in_training_confirmation']: raise ValueError('Do not activate an unconfirmed LightGBM change.')
    models={}
    from backend.modeling import MODEL_NAMES
    for key,entry in catalog['models'].items():
        old=load_json(comparison/'final'/key/'results.json')
        path=ROOT/entry['model_path']; metrics=comparison/'final'/key/'results.json'
        if key=='lightgbm': path=ROOT/result['model_path']; metrics=tuning/'results.json'
        models[key]={'path':path.relative_to(ROOT).as_posix(),'sha256':sha256(path),
            'name':MODEL_NAMES[key],'metrics_path':metrics.relative_to(ROOT).as_posix(),
            'metadata':{'params':result['parameters'] if key=='lightgbm' else entry['parameters'],
                        'source_run':tuning.name if key=='lightgbm' else (path.parent.parent.name if key in {'forest','xgboost'} else comparison.name)}}
    checkpoint=ROOT/'runtime/experiments/torch-cache/hub/checkpoints/resnet18-f37072fd.pth'
    cfg={'descriptor':RESNET_ID,'dimensions':512,'run_id':tuning.name,'split_fingerprint':catalog['split_fingerprint'],
        'classes':catalog['classes'],'models':models,'checkpoint_path':checkpoint.relative_to(ROOT).as_posix(),
        'checkpoint_sha256':sha256(checkpoint),'versions':load_json(tuning/'protocol.json')['versions'],
        'comparison_results':(comparison/'results.json').relative_to(ROOT).as_posix(),
        'tuning_protocol':(tuning/'protocol.json').relative_to(ROOT).as_posix(),
        'tuning_confirmation':(tuning/'internal_confirmation.json').relative_to(ROOT).as_posix(),
        'tuning_selection':(tuning/'selection.json').relative_to(ROOT).as_posix(),
        'previous_lightgbm':catalog['models']['lightgbm']}
    target=ROOT/'runtime/descriptor_registry.json'
    if target.exists():
        backup=target.with_name('descriptor_registry.previous.'+tuning.name+'.json')
        if not backup.exists(): backup.write_bytes(target.read_bytes())
    temp=target.with_suffix('.tmp'); save_json(temp,cfg); temp.replace(target)
    print('ResNet registry configured. HSV remains default; METRICLAB_ENABLE_RESNET=0 disables ResNet.')

if __name__=='__main__': main()
