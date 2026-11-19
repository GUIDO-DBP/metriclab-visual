"""Build versioned release assets from a dependency allowlist. Never train."""
import json
import zipfile
from pathlib import Path
from scripts.resource_io import ROOT,sha256,read_json

VERSION='2026-11-19-v2'
def write_json(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')

def main():
    previous=read_json(ROOT/'resources/manifest.json') if (ROOT/'resources/manifest.json').exists() else {}
    registry=read_json(ROOT/'runtime/descriptor_registry.json')
    data=ROOT/'runtime/data'; active=read_json(data/'active_classifiers.json')
    from backend.modeling import artifact_directory
    run=artifact_directory(data)
    models={p for p in run.iterdir() if p.is_file()}
    models|={data/n for n in ['manifest.json','train.bin','query.bin','active_classifiers.json']}
    models.add(ROOT/'runtime/descriptor_registry.json')
    for entry in registry['models'].values(): models|={ROOT/entry['path'],ROOT/entry['metrics_path']}
    for key in ['comparison_results','tuning_protocol','tuning_confirmation','tuning_selection']: models.add(ROOT/registry[key])
    research={p for p in data.rglob('*') if p.is_file()}; excluded_logs=[]
    for relative in ['resnet18/20261118T195917Z-42d3213e','resnet18_comparison/20261118T204643Z-8efb68c2','lightgbm_resnet/20261118T211652Z-600d0ec2']:
        for p in (ROOT/'runtime/experiments'/relative).rglob('*'):
            if not p.is_file(): continue
            if p.suffix=='.log':
                excluded_logs.append(p.relative_to(ROOT).as_posix()); continue
            research.add(p)
    research.add(ROOT/'runtime/integration_backups/20261118T211632Z/immutable_artifacts.json')
    research-=models
    entries=[]; output=ROOT/'runtime/distribution'/VERSION; output.mkdir(parents=True,exist_ok=True)
    for kind,paths in [('models',models),('research',research)]:
        inventory={'id':kind,'version':VERSION,'files':{p.relative_to(ROOT).as_posix():{'bytes':p.stat().st_size,'sha256':sha256(p)} for p in sorted(paths)}}
        filename=f'metriclab-{kind}-{VERSION}.zip'; target=output/filename
        if target.exists():
            with zipfile.ZipFile(target) as z:
                if json.loads(z.read('PACKAGE_MANIFEST.json'))!=inventory: raise ValueError('Versión ya creada con contenido diferente; utiliza una versión nueva.')
        else:
            temporary=target.with_suffix('.tmp')
            with zipfile.ZipFile(temporary,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
                z.writestr('PACKAGE_MANIFEST.json',json.dumps(inventory,sort_keys=True))
                for p in sorted(paths): z.write(p,p.relative_to(ROOT).as_posix())
            temporary.replace(target)
        digest=sha256(target)
        old=next((e for e in previous.get('packages',[]) if e['id']==kind and e['sha256']==digest),{})
        entries.append({'id':kind,'filename':filename,'bytes':target.stat().st_size,'sha256':digest,'url':old.get('url'),'file_count':len(paths),'uncompressed_bytes':sum(p.stat().st_size for p in paths)})
        print(kind,len(paths),target.stat().st_size,sha256(target))
    manifest=read_json(data/'manifest.json')
    parent=ROOT/'runtime/experiments/resnet18/20261118T195917Z-42d3213e'
    image_rows=[]
    for split in ['train','query']:
        for row in read_json(parent/f'{split}_rows.json'):
            image_rows.append({'path':row['path'],'category':row['category'],'split':split,'id':row['id'],'sha256':row['image_sha256']})
    write_json(ROOT/'resources/image_manifest.json',{'fingerprint':manifest['fingerprint'],'images':image_rows})
    extraction=read_json(parent/'extraction.json')
    catalog={'version':VERSION,'publication_status':previous.get('publication_status','prepared locally; remote/release not yet published'),'packages':entries,
        'dataset':{'record_url':'https://data.caltech.edu/records/mzrjq-6wc02','url':'https://data.caltech.edu/records/mzrjq-6wc02/files/caltech-101.zip?download=1','md5':'3138e1922a9193bfa496528edbbc45d0','license':'CC-BY-4.0','doi':'10.22002/D1.20086'},
        'extractor_weights':{'url':extraction['weights_url'],'sha256':extraction['checkpoint_sha256'],'path':registry['checkpoint_path']},
        'split_fingerprint':manifest['fingerprint'],'dimensions':{'hsv':144,'resnet18':512}}
    catalog['excluded_private_logs']=sorted(excluded_logs)
    archive=ROOT/'runtime/downloads/caltech-101.zip'
    dataset_hash=sha256(archive) if archive.exists() else previous.get('dataset',{}).get('sha256')
    if dataset_hash: catalog['dataset']['sha256']=dataset_hash
    write_json(ROOT/'resources/manifest.json',catalog)
    (output/'SHA256SUMS.txt').write_text(''.join(f"{e['sha256']}  {e['filename']}\n" for e in entries),encoding='utf-8')

if __name__=='__main__': main()
