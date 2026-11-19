"""Install trusted, versioned artifacts without training or overwriting models."""
import argparse
from pathlib import Path
from scripts.resource_io import ROOT,CATALOG,read_json,download,install_package,release_asset

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--kind',choices=['models','research'],default='models')
    p.add_argument('--archive',type=Path)
    p.add_argument('--repository',help='Repositorio real propietario/nombre; no se supone una URL inexistente.')
    p.add_argument('--tag',default='resources-v1')
    p.add_argument('--weights',action='store_true',help='Descarga pesos oficiales PyTorch; se verifican separadamente.')
    args=p.parse_args(); catalog=read_json(CATALOG)
    entry=next(x for x in catalog['packages'] if x['id']==args.kind)
    archive=args.archive
    if not archive:
        url=entry.get('url'); token=None
        if args.repository: url,token=release_asset(args.repository,args.tag,entry['filename'])
        if not url: raise SystemExit('Paquete aún sin URL publicada. Proporciona --archive RUTA_ZIP o --repository propietario/repositorio con una Release existente.')
        archive=download(url,ROOT/'runtime/downloads'/entry['filename'],entry['sha256'],token=token)
    install_package(archive,entry)
    if args.weights:
        weights=catalog['extractor_weights']; download(weights['url'],ROOT/weights['path'],weights['sha256'])

if __name__=='__main__': main()
