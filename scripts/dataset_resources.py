"""Obtain exact Caltech images independently; preserve the packaged historical split."""
import argparse
import json
import shutil
import tarfile
import tempfile
import zipfile
from pathlib import Path,PurePosixPath
from scripts.resource_io import ROOT,CATALOG,read_json,download,sha256

def expected_images(): return read_json(ROOT/'resources/image_manifest.json')['images']

def validate(root):
    rows=expected_images(); missing=[]; changed=[]
    for row in rows:
        path=root/row['path']
        if not path.is_file(): missing.append(row['path'])
        elif sha256(path)!=row['sha256']: changed.append(row['path'])
    if missing or changed: raise ValueError(f'Dataset incompleto o diferente: faltan {len(missing)}, hashes distintos {len(changed)}. Primer caso: {(missing+changed)[0]}')
    print(f'Dataset verificado: {len(rows)} imágenes; separación histórica conservada.')

def install_from_folder(source):
    source=Path(source).resolve()
    nested=source/'101_ObjectCategories'
    if nested.is_dir(): source=nested
    plans=[]
    for row in expected_images():
        dest=ROOT/row['path']; relative=Path(row['path']).relative_to('dataset/101_ObjectCategories'); src=source/relative
        if not src.is_file() or sha256(src)!=row['sha256']: raise ValueError(f'Imagen ausente o distinta: {relative.as_posix()}')
        if dest.exists() and sha256(dest)!=row['sha256']: raise ValueError(f'No se sobrescribe una imagen diferente: {relative.as_posix()}')
        plans.append((src,dest))
    for src,dest in plans:
        if dest.exists(): continue
        dest.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(src,dest)
    validate(ROOT)

def unpack_official(archive):
    staging=ROOT/'runtime/dataset-staging'; staging.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=staging) as temp:
        temp=Path(temp)
        # The official ZIP contains the category TAR.GZ. Never extract arbitrary paths or links.
        with zipfile.ZipFile(archive) as z:
            member=next((n for n in z.namelist() if n.endswith('/101_ObjectCategories.tar.gz') or n=='101_ObjectCategories.tar.gz'),None)
            if member:
                tarpath=temp/'categories.tar.gz'
                with z.open(member) as src,tarpath.open('wb') as out: shutil.copyfileobj(src,out)
                with tarfile.open(tarpath,'r:gz') as tar:
                    for m in tar.getmembers():
                        parts=PurePosixPath(m.name).parts
                        if not m.isfile() or '101_ObjectCategories' not in parts: continue
                        relative=PurePosixPath(*parts[parts.index('101_ObjectCategories')+1:])
                        if '..' in relative.parts or relative.is_absolute(): raise ValueError('Ruta TAR insegura.')
                        dest=temp/'images'/relative
                        dest.parent.mkdir(parents=True,exist_ok=True)
                        with tar.extractfile(m) as src,dest.open('wb') as out: shutil.copyfileobj(src,out)
            else:
                for n in z.namelist():
                    parts=PurePosixPath(n).parts
                    if n.endswith('/') or '101_ObjectCategories' not in parts: continue
                    relative=PurePosixPath(*parts[parts.index('101_ObjectCategories')+1:])
                    if '..' in relative.parts or relative.is_absolute() or '\\' in n: raise ValueError('Ruta ZIP insegura.')
                    dest=temp/'images'/relative; dest.parent.mkdir(parents=True,exist_ok=True)
                    with z.open(n) as src,dest.open('wb') as out: shutil.copyfileobj(src,out)
        install_from_folder(temp/'images')

def main():
    p=argparse.ArgumentParser(); p.add_argument('--source',type=Path); p.add_argument('--archive',type=Path); p.add_argument('--download',action='store_true'); p.add_argument('--verify',action='store_true')
    args=p.parse_args()
    if args.verify: validate(ROOT); return
    if args.source: install_from_folder(args.source); return
    archive=args.archive
    if args.download:
        d=read_json(CATALOG)['dataset']
        algorithm='sha256' if d.get('sha256') else 'md5'
        archive=download(d['url'],ROOT/'runtime/downloads/caltech-101.zip',d[algorithm],algorithm)
    if not archive: raise SystemExit('Usa --source CARPETA, --archive ZIP_OFICIAL, --download o --verify.')
    d=read_json(CATALOG)['dataset']
    import hashlib
    with archive.open('rb') as f:
        if hashlib.file_digest(f,'md5').hexdigest()!=d['md5']: raise ValueError('El ZIP no coincide con el checksum oficial publicado.')
    if d.get('sha256') and sha256(archive)!=d['sha256']: raise ValueError('SHA256 del ZIP distinto del catálogo verificado.')
    unpack_official(archive)

if __name__=='__main__': main()
