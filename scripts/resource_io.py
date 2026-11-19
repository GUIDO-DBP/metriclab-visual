"""Verified downloads and additive ZIP installation, standard library only."""
import hashlib
import json
import os
import shutil
import stat
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path,PurePosixPath

ROOT=Path(__file__).resolve().parents[1]
CATALOG=ROOT/'resources/manifest.json'

class SafeRedirect(urllib.request.HTTPRedirectHandler):
    """Do not forward a private GitHub credential to a signed storage redirect."""
    def redirect_request(self,req,fp,code,msg,headers,newurl):
        from urllib.parse import urlsplit
        if urlsplit(newurl).scheme!='https': raise ValueError('Redirección de descarga sin HTTPS rechazada.')
        redirected=super().redirect_request(req,fp,code,msg,headers,newurl)
        if redirected and urlsplit(req.full_url).hostname!=urlsplit(newurl).hostname:
            redirected.remove_header('Authorization')
        return redirected

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(1024*1024),b''): h.update(part)
    return h.hexdigest()

def read_json(path): return json.loads(Path(path).read_text(encoding='utf-8'))

def download(url,target,expected,algorithm='sha256',token=None):
    target=Path(target)
    if target.exists():
        with target.open('rb') as stream: existing=hashlib.file_digest(stream,algorithm).hexdigest()
        if existing!=expected:
            raise ValueError(f'Checksum diferente en {target.name}; conserva el archivo y revisa su procedencia.')
        print(f'Ya verificado: {target.name}'); return target
    if not url.startswith('https://'): raise ValueError('La descarga requiere HTTPS.')
    target.parent.mkdir(parents=True,exist_ok=True)
    headers={'User-Agent':'MetricLab-Visual-installer','Accept':'application/octet-stream'}
    # A credential is used only for the exact API host; never log it or put it in a URL.
    from urllib.parse import urlsplit
    if token and urlsplit(url).hostname=='api.github.com': headers['Authorization']='Bearer '+token
    if urlsplit(url).hostname in {'data.caltech.edu','download.pytorch.org'}:
        import time
        url+=('&' if '?' in url else '?')+'metriclab_cache='+str(time.time_ns())
        headers['Cache-Control']='no-cache'
    request=urllib.request.Request(url,headers=headers)
    temp=target.with_suffix(target.suffix+'.part')
    try:
        with urllib.request.build_opener(SafeRedirect()).open(request,timeout=90) as response,temp.open('wb') as f:
            shutil.copyfileobj(response,f)
        with temp.open('rb') as f: actual=hashlib.file_digest(f,algorithm).hexdigest()
        if actual!=expected: raise ValueError(f'La descarga de {target.name} no coincide con el hash esperado.')
        temp.replace(target)
    finally:
        temp.unlink(missing_ok=True)
    print(f'Descargado y verificado: {target.name}'); return target

def install_package(archive,entry,root=ROOT):
    root=Path(root).resolve(); archive=Path(archive)
    if sha256(archive)!=entry['sha256']: raise ValueError('SHA256 del paquete incompatible con el catálogo versionado.')
    with zipfile.ZipFile(archive) as z:
        names=z.namelist()
        if len(names)!=len(set(names)): raise ValueError('Entradas ZIP duplicadas.')
        inventory=json.loads(z.read('PACKAGE_MANIFEST.json'))
        expected=set(inventory['files'])|{'PACKAGE_MANIFEST.json'}
        if set(names)!=expected: raise ValueError('Contenido del ZIP diferente de su inventario.')
        if inventory['id']!=entry['id']: raise ValueError('Tipo de paquete incorrecto.')
        destinations=[]
        # Preflight the complete package before making any destination change.
        for name,info in inventory['files'].items():
            path=PurePosixPath(name); dest=(root/name).resolve()
            zi=z.getinfo(name)
            if '\\' in name or ':' in name or path.is_absolute() or '..' in path.parts or path.parts[0]!='runtime' or not dest.is_relative_to(root/'runtime'):
                raise ValueError('Ruta ZIP no permitida.')
            if stat.S_ISLNK(zi.external_attr>>16) or zi.file_size!=info['bytes']:
                raise ValueError('Tipo o tamaño ZIP incompatible.')
            if dest.exists() and (not dest.is_file() or sha256(dest)!=info['sha256']):
                raise ValueError(f'No se sobrescribe un recurso diferente: {name}')
            destinations.append((name,dest,info))
        staging=root/'runtime/resource-staging'; staging.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=staging) as temp:
            for name,dest,info in destinations:
                staged=Path(temp)/name; staged.parent.mkdir(parents=True,exist_ok=True)
                with z.open(name) as src,staged.open('wb') as out: shutil.copyfileobj(src,out)
                if sha256(staged)!=info['sha256']: raise ValueError(f'Archivo dañado: {name}')
            for name,dest,info in destinations:
                if dest.exists(): continue
                dest.parent.mkdir(parents=True,exist_ok=True)
                (Path(temp)/name).replace(dest)
    print(f"Paquete {entry['id']}: {len(destinations)} archivos verificados; existentes idénticos conservados.")

def release_asset(repository,tag,filename):
    import re
    if not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+',repository): raise ValueError('Repositorio esperado: propietario/nombre.')
    token=os.getenv('GITHUB_TOKEN') or os.getenv('GH_TOKEN')
    headers={'User-Agent':'MetricLab-Visual-installer','Accept':'application/vnd.github+json'}
    if token: headers['Authorization']='Bearer '+token
    from urllib.parse import quote
    req=urllib.request.Request(f'https://api.github.com/repos/{repository}/releases/tags/{quote(tag,safe="")}',headers=headers)
    try:
        with urllib.request.urlopen(req,timeout=30) as response: release=json.load(response)
    except urllib.error.HTTPError as exc:
        raise ValueError(f'Release inaccesible (HTTP {exc.code}). Comprueba tag, publicación y acceso al repositorio privado; usa -Archive para un ZIP local.') from None
    asset=next((a for a in release['assets'] if a['name']==filename),None)
    if not asset: raise ValueError('El paquete no está publicado en esa Release.')
    return (asset['url'] if token else asset['browser_download_url']),token
