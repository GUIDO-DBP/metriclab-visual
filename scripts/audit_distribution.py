"""Audit exactly the files Git would include, without printing potential secret values."""
import json
import re
import subprocess
import zipfile
from collections import defaultdict
from pathlib import Path
from scripts.resource_io import ROOT,read_json

PATTERNS={
 'github credential':re.compile(r'(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{30,})'),
 'private key':re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
 'api credential':re.compile(r'\bsk-[A-Za-z0-9_-]{24,}'),
 'private Windows path':re.compile(r'[A-Za-z]:[\\/]Users[\\/]'),
}
def scan(text,name):
    return [{'file':name,'kind':kind} for kind,pattern in PATTERNS.items() if pattern.search(text)]

def main():
    raw=subprocess.check_output(['git','ls-files','--cached','--others','--exclude-standard','-z'],cwd=ROOT).decode()
    paths=[ROOT/n for n in raw.split('\0') if n and (ROOT/n).is_file()]
    groups=defaultdict(lambda:{'files':0,'bytes':0});findings=[]
    for p in paths:
        relative=p.relative_to(ROOT);group=relative.parts[0] if len(relative.parts)>1 else '(raíz)'
        groups[group]['files']+=1;groups[group]['bytes']+=p.stat().st_size
        if p.stat().st_size>=50*1024*1024: findings.append({'file':relative.as_posix(),'kind':'large file in Git selection'})
        if p.suffix.lower() in {'.py','.md','.json','.txt','.ps1','.sql','.html','.ts','.tsx','.css','.cpp','.hpp','.sh'} or p.name in {'.env','.env.example','.gitignore'}:
            findings+=scan(p.read_text(encoding='utf-8-sig',errors='replace'),relative.as_posix())
    catalog=read_json(ROOT/'resources/manifest.json')
    for entry in catalog['packages']:
        archive=ROOT/'runtime/distribution'/catalog['version']/entry['filename']
        with zipfile.ZipFile(archive) as z:
            for name in z.namelist():
                if Path(name).suffix in {'.json','.md','.csv','.txt','.log'}: findings+=scan(z.read(name).decode('utf-8',errors='replace'),entry['filename']+'!'+name)
    # Inspect all historic source blobs, preserving commit history rather than rewriting it.
    history_findings=[]
    for commit in subprocess.check_output(['git','rev-list','HEAD'],cwd=ROOT,text=True).splitlines():
        for name in subprocess.check_output(['git','ls-tree','-r','--name-only',commit],cwd=ROOT,text=True).splitlines():
            if Path(name).suffix in {'.py','.md','.json','.txt','.ps1','.sql','.html','.ts','.tsx','.cpp','.hpp'}:
                contents=subprocess.check_output(['git','show',commit+':'+name],cwd=ROOT).decode('utf-8',errors='replace')
                history_findings+=scan(contents,commit[:8]+':'+name)
    findings+=history_findings
    summary={'selected_files':len(paths),'selected_bytes':sum(p.stat().st_size for p in paths),'groups':dict(groups),'findings':findings,'history_scanned':True,'package_text_scanned':True,'excluded_historical_logs':len(catalog['excluded_private_logs'])}
    (ROOT/'resources/distribution_audit.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    prior=read_json(ROOT/'runtime/github_preparation_backup/inventory.json')
    lines=['# Auditoría de preparación GitHub','', 'Fecha: 19 de noviembre de 2026. Repositorio de destino: `GUIDO-DBP/metriclab-visual`; rama local `feature/clasificacion-u2`. Se conserva el commit anterior `621253d` y el historial, sin force push. Se hizo una copia ZIP del código antes de modificarlo, en `runtime/github_preparation_backup/`, fuera de la publicación.','',
    '## Contenido seleccionado para Git','', '| Grupo | Archivos | MB decimales | Decisión |','| --- | ---: | ---: | --- |']
    for group,values in sorted(groups.items()): lines.append(f"| {group} | {values['files']} | {values['bytes']/1e6:.3f} | Código, metadatos pequeños o documentación seleccionada. |")
    lines+=['','El inventario exacto puede variar al añadir este informe; `resources/distribution_audit.json` registra la última pasada. No se añaden archivos por debajo del límite de Git solo porque quepan. PDF, capturas definitivas y figuras educativas se conservan; las temporales de QA no.','',
    '## Grupos excluidos y recuperación','', '| Grupo local inicial | Tamaño medido antes de las pruebas nuevas | Motivo / recuperación |','| --- | ---: | --- |']
    for name,why in [('.venv','Entorno ligado al equipo. Rehacer con setup y lock; no usa paquetes globales en una instalación nueva.'),('frontend','El total inicial incluye node_modules y dist. Solo se publica código/config/lock; npm ci y build regeneran dependencias y compilación.'),('dataset','138,6 MB de fotografías/archivos originales. Descargar Caltech oficial, verificar archivo e imágenes; no se copia completo a Git.'),('runtime','Incluye entornos, cachés y artefactos esenciales. No se borra: los recursos necesarios se empaquetan por allowlist, el resto queda local.')]:
        size=prior['groups'][name];lines.append(f'| {name} | {size/1e6:.1f} MB | {why} |')
    lines+=['','Se excluyen también .git del contenido de trabajo, cachés Python/pip/npm, binarios compilados, .env reales, archivos temporales, logs, copias de seguridad, ruedas y checkpoints descargables. El historial Git se transfiere por Git, no como copia manual de .git. Las reglas están en `.gitignore` y `.gitattributes`.','',
    '## Paquetes de recursos','', '| Paquete | Archivos | Descarga MB | Tamaño sin comprimir MB | SHA256 |','| --- | ---: | ---: | ---: | --- |']
    for e in catalog['packages']:lines.append(f"| {e['id']} | {e['file_count']} | {e['bytes']/1e6:.3f} | {e['uncompressed_bytes']/1e6:.3f} | `{e['sha256']}` |")
    lines+=['','`models` contiene el bundle HSV activo, formatos nativos, manifiesto, vectores HSV, registro y cinco modelos ResNet con métricas/protocolos que utiliza la API. `research` añade embeddings, etiquetas, predicciones, históricos, curvas e inventarios de preservación; no es obligatorio para una consulta normal. Los pesos CNN se descargan de PyTorch, no se duplican en los paquetes. La DLL se compila desde el código.','',
    'Se eligieron Releases frente a LFS: separan código de recursos y permiten descargar modelos sin instalar LFS ni reentrenar. GitHub bloquea archivos Git de más de 100 MiB y advierte desde 50 MiB; los assets de Release deben ser menores de 2 GiB ([documentación oficial](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases)). Ambos ZIP están muy por debajo de ese límite. No se afirma que el acceso sea funcional hasta comprobar su publicación; ver `PUBLICACION_GITHUB.md`.','',
    '## Privacidad, integridad y límites','',
    'La auditoría encontró rutas privadas en tres logs históricos de investigación. Se excluyeron todos los logs de los ZIP públicos; los originales locales conservan sus hashes. La versión preliminar v1 de paquetes no se publica. El catálogo final v2 declara las omisiones; la prueba de preservación comprueba todos los originales donde existen y permite únicamente logs declarados ausentes en instalaciones distribuidas. No se alteraron modelos ni resultados para redaccionar logs.','',
    'Se buscaron credenciales con patrones conocidos, claves privadas y rutas privadas tanto en selección actual como en fuentes del historial y textos de ZIP. Los ejemplos `usuario:clave` de `.env.example` son placeholders, no credenciales. La búsqueda automática no demuestra ausencia de todo secreto concebible; se revisó además el contenido seleccionado y los enlaces de PDF. No se imprimen valores potencialmente secretos.','',
    f"Resultado de la última pasada: **{len(findings)} hallazgos pendientes**. Los paquetes contienen inventario por archivo y checksum; los instaladores verifican antes de mutar y rechazan rutas inseguras o archivos existentes diferentes. Joblib solo debe cargarse desde procedencia confiable.",'',
    'Las licencias se documentan en `LICENCIAS_Y_RECURSOS.md`: Caltech CC BY 4.0 verificada mediante el DOI; código de bibliotecas y condiciones de pesos tratados por separado. No se asignó una licencia abierta nueva al código propio ni se inventaron identidades. La distribución no incluye corpus ImageNet ni fotos privadas de usuarios.','']
    (ROOT/'docs/AUDITORIA_GITHUB.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({'files':len(paths),'bytes':summary['selected_bytes'],'findings':findings},ensure_ascii=False))
    if findings: raise SystemExit(1)

if __name__=='__main__': main()
