"""Render editable Markdown to standalone HTML consumed by the PDF browser."""
import base64
import json
import re
from pathlib import Path
import markdown

ROOT=Path(__file__).resolve().parents[1]; DOCS=ROOT/'documentacion_entrega'
CSS='''
.academic h2#referencias{break-before:page}
*{box-sizing:border-box} html{font-size:11pt} body{margin:0;color:#172336;font-family:Arial,sans-serif;line-height:1.55}
h1{font-size:23pt;line-height:1.2;color:#253b63;margin:0 0 20pt} h2{font-size:16pt;margin-top:24pt;color:#304a75} h3{font-size:12.5pt;margin-top:17pt}h4{font-size:11pt}
h1,h2,h3,h4{break-after:avoid}p{orphans:3;widows:3}a{color:#3a58a6;text-decoration:none;overflow-wrap:anywhere} code{font-family:Consolas,monospace;font-size:8.7pt;overflow-wrap:anywhere}pre{background:#f2f4fa;border:1px solid #d7deea;padding:10pt;white-space:pre-wrap;overflow-wrap:anywhere;word-break:break-word;line-height:1.4;break-inside:auto}
table{border-collapse:collapse;width:100%;font-size:9pt;margin:16pt 0;table-layout:auto}thead{display:table-header-group}tr{break-inside:avoid}th,td{padding:6pt 5pt;text-align:left;vertical-align:top;border-bottom:1px solid #d6deeb;overflow-wrap:anywhere}th{background:#edf1fa}td code,th code{font-size:8pt}td:has(code){word-break:break-word}
img{max-width:100%;height:auto;max-height:6.5in;object-fit:contain;display:block;margin:12pt auto 6pt;break-inside:avoid}p:has(>img){break-inside:avoid}figcaption,.caption{font-size:9pt;color:#3e4d63;line-height:1.4;break-before:avoid;margin-bottom:16pt}.toc{font-size:10pt;line-height:1.5;border-left:3px solid #8b85c5;padding-left:15pt;margin:20pt 0}.toc ul{padding-left:16pt}blockquote{border-left:3px solid #b9c4e0;margin-left:0;padding-left:12pt}li{margin:4pt 0}
body.academic{font-family:'Times New Roman',serif;font-size:12pt;line-height:2;color:#000} .academic h1,.academic h2,.academic h3,.academic h4{color:#000;font-size:12pt;line-height:2}.academic h1{text-align:center}.academic h2{text-align:center}.academic h3{text-align:left}.academic h4{font-style:italic}.academic p{text-indent:.5in;margin:0}.academic .academic-cover{break-after:page;text-align:center;padding-top:1.2in}.academic-cover p{text-indent:0;text-align:center;margin:0}.academic-cover h1{margin:0 0 .6in;font-size:12pt}.academic table{font-family:'Times New Roman',serif;font-size:10pt;line-height:1.4;border-top:1px solid #000;border-bottom:1px solid #000}.academic th{background:none;border-bottom:1px solid #000}.academic td{border:0}.academic .caption{text-indent:0;font-size:10pt;line-height:1.5}.academic p:has(img){text-indent:0}.academic code{font-size:8pt}.academic .toc{line-height:1.5}.references p{padding-left:.5in;text-indent:-.5in!important}.academic .toc li{margin:0}
'''

def main():
    build=DOCS/'.build'; build.mkdir(exist_ok=True); entries=[]
    sources=sorted(DOCS.glob('[0-9][0-9]_*.md'))+[DOCS/'INFORME_GENERAL_METRICLAB_VISUAL.md']
    for source in sources:
        raw=source.read_text(encoding='utf-8').replace('<div class="academic-cover">','<div class="academic-cover" markdown="1">')
        html=markdown.markdown(raw,extensions=['tables','fenced_code','toc','md_in_html'],extension_configs={'toc':{'toc_depth':'2-3','title':''}})
        def img(match):
            before,relative,after=match.groups(); path=(DOCS/relative).resolve()
            if not path.is_relative_to(DOCS/'assets') or not path.is_file(): raise ValueError('Imagen de documentación ausente o fuera de assets: '+relative)
            data=base64.b64encode(path.read_bytes()).decode(); return before+'data:image/png;base64,'+data+after
        html=re.sub(r'(<img[^>]*src=")([^"]+)("[^>]*>)',img,html)
        # Captions remain selectable text in PDF, separate from raster pixels.
        html=re.sub(r'(<p><img[^>]*alt="([^"]*)"[^>]*></p>)',lambda m:m[1]+'<p class="caption">'+m[2]+'</p>',html)
        def links(match):
            href=match[1]
            if href.startswith(('#','https://','http://','mailto:')): return match[0]
            import posixpath
            target=posixpath.normpath('documentacion_entrega/'+href)
            return 'href="https://github.com/GUIDO-DBP/metriclab-visual/blob/main/'+target+'"'
        html=re.sub(r'href="([^"]+)"',links,html)
        academic=source.stem.startswith('INFORME_')
        if academic:
            html=html.replace('<h2 id="referencias">Referencias</h2>','<h2 id="referencias">Referencias</h2><div class="references">')
            html=html.replace('<h2 id="apendice-fuentes-editables-y-reproducibilidad">','</div><h2 id="apendice-fuentes-editables-y-reproducibilidad">')
        output=build/(source.stem+'.html');output.write_text('<!doctype html><html lang="es"><meta charset="utf-8"><title>'+source.stem+'</title><style>'+CSS+'</style><body class="'+('academic' if academic else 'manual')+'">'+html+'</body></html>',encoding='utf-8')
        entries.append({'name':source.stem,'html':output.name,'academic':academic})
    (build/'manifest.json').write_text(json.dumps(entries),encoding='utf-8')
    print('HTML independiente preparado:',len(entries),'documentos.')

if __name__=='__main__': main()
