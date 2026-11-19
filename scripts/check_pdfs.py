"""Check every PDF page, extract text and render contact sheets for visual QA."""
import json
from pathlib import Path
import pymupdf

ROOT=Path(__file__).resolve().parents[1];DOCS=ROOT/'documentacion_entrega'
def main():
    out=DOCS/'qa';out.mkdir(exist_ok=True);records=[]
    from PIL import Image,ImageDraw
    for path in sorted((DOCS/'pdf').glob('*.pdf'))+[DOCS/'INFORME_GENERAL_METRICLAB_VISUAL.pdf']:
        pdf=pymupdf.open(path);issues=[];texts=[];thumbs=[]
        for i,page in enumerate(pdf):
            text=page.get_text();texts.append(text)
            if '\ufffd' in text: issues.append(f'page {i+1}: replacement glyph')
            if len(text.strip())<8 and not page.get_images(): issues.append(f'page {i+1}: empty')
            for block in page.get_text('dict')['blocks']:
                for line in block.get('lines',[]):
                    for span in line['spans']:
                        x0,y0,x1,y1=span['bbox']
                        if x0<-1 or y0<-1 or x1>page.rect.width+1 or y1>page.rect.height+1: issues.append(f'page {i+1}: text beyond page')
            pix=page.get_pixmap(matrix=pymupdf.Matrix(.24,.24),alpha=False)
            image=Image.frombytes('RGB',(pix.width,pix.height),pix.samples)
            thumb=Image.new('RGB',(170,230),'#e5e8ef');thumb.paste(image,((170-image.width)//2,17));ImageDraw.Draw(thumb).text((5,2),str(i+1),fill='black');thumbs.append(thumb)
        cols=4;rows=(len(thumbs)+cols-1)//cols;sheet=Image.new('RGB',(cols*170,rows*230),'white')
        for i,thumb in enumerate(thumbs):sheet.paste(thumb,((i%cols)*170,(i//cols)*230))
        sheet.save(out/(path.stem+'-pages.png'))
        record={'file':path.relative_to(ROOT).as_posix(),'pages':len(pdf),'bytes':path.stat().st_size,'text_characters':sum(map(len,texts)),'issues':issues};records.append(record)
        (out/(path.stem+'.txt')).write_text('\n'.join(texts),encoding='utf-8')
        print(path.name,len(pdf),'pages',len(issues),'issues')
    (out/'pdf-verification.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
    assert len(records)==11 and not any(r['issues'] for r in records),'PDF QA failed'

if __name__=='__main__': main()
