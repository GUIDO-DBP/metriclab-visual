import {createRequire} from 'node:module'
import {pathToFileURL} from 'node:url'
import fs from 'node:fs'
import path from 'node:path'
const root=path.resolve(import.meta.dirname,'..'), docs=path.join(root,'documentacion_entrega')
const require=createRequire(path.join(root,'frontend/package.json'))
const {chromium}=require('@playwright/test')
const browser=await chromium.launch({channel:'msedge',headless:true})
const entries=JSON.parse(fs.readFileSync(path.join(docs,'.build/manifest.json'),'utf8'))
fs.mkdirSync(path.join(docs,'pdf'),{recursive:true})
const qa=[]
try {
 for(const e of entries) {
  const page=await browser.newPage({viewport:{width:e.academic?624:658,height:1000}})
  await page.goto(pathToFileURL(path.join(docs,'.build',e.html)).href)
  await page.emulateMedia({media:'print'})
  await page.evaluate(()=>document.fonts.ready)
  const broken=await page.locator('img').evaluateAll(imgs=>imgs.filter(i=>!i.complete||!i.naturalWidth).length)
  if(broken)throw Error('Imágenes rotas: '+e.name)
  const overflow=await page.evaluate(()=>[...document.querySelectorAll('table,pre')].filter(x=>x.scrollWidth>x.clientWidth+2).map(x=>x.tagName))
  if(overflow.length)throw Error('Desbordamiento antes de PDF '+e.name+': '+overflow)
  const target=e.academic?path.join(docs,e.name+'.pdf'):path.join(docs,'pdf',e.name+'.pdf')
  await page.pdf({path:target,format:e.academic?'Letter':'A4',printBackground:true,displayHeaderFooter:true,
   headerTemplate:e.academic?'<div style="width:100%;font-family:Times New Roman;font-size:12pt;text-align:right;margin-right:1in"><span class="pageNumber"></span></div>':'<span></span>',
   footerTemplate:e.academic?'<span></span>':'<div style="font:9px Arial;width:100%;text-align:center;color:#5a6883">MetricLab Visual · <span class="pageNumber"></span> / <span class="totalPages"></span></div>',
   margin:e.academic?{top:'1in',bottom:'1in',left:'1in',right:'1in'}:{top:'18mm',bottom:'18mm',left:'18mm',right:'18mm'},tagged:true,outline:true})
  qa.push({name:e.name,broken_images:broken,overflow_elements:overflow,bytes:fs.statSync(target).size})
  console.log(e.name,fs.statSync(target).size)
  await page.close()
 }
 fs.mkdirSync(path.join(docs,'qa'),{recursive:true})
 fs.writeFileSync(path.join(docs,'qa/browser-layout.json'),JSON.stringify(qa,null,2))
} finally {await browser.close()}
