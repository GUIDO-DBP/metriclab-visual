import { useEffect, useRef, useState } from 'react'
import type { ReactNode } from 'react'
import { CheckCircle2, ImagePlus, Maximize2, UploadCloud, X } from 'lucide-react'
import { category, format } from './api'
import type { Neighbor, Sample, Criterion, GalleryImage } from './types'

// Extracted from the original interface; shared by classification and metric search.
export function UploadCard({ file, preview, busy, samples, onChoose, onError }: {
  file: File | null; preview: string; busy: boolean; samples: Sample[];
  onChoose: (file?: File, sample?: Sample) => void; onError: (message: string) => void;
}) {
  const input = useRef<HTMLInputElement>(null)
  const [sampleLoading,setSampleLoading]=useState(false)
  const sampleRequest=useRef<AbortController|null>(null)
  useEffect(()=>()=>sampleRequest.current?.abort(),[])
  const disabled=busy || sampleLoading
  async function sample(selected: Sample) {
    sampleRequest.current?.abort()
    const controller=new AbortController(); sampleRequest.current=controller
    setSampleLoading(true)
    try {
      const response = await fetch(selected.url,{signal:controller.signal})
      if (!response.ok) throw new Error('Imagen de muestra no disponible.')
      const blob = await response.blob()
      onChoose(new File([blob], 'consulta-caltech.jpg', { type: blob.type }), selected)
    } catch (error) { if(!controller.signal.aborted)onError((error as Error).message) }
    finally { if(!controller.signal.aborted)setSampleLoading(false) }
  }
  return <div className="panel upload-panel">
    <div className="panel-title"><span className="step-badge">01</span><h3>Tu imagen</h3><span className="small muted">JPG / PNG</span></div>
    <input ref={input} type="file" accept="image/jpeg,image/png" hidden disabled={disabled}
      onChange={e => { onChoose(e.target.files?.[0]); e.target.value = '' }} />
    <button className={`dropzone ${preview ? 'has-image' : ''}`} disabled={disabled} onClick={() => input.current?.click()}
      onDragOver={e => e.preventDefault()} onDrop={e => { e.preventDefault(); if (!disabled) onChoose(e.dataTransfer.files[0]) }}>
      {preview ? <><img src={preview} alt="Imagen seleccionada"/><span className="replace-image"><ImagePlus size={16}/>Cambiar imagen</span></>
        : <><span className="upload-icon"><UploadCloud size={28}/></span><strong>Arrastra tu imagen aquí</strong><span>o <b>explora tus archivos</b></span><small>Hasta 10 MB · máximo 24 megapíxeles</small></>}
    </button>
    {file && <div className="file-label"><CheckCircle2 size={15}/><span>{file.name}</span><span className="muted">{format(file.size/1024, 0)} KB</span></div>}
    {!file && !!samples.length && <div className="sample-row"><span className="small muted">O elige una muestra identificada de prueba</span><div>{samples.slice(0,4).map(s =>
      <button title={category(s.category)} key={s.id} disabled={disabled} onClick={() => void sample(s)}><img src={s.url} alt={category(s.category)}/></button>)}</div><label className="field sample-selector">Muestra de prueba<select aria-label="Muestra identificada del conjunto de prueba" value="" disabled={disabled} onChange={e=>{const chosen=samples.find(s=>String(s.id)===e.target.value);if(chosen)void sample(chosen)}}><option value="" disabled>Elegir una categoría de las muestras disponibles</option>{samples.map(s=><option key={s.id} value={s.id}>{category(s.category)} · muestra #{s.id}</option>)}</select></label></div>}
    {sampleLoading && <p role="status" className="small muted">Cargando muestra identificada del conjunto de prueba…</p>}
  </div>
}
export function Criteria({ value, onChange, disabled = false }: { value: Criterion; onChange: (criterion: Criterion) => void; disabled?: boolean }) {
  return <div className="criterion-control"><label className="field">Criterio de Decision Tree</label><div className="segmented" aria-label="Criterio de Decision Tree">
    {(['gini', 'entropy'] as const).map(key => <button key={key} aria-pressed={value === key} disabled={disabled} className={value === key ? 'selected' : ''} onClick={() => onChange(key)}>{key === 'gini' ? 'Gini' : 'Entropía'}</button>)}
  </div><p className="field-help">Dos criterios de división del mismo algoritmo.</p></div>
}
export function Stat({ value, label, icon }: { value: string; label: string; icon?: ReactNode }) {
  return <div className="stat">{icon && <span className="stat-icon">{icon}</span>}<div><strong>{value}</strong><span>{label}</span></div></div>
}
export function Empty({ icon, title, text }: { icon: ReactNode; title: string; text: string }) {
  return <div className="empty-state"><div>{icon}</div><h3>{title}</h3><p>{text}</p></div>
}
export function Descriptor({ values }: { values: number[] }) {
  return <details className="panel descriptor"><summary>Inspeccionar el vector extraído · {values.length} dimensiones</summary><p className="small muted">{values.length===144 ? 'Azul: histograma HSV; violeta: orientación de bordes.' : 'Componentes del embedding aprendido por ResNet-18; no representan categorías ni partes identificables del objeto.'}</p><div className="vector-bars">{values.map((v,i) => <div key={i} title={`Dimensión ${i}: ${v}`} style={{ height: `${Math.max(2, v/Math.max(...values)*100)}%`, background: values.length===144 && i<128 ? '#5b7aea' : '#a06add' }}/>)}</div></details>
}
export function Gallery({ neighbors, onOpen }: { neighbors: Neighbor[]; onOpen: (image: Neighbor) => void }) {
  return <GalleryGrid images={neighbors} onOpen={onOpen} metric/>
}
export function CategoryGallery({images,onOpen}: {images:GalleryImage[];onOpen:(image:GalleryImage)=>void}) {
  return <GalleryGrid images={images} onOpen={onOpen}/>
}
function GalleryGrid<T extends GalleryImage>({images,onOpen,metric=false}: {images:T[];onOpen:(image:T)=>void;metric?:boolean}) {
  return <div className={`gallery ${metric ? '' : 'category-gallery'}`}>{images.map((n,i) => <button className="image-card" key={n.id} onClick={() => onOpen(n)} data-category={n.category} data-split={metric ? undefined : n.split}><div className="image-wrap"><img src={n.url} alt={category(n.category)} loading="lazy"/>{metric && <span className="rank">{String(i+1).padStart(2,'0')}</span>}<Maximize2 className="enlarge" size={17}/></div><div className="image-caption"><strong>{category(n.category)}</strong><span>{metric && typeof n.distance==='number' ? <>Distancia L2 <b>{format(n.distance,4)}</b></> : 'Imagen real · entrenamiento'}</span></div></button>)}</div>
}
export function ImageModal({ image, onClose }: { image: GalleryImage; onClose: () => void }) {
  const close = useRef<HTMLButtonElement>(null)
  useEffect(() => { const previous = document.activeElement as HTMLElement; close.current?.focus(); return () => previous?.focus() }, [])
  return <div className="modal" role="dialog" aria-modal="true" aria-label={`Imagen: ${category(image.category)}`} onClick={onClose} onKeyDown={e => { if (e.key === 'Escape') onClose(); if (e.key === 'Tab') { e.preventDefault(); close.current?.focus() } }}><button ref={close} className="modal-close" aria-label="Cerrar imagen ampliada" onClick={onClose}><X/></button><figure onClick={e => e.stopPropagation()}><img src={image.url} alt={category(image.category)}/><figcaption>{category(image.category)} · {typeof image.distance==='number' ? `distancia ${format(image.distance,5)}` : 'Fotografía real de entrenamiento'}</figcaption></figure></div>
}
