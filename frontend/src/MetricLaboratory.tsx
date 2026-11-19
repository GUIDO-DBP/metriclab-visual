import { useEffect, useState } from 'react'
import { ArrowRight, CheckCircle2, ChevronLeft, ChevronRight, FlaskConical, GitBranch, LoaderCircle, Search } from 'lucide-react'
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { api, category, format } from './api'
import { Descriptor, Empty, Gallery, ImageModal, Stat, UploadCard } from './components'
import { MetricTree } from './MetricTree'
import type { MetricNode, Neighbor, SearchResult, Sample, Status } from './types'

export function MetricLaboratory({ file, preview, onChoose, onError, samples, status }: {
  file: File | null; preview: string; onChoose: (file?: File) => void; onError: (message: string) => void;
  samples: Sample[]; status: Status | null;
}) {
  const [mode, setMode] = useState<'search' | 'compare'>('search')
  const [algorithm, setAlgorithm] = useState(1)
  const [k, setK] = useState(10)
  const [busy, setBusy] = useState(false)
  const [result, setResult] = useState<SearchResult | null>(null)
  const [tree, setTree] = useState<MetricNode[]>([])
  const [step, setStep] = useState(0)
  const [modal, setModal] = useState<Neighbor | null>(null)
  const [knn, setKnn] = useState<{ category: string; neighbors: Neighbor[] } | null>(null)
  useEffect(() => { setResult(null); setKnn(null); setStep(0) }, [file, k, algorithm])
  useEffect(() => { if (status?.metric_ready) api<MetricNode[]>('/api/tree').then(setTree).catch(e => onError(e.message)) }, [status?.metric_ready])
  async function execute(secondaryKnn = false) {
    if (!file) return
    setBusy(true)
    const form = new FormData(); form.append('file',file); form.append('k',String(secondaryKnn ? 5 : k)); form.append('algorithm',String(algorithm)); form.append('model','knn')
    try {
      if (secondaryKnn) { const response = await api<{predictions: {category: string; neighbors: Neighbor[]}[]}>('/api/classify',{method:'POST',body:form}); setKnn(response.predictions[0]) }
      else { setResult(await api<SearchResult>(`/api/${mode}`,{method:'POST',body:form})); setStep(0) }
    } catch (e) { onError((e as Error).message) } finally { setBusy(false) }
  }
  const trace = result?.trace || []
  const event = trace[step]
  const run = result?.runs[0]
  return <>
    <div className="notice complement-notice"><GitBranch size={20}/><div><strong>Trabajo complementario de la primera entrega</strong><p>Secuencial, VP-Tree y LAESA conservan su motor C++ y sus pruebas. Este laboratorio está separado de los cuatro clasificadores principales.</p></div></div>
    {!status?.metric_ready && <div className="notice warning">El motor métrico no está disponible. La clasificación principal puede utilizarse de forma independiente.</div>}
    <div className="tab-row"><button className={mode === 'search' ? 'selected' : ''} onClick={() => { setMode('search'); setResult(null) }}><Search size={16}/>Búsqueda por similitud</button><button className={mode === 'compare' ? 'selected' : ''} onClick={() => { setMode('compare'); setResult(null) }}><FlaskConical size={16}/>Comparar índices</button></div>
    <div className="workspace-grid"><UploadCard {...{file,preview,busy,samples,onChoose,onError}}/><div className="panel config-panel">
      <div className="panel-title"><span className="step-badge">02</span><h3>Configura la búsqueda métrica</h3></div>
      <label className="field">Descriptor<select disabled><option>HSV + orientación de bordes · 144D</option></select></label>
      <label className="field">Algoritmo</label><div className="segmented">{['Secuencial','VP-Tree','LAESA'].map((name,i) => <button key={name} className={mode === 'compare' || algorithm === i ? 'selected' : ''} disabled={mode === 'compare' || busy} onClick={() => setAlgorithm(i)}>{name}</button>)}</div>
      <label className="field">Cantidad de resultados<select value={k} disabled={busy} onChange={e => setK(Number(e.target.value))}>{[5,10,20].map(n => <option key={n} value={n}>{n} vecinos más cercanos</option>)}</select></label>
      <div className="explanation"><GitBranch size={18}/><p>La misma métrica euclidiana y los mismos vectores de la primera entrega. Cada resultado se verifica con búsqueda secuencial; las decisiones de poda proceden del C++ real.</p></div>
      <button className="button primary wide" disabled={busy || !file || !status?.metric_ready} onClick={() => void execute()}>{busy ? <LoaderCircle className="spin" size={18}/> : <Search size={18}/>} {mode === 'compare' ? 'Comparar los tres índices' : 'Buscar imágenes similares'}<ArrowRight size={17}/></button>
    </div></div>
    {!result && <Empty icon={<Search/>} title="Los índices métricos siguen aquí." text="Carga una imagen para buscar vecinos o comparar el costo de los tres algoritmos."/>}
    {run && <><div className="section-heading"><h2>{mode === 'compare' ? 'Comparación de índices' : `Vecinos recuperados · ${run.name}`}</h2><span className="success-badge"><CheckCircle2 size={15}/>{result!.runs.length === 3 ? '3 índices verificados' : 'Verificado con secuencial'}</span></div>
      {result!.runs.length === 1 ? <><Gallery neighbors={run.neighbors} onOpen={setModal}/><div className="stats-grid"><Stat value={`${format(run.query_ms,4)} ms`} label="Tiempo de consulta"/><Stat value={format(run.distances,0)} label="Distancias calculadas"/><Stat value={`${format(run.avoided_pct)} %`} label="Comparaciones evitadas"/><Stat value={`${format(status?.indexed || 0,0)} × 144`} label="Vectores indexados"/></div></> : <section className="panel"><div className="chart"><ResponsiveContainer width="100%" height={250}><BarChart data={result!.runs}><CartesianGrid strokeDasharray="3 3" vertical={false}/><XAxis dataKey="name"/><YAxis/><Tooltip/><Bar dataKey="distances" name="Distancias calculadas" fill="#7764e8" radius={[5,5,0,0]}/></BarChart></ResponsiveContainer></div><div className="table-scroll"><table><thead><tr><th>Índice</th><th>Consulta (ms)</th><th>Distancias</th><th>Evitadas</th><th>Construcción (ms)</th><th>Dist. construcción</th><th>Estructura (KiB)</th></tr></thead><tbody>{result!.runs.map(r => <tr key={r.algorithm}><td>{r.name}</td><td>{format(r.query_ms,4)}</td><td>{format(r.distances,0)}</td><td>{format(r.avoided_pct)} %</td><td>{format(r.build_ms)}</td><td>{format(r.build_distances,0)}</td><td>{format(r.memory_bytes/1024)}</td></tr>)}</tbody></table></div><p className="small muted">Memoria de la estructura, sin vectores ni aplicación. Construcción medida al iniciar. Tiempo de consulta sin JSON, HTTP, verificación ni traza. Una consulta no constituye un benchmark agregado.</p><details><summary>Vecinos devueltos por cada índice</summary>{result!.runs.map(r => <div key={r.algorithm}><h3>{r.name}</h3><Gallery neighbors={r.neighbors} onOpen={setModal}/></div>)}</details></section>}
      <Descriptor values={result!.descriptor}/></>}
    <section className="panel tree-panel"><span className="eyebrow">VISUALIZADOR CONSERVADO</span><h2>Recorre el VP-Tree</h2>{event ? <>
      <p className="muted">Topología real, cuatro niveles a la vez. Pulsa un nodo para explorar su rama.</p><MetricTree nodes={tree} events={trace.slice(0,step+1)} current={event.node}/>
      <div className="trace-controls"><button className="button secondary" aria-label="Paso anterior" disabled={!step} onClick={() => setStep(s => s-1)}><ChevronLeft size={18}/></button><input aria-label="Paso de la traza" type="range" min="0" max={trace.length-1} value={step} onChange={e => setStep(Number(e.target.value))}/><button className="button secondary" aria-label="Paso siguiente" disabled={step === trace.length-1} onClick={() => setStep(s => s+1)}><ChevronRight size={18}/></button><span>{step+1} / {trace.length}</span></div>
      <div className={`trace-event ${event.kind === 'prune' ? 'pruned' : ''}`}><strong>{event.kind === 'prune' ? 'Rama descartada' : event.kind === 'dist' ? 'Distancia calculada' : event.kind === 'candidate' ? 'Nuevo candidato' : 'Nodo visitado'} · nodo {event.node}</strong><p>{event.reason}</p><div>{event.kind !== 'visit' && <>d = {format(event.dist,5)} · </>}μ = {format(event.radius,5)} · τ = {event.tau === null ? '∞' : format(event.tau,5)}{event.kind === 'prune' && <> · cota = {format(event.bound,5)}</>}</div></div>
    </> : <p className="muted">Ejecuta VP-Tree o la comparación para generar una traza de tu consulta.</p>}</section>
    <section className="panel"><h3>k-NN — comparación secundaria</h3><p className="muted">Clasifica por votación entre cinco vecinos del VP-Tree. No forma parte de los cuatro modelos solicitados.</p><button className="button secondary" disabled={!file || !status?.metric_ready || busy} onClick={() => void execute(true)}>Clasificar con k-NN (complementario)</button>{knn && <><h2>{category(knn.category)}</h2><Gallery neighbors={knn.neighbors} onOpen={setModal}/></>}</section>
    {modal && <ImageModal image={modal} onClose={() => setModal(null)}/>}
  </>
}
