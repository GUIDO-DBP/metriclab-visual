import { useEffect, useRef, useState } from 'react'
import { ArrowRight, ArrowUpRight, Atom, BarChart3, CheckCircle2, ChevronRight, Database, FlaskConical, GitBranch, Home, ImagePlus, Info, Layers, LoaderCircle, ScanLine, Sparkles, Timer, X } from 'lucide-react'
import { api, category, descriptions, format, names } from './api'
import { Criteria, Descriptor, Empty, Stat, UploadCard, CategoryGallery, ImageModal } from './components'
import { ClassifierEvaluation } from './ClassifierEvaluation'
import { ClassifierExplorer } from './ClassifierExplorer'
import { MetricLaboratory } from './MetricLaboratory'
import { HSV_ID, RESNET_ID } from './types'
import type { Classification, Criterion, DatasetInfo, Experiments, ModelKey, Page, Status, DescriptorInfo, GalleryImage, Sample } from './types'

const navigation = [
  {id:'home',name:'Inicio',icon:Home},
  {id:'classify',name:'Clasificar imagen',icon:ScanLine},
  {id:'evaluation',name:'Comparación de clasificadores',icon:BarChart3},
  {id:'explorer',name:'Explorador de árboles',icon:GitBranch},
  {id:'dataset',name:'Dataset y metodología',icon:Database},
] as const
const families = [
  {key:'gini',name:'Decision Tree',tag:'UNA SECUENCIA DE DECISIONES',icon:GitBranch},
  {key:'forest',name:'Random Forest',tag:'MUCHOS ÁRBOLES, UNA PREDICCIÓN',icon:Layers},
  {key:'xgboost',name:'XGBoost',tag:'APRENDER DE LOS ERRORES',icon:Sparkles},
  {key:'lightgbm',name:'LightGBM',tag:'BOOSTING CON HISTOGRAMAS',icon:FlaskConical},
] as const

export default function App() {
  const [page,setPage] = useState<Page>('home')
  const [descriptorId,setDescriptorId] = useState<string>(HSV_ID)
  const refreshSequence=useRef(0)
  const [descriptorOptions,setDescriptorOptions] = useState<DescriptorInfo[]>([])
  const [evaluationScope,setEvaluationScope] = useState('historical_test')
  const [modal,setModal] = useState<GalleryImage | null>(null)
  const [status,setStatus] = useState<Status | null>(null)
  const [dataset,setDataset] = useState<DatasetInfo | null>(null)
  const [experiments,setExperiments] = useState<Experiments | null>(null)
  const [connectionError,setConnectionError] = useState('')
  const [error,setError] = useState('')
  const [file,setFile] = useState<File | null>(null)
  const [knownSample,setKnownSample] = useState<Sample|null>(null)
  const [preview,setPreview] = useState('')
  const [family,setFamily] = useState('decision_tree')
  const [criterion,setCriterion] = useState<Criterion>('gini')
  const [busy,setBusy] = useState(false)
  const [classification,setClassification] = useState<Classification | null>(null)
  const [predictionKey,setPredictionKey] = useState<ModelKey>('gini')
  const [focusKey,setFocusKey] = useState<ModelKey>('gini')
  const [explorerKey,setExplorerKey] = useState<ModelKey>('gini')
  const selectedKey = (family === 'decision_tree' ? criterion : family) as ModelKey
  const selectedModel = status?.models.find(model => model.key === selectedKey)
  const title = page === 'metric' ? 'Laboratorio de búsqueda métrica' : navigation.find(n => n.id === page)!.name
  const prediction = classification?.predictions.find(p => p.key === predictionKey) || classification?.predictions[0]

  async function refresh() {
    const sequence=++refreshSequence.current
    try {
      const current = await api<Status>(`/api/status?descriptor_id=${descriptorId}`)
      if(sequence!==refreshSequence.current) return
      setStatus(current); setConnectionError('')
      if (current.ready) {
        const [data,evaluation] = await Promise.all([api<DatasetInfo>(`/api/dataset?descriptor_id=${descriptorId}`),api<Experiments>(`/api/experiments?descriptor_id=${descriptorId}&evaluation=historical_test`)])
        if(sequence!==refreshSequence.current) return
        setDataset(data); setExperiments(evaluation)
      } else { setDataset(null); setExperiments(null) }
    } catch { if(sequence===refreshSequence.current) setConnectionError('No se pudo conectar con la aplicación. Inicia el servidor local y vuelve a comprobar el estado.') }
  }
  useEffect(() => { void refresh() },[descriptorId])
  useEffect(() => { api<{descriptors:DescriptorInfo[]}>('/api/descriptors').then(data => setDescriptorOptions(data.descriptors)).catch(() => {}) },[])
  useEffect(() => { if (!file) { setPreview(''); return }; const url = URL.createObjectURL(file); setPreview(url); return () => URL.revokeObjectURL(url) },[file])
  useEffect(() => { setError(''); window.scrollTo({top:0,behavior:'instant'}) },[page])
  function chooseFile(selected?: File, sample?: Sample) {
    if (!selected) return
    if (!['image/jpeg','image/png'].includes(selected.type) || selected.size > 10*1024*1024) { setError('Selecciona una imagen JPG o PNG de hasta 10 MB.'); return }
    setFile(selected); setKnownSample(sample && dataset?.samples.some(s=>s.id===sample.id && s.url===sample.url && s.category===sample.category) ? sample : null); setClassification(null); setError('')
  }
  async function classify(compareFour = false, overrideKey?: ModelKey) {
    if (!file) return
    setBusy(true); setError('')
    const form = new FormData(); form.append('file',file); form.append('model',overrideKey || selectedKey); form.append('descriptor_id',descriptorId); form.append('criterion',criterion); form.append('compare',String(compareFour))
    try { const result = await api<Classification>('/api/classify',{method:'POST',body:form}); setClassification(result); setPredictionKey(overrideKey || selectedKey) }
    catch (e) { setError((e as Error).message) } finally { setBusy(false) }
  }
  function openExplorer(key: ModelKey) { setExplorerKey(key); setPage('explorer') }
  function openEvaluation(key: ModelKey) { setFocusKey(key); setPage('evaluation') }

  const isResnet = descriptorId===RESNET_ID
  const dimensions = isResnet ? 512 : 144
  const descriptorName = isResnet ? 'ResNet-18' : 'HSV + Sobel'
  return <div className="app-shell classification-app">
    <aside className="sidebar">
      <a className="brand" href="#inicio" onClick={e => { e.preventDefault(); setPage('home') }}><span className="brand-icon"><Atom size={27}/></span><span>MetricLab<span className="brand-sub">VISUAL</span></span></a>
      <div className="workspace-label">CLASIFICACIÓN DE IMÁGENES</div>
      <nav aria-label="Navegación principal">{navigation.map(item => <button key={item.id} title={item.name} aria-label={item.name} className={page === item.id ? 'active' : ''} aria-current={page === item.id ? 'page' : undefined} onClick={() => setPage(item.id)}><item.icon size={19}/><span>{item.name}</span>{page === item.id && <span className="nav-dot"/>}</button>)}</nav>
      <div className="nav-complement"><span>TRABAJO COMPLEMENTARIO</span><button title="Laboratorio de búsqueda métrica — Trabajo complementario" aria-label="Laboratorio de búsqueda métrica — Trabajo complementario" className={page === 'metric' ? 'active' : ''} onClick={() => setPage('metric')}><FlaskConical size={18}/><span>Laboratorio de búsqueda métrica</span></button></div>
      <div className="sidebar-bottom"><div className="course-icon"><GitBranch size={21}/></div><strong>Una imagen. Cuatro perspectivas.</strong><p>Clasifica, compara y comprende cómo deciden los modelos.</p><div className="course-label">ESTRUCTURAS DE DATOS AVANZADAS · U2</div><span className="version">MetricLab Visual · v2.0</span></div>
    </aside>
    <div className="main-shell"><header className="topbar"><div><span className="muted">MetricLab</span><ChevronRight size={14}/><strong>{title}</strong></div><span className={`connection ${status?.classification_ready && !connectionError ? 'online' : ''}`}><i/>{status?.classification_ready && !connectionError ? '4 clasificadores disponibles' : 'Modelos pendientes'}</span></header>
    <main>
      <div className="page-heading"><div className="eyebrow">CLASIFICA. COMPARA. COMPRENDE.</div><h1>{page === 'home' ? <>Una imagen.<br className="mobile-break"/> Cuatro formas de decidir.</> : title}</h1><p>{page === 'home' ? 'Explora la clasificación de imágenes con modelos basados en árboles y resultados que puedes explicar.' : page === 'classify' ? 'Carga una fotografía y descubre qué categoría predice cada modelo.' : page === 'evaluation' ? 'Una comparación justa sobre las mismas imágenes de entrenamiento y prueba.' : page === 'explorer' ? 'Abre los modelos entrenados y sigue sus decisiones, nodo a nodo.' : page === 'dataset' ? 'Conoce los datos, el proceso experimental y los límites de esta aplicación.' : 'Trabajo complementario · los índices métricos de la primera entrega.'}</p></div>
      {page!=='metric' && <section className="descriptor-toolbar panel"><label className="field">Descriptor visual<select aria-label="Descriptor visual" value={descriptorId} disabled={busy} onChange={e => { setDescriptorId(e.target.value); setStatus(null); setDataset(null); setExperiments(null); setClassification(null); setModal(null); setEvaluationScope('historical_test') }}>{descriptorOptions.length ? descriptorOptions.map(d => <option key={d.id} value={d.id} disabled={!d.available}>{d.name} · {d.dimensions} dimensiones{!d.available ? ' · No disponible' : ''}</option>) : <option value={HSV_ID}>HSV + Sobel · 144 dimensiones</option>}</select></label><div><span className="success-badge">Modelos separados por descriptor</span><p className="small muted">{isResnet ? 'Extractor preentrenado congelado. La primera carga puede tardar; CPU/CUDA automático.' : 'Descriptor original de color y bordes. HSV permanece como opción predeterminada.'}</p></div></section>}
      {(connectionError || (status && !status.ready)) && <div className="notice warning"><Info size={20}/><div><strong>{connectionError || 'Dataset pendiente de preparación'}</strong><p>{status?.error || 'Consulta las instrucciones de inicio y preparación en README.md.'}</p></div><button className="button secondary" onClick={() => void refresh()}>Comprobar estado</button></div>}
      {status?.model_error && <div className="notice warning"><Info size={18}/><div><strong>No se pudieron cargar los modelos.</strong><p>{status.model_error}</p></div></div>}
      {status?.kind === 'synthetic' && <div className="notice warning"><Info size={18}/>Modo de demostración sintética. Los resultados no corresponden a Caltech-101.</div>}
      {error && <div className="notice danger" role="alert"><Info size={19}/>{error}<button aria-label="Cerrar error" onClick={() => setError('')}><X size={18}/></button></div>}

      {page === 'home' && <>
        <section className="classification-hero"><div><span className="hero-kicker"><span/>LABORATORIO DE MACHINE LEARNING</span><h2>De los píxeles<br/>a una categoría.</h2><p>Sube una imagen, elige un clasificador y compara sus decisiones. Todos los modelos trabajan con las mismas características visuales.</p><button className="button primary hero-cta" onClick={() => setPage('classify')}><ScanLine size={19}/>Clasificar imagen<ArrowRight size={18}/></button><span className="hero-footnote">JPG o PNG · predicciones de modelos entrenados</span></div><div className="hero-diagram" aria-label="Flujo de clasificación"><div className="hero-image-stack"><ImagePlus size={40}/><span>TU IMAGEN</span></div><div className="hero-vector">{Array.from({length:12},(_,i) => <i key={i}/>)}<span>{dimensions} CARACTERÍSTICAS</span></div><div className="hero-models">{families.map(f => <div key={f.key}><f.icon size={19}/><span>{f.name}</span></div>)}</div><div className="diagram-caption">Una representación del proceso.<br/>Tus predicciones aparecen al clasificar una imagen.</div></div></section>
        <div className="stats-grid"><Stat icon={<Database/>} value={dataset ? format(dataset.total_processed,0) : '—'} label="Imágenes procesadas"/><Stat icon={<Layers/>} value={status ? String(status.categories.length) : '—'} label="Categorías disponibles"/><Stat icon={<BarChart3/>} value={String(dimensions)} label="Características por imagen"/><Stat icon={<GitBranch/>} value="4" label="Algoritmos de clasificación"/></div>
        <div className="section-heading"><div><span className="eyebrow">CONOCE A LOS CLASIFICADORES</span><h2>Diferentes estrategias, la misma imagen.</h2></div><span className="small muted">Decision Tree incluye Gini y entropía</span></div>
        <div className="classifier-cards">{families.map((f,index) => { const available = status?.models.find(m => m.key === f.key)?.available; const metric = experiments?.metrics.find(m => m.key === f.key); return <section className={`classifier-card classifier-${index}`} key={f.key}><div className="feature-top"><span className="feature-icon"><f.icon size={24}/></span><span className={available ? 'available-label' : 'pending-label'}><i/>{available ? 'Entrenado' : 'Pendiente'}</span></div><span className="eyebrow">{f.tag}</span><h3>{f.name}</h3><p>{descriptions[f.key]}</p><div className="model-card-result"><span>{descriptorName} · prueba histórica</span><strong>{metric ? `${format(metric.accuracy*100,2)} %` : 'Sin evaluar'}</strong></div><button className="text-button" onClick={() => openEvaluation(f.key)}>Consultar resultados<ArrowUpRight size={16}/></button></section> })}</div>
        <section className="panel workflow-panel"><div className="section-heading"><div><span className="eyebrow">EL PROCESO, SIN CAJAS NEGRAS</span><h2>¿Cómo llega una imagen a una predicción?</h2></div></div><ProcessFlow descriptorId={descriptorId}/></section>
        <div className="home-bottom"><div className="panel dataset-summary"><span className="eyebrow">TU DATASET ACADÉMICO</span><h3>{status?.dataset || 'Preparación pendiente'}</h3><p>{status ? `${format(status.indexed,0)} imágenes de entrenamiento y ${format(status.queries,0)} de prueba histórica reutilizada.` : 'Conoce la colección y el protocolo experimental.'}</p><button className="text-button" onClick={() => setPage('dataset')}>Explorar la metodología<ArrowUpRight size={16}/></button></div><button className="lab-banner" onClick={() => setPage('explorer')}><GitBranch size={32}/><div><h3>Abre el árbol. Sigue la decisión.</h3><p>Explora divisiones, clases e importancias de modelos reales.</p></div><ArrowRight size={23}/></button></div>
      </>}

      {page === 'classify' && <>
        <div className="workspace-grid"><UploadCard {...{file,preview,busy,onError:setError}} samples={dataset?.samples || []} onChoose={chooseFile}/><section className="panel config-panel classification-config"><div className="panel-title"><span className="step-badge">02</span><h3>Elige cómo clasificar</h3></div>
          <label className="field">Modelo<select aria-label="Modelo de clasificación" value={family} disabled={busy} onChange={e => { setFamily(e.target.value); const hadResult=!!classification; setClassification(null); if(hadResult && file) void classify(false,(e.target.value==='decision_tree' ? criterion : e.target.value) as ModelKey) }}>{families.map(f => <option key={f.key} value={f.key === 'gini' ? 'decision_tree' : f.key} disabled={!status?.models.find(m => m.key === f.key)?.available}>{f.name}{!status?.models.find(m => m.key === f.key)?.available ? ' · Pendiente' : ''}</option>)}</select></label>
          {family === 'decision_tree' && <Criteria value={criterion} onChange={value => { setCriterion(value); const hadResult=!!classification; setClassification(null); if(hadResult && file) void classify(false,value) }} disabled={busy}/>}
          <div className="explanation"><Info size={18}/><p>{descriptions[selectedKey]}</p></div>
          <div className="descriptor-inline"><BarChart3 size={16}/><span>Descriptor: {descriptorName} · {dimensions} dimensiones</span></div>
          <button className="button primary wide" disabled={busy || !file || !selectedModel?.available} onClick={() => void classify()}>{busy ? <LoaderCircle className="spin" size={18}/> : <ScanLine size={18}/>} {busy ? 'Clasificando…' : 'Clasificar imagen'}<ArrowRight size={17}/></button>
          <button className="button secondary wide compare-four-button" disabled={busy || !file || !status?.classification_ready} onClick={() => void classify(true)}><Layers size={17}/>Comparar los cuatro modelos</button>
          <p className="field-help">La comparación usa una sola extracción del vector. Decision Tree utiliza el criterio {criterion === 'gini' ? 'Gini' : 'entropía'}.</p>
        </section></div>
        {classification && classification.predictions.length > 1 && <section className="panel same-image-comparison"><div className="section-heading"><div><span className="eyebrow">MISMA FOTOGRAFÍA · MISMO VECTOR</span><h2>Cuatro modelos, cuatro predicciones</h2></div><span className={classification.unanimous ? 'success-badge' : 'disagreement-badge'}>{classification.unanimous ? 'Coincidencia entre modelos' : 'Hay desacuerdo'}</span></div><p className="muted">{classification.unanimous ? 'Los cuatro modelos proponen la misma categoría. La coincidencia no garantiza que sea correcta.' : 'Los modelos proponen categorías diferentes. Sin conocer la etiqueta real de esta foto, no podemos afirmar cuál acertó.'}</p><div className="table-scroll"><table aria-label="Comparación de predicciones"><thead><tr><th>Modelo</th><th>Categoría predicha</th><th>Probabilidad estimada</th><th>Inferencia (ms)</th><th>Coincidencias</th></tr></thead><tbody>{classification.predictions.map(p => <tr key={p.key} className={prediction?.key === p.key ? 'focused-row' : ''}><td><button className="text-button" onClick={() => setPredictionKey(p.key)}>{p.name}</button></td><td><span className={!classification.unanimous ? 'category-disagreement' : ''}>{category(p.category)}</span></td><td>{format(p.probabilities[0]?.value*100 || 0,2)} %</td><td>{format(p.inference_ms,3)}</td><td>{classification.agreement[p.category]} de {classification.predictions.length}</td></tr>)}</tbody></table></div><p className="small muted">Probabilidades sin calibración: su valor no es directamente comparable como garantía de acierto. Selecciona una fila para inspeccionar ese resultado.</p></section>}
        {!!classification?.unavailable.length && <div className="notice warning">No se ejecutaron: {classification.unavailable.map(m => m.name).join(', ')}. Sus resultados no se han simulado.</div>}
        {prediction ? <section className="panel prediction"><div className="section-heading"><div><span className="eyebrow">PREDICCIÓN CALCULADA · {prediction.name}</span><h2>{category(prediction.category)}</h2></div><span className="inference-badge"><Timer size={15}/>{format(prediction.inference_ms,3)} ms de inferencia</span></div><p className="muted">El modelo considera que esta es la categoría más probable entre las {status?.categories.length} clases conocidas. {isResnet ? 'Utiliza patrones visuales de ResNet-18' : 'Utiliza color y orientación de bordes'}; puede equivocarse ante fotografías distintas al dataset.</p><div className="prediction-tabs">{classification!.predictions.length > 1 && classification!.predictions.map(p => <button key={p.key} className={prediction.key === p.key ? 'selected' : ''} onClick={() => setPredictionKey(p.key)}>{p.name}</button>)}</div>
          <div className={`ground-truth ${knownSample ? (knownSample.category===prediction.category ? 'correct' : 'incorrect') : ''}`} data-testid="ground-truth">
            {knownSample ? <><strong>Muestra identificada del conjunto de prueba · #{knownSample.id}</strong><p>Categoría verdadera: <b>{category(knownSample.category)}</b> · Predicción: <b>{category(prediction.category)}</b>.</p><p>{knownSample.category===prediction.category ? 'El modelo acertó en esta muestra.' : 'El modelo no acertó en esta muestra.'} Este caso individual no reemplaza las métricas del conjunto completo.</p></> : <><strong>Categoría verdadera desconocida</strong><p>Imagen cargada por el usuario. No inferimos su etiqueta a partir del nombre del archivo ni de la predicción.</p></>}
          </div>
          <h3>Categorías más probables</h3><p className="small muted">{prediction.probability_label}. No es una garantía de acierto.</p>
          {prediction.probabilities.slice(0,5).map(p => <div className="probability" key={p.category}><span>{category(p.category)}</span><div><i style={{width:`${p.value*100}%`}}/></div><b>{format(p.value*100,2)} %</b></div>)}
          <details><summary>Ver probabilidades de las {prediction.probabilities.length} categorías</summary><div className="probability-list">{prediction.probabilities.map(p => <div key={p.category}><span>{category(p.category)}</span><strong>{format(p.value*100,4)} %</strong></div>)}</div></details>
          {prediction.decisions && <details><summary>Ruta real de decisión · {prediction.decisions.length} nodos</summary><div className="decision-path">{prediction.decisions.map(node => <div key={node.id}><GitBranch size={17}/>{node.leaf ? <strong>Hoja → {category(prediction.category)}</strong> : <span>{node.feature_name}: {format(node.value!,5)} {node.direction === 'left' ? '≤' : '>'} {format(node.threshold!,5)}<small>{prediction.key === 'gini' ? 'Gini' : 'Entropía'}: {format(node.impurity,5)}</small></span>}</div>)}</div></details>}
          <section className="category-examples" aria-label="Ejemplos de la categoría predicha"><div className="section-heading"><h3>Ejemplos reales del dataset: {category(prediction.category)}</h3><span className="success-badge">Imágenes de entrenamiento</span></div><p className="small muted">Ilustran la categoría predicha. No son vecinos recuperados por similitud ni demuestran que la consulta esté bien clasificada.</p><CategoryGallery images={prediction.examples || []} onOpen={setModal}/></section>
          <div className="action-row"><button className="button secondary" onClick={() => openExplorer(prediction.key)}><GitBranch size={17}/>{prediction.key === 'xgboost' || prediction.key === 'lightgbm' ? 'Explorar importancias' : 'Explorar este árbol'}</button><button className="text-button" onClick={() => openEvaluation(prediction.key)}>Ver evaluación del modelo<ArrowUpRight size={16}/></button></div>
        </section> : <Empty icon={<ScanLine/>} title="Tu imagen es el punto de partida." text="Carga una fotografía, selecciona un modelo entrenado y obtén una predicción real."/>}
        {classification && <><div className="panel timing-summary"><span>Extracción: {format(classification.extraction.extraction_ms,2)} ms · {classification.extraction.device}{classification.extraction.cache_hit ? ' · vector en caché' : ''}</span><span>Total de solicitud: {format(classification.total_ms,2)} ms</span></div><Descriptor values={classification.descriptor}/></>}
        <section className="panel availability-panel"><h3>Estado de los cuatro algoritmos</h3><div className="model-list">{families.map(f => { const model = status?.models.find(m => m.key === f.key); return <div key={f.key}><span className={`model-dot ${model?.available ? 'available' : ''}`}/><div><strong>{f.name}</strong><p>{model?.available ? 'Entrenado y disponible' : model?.reason || 'Sin conexión'}</p></div></div> })}</div><p className="small muted">Gini y entropía son criterios del mismo Decision Tree. El k-NN se conserva en el laboratorio complementario.</p></section>
      </>}
      {page === 'evaluation' && <><div className="tab-row"><button className={evaluationScope==='historical_test' ? 'selected' : ''} onClick={() => setEvaluationScope('historical_test')}>Prueba histórica reutilizada</button><button className={evaluationScope==='training_cv' ? 'selected' : ''} onClick={() => setEvaluationScope('training_cv')}>Validación interna — entrenamiento</button></div><EvaluationView key={descriptorId+':'+evaluationScope} descriptorId={descriptorId} scope={evaluationScope} fallback={experiments} focusKey={focusKey}/></>}
      {page === 'explorer' && <ClassifierExplorer key={descriptorId} descriptorId={descriptorId} initialKey={explorerKey} {...{file,preview,status}} onClassify={() => setPage('classify')}/>}
      {page === 'dataset' && <DatasetMethodology descriptorId={descriptorId} dataset={dataset} experiments={experiments}/>}
      {page === 'metric' && <MetricLaboratory {...{file,preview,status}} samples={dataset?.samples || []} onChoose={chooseFile} onError={setError}/>}
      <footer><span><Atom size={15}/>MetricLab Visual</span><span>Clasificación de imágenes · Cuatro algoritmos · Resultados reales.</span></footer>
    </main></div>{modal && <ImageModal image={modal} onClose={() => setModal(null)}/>}
  </div>
}

function ProcessFlow({descriptorId}: {descriptorId:string}) {
  const isResnet=descriptorId===RESNET_ID
  return <div className="workflow classification-flow">{[
    {icon:ImagePlus,title:'Imagen',text:'Una fotografía JPG o PNG.'},
    {icon:ScanLine,title:'Preprocesamiento',text:isResnet ? 'RGB, resize 256 y recorte 224.' : 'Normalización a 128 × 128.'},
    {icon:BarChart3,title:'Vector numérico',text:isResnet ? '512 componentes de ResNet-18.' : '128 valores HSV + 16 de bordes.'},
    {icon:GitBranch,title:'Clasificador',text:'Uno de los cuatro modelos.'},
    {icon:CheckCircle2,title:'Categoría predicha',text:'Una clase y sus probabilidades.'},
  ].map((item,i) => <div className="workflow-item" key={item.title}><div className="workflow-icon"><item.icon size={22}/><span>{i+1}</span></div><strong>{item.title}</strong><p>{item.text}</p></div>)}</div>
}

function DatasetMethodology({dataset,experiments,descriptorId}: {dataset: DatasetInfo | null; experiments: Experiments | null; descriptorId:string}) {
  if (!dataset) return <Empty icon={<Database/>} title="Dataset pendiente" text="Prepara las imágenes para consultar sus estadísticas y metodología."/>
  const counts = Object.entries(dataset.counts).sort((a,b) => b[1]-a[1])
  return <>
    <div className="stats-grid"><Stat value={format(dataset.total_processed,0)} label="Imágenes procesadas"/><Stat value={String(dataset.categories.length)} label="Categorías"/><Stat value={format(dataset.indexed,0)} label="Entrenamiento"/><Stat value={format(dataset.queries,0)} label="Prueba histórica reutilizada"/></div>
    <section className="panel workflow-panel"><span className="eyebrow">DE LA IMAGEN AL RESULTADO</span><h2>Un proceso común para todos los modelos</h2><ProcessFlow descriptorId={descriptorId}/></section>
    <div className="info-grid"><section className="panel"><Database className="info-icon"/><h2>{dataset.dataset} · datos locales</h2><p>Fuente: colección académica Caltech-101, organizada por categoría. Se excluye BACKGROUND_Google porque representa fondo y no una clase de objeto.</p><p>Entrenamiento y prueba se separaron por categoría con semilla {dataset.seed}, aproximadamente {format((1-dataset.query_fraction)*100,0)}/{format(dataset.query_fraction*100,0)}. No comparten rutas de imágenes.</p><p>{dataset.max_per_class ? `Subconjunto limitado a ${dataset.max_per_class} imágenes por clase.` : 'Se utilizaron todas las imágenes válidas de las clases incluidas, sin límite por categoría.'} Archivos omitidos: {dataset.skipped}.</p><span className="success-badge">{dataset.kind === 'caltech101' ? 'Dataset académico real' : 'Datos sintéticos de demostración'}</span></section>
    <section className="panel"><BarChart3 className="info-icon"/><h2>Dos descriptores compatibles</h2><p><strong>HSV + Sobel · 144D:</strong> BGR a 128 × 128, histograma HSV de 128 valores y 16 orientaciones de bordes, normalizados por separado.</p><p><strong>ResNet-18 · 512D:</strong> pesos ImageNet oficiales, imagen RGB con resize 256, center crop 224 y normalización oficial. CNN congelada; salida previa a la capa final.</p><code>{dataset.descriptor} · {dataset.dimensions}D</code><p>Cada descriptor carga exclusivamente sus modelos. La prueba histórica ya se utilizó; no es una evaluación independiente nueva.</p></section>
    <section className="panel"><GitBranch className="info-icon"/><h2>Entrenamiento y evaluación</h2><p>Decision Tree (Gini/entropía), Random Forest, XGBoost y LightGBM se evalúan sobre las mismas imágenes de prueba. Semilla de modelos: 0.</p><p>{experiments?.protocol || 'Evaluación pendiente.'}</p><p>Los modelos se guardan y se cargan al iniciar la aplicación. Ninguna consulta vuelve a entrenarlos. Se registran parámetros, versiones, tiempos y huella de la división.</p></section>
    <section className="panel"><Info className="info-icon"/><h2>Qué puede y qué no puede concluirse</h2><p>El descriptor resume color y bordes; no comprende el significado de una imagen. Una foto ajena a las clases conocidas puede recibir una etiqueta incorrecta con alta probabilidad.</p><p>La colección está desbalanceada: entre {dataset.imbalance.min_train} y {dataset.imbalance.max_train} imágenes de entrenamiento por categoría (relación {format(dataset.imbalance.ratio,1)}×). Por eso mostramos balanced accuracy, precision/recall macro y F1 macro.</p><p>Probabilidades sin calibrar, una sola división fija y tiempos dependientes del equipo. La demo HTML anterior es sintética y sus métricas no se mezclan con Caltech.</p></section></div>
    <section className="panel"><div className="section-heading"><h2>Distribución por categoría</h2><span className="small muted">{dataset.categories.length} categorías reales</span></div><div className="table-scroll distribution-table"><table><thead><tr><th>Categoría</th><th>Entrenamiento</th><th>Prueba</th><th>Distribución en entrenamiento</th></tr></thead><tbody>{counts.map(([name,count]) => <tr key={name}><td>{category(name)}</td><td>{count}</td><td>{dataset.class_counts.query[name]}</td><td><div className="distribution-bar"><i style={{width:`${count/dataset.imbalance.max_train*100}%`}}/></div></td></tr>)}</tbody></table></div></section>
    <details className="panel"><summary>Identidad de los datos y versiones</summary><p className="small muted">Huella: {dataset.fingerprint}</p><pre className="json-view">{JSON.stringify(experiments?.versions || {},null,2)}</pre></details>
  </>
}

function EvaluationView({descriptorId,scope,fallback,focusKey}:{descriptorId:string;scope:string;fallback:Experiments|null;focusKey:ModelKey}) {
  const [data,setData]=useState<Experiments|null>(()=>fallback?.descriptor_id===descriptorId && fallback.evaluation_scope===scope ? fallback : null)
  const [error,setError]=useState('')
  useEffect(() => { let active=true; setData(null); setError(''); api<Experiments>(`/api/experiments?descriptor_id=${descriptorId}&evaluation=${scope}`).then(r => {if(active) setData(r)}).catch(e => {if(active) setError(e.message)}); return () => {active=false} },[descriptorId,scope])
  if(error) return <div className="notice warning">{error}</div>
  if(!data) return <div className="loading-line"><LoaderCircle className="spin" size={18}/>Cargando evaluación…</div>
  if(scope==='historical_test') return <ClassifierEvaluation experiments={data} focusKey={focusKey}/>
  return <section className="panel"><h2>Validación interna, solo entrenamiento</h2><p className="muted">{data.protocol || data.message}</p>{data.cv_metrics && <div className="table-scroll"><table><thead><tr><th>Configuración original</th><th>Accuracy media ± DE</th><th>Balanced acc. media ± DE</th><th>F1 macro media ± DE</th></tr></thead><tbody>{data.cv_metrics.map(r => <tr key={r.key}><td>{names[r.key]}{r.key==='lightgbm' ? ' · antes del ajuste' : ''}</td><td>{format(r.mean.accuracy*100)} ± {format(r.std.accuracy*100)} %</td><td>{format(r.mean.balanced_accuracy*100)} ± {format(r.std.balanced_accuracy*100)} %</td><td>{format(r.mean.f1_macro,4)} ± {format(r.std.f1_macro,4)}</td></tr>)}</tbody></table></div>}{data.internal_confirmation && <><h3>LightGBM: confirmación interna reservada tras la búsqueda</h3><p className="small muted">Partición exclusivamente de entrenamiento, no usada para seleccionar parámetros. No es el mismo alcance que la CV de cinco folds anterior.</p><div className="table-scroll"><table><thead><tr><th>Configuración</th><th>Accuracy</th><th>Balanced accuracy</th><th>F1 macro</th></tr></thead><tbody>{Object.entries(data.internal_confirmation).map(([key,r]) => <tr key={key}><td>{key==='original' ? 'Original' : 'Regularizada seleccionada'}</td><td>{format(r.accuracy*100)} %</td><td>{format(r.balanced_accuracy*100)} %</td><td>{format(r.f1_macro,4)}</td></tr>)}</tbody></table></div></>}<p className="small muted">DE describe dispersión entre folds; no es un intervalo de confianza. El descriptor ya fue explorado antes con la prueba histórica.</p></section>
}
