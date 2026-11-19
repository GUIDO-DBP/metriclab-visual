import { useEffect, useRef, useState } from 'react'
import { ArrowLeft, ChevronLeft, ChevronRight, GitBranch, Info, Layers, LoaderCircle, Route, Sparkles } from 'lucide-react'
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { api, category, descriptions, format, names } from './api'
import { Criteria, Empty } from './components'
import type { ClassifierTree, DecisionNode, Importance, ModelInfo, ModelKey, Status } from './types'
import { RESNET_ID } from './types'

export function ClassifierExplorer({ initialKey, file, preview, status, onClassify, descriptorId }: {
  descriptorId: string; initialKey: ModelKey; file: File | null; preview: string; status: Status | null; onClassify: () => void;
}) {
  const [key, setKey] = useState<ModelKey>(initialKey)
  const [treeIndex, setTreeIndex] = useState(0)
  const [root, setRoot] = useState(0)
  const [history, setHistory] = useState<number[]>([])
  const [info, setInfo] = useState<ModelInfo | null>(null)
  const [importance, setImportance] = useState<Importance | null>(null)
  const [tree, setTree] = useState<ClassifierTree | null>(null)
  const [selected, setSelected] = useState<DecisionNode | null>(null)
  const [path, setPath] = useState<DecisionNode[]>([])
  const [step, setStep] = useState(0)
  const [busy, setBusy] = useState(false)
  const [loading, setLoading] = useState(false)
  const [treeLoading,setTreeLoading]=useState(false)
  const traceSequence=useRef(0)
  const [error, setError] = useState('')
  const isTree = ['gini','entropy','forest'].includes(key)
  useEffect(() => { setKey(initialKey) }, [initialKey])
  useEffect(() => { setTreeIndex(0); setRoot(0); setHistory([]); setPath([]); setStep(0); setSelected(null) }, [key,descriptorId])
  useEffect(() => { traceSequence.current++; setBusy(false); setPath([]); setStep(0) }, [file,key,treeIndex,descriptorId])
  useEffect(() => {
    let cancelled = false
    setLoading(true); setError(''); setInfo(null); setImportance(null)
    Promise.all([api<ModelInfo>(`/api/models/${key}?descriptor_id=${descriptorId}`),api<Importance>(`/api/models/${key}/importance?descriptor_id=${descriptorId}`)]).then(([m,i]) => { if (!cancelled) { setInfo(m); setImportance(i) } }).catch(e => { if (!cancelled) setError(e.message) }).finally(() => { if (!cancelled) setLoading(false) })
    return () => { cancelled = true }
  },[key,status?.run_id,descriptorId])
  useEffect(() => {
    let cancelled = false
    setTree(null); setSelected(null)
    setTreeLoading(isTree)
    if (isTree) api<ClassifierTree>(`/api/models/${key}/tree?tree_index=${treeIndex}&root=${root}&max_depth=3&descriptor_id=${descriptorId}`).then(t => { if (!cancelled) { setTree(t); setSelected(t.nodes[0]) } }).catch(e => { if (!cancelled) setError(e.message) }).finally(()=>{if(!cancelled)setTreeLoading(false)})
    return () => { cancelled = true }
  },[key,treeIndex,root,status?.run_id,descriptorId])
  async function traceImage() {
    if (!file) return
    const sequence=++traceSequence.current
    setBusy(true); setError('')
    const form = new FormData(); form.append('file',file); form.append('tree_index',String(treeIndex)); form.append('descriptor_id',descriptorId)
    try { const result = await api<{path: DecisionNode[]}>(`/api/models/${key}/path`,{method:'POST',body:form}); if(sequence!==traceSequence.current)return; setPath(result.path); setStep(0); setRoot(0); setHistory([]); setSelected(result.path[0]) }
    catch (e) { if(sequence===traceSequence.current)setError((e as Error).message) } finally { if(sequence===traceSequence.current)setBusy(false) }
  }
  function goStep(next: number) {
    setStep(next); const node = path[next]; setSelected(node)
    if (tree && !tree.nodes.some(n => n.id === node.id)) { setHistory(h => [...h,root]); setRoot(node.id) }
  }
  const active = path.length ? path[step] : null
  return <>
    <section className="panel explorer-intro"><div><span className="eyebrow">MODELOS ENTRENADOS, NO EJEMPLOS SIMULADOS</span><h2>Explora cómo se construye una decisión</h2><p className="muted">{descriptions[key]}</p></div><label className="field">Algoritmo<select aria-label="Algoritmo del explorador" value={key === 'entropy' ? 'gini' : key} onChange={e => setKey(e.target.value as ModelKey)} disabled={busy}>{(['gini','forest','xgboost','lightgbm'] as const).map(k => <option key={k} value={k} disabled={!status?.models.find(m => m.key === k)?.available}>{k === 'gini' ? 'Decision Tree' : names[k]}</option>)}</select></label></section>
    {error && <div className="notice warning" role="alert"><Info size={18}/>{error}</div>}
    {loading && <div className="loading-line"><LoaderCircle className="spin" size={18}/>Leyendo el modelo entrenado…</div>}
    {isTree && <details className="panel tree-reading-guide"><summary>Cómo leer un árbol de decisión</summary><dl><dt>Nodo raíz</dt><dd>Es el nodo 0: el punto inicial del árbol completo. Reúne las muestras que llegaron al ajuste de ese árbol.</dd><dt>Característica y umbral</dt><dd>El nodo compara un componente numérico con un límite aprendido. Si el valor es menor o igual, sigue la rama izquierda; si es mayor, la derecha.</dd><dt>Impureza: Gini o entropía</dt><dd>Describe la mezcla de etiquetas en el nodo. Gini = 1 − Σp²; entropía = −Σp log₂p. Ambas valen 0 en un nodo puro. No son el porcentaje de error de una fotografía nueva.</dd><dt>Nodo hoja y predicción</dt><dd>La hoja no divide más. Su distribución de clases determina la predicción del árbol; no garantiza el acierto. El bosque promedia probabilidades de todos sus árboles, por lo que una hoja individual puede discrepar del bosque.</dd></dl>{descriptorId===RESNET_ID && <p>Las características son componentes del embedding ResNet-18. No se les atribuyen ojos, ruedas u otros significados visuales no comprobados.</p>}</details>}
    {isTree && <section className="panel"><div className="explorer-controls">
      {key !== 'forest' ? <Criteria value={key === 'entropy' ? 'entropy' : 'gini'} onChange={setKey} disabled={busy}/> : <label className="field">Árbol individual del bosque<select value={treeIndex} onChange={e => { setTreeIndex(Number(e.target.value)); setRoot(0); setHistory([]) }} disabled={busy}>{Array.from({length:info?.explorable_trees || 1},(_,i) => <option key={i} value={i}>Árbol {i+1} de {info?.tree_count || '…'}</option>)}</select></label>}
      <div className="explanation"><Info size={18}/><p>{key === 'forest' ? 'Se pueden explorar los primeros diez árboles reales. El bosque combina las probabilidades de todos sus árboles; un árbol individual puede decidir otra categoría.' : key === 'gini' ? 'Gini = 1 − Σp². Vale 0 cuando el nodo contiene una sola clase. Se busca reducir la impureza tras cada división.' : 'Entropía = −Σp log₂p. Vale 0 cuando no hay incertidumbre. La ganancia de información mide cuánto disminuye al dividir.'}</p></div>
    </div>
    {tree ? <><div className="tree-toolbar"><button className="text-button" disabled={!history.length} onClick={() => { setRoot(history.at(-1)!); setHistory(h => h.slice(0,-1)) }}><ArrowLeft size={16}/>Volver a la rama anterior</button><span className="small muted">{format(tree.total_nodes,0)} nodos totales · profundidad {tree.total_depth} · hasta 4 niveles visibles</span><button className="text-button" disabled={root === 0} onClick={() => { setRoot(0); setHistory([]) }}>Ir a la raíz</button></div>
      <DecisionDiagram tree={tree} selected={selected?.id} path={path.slice(0,step+1).map(n => n.id)} onSelect={setSelected}/>
      <div className="node-inspector" data-testid="node-inspector">{selected && <><div className="section-heading"><h3>{selected.leaf ? 'Hoja' : 'Nodo de decisión'} #{selected.id}</h3>{!selected.leaf && <button className="button secondary" disabled={selected.id === root} onClick={() => { setHistory(h => [...h,root]); setRoot(selected.id) }}>Explorar hijos de este nodo</button>}</div><div className="node-stats"><div><span>Característica</span><strong>{selected.feature_name}</strong></div><div><span>Umbral</span><strong>{selected.threshold === null ? 'Hoja final' : format(selected.threshold,6)}</strong></div><div><span>{tree.criterion === 'gini' ? 'Gini' : 'Entropía'}</span><strong>{format(selected.impurity,5)}</strong></div><div><span>Muestras / peso</span><strong>{format(selected.samples,0)} / {format(selected.weighted_samples,0)}</strong></div></div>
      <div className="node-reading" data-testid="node-explanation"><strong>{selected.id===0 ? 'Nodo raíz del árbol completo.' : selected.leaf ? 'Nodo hoja: aquí termina la ruta.' : 'Nodo interno: una pregunta aprendida.'}</strong>{selected.leaf ? <p>Este nodo no tiene otra división. La clase con mayor peso es <b>{category(selected.classes[0]?.category || '')}</b>; determina la respuesta de este árbol, no una certeza sobre la fotografía.</p> : <p>Compara <b>{selected.feature_name}</b> con <b>{format(selected.threshold!,6)}</b>. Valor ≤ umbral: rama izquierda (nodo {selected.left}). Valor &gt; umbral: rama derecha (nodo {selected.right}).</p>}<p>Impureza {tree.criterion==='gini' ? 'Gini' : 'entropía'}: {format(selected.impurity,5)}. Un valor 0 indica una única clase en las muestras del nodo. Los números mostrados se redondean; el modelo compara con su precisión original.</p>{descriptorId===RESNET_ID && !selected.leaf && <p>Esta característica es un componente numérico del embedding; no tiene una interpretación visual demostrada.</p>}</div>
      <p className="small muted">Las ramas izquierda y derecha corresponden a ≤ y &gt; del umbral. En Random Forest, el peso incluye repeticiones de la muestra bootstrap.</p>
      <details><summary>Clases presentes en el nodo · {selected.classes.length}</summary><div className="node-classes">{selected.classes.map(c => <div key={c.category}><strong>{category(c.category)}</strong><span>{format(c.proportion*100)} % · peso {format(c.weight,1)}</span></div>)}</div></details></>}</div>
    </> : treeLoading ? <div className="loading-line" role="status"><LoaderCircle className="spin" size={18}/>Leyendo nodos del árbol real…</div> : <Empty icon={<GitBranch/>} title="Árbol pendiente" text="Selecciona un modelo disponible para consultar sus nodos reales."/>}
    <div className="image-path-section"><div className="section-heading"><div><h3>La ruta de tu imagen</h3><p className="small muted">{key === 'forest' ? 'Recorrido del árbol individual seleccionado.' : 'Sigue las preguntas que responde el árbol para tu vector.'}</p></div>{file ? <button className="button primary" disabled={busy || !tree} onClick={() => void traceImage()}>{busy ? <LoaderCircle size={17} className="spin"/> : <Route size={17}/>}Ver ruta de imagen cargada</button> : <button className="button secondary" onClick={onClassify}>Cargar una imagen</button>}</div>
      {file && <div className="path-image"><img src={preview} alt="Imagen para explorar su ruta"/><span>{file.name}</span></div>}
      {!!path.length && <><div className="trace-controls"><button className="button secondary" aria-label="Decisión anterior" disabled={step === 0} onClick={() => goStep(step-1)}><ChevronLeft size={18}/></button><input aria-label="Paso de la ruta de clasificación" type="range" min="0" max={path.length-1} value={step} onChange={e => goStep(Number(e.target.value))}/><button className="button secondary" aria-label="Decisión siguiente" disabled={step === path.length-1} onClick={() => goStep(step+1)}><ChevronRight size={18}/></button><span>{step+1} / {path.length}</span></div>{active && <div className="trace-event"><strong>{active.leaf ? `Hoja → ${category(active.classes[0]?.category || '')}` : `Nodo ${active.id} · ${active.feature_name}`}</strong><p>{active.leaf ? 'La ruta termina aquí. La clase más representada determina la predicción de este árbol.' : `Valor ${format(active.value!,6)} ${active.direction === 'left' ? '≤' : '>'} umbral ${format(active.threshold!,6)} → rama ${active.direction === 'left' ? 'izquierda' : 'derecha'}.`}</p></div>}</>}
    </div></section>}
    {!isTree && info && <section className="panel boosting-explanation"><span className="feature-icon"><Sparkles size={25}/></span><div><h2>{names[key]} · {info.boosting_rounds} etapas reales de boosting</h2><p>En clasificación multiclase, cada etapa agrega árboles que ajustan las puntuaciones de las categorías. El resultado final combina todas las etapas, no una única ruta de árbol.</p><p className="small muted">{info.tree_count ? `${format(info.tree_count,0)} árboles en el modelo guardado. ` : ''}{info.n_classes} clases, {info.dimensions} características. A continuación se muestran importancias calculadas por el modelo entrenado.</p></div></section>}
    {importance && <section className="panel"><div className="section-heading"><div><span className="eyebrow">QUÉ CARACTERÍSTICAS UTILIZA EL MODELO</span><h2>Importancia de características · {names[key]}</h2></div><span className="small muted">{importance.measure}</span></div><p className="muted">{importance.note}</p><div className="importance-groups">{Object.entries(importance.groups).map(([group,value]) => <span key={group}><i/>{group==='hsv' ? 'HSV' : group==='edges' ? 'Bordes' : 'Embedding ResNet'} <strong>{format(value*100)} %</strong></span>)}</div><div className="chart"><ResponsiveContainer width="100%" height={360}><BarChart layout="vertical" data={importance.features.slice(0,12).map(f => ({...f,label:`f${f.index} · ${f.group}`}))} margin={{left:5,right:25}}><CartesianGrid horizontal={false} strokeDasharray="3 3"/><XAxis type="number" tickFormatter={value => `${format(value*100,1)} %`}/><YAxis type="category" dataKey="label" width={95}/><Tooltip formatter={value => `${format(Number(value)*100,3)} %`}/><Bar dataKey="value" name="Importancia normalizada" fill="#8773da" radius={[0,4,4,0]}/></BarChart></ResponsiveContainer></div><details><summary>Consultar las {importance.features.length} características</summary><div className="table-scroll feature-table"><table><thead><tr><th>Dimensión</th><th>Característica</th><th>Importancia</th></tr></thead><tbody>{importance.features.map(f => <tr key={f.index}><td>{f.index}</td><td>{f.name}</td><td>{format(f.value*100,4)} %</td></tr>)}</tbody></table></div></details></section>}
    {info && <details className="panel"><summary>Configuración y versión del modelo guardado</summary><pre className="json-view">{JSON.stringify(info.metadata,null,2)}</pre></details>}
  </>
}

function DecisionDiagram({ tree, selected, path, onSelect }: { tree: ClassifierTree; selected?: number; path: number[]; onSelect: (node: DecisionNode) => void }) {
  const byId = new Map(tree.nodes.map(n => [n.id,n]))
  const positions: {node:DecisionNode;x:number;y:number;parent?:{x:number;y:number};side?:string}[] = []
  function layout(id:number,x:number,y:number,width:number,parent?:{x:number;y:number},side?:string) {
    const node = byId.get(id); if (!node) return
    positions.push({node,x,y,parent,side}); layout(node.left,x-width/2,y+95,width/2,{x,y},'≤'); layout(node.right,x+width/2,y+95,width/2,{x,y},'>')
  }
  layout(tree.root,550,45,520)
  return <div className="tree-scroll classifier-tree"><svg viewBox="0 0 1100 385" aria-label="Árbol de decisión entrenado">{positions.map(p => p.parent && <g key={`line${p.node.id}`}><line x1={p.parent.x} y1={p.parent.y+28} x2={p.x} y2={p.y-28} stroke={path.includes(p.node.id) ? '#8872d7' : '#dde1ed'} strokeWidth={path.includes(p.node.id) ? 3 : 1.5}/><text x={(p.parent.x+p.x)/2+7} y={(p.parent.y+p.y)/2+3} fontSize="11" fill="#8d96ac">{p.side}</text></g>)}{positions.map(({node,x,y}) => <g role="button" tabIndex={0} aria-label={`Inspeccionar nodo ${node.id}`} key={node.id} onClick={() => onSelect(node)} onKeyDown={e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); onSelect(node) } }}><title>{`${node.feature_name}; ${node.leaf ? 'hoja' : 'umbral '+node.threshold}`}</title><rect x={x-59} y={y-28} width="118" height="57" rx="9" fill={path.includes(node.id) ? '#eee8fc' : node.leaf ? '#eef8f4' : '#fafbff'} stroke={selected === node.id ? '#7053ca' : '#d9dce9'} strokeWidth={selected === node.id ? 2.5 : 1.2}/><text x={x} y={y-9} textAnchor="middle" fontSize="10" fill="#8a80a5">{node.leaf ? 'Hoja' : 'Nodo'} #{node.id}</text><text x={x} y={y+6} textAnchor="middle" fontSize="10" fill="#57627c">{node.leaf ? category(node.classes[0]?.category || '').slice(0,18) : `f${node.feature} ≤ ${format(node.threshold!,4)}`}</text><text x={x} y={y+20} textAnchor="middle" fontSize="9" fill="#98a0b2">{node.samples} muestras</text></g>)}</svg></div>
}
