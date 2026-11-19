import { useEffect, useState } from 'react'
import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { BarChart3, CheckCircle2, Download, Info } from 'lucide-react'
import { category, format, names } from './api'
import { Empty, Stat } from './components'
import type { Experiments, ModelKey } from './types'

const metricExplanations = [
  ['Accuracy', 'Fracción de imágenes clasificadas correctamente. Las clases grandes tienen más peso.'],
  ['Balanced accuracy', 'Promedia los aciertos dentro de cada clase y da igual peso a categorías pequeñas y grandes.'],
  ['Precision macro', 'Por clase, qué proporción de sus predicciones fueron correctas; luego promedia sin ponderar.'],
  ['Recall macro', 'Por clase, qué proporción de sus imágenes se recuperó correctamente. En esta evaluación coincide con balanced accuracy.'],
  ['F1 macro', 'Combina precisión y recall dentro de cada clase y después promedia. Es útil para datos desbalanceados.'],
  ['Tiempo de inferencia', 'Costo de obtener probabilidades para una imagen. Se excluyen carga, extracción y dibujo de la interfaz.'],
]
export function ClassifierEvaluation({ experiments, focusKey }: { experiments: Experiments | null; focusKey: ModelKey }) {
  const [matrixKey, setMatrixKey] = useState<string>(focusKey)
  const [classFilter, setClassFilter] = useState('')
  useEffect(() => { setMatrixKey(focusKey) }, [focusKey])
  if (!experiments?.metrics.length) return <Empty icon={<BarChart3/>} title="Evaluación pendiente" text={experiments?.message || 'Entrena los clasificadores para consultar una evaluación real.'}/>
  const metrics = experiments.metrics.filter(m => m.key !== 'knn')
  const selected = metrics.find(m => m.key === matrixKey) || metrics[0]
  const classes = experiments.classes || []
  const indices = classes.map((name,index) => ({name,index})).filter(c => category(c.name).toLowerCase().includes(classFilter.toLowerCase()))
  const chart = metrics.map(m => ({...m, short_name: ({gini:'DT · Gini',entropy:'DT · Entropía',forest:'Random Forest',xgboost:'XGBoost',lightgbm:'LightGBM'} as Record<string,string>)[m.key] || m.model}))
  function downloadResults() {
    const blob = new Blob([JSON.stringify(experiments,null,2)], {type:'application/json'})
    const url = URL.createObjectURL(blob); const link = document.createElement('a'); link.href = url; link.download = `metriclab-evaluacion-${experiments!.run_id || 'local'}.json`; link.click(); URL.revokeObjectURL(url)
  }
  return <>
    <div className="stats-grid"><Stat value={format(experiments.n_train || 0,0)} label="Imágenes de entrenamiento"/><Stat value={format(experiments.n_test || 0,0)} label="Imágenes de prueba"/><Stat value={String(experiments.n_classes || classes.length)} label="Categorías evaluadas"/><Stat value="4" label="Algoritmos · DT con dos criterios"/></div>
    <section className="panel"><div className="section-heading"><div><span className="eyebrow">MISMO DATASET · MISMA DIVISIÓN · MISMO DESCRIPTOR</span><h2>¿Cómo se comporta cada clasificador?</h2></div><button className="button secondary" onClick={downloadResults}><Download size={16}/>Descargar evaluación</button></div>
      <p className="muted">{experiments.protocol}</p>
      <div className="chart"><ResponsiveContainer width="100%" height={310}><BarChart data={chart} margin={{left:0,right:6,bottom:15}}><CartesianGrid vertical={false} strokeDasharray="3 3"/><XAxis dataKey="short_name" interval={0} tick={{fontSize:10}}/><YAxis domain={[0,1]} tickFormatter={value => `${Math.round(value*100)} %`}/><Tooltip formatter={value => `${format(Number(value)*100)} %`}/><Legend/><Bar dataKey="accuracy" name="Accuracy" fill="#5f7de4" radius={[3,3,0,0]}/><Bar dataKey="balanced_accuracy" name="Balanced accuracy" fill="#a38be3" radius={[3,3,0,0]}/><Bar dataKey="f1_macro" name="F1 macro" fill="#6bb4ac" radius={[3,3,0,0]}/></BarChart></ResponsiveContainer></div>
      <div className="table-scroll evaluation-table"><table><caption>Cuatro algoritmos; Gini y entropía son dos variantes del mismo Decision Tree.</caption><thead><tr><th>Modelo / criterio</th><th>Accuracy</th><th>Balanced acc.</th><th>Precision macro</th><th>Recall macro</th><th>F1 macro</th><th>Entrenamiento (s)</th><th>Inferencia media (ms)</th></tr></thead><tbody>{metrics.map(m => <tr className={matrixKey === m.key ? 'focused-row' : ''} key={m.key}><td><button className="text-button" onClick={() => setMatrixKey(m.key)}>{m.model}</button>{m.reused && <small className="reused-label">Modelo conservado</small>}</td><td>{format(m.accuracy*100)} %</td><td>{format(m.balanced_accuracy*100)} %</td><td>{m.precision_macro === undefined ? 'Pendiente' : `${format(m.precision_macro*100)} %`}</td><td>{m.recall_macro === undefined ? 'Pendiente' : `${format(m.recall_macro*100)} %`}</td><td>{format(m.f1_macro,4)}</td><td>{m.train_s === null ? 'No registrado' : format(m.train_s,3)}</td><td>{m.inference_ms === undefined ? 'Pendiente' : format(m.inference_ms,3)}</td></tr>)}</tbody></table></div>
      <div className="explanation"><Info size={18}/><p>{experiments.timing_protocol} Los tiempos de entrenamiento de los modelos conservados proceden de su ejecución anterior; no se reentrenaron para cada predicción.</p></div>
    </section>
    <section className="panel"><div className="section-heading"><div><span className="eyebrow">DE LOS ACIERTOS A LAS CONFUSIONES</span><h2>Matriz de confusión</h2></div><span className="success-badge"><CheckCircle2 size={15}/>{format(experiments.n_test || 0,0)} imágenes evaluadas</span></div>
      <div className="matrix-settings"><label className="field">Modelo evaluado<select value={selected.key} onChange={e => setMatrixKey(e.target.value)}>{metrics.map(m => <option value={m.key} key={m.key}>{names[m.key as ModelKey] || m.model}</option>)}</select></label><label className="field">Filtrar categorías<input type="search" placeholder="Por ejemplo: accordion" value={classFilter} onChange={e => setClassFilter(e.target.value)}/></label></div>
      <p className="muted">Filas: categoría real. Columnas: categoría predicha. La diagonal contiene los aciertos; las demás celdas muestran confusiones. La intensidad se normaliza dentro de cada fila.</p>
      {classFilter && <p className="small muted">Filtro visual de filas y columnas; no recalcula métricas ni cambia la evaluación completa.</p>}
      <div className="confusion table-scroll"><table aria-label="Matriz de confusión"><thead><tr><th>Real ↓ / predicha →</th>{indices.map(c => <th key={c.index}>{category(c.name)}</th>)}</tr></thead><tbody>{indices.map(c => <tr key={c.index}><th>{category(c.name)}</th>{indices.map(p => { const value = selected.confusion[c.index][p.index]; const maximum = Math.max(...selected.confusion[c.index],1); return <td key={p.index} title={`Real: ${category(c.name)}; predicha: ${category(p.name)}; ${value} imágenes`} style={{background:`rgba(103,86,218,${value/maximum*.65})`}}>{value}</td> })}</tr>)}</tbody></table></div>
      {!indices.length && <p className="muted">No hay categorías con ese nombre.</p>}
    </section>
    <section className="panel"><span className="eyebrow">GUÍA PARA LEER LOS RESULTADOS</span><h2>Las métricas, en palabras sencillas</h2><div className="metric-glossary">{metricExplanations.map(([title,text]) => <div key={title}><h3>{title}</h3><p>{text}</p></div>)}</div><p className="small muted">El conjunto de prueba no se usa para entrenar, seleccionar características ni ajustar hiperparámetros. Estos resultados corresponden a una división fija; no son validación cruzada ni garantizan precisión con fotos externas.</p></section>
    <details className="panel"><summary>Parámetros, versiones y trazabilidad de esta ejecución</summary><p className="small muted">Ejecución: {experiments.run_id}. Huella de datos: {experiments.fingerprint}.</p><pre className="json-view">{JSON.stringify(experiments.versions,null,2)}</pre></details>
  </>
}
