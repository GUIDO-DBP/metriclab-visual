import { useState } from 'react'
import { ChevronLeft } from 'lucide-react'
import type { MetricNode as TreeNode, TraceEvent as Event } from './types'
export function MetricTree({ nodes, events, current }: { nodes: TreeNode[]; events: Event[]; current?: number }) {
  const [root, setRoot] = useState(0)
  const [history, setHistory] = useState<number[]>([])
  const positions: { node: TreeNode; x: number; y: number; parent?: { x: number; y: number } }[] = []
  function layout(id: number, x: number, y: number, width: number, level: number, parent?: { x: number; y: number }) { const node = nodes[id]; if (!node || level>3) return; positions.push({node,x,y,parent}); layout(node.left,x-width/2,y+85,width/2,level+1,{x,y}); layout(node.right,x+width/2,y+85,width/2,level+1,{x,y}) }
  layout(root, 480, 40, 460, 0)
  const visited = new Set(events.filter(e => e.kind === 'visit').map(e => e.node))
  const pruned = new Set(events.filter(e => e.kind === 'prune').map(e => e.node))
  return <><div className="tree-toolbar"><button className="text-button" disabled={!history.length} onClick={() => { setRoot(history.at(-1)!); setHistory(h => h.slice(0,-1)) }}><ChevronLeft size={16}/>Remontar</button><span className="small muted">4 niveles visibles · pulsa un nodo para explorar sus hijos</span><div className="tree-legend"><span>○ Pendiente</span><span className="purple">● Visitado</span><span className="red">● Podado</span></div></div><div className="tree-scroll"><svg viewBox="0 0 960 330" className="tree" aria-label="Topología real del VP-Tree">{positions.map(p => p.parent && <line key={`e${p.node.id}`} x1={p.parent.x} y1={p.parent.y} x2={p.x} y2={p.y} stroke="#dce1ed" strokeWidth="2"/>)}{positions.map(({node,x,y}) => <g key={node.id} role="button" tabIndex={0} aria-label={`Explorar nodo ${node.id}`} onClick={() => { if (node.left>=0 || node.right>=0) { setHistory(h => [...h,root]); setRoot(node.id) } }} onKeyDown={e => { if (e.key === 'Enter' && !node.leaf) { setHistory(h => [...h,root]); setRoot(node.id) } }}><title>{`Nodo ${node.id}, ${node.leaf ? 'hoja' : 'pivote ' + node.pivot}, μ=${node.radius}`}</title><circle cx={x} cy={y} r="20" fill={pruned.has(node.id) ? '#ffe3e3' : visited.has(node.id) ? '#7764e8' : '#f5f6fc'} stroke={current===node.id ? '#26334e' : pruned.has(node.id) ? '#e28989' : '#c9c8e3'} strokeWidth={current===node.id ? 3 : 1.5}/><text x={x} y={y+4} textAnchor="middle" fontSize="11" fill={visited.has(node.id) && !pruned.has(node.id) ? 'white' : '#566079'}>{node.id}</text></g>)}</svg></div></>
}
