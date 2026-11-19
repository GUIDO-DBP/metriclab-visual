import type { Metric } from './types'

// Descriptive display rules only: never select, fit or change a model.
export function compareMetrics(metrics: Metric[]) {
  const leaders = (key: 'accuracy'|'f1_macro') => {
    const valid=metrics.filter(m=>Number.isFinite(m[key]))
    const best=Math.max(...valid.map(m=>m[key]))
    return valid.filter(m=>Math.abs(m[key]-best)<1e-12)
  }
  const close: {a:Metric;b:Metric;accuracyGap:number;f1Gap:number}[]=[]
  metrics.forEach((a,i)=>metrics.slice(i+1).forEach(b=>{
    const accuracyGap=Math.abs(a.accuracy-b.accuracy),f1Gap=Math.abs(a.f1_macro-b.f1_macro)
    if(accuracyGap<=.01+1e-12 && f1Gap<=.01+1e-12) close.push({a,b,accuracyGap,f1Gap})
  }))
  return {accuracy:leaders('accuracy'),f1:leaders('f1_macro'),close}
}

export function analyzeConfusion(classes:string[],matrix:number[][]) {
  if(!classes.length || matrix.length!==classes.length || matrix.some(row=>row.length!==classes.length || row.some(v=>!Number.isFinite(v)||v<0||!Number.isInteger(v)))) return null
  const categories=classes.map((name,index)=>{
    const support=matrix[index].reduce((a,b)=>a+b,0),correct=matrix[index][index]
    return {name,index,support,correct,errors:support-correct,recall:support ? correct/support : null,maximum:Math.max(...matrix[index],1)}
  })
  const supported=categories.filter(c=>c.recall!==null)
  const alphabetical=(a:{name:string},b:{name:string})=>a.name.localeCompare(b.name)
  const highest=[...supported].sort((a,b)=>b.recall!-a.recall! || alphabetical(a,b)).slice(0,5)
  const lowest=[...supported].sort((a,b)=>a.recall!-b.recall! || alphabetical(a,b)).slice(0,5)
  const confusions:{actual:string;predicted:string;count:number;support:number}[]=[]
  matrix.forEach((row,i)=>row.forEach((count,j)=>{
    if(i!==j && count>0) confusions.push({actual:classes[i],predicted:classes[j],count,support:categories[i].support})
  }))
  confusions.sort((a,b)=>b.count-a.count || a.actual.localeCompare(b.actual) || a.predicted.localeCompare(b.predicted))
  return {categories,highest,lowest,confusions:confusions.slice(0,5),total:categories.reduce((n,c)=>n+c.support,0),errors:categories.reduce((n,c)=>n+c.errors,0)}
}
