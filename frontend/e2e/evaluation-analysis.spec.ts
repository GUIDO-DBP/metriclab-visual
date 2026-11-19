import {test,expect} from '@playwright/test'
import {analyzeConfusion,compareMetrics} from '../src/evaluationAnalysis'
import type {Metric} from '../src/types'
import fs from 'node:fs'
import path from 'node:path'

test('cálculos resueltos: soporte, recall, errores direccionales y clases sin muestras',()=>{
  const r=analyzeConfusion(['A','B','C','D'],[[18,2,0,0],[1,2,1,0],[0,1,1,0],[0,0,0,0]])!
  expect(r.total).toBe(26);expect(r.errors).toBe(5)
  expect(r.highest.map(c=>c.name)).toEqual(['A','B','C'])
  expect(r.lowest.map(c=>c.name)).toEqual(['B','C','A'])
  expect(r.categories.map(c=>c.support)).toEqual([20,4,2,0])
  expect(r.categories.map(c=>c.recall)).toEqual([.9,.5,.5,null])
  expect(r.confusions[0]).toEqual({actual:'A',predicted:'B',count:2,support:20})
  expect(r.confusions.every(c=>c.actual!==c.predicted)).toBeTruthy()
})

test('una matriz inválida no se transforma en datos ficticios',()=>{
  for(const matrix of [[[1]],[[1,-1],[0,2]],[[1,.5],[0,2]],[[NaN,1],[0,2]]]) expect(analyzeConfusion(['A','B'],matrix)).toBeNull()
  expect(analyzeConfusion([],[])).toBeNull()
})

test('líderes empatados y cercanía descriptiva con límites explícitos',()=>{
  const row=(key:string,accuracy:number,f1_macro:number)=>({key,model:key,accuracy,f1_macro}) as Metric
  const r=compareMetrics([row('A',.9,.8),row('B',.9,.81),row('C',.89,.8),row('D',.7,.7)])
  expect(r.accuracy.map(m=>m.key)).toEqual(['A','B'])
  expect(r.f1.map(m=>m.key)).toEqual(['B'])
  expect(r.close.map(c=>c.a.key+c.b.key)).toEqual(['AB','AC','BC'])
})

test('las matrices reales de ambos descriptores conservan soportes y métricas históricas',()=>{
  const root=path.resolve('..'),read=(p:string)=>JSON.parse(fs.readFileSync(path.join(root,p),'utf8'))
  const manifest=read('runtime/data/manifest.json'),active=read('runtime/data/active_classifiers.json')
  const hsv=read('runtime/data/'+active.directory+'/experiments.json'),registry=read('runtime/descriptor_registry.json')
  const expected=new Map<string,number>()
  manifest.query.forEach((r:{category:string})=>expected.set(r.category,(expected.get(r.category)||0)+1))
  const resnet=Object.entries(registry.models).map(([key,e])=>({...read((e as {metrics_path:string}).metrics_path),key}))
  for(const [classes,metrics] of [[hsv.classes,hsv.metrics],[registry.classes,resnet]] as [string[],Metric[]][]) {
    for(const m of metrics) {
      const r=analyzeConfusion(classes,m.confusion)!
      expect(r.total).toBe(1736)
      for(const c of r.categories) expect(c.support).toBe(expected.get(c.name))
      expect((r.total-r.errors)/r.total).toBeCloseTo(m.accuracy,12)
      expect(r.categories.reduce((s,c)=>s+c.recall!,0)/101).toBeCloseTo(m.balanced_accuracy,12)
    }
    expect(compareMetrics(metrics).f1[0].key).toBe('xgboost')
  }
  expect(compareMetrics(resnet).accuracy[0].key).toBe('lightgbm')
})
