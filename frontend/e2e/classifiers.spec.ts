import { test, expect } from '@playwright/test'
import fs from 'node:fs'
import path from 'node:path'
const root = path.resolve('..')
const manifest = JSON.parse(fs.readFileSync(path.join(root,'runtime/data/manifest.json'),'utf8'))
const sample = path.join(root,manifest.query[0].path)

test('cinco configuraciones reales, cuatro algoritmos y comparación sin reextraer', async ({page}) => {
  const errors: string[] = []
  page.on('pageerror',e => errors.push(e.message))
  await page.goto('/')
  await expect(page.locator('.classifier-card')).toHaveCount(4)
  await expect(page.getByText('4 clasificadores disponibles')).toBeVisible()
  await page.locator('.hero-cta').click()
  await page.locator('input[type=file]').setInputFiles(sample)
  for (const model of ['decision_tree','forest','xgboost','lightgbm']) {
    await page.getByLabel('Modelo de clasificación').selectOption(model)
    const response = page.waitForResponse(r => r.url().endsWith('/api/classify') && r.request().method() === 'POST')
    await page.locator('.classification-config .primary').click()
    const prediction = (await (await response).json()).predictions[0]
    await expect(page.locator('.prediction h2')).toHaveText(prediction.category.replaceAll('_',' '))
    await expect(page.locator('.inference-badge')).toContainText('ms de inferencia')
  }
  await page.getByLabel('Modelo de clasificación').selectOption('decision_tree')
  await page.getByRole('button',{name:'Entropía',exact:true}).click()
  await page.locator('.classification-config .primary').click()
  await expect(page.locator('.prediction .eyebrow')).toContainText('Entropía')
  await page.getByRole('button',{name:'Comparar los cuatro modelos',exact:true}).click()
  await expect(page.getByRole('table',{name:'Comparación de predicciones'}).locator('tbody tr')).toHaveCount(4)
  await expect(page.getByRole('table',{name:'Comparación de predicciones'})).toContainText('Entropía')
  await page.screenshot({path:'../runtime/classification-comparison-desktop.png',fullPage:true})
  expect(errors).toEqual([])
})

test('explorador: Gini, entropía, árboles del bosque y boosting', async ({page}) => {
  const errors: string[] = []; page.on('pageerror',e => errors.push(e.message))
  await page.goto('/')
  await expect(page.getByText('4 clasificadores disponibles')).toBeVisible()
  await page.locator('.hero-cta').click()
  await page.locator('input[type=file]').setInputFiles(sample)
  await page.getByRole('button',{name:'Explorador de árboles',exact:true}).click()
  await expect(page.getByRole('img',{name:'Árbol de decisión entrenado'})).toBeVisible()
  await expect(page.locator('.classifier-tree svg')).toBeVisible()
  await page.getByRole('button',{name:'Inspeccionar nodo 0',exact:true}).click()
  await expect(page.getByTestId('node-inspector')).toContainText('Característica')
  await page.getByRole('button',{name:'Ver ruta de imagen cargada',exact:true}).click()
  const slider = page.getByLabel('Paso de la ruta de clasificación')
  await expect(slider).toBeVisible()
  await slider.press('End')
  await expect(page.locator('.trace-event strong')).toContainText('Hoja →')
  await page.getByRole('button',{name:'Entropía',exact:true}).click()
  await expect(page.locator('.explorer-controls')).toContainText('ganancia de información')
  await expect(page.getByTestId('node-inspector')).toContainText('Entropía')
  await page.getByLabel('Algoritmo del explorador').selectOption('forest')
  await page.getByLabel('Árbol individual del bosque').selectOption('9')
  await expect(page.getByLabel('Árbol individual del bosque')).toContainText('Árbol 10 de 200')
  await expect(page.locator('.classifier-tree svg')).toBeVisible()
  await page.screenshot({path:'../runtime/classifier-explorer-desktop.png',fullPage:true})
  for (const key of ['xgboost','lightgbm']) {
    await page.getByLabel('Algoritmo del explorador').selectOption(key)
    await expect(page.locator('.boosting-explanation')).toContainText('80 etapas reales')
    await page.getByText('Consultar las 144 características',{exact:true}).click()
    await expect(page.locator('.feature-table tbody tr')).toHaveCount(144)
  }
  expect(errors).toEqual([])
})

test('evaluación real, matrices filtrables, exportación y metodología', async ({page}) => {
  await page.goto('/')
  await expect(page.getByText('4 clasificadores disponibles')).toBeVisible()
  await page.getByRole('button',{name:'Comparación de clasificadores',exact:true}).click()
  await expect(page.locator('.evaluation-table tbody tr')).toHaveCount(5)
  await page.getByLabel('Modelo evaluado').selectOption('xgboost')
  await page.getByLabel('Filtrar categorías').fill('accordion')
  await expect(page.getByRole('table',{name:'Matriz de confusión'}).locator('tbody tr')).toHaveCount(1)
  const download = page.waitForEvent('download')
  await page.getByRole('button',{name:'Descargar evaluación',exact:true}).click()
  const file = await download
  const data = JSON.parse(fs.readFileSync((await file.path())!,'utf8'))
  expect(data.n_test).toBe(1736)
  expect(data.metrics.map((m: {key:string}) => m.key)).toEqual(['gini','entropy','forest','xgboost','lightgbm'])
  await page.screenshot({path:'../runtime/classifier-evaluation-desktop.png',fullPage:true})
  await page.getByRole('button',{name:'Dataset y metodología',exact:true}).click()
  await expect(page.locator('.distribution-table tbody tr')).toHaveCount(101)
  await expect(page.getByText('Dataset académico real')).toBeVisible()
})

test('carga por arrastre, rechazo de imagen corrupta y recuperación', async ({page}) => {
  await page.goto('/')
  await expect(page.getByText('4 clasificadores disponibles')).toBeVisible()
  await page.locator('.hero-cta').click()
  await page.locator('input[type=file]').setInputFiles({name:'corrupta.jpg',mimeType:'image/jpeg',buffer:Buffer.from('not an image')})
  await page.locator('.classification-config .primary').click()
  await expect(page.getByRole('alert')).toContainText('Error de procesamiento')
  const base64 = fs.readFileSync(sample).toString('base64')
  const transfer = await page.evaluateHandle(encoded => {
    const data = Uint8Array.from(atob(encoded),c => c.charCodeAt(0))
    const transfer = new DataTransfer(); transfer.items.add(new File([data],'consulta.jpg',{type:'image/jpeg'})); return transfer
  },base64)
  await page.locator('.dropzone').dispatchEvent('drop',{dataTransfer:transfer})
  await expect(page.getByRole('alert')).toHaveCount(0)
  await page.locator('.classification-config .primary').click()
  await expect(page.locator('.prediction')).toHaveCount(1)
})
