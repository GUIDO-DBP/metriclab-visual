import { test, expect } from '@playwright/test'
import fs from 'node:fs'
import path from 'node:path'
const root=path.resolve('..')
const manifest=JSON.parse(fs.readFileSync(path.join(root,'runtime/data/manifest.json'),'utf8'))
const resnet='torchvision-resnet18-imagenet1k-v1-avgpool512-fp32'
const hsv='hsv128-edge16-v1'

test('ResNet: modelos, galería de categoría, alcance experimental y reversión HSV',async ({page}) => {
  test.setTimeout(180000)
  const errors:string[]=[]; page.on('pageerror',e=>errors.push(e.message))
  await page.goto('/')
  await page.getByLabel('Descriptor visual').selectOption(resnet)
  await expect(page.getByText('4 clasificadores disponibles')).toBeVisible()
  await page.locator('.hero-cta').click()
  await page.locator('input[type=file]').setInputFiles(path.join(root,manifest.query[0].path))
  for(const model of ['decision_tree','forest','xgboost','lightgbm']) {
    await page.getByLabel('Modelo de clasificación').selectOption(model)
    await page.locator('.classification-config .primary').click()
    await expect(page.locator('.category-gallery .image-card')).toHaveCount(6,{timeout:90000})
    const category=await page.locator('.category-gallery .image-card').first().getAttribute('data-category')
    await expect(page.locator('.prediction h2')).toHaveText(category!.replaceAll('_',' '))
    for(const card of await page.locator('.category-gallery .image-card').all()) {
      await expect(card).toHaveAttribute('data-category',category!)
      await expect(card).toHaveAttribute('data-split','train')
      const url=await card.locator('img').getAttribute('src')
      const id=Number(url!.split('/').pop())
      expect(manifest.train[id].category).toBe(category)
      await expect.poll(()=>card.locator('img').evaluate((img:HTMLImageElement)=>img.complete && img.naturalWidth>0)).toBeTruthy()
    }
    await expect(page.locator('.category-gallery')).not.toContainText('Distancia L2')
  }
  await page.getByLabel('Modelo de clasificación').selectOption('decision_tree')
  await page.getByRole('button',{name:'Entropía',exact:true}).click()
  await page.locator('.classification-config .primary').click()
  await expect(page.locator('.prediction .eyebrow')).toContainText('Entropía')
  await page.getByRole('button',{name:'Comparar los cuatro modelos',exact:true}).click()
  const table=page.getByRole('table',{name:'Comparación de predicciones'})
  await expect(table.locator('tbody tr')).toHaveCount(4)
  for(const row of await table.locator('tbody tr').all()) {
    const category=(await row.locator('td').nth(1).innerText()).trim()
    await row.locator('button').click()
    await expect(page.locator('.prediction h2')).toHaveText(category)
    await expect(page.locator('.category-gallery .image-card strong').first()).toHaveText(category)
  }
  await page.locator('.category-gallery .image-card').first().click()
  await expect(page.getByRole('dialog')).toContainText('Fotografía real de entrenamiento')
  await page.keyboard.press('Escape')
  await page.screenshot({path:'../runtime/resnet-classification-desktop.png',fullPage:true})
  await page.setViewportSize({width:390,height:844})
  expect(await page.evaluate(()=>document.documentElement.scrollWidth<=window.innerWidth)).toBeTruthy()
  await page.screenshot({path:'../runtime/resnet-classification-mobile.png',fullPage:true})
  await page.setViewportSize({width:1440,height:1000})
  await page.getByRole('button',{name:'Comparación de clasificadores',exact:true}).click()
  await expect(page.locator('.evaluation-table tbody tr')).toHaveCount(5)
  await expect(page.getByText('Estos resultados no son una evaluación independiente nueva.',{exact:false})).toBeVisible()
  await page.getByRole('button',{name:'Validación interna — entrenamiento',exact:true}).click()
  await expect(page.getByRole('heading',{name:'LightGBM: confirmación interna reservada tras la búsqueda'})).toBeVisible()
  await page.getByLabel('Descriptor visual').selectOption(hsv)
  await expect(page.locator('.evaluation-table tbody tr')).toHaveCount(5)
  await page.getByRole('button',{name:'Clasificar imagen',exact:true}).first().click()
  await expect(page.locator('.prediction')).toHaveCount(0)
  await page.locator('.classification-config .primary').click()
  await expect(page.locator('.descriptor-inline')).toContainText('144 dimensiones')
  await expect(page.locator('.category-gallery .image-card')).toHaveCount(6)
  expect(errors).toEqual([])
})
