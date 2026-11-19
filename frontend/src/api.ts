import type { ModelKey } from './types'
export const format = (v: number, digits = 2) => v.toLocaleString('es-PE', { maximumFractionDigits: digits })
export const category = (name: string) => name.replaceAll('_', ' ')
export const names: Record<ModelKey, string> = {
  gini: 'Decision Tree — Gini', entropy: 'Decision Tree — Entropía', forest: 'Random Forest', xgboost: 'XGBoost', lightgbm: 'LightGBM',
}
export const descriptions: Record<ModelKey, string> = {
  gini: 'Un árbol divide las imágenes con preguntas sobre sus características. Gini mide la mezcla de categorías: cuanto menor, más homogéneo es el nodo.',
  entropy: 'El mismo algoritmo Decision Tree utiliza entropía para medir incertidumbre. Busca divisiones que aportan información sobre la categoría.',
  forest: 'Random Forest combina las probabilidades de 200 árboles entrenados con distintas muestras y subconjuntos de características.',
  xgboost: 'XGBoost construye árboles de forma sucesiva: cada etapa corrige errores de las anteriores mediante boosting.',
  lightgbm: 'LightGBM aplica boosting con histogramas y crecimiento por hojas. Prioriza las divisiones con mayor ganancia.',
}
export async function api<T>(url: string, options?: RequestInit): Promise<T> {
  const response = await fetch(url, options)
  if (!response.ok) {
    const data = await response.json().catch(() => ({}))
    throw new Error(typeof data.detail === 'string' ? data.detail : 'No se pudo completar la solicitud. Revisa la imagen o vuelve a intentarlo.')
  }
  return response.json()
}
