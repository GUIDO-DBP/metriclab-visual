"""Assemble an editable academic report from the authored chapter sources."""
import re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
DOCS=ROOT/'documentacion_entrega'

def excerpt(filename,prefix,heading):
    source=(DOCS/filename).read_text(encoding='utf-8')
    lines=source.splitlines(); start=next(i for i,l in enumerate(lines) if l.startswith('## '+prefix))+1
    end=next((i for i in range(start,len(lines)) if lines[i].startswith('## ')),len(lines))
    body='\n'.join(lines[start:end]).strip()
    # Keep mathematical figures, but avoid carrying chapter figure numbers into APA report.
    body=re.sub(r'^!\[Figura[^\n]+\n?', '',body,flags=re.M)
    body=re.sub(r'^### \d+\.\d+\.\d+\. (.*)',r'#### \1',body,flags=re.M)
    return '### '+heading+'\n\n'+body+'\n\n'

def main():
    text='''<div class="academic-cover">

# MetricLab Visual: clasificación de imágenes mediante descriptores visuales y conjuntos de árboles

**[Nombres y apellidos de los autores]**

[Institución verificada]

[Facultad o escuela]

Estructuras de Datos Avanzadas

[Nombre del docente]

19 de noviembre de 2026

</div>

## Resumen

Se presenta MetricLab Visual, una aplicación local que compara Decision Tree, Random Forest, XGBoost y LightGBM en 101 categorías de Caltech-101. Se conservan 6.941 imágenes de entrenamiento y 1.736 de prueba histórica. La representación original concatena histogramas HSV y Sobel en 144 dimensiones; la alternativa utiliza 512 activaciones de ResNet-18 preentrenada y congelada. Los resultados descriptivos muestran mejoras amplias con el nuevo descriptor para bosque y XGBoost. Un diagnóstico posterior de LightGBM identificó márgenes excesivos y probabilidades saturadas, y una búsqueda controlada exclusivamente dentro de entrenamiento seleccionó una configuración regularizada. Esta alcanzó 87,62 % de accuracy en confirmación interna reservada y 87,33 % en la prueba histórica reutilizada. Se integraron ambos descriptores sin sustituir modelos anteriores, con galerías reales por categoría y trazabilidad. La entrega incluye instaladores y paquetes verificados para ejecutar modelos sin reentrenar. Se reconoce que la prueba histórica no constituye validación independiente nueva y que faltan evaluación externa y calibración.

*Palabras clave:* clasificación de imágenes, ResNet-18, árboles de decisión, boosting, reproducibilidad.

## Índice

[TOC]

## Introducción

'''
    text+=excerpt('01_INTRODUCCION_GENERAL.md','1.1.','Contexto y problema')
    text+='''## Problema, justificación y objetivos

El problema consiste en predecir una de 101 categorías a partir de fotografías con resoluciones y apariencias variables, manteniendo una evaluación interpretable. Una representación global simple puede perder organización espacial; una representación aprendida puede conservar patrones útiles, pero añade dependencias y costos. Comparar clasificadores sin controlar descriptor, separación y versiones produciría conclusiones poco trazables.

El objetivo general es desarrollar y estudiar una aplicación de clasificación que permita comparar métodos con datos reales. Los objetivos específicos son preservar separación e integridad; contrastar dos descriptores y cuatro familias; investigar el fallo de LightGBM sin ajustar sobre prueba; integrar resultados de forma reversible; y facilitar su estudio y ejecución por otro estudiante. El laboratorio métrico permite relacionar estas actividades con pivotes, distancias y poda, sin confundir índices de búsqueda con clasificadores supervisados.

## Marco teórico

ResNet fue propuesta por He et al. (2016) como una arquitectura de aprendizaje residual. Random Forest combina árboles aleatorizados (Breiman, 2001). XGBoost implementa boosting regularizado y optimizaciones para construir conjuntos de árboles (Chen & Guestrin, 2016). LightGBM desarrolla un sistema eficiente basado en árboles y técnicas de entrenamiento (Ke et al., 2017). El proyecto utiliza estas bibliotecas y mantiene sus parámetros reales; no implementa desde cero esos cuatro estimadores.

'''
    for prefix,heading in [('3.4.','Representaciones aprendidas'),('3.5.','Preentrenamiento y congelación'),('3.6.','Preprocesamiento reproducible'),('3.7.','Pooling y embedding')]: text+=excerpt('03_EXTRACCION_CARACTERISTICAS.md',prefix,heading)
    text+=excerpt('04_ALGORITMOS_CLASIFICACION.md','4.2.','Árboles de decisión e impureza')
    text+=excerpt('04_ALGORITMOS_CLASIFICACION.md','4.3.','Bosques y agregación')
    text+='## Metodología\n\n'
    for prefix,heading in [('2.1.','Dataset y atribución'),('2.4.','Separación fija y verificación')]: text+=excerpt('02_DATASET_CALTECH101.md',prefix,heading)
    for prefix,heading in [('5.3.','Validación interna original'),('5.4.','Búsqueda controlada y confirmación'),('5.5.','Prevención de fuga y alcance de evaluación')]: text+=excerpt('05_ENTRENAMIENTO_MODELOS.md',prefix,heading)
    text+='## Implementación\n\n'
    text+=excerpt('07_ARQUITECTURA_CODIGO.md','7.4.','Recorrido de una solicitud')
    text+=excerpt('07_ARQUITECTURA_CODIGO.md','7.5.','Worker y recursos persistentes')
    text+='''### Integración de ejemplos reales

Cada predicción se asocia con seis fotografías de entrenamiento cuya categoría exacta se obtiene del manifiesto. No se calculan distancias para elegirlas ni se presentan como validación del acierto. Los modelos se seleccionan por descriptor, con controles de clases, dimensión, fingerprint y hashes. El laboratorio C++ permanece limitado a HSV144 y mantiene su oráculo de búsqueda.

## Resultados

**Tabla 1**

*Resultados históricos HSV + Sobel (144 dimensiones)*

'''
    source=(DOCS/'06_ESTADISTICAS_RESULTADOS.md').read_text(encoding='utf-8')
    def table(section):
        content=source.split('## '+section,1)[1].split('\n## ',1)[0]
        return '\n'.join(l for l in content.splitlines() if l.startswith('|'))+'\n\n'
    text+=table('6.5.')
    text+='''*Nota.* N=1.736 imágenes de prueba histórica reutilizada. Los tiempos de entrenamiento de árboles y bosque se conservaron de ajustes anteriores. Inferencia excluye extracción y HTTP. Fuente: artefactos versionados de MetricLab Visual (2026).

**Tabla 2**

*Resultados descriptivos ResNet-18 integrados (512 dimensiones)*

'''+table('6.6.')
    text+='''*Nota.* Misma prueba histórica; no es evaluación independiente nueva. RF/XGB completos se conservaron. Los tiempos de las ejecuciones respectivas no son un benchmark simultáneo. Las diferencias pequeñas no establecen superioridad estadística.

**Figura 1**

*Accuracy y F1 macro por descriptor y clasificador*

![Comparación real de modelos activos.](assets/informe_resultados.png)

*Nota.* Elaboración propia desde los artefactos conservados. La tabla separa sus alcances; no se ajustaron los modelos para esta entrega.

'''
    text+=excerpt('10_EXPERIMENTOS_CONCLUSIONES.md','10.5.','Diagnóstico y mejora de LightGBM')
    text+='## Discusión\n\n'
    text+=excerpt('10_EXPERIMENTOS_CONCLUSIONES.md','10.8.','Interpretación de los hallazgos')
    text+=excerpt('10_EXPERIMENTOS_CONCLUSIONES.md','10.9.','Limitaciones')
    text+='''## Conclusiones

La comparación apoya mantener ResNet-18 como descriptor alternativo: conserva modelos anteriores y mejora las métricas descriptivas de conjuntos de árboles respecto de HSV. La representación no garantiza por sí sola un buen ajuste: LightGBM necesitó investigación y regularización. Su mejora se confirmó dentro de entrenamiento antes de volver a medir prueba.

La aplicación permite estudiar decisiones, importancias, matrices y ejemplos reales sin equiparar probabilidades con certeza ni galerías con recuperación métrica. La persistencia y los paquetes con hashes permiten ejecutar sin volver a entrenar. Las conclusiones de generalización siguen limitadas por reutilizar la prueba histórica y por la ausencia de una evaluación externa prospectiva.

'''
    text+=excerpt('10_EXPERIMENTOS_CONCLUSIONES.md','10.10.','Trabajo futuro')
    text+='## Referencias\n\n'
    refs=(DOCS/'BIBLIOGRAFIA.md').read_text(encoding='utf-8').split('\n\n')
    wanted=('Breiman,','Chen,','He,','Ke,','Li,','Los autores')
    text+='\n\n'.join(p for p in refs if p.startswith(wanted))+'\n\n'
    text+='''## Apéndice: fuentes editables y reproducibilidad

Los diez manuales complementarios desarrollan cálculos, código, pantallas e instalación. `resources/experiment_summary.json` y los paquetes versionados contienen la evidencia numérica. Los campos de portada son marcadores deliberados y deben completarse antes de presentar el trabajo; no se ha supuesto una institución a partir de documentos antiguos. El registro de prueba limpia distingue la comprobación en un segundo directorio de una prueba física en otro equipo.
'''
    (DOCS/'INFORME_GENERAL_METRICLAB_VISUAL.md').write_text(text,encoding='utf-8')
    # Academic chart with no chapter figure numbering embedded inside the image.
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import json,numpy as np
    s=json.loads((ROOT/'resources/experiment_summary.json').read_text(encoding='utf-8'));fig,axes=plt.subplots(1,2,figsize=(10,4));x=np.arange(5)
    for ax,k in zip(axes,['accuracy','f1_macro']):
        ax.bar(x-.18,[r[k] for r in s['hsv']],.36,label='HSV144',color='#87a4e5');ax.bar(x+.18,[r[k] for r in s['resnet18']],.36,label='ResNet512',color='#7765d0');ax.set_xticks(x,['Gini','Entropía','RF','XGB','LGBM'],rotation=25);ax.set_ylim(0,1);ax.set_title(k);ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(DOCS/'assets/informe_resultados.png',dpi=190,bbox_inches='tight');plt.close(fig)
    print('Informe académico editable ensamblado.')

if __name__=='__main__': main()
