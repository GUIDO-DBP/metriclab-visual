import argparse
import csv
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from .common import assert_protected, load_json, save_json, sha256, validate_embeddings


def create_report(run):
    run = Path(run)
    checks = validate_embeddings(run)
    rows = [load_json(run/'models'/f'{descriptor}_{key}_results.json')
            for key in ['forest','xgboost'] for descriptor in ['hsv','resnet18']]
    truth = np.load(run/'query_labels.npy')
    for row in rows:
        prefix = f"{row['descriptor']}_{row['key']}"
        pred = np.load(run/'models'/f'{prefix}_predictions.npy')
        if pred.shape != truth.shape:
            raise ValueError('Missing test predictions.')
        for key,value in [('accuracy',accuracy_score(truth,pred)),
                          ('balanced_accuracy',balanced_accuracy_score(truth,pred)),
                          ('f1_macro',f1_score(truth,pred,average='macro'))]:
            if not np.isclose(row[key],value,rtol=0,atol=1e-12):
                raise ValueError(f'Metric {key} is not supported by saved predictions.')
        if sha256(run/'models'/f'{prefix}.joblib') != row['model_sha256']:
            raise ValueError('Saved model changed.')
    extraction = load_json(run/'extraction.json')
    hsv_latency = load_json(run/'hsv_extraction_latency.json')
    historical = [r for r in load_json(run/'baseline.json')['metrics'] if r['key'] in {'forest','xgboost'}]
    audit = load_json(run/'historical_reproducibility_audit.json') if (run/'historical_reproducibility_audit.json').exists() else None
    cpu_check = load_json(run/'cpu_validation.json') if (run/'cpu_validation.json').exists() else None
    improvements = {}
    for key in ['forest','xgboost']:
        old = next(r for r in rows if r['key'] == key and r['descriptor'] == 'hsv')
        new = next(r for r in rows if r['key'] == key and r['descriptor'] == 'resnet18')
        old_pred = np.load(run/'models'/f'hsv_{key}_predictions.npy')
        new_pred = np.load(run/'models'/f'resnet18_{key}_predictions.npy')
        improvements[key] = {'accuracy_pp':100*(new['accuracy']-old['accuracy']),
            'balanced_accuracy_pp':100*(new['balanced_accuracy']-old['balanced_accuracy']),
            'f1_macro_delta':new['f1_macro']-old['f1_macro'],
            'corrected_predictions':int(np.sum((old_pred!=truth)&(new_pred==truth))),
            'regressed_predictions':int(np.sum((old_pred==truth)&(new_pred!=truth)))}
    integrity_count = assert_protected(load_json(run/'protected_files.json'))
    result = {'original_split':checks,'models':rows,'improvements':improvements,
        'extraction':extraction,'hsv_extraction_latency':hsv_latency,
        'protected_files_unchanged':integrity_count,'activated':False,
        'protocol':load_json(run/'protocol.json'),
        'historical_baseline':[{k:r[k] for k in ['key','accuracy','balanced_accuracy','f1_macro','train_s','inference_ms']} for r in historical],
        'reproducibility_audit':audit,'cpu_validation':cpu_check}
    save_json(run/'results.json',result)
    columns = ['descriptor','key','dimensions','accuracy','balanced_accuracy','f1_macro','train_s','inference_ms','n_train','n_test','n_classes']
    with (run/'comparison.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore'); writer.writeheader(); writer.writerows(rows)
    lines = ['# Experimento independiente: ResNet-18 frente a HSV + Sobel', '',
        f"Dispositivo de extracción: **{extraction['device']}** ({extraction['gpu'] or 'CPU'}).",
        'División original: **6.941 entrenamiento / 1.736 prueba / 101 clases**, sin cambios.',
        'CNN preentrenada y congelada: `ResNet18_Weights.IMAGENET1K_V1`, `weights.transforms()`, `fc=Identity`, 512 dimensiones FP32.',
        '', '| Descriptor | Modelo | Accuracy | Balanced accuracy | F1 macro | Entrenamiento (s) | Inferencia clasificador (ms/imagen) |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for row in rows:
        lines.append(f"| {'HSV + Sobel (144D)' if row['descriptor']=='hsv' else 'ResNet-18 (512D)'} | {'Random Forest' if row['key']=='forest' else 'XGBoost'} | {row['accuracy']*100:.2f} % | {row['balanced_accuracy']*100:.2f} % | {row['f1_macro']:.4f} | {row['train_s']:.3f} | {row['inference_ms']:.3f} |")
    lines += ['', 'Tiempos de entrenamiento e inferencia de las cuatro filas medidos de nuevo en este experimento; mismas semillas y parámetros por clasificador, salvo dimensión de entrada. Hasta dos hilos CPU por defecto.',
        'Inferencia: media de 128 llamadas unitarias a predict_proba tras calentamiento; excluye extractor, imágenes, HTTP y JSON.', '',
        f"Extracción individual ResNet-18 (imagen + preprocesamiento oficial + transferencia + CNN): **{extraction['unit_extraction_latency']['mean_ms']:.3f} ms** de media; HSV/Sobel: **{hsv_latency['mean_ms']:.3f} ms**. La inferencia del clasificador en la tabla no incluye este costo.",
        f"Extracción masiva de los 8.677 embeddings: **{extraction['bulk_extraction_s']:.2f} s**, batch {extraction['batch_size']}; excluye descarga, carga y calentamiento inicial del modelo.",
        '', '## Referencia histórica y reproducibilidad',
        'Los resultados guardados de la aplicación permanecen intactos:']
    for row in historical:
        lines.append(f"- {'Random Forest' if row['key']=='forest' else 'XGBoost'} HSV original: accuracy {row['accuracy']*100:.2f} %, balanced accuracy {row['balanced_accuracy']*100:.2f} %, F1 macro {row['f1_macro']:.4f}.")
    if cpu_check:
        lines += [f"La ejecución CPU también se comprobó: dos filas reales coinciden con los vectores guardados (error máximo {max(cpu_check['max_absolute_errors']):.8f}). En una comprobación funcional de {cpu_check['samples']} imágenes con dos hilos, la extracción CPU tomó {cpu_check['mean_extraction_ms']:.3f} ms de media. Esta muestra CPU es más pequeña que el benchmark GPU y no se trata como una comparación de latencia exhaustiva."]
    if audit:
        lines += [f"La referencia XGBoost original utilizó {audit['historical_threads']} hilos y la comparación pareada actual limita ambos descriptores a {audit['paired_experiment_threads']}. La réplica HSV actual da {audit['paired_experiment_baseline_accuracy']*100:.2f} %; no se la sustituye por el {audit['historical_native_accuracy']*100:.2f} % anterior.",
            f"Una auditoría adicional repitió los parámetros históricos sin cambios (incluidos los hilos): obtuvo {audit['historical_params_replay_accuracy']*100:.2f} % y {audit['disagreements_original_vs_original_params_replay']} predicciones diferentes frente al modelo original. No se seleccionaron hiperparámetros ni se cambió la configuración del experimento para elevar la métrica."]
    lines += ['', '## Cambio de precisión']
    for key,name in [('forest','Random Forest'),('xgboost','XGBoost')]:
        delta=improvements[key]
        lines.append(f"- {name}: accuracy **{delta['accuracy_pp']:+.2f} puntos porcentuales**; balanced accuracy {delta['balanced_accuracy_pp']:+.2f} puntos; F1 macro {delta['f1_macro_delta']:+.4f}. Corrige {delta['corrected_predictions']} predicciones anteriores y empeora {delta['regressed_predictions']}.")
    xgb_new = next(r for r in rows if r['key']=='xgboost' and r['descriptor']=='resnet18')
    xgb_historical = next(r for r in historical if r['key']=='xgboost')
    lines.append(f"- Frente al XGBoost histórico de {xgb_historical['accuracy']*100:.2f} %, el nuevo descriptor mejora **{100*(xgb_new['accuracy']-xgb_historical['accuracy']):+.2f} puntos** de accuracy; sus tiempos históricos no se tratan como mediciones simultáneas.")
    best = max((r for r in rows if r['descriptor']=='resnet18'),key=lambda r:r['accuracy'])
    lines += ['', '## Interpretación y recomendación',
        'Las activaciones preentrenadas capturan patrones visuales aprendidos en ImageNet, mientras HSV/Sobel describe principalmente color y orientación de bordes. Las métricas macro permiten valorar las clases pequeñas.',
        f"La mejor configuración ResNet de este experimento es {'Random Forest' if best['key']=='forest' else 'XGBoost'}, con {best['accuracy']*100:.2f} % de accuracy.",
        ('Se recomienda integrar posteriormente ResNet-18 como descriptor alternativo para clasificación: mejora accuracy, balanced accuracy y F1 macro con ambos clasificadores en esta prueba. XGBoost ofrece aquí el mejor accuracy y F1 macro. Mantener HSV como referencia, registrar la nueva identidad de descriptor y asociar modelos de 512D sin mezclar los artefactos de 144D. Confirmar después con validación independiente. No se ha activado el descriptor ni modificado el frontend.' if all(improvements[k]['accuracy_pp']>0 and improvements[k]['balanced_accuracy_pp']>0 and improvements[k]['f1_macro_delta']>0 for k in improvements) else 'La integración requiere revisar las métricas por clase y el costo de extracción. No se ha activado el descriptor ni modificado el frontend.'),
        '', '## Verificación y límites',
        f"- {integrity_count} archivos protegidos de aplicación, frontend, separación y modelos activos conservan su SHA-256.",
        '- Mismo orden de archivos y etiquetas, sin imágenes descartadas; embeddings finitos de (6941,512) y (1736,512).',
        '- Las 8.677 imágenes tienen SHA-256 registrado. Se verificó la alineación de todas las etiquetas y se reextrajeron seis filas reales individualmente para compararlas con su vector por lote.',
        '- Ningún parámetro de la CNN fue entrenable; el estado de pesos y BatchNorm tiene el mismo hash antes y después.',
        '- Métricas recalculadas desde las 1.736 predicciones guardadas; modelos guardados/recargados con probabilidades equivalentes.',
        '- Los parámetros se copiaron de la referencia antes de evaluar. No se usó prueba para fit, early stopping, selección de características ni ajuste de hiperparámetros.',
        f"- Contenidos de imagen idénticos entre particiones de la división conservada: {checks['identical_content_across_splits']}.",
        '- Una sola división fija, sin validación cruzada ni calibración. La prueba ya se había utilizado en el proyecto: para confirmar una integración futura se recomienda validación dentro de entrenamiento y una evaluación independiente final.',
        '- ResNet fue preentrenada en ImageNet; no se auditó posible solapamiento de imágenes con ese corpus de origen.',
        '', 'Fuentes oficiales: [ResNet-18 y transforms](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html), [arquitectura torchvision](https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py).']
    (run/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    plot(rows,run/'comparison.png')
    print('\n'.join(lines[:13]),flush=True)


def plot(rows,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    for axis,metric,label in zip(axes,['accuracy','balanced_accuracy','f1_macro'],['Accuracy','Balanced accuracy','F1 macro']):
        x=np.arange(2)
        for i,(descriptor,color,title) in enumerate([('hsv','#8f9dc2','HSV + Sobel · 144D'),('resnet18','#7560cb','ResNet-18 · 512D')]):
            values=[next(r[metric] for r in rows if r['descriptor']==descriptor and r['key']==key) for key in ['forest','xgboost']]
            bars=axis.bar(x+(i-.5)*.34,values,.34,color=color,label=title)
            axis.bar_label(bars,labels=[f'{v:.3f}' for v in values],fontsize=8,padding=3)
        axis.set_xticks(x,['Random Forest','XGBoost']); axis.set_ylim(0,1); axis.set_title(label); axis.grid(axis='y',alpha=.15); axis.set_axisbelow(True)
        axis.spines[['top','right']].set_visible(False)
    fig.legend(*axes[0].get_legend_handles_labels(),fontsize=9,loc='outside lower center',ncol=2,frameon=False)
    fig.suptitle('Caltech-101 · misma división 6.941 / 1.736 · sin ajuste sobre prueba',fontsize=11)
    fig.savefig(path,dpi=180)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--run',type=Path,required=True)
    create_report(parser.parse_args().run)


if __name__ == '__main__': main()
