import argparse
import csv
from pathlib import Path

import numpy as np

from experiments.resnet18.common import DESCRIPTOR, ROOT, assert_protected, load_json, save_json, sha256
from .core import KEYS, NAMES, assert_parent, calculate_metrics, parent_directory, validate_folds


def report(run):
    run=Path(run); parent=parent_directory(run)
    protocol=load_json(run/'protocol.json')
    selection=load_json(run/'cv_selection.json')
    folds=validate_folds(run)
    train_truth=np.load(parent/'train_labels.npy')
    test_truth=np.load(parent/'query_labels.npy')
    cv=[load_json(run/'cv'/key/'summary.json') for key in KEYS]
    final=[load_json(run/'final'/key/'results.json') for key in KEYS]
    for key,summary,result in zip(KEYS,cv,final):
        if summary['folds_sha256']!=sha256(run/'folds.json'): raise ValueError('Fold identity mismatch.')
        for record,fold in zip(summary['folds'],folds['folds']):
            prediction=np.load(run/'cv'/key/f"fold_{record['fold']}_predictions.npy")
            expected=calculate_metrics(train_truth[fold['validation']],prediction)
            for metric,value in expected.items():
                if not np.isclose(record[metric],value,atol=1e-12,rtol=0): raise ValueError('CV metric not supported by saved fold predictions.')
        oof=np.load(run/'cv'/key/'oof_predictions.npy')
        for metric,value in calculate_metrics(train_truth,oof).items():
            if not np.isclose(summary['oof'][metric],value,atol=1e-12,rtol=0): raise ValueError('OOF metric mismatch.')
        prediction=np.load(run/'final'/key/'predictions.npy')
        for metric,value in calculate_metrics(test_truth,prediction).items():
            if not np.isclose(result[metric],value,atol=1e-12,rtol=0): raise ValueError('Historical test metric mismatch.')
        if result['model_sha256']!=sha256(ROOT/result['model_path']): raise ValueError('Full model changed.')
    recommended=sorted(cv,key=lambda r:(-r['mean']['f1_macro'],-r['mean']['balanced_accuracy'],r['key']))[0]
    if recommended['key']!=selection['candidate']: raise ValueError('Recommendation must be supported by training-only CV.')
    counts={'application':assert_protected(load_json(run/'protected_files.json')),
            'parent_experiment':assert_parent(load_json(run/'parent_files.json'))}
    result={'run_id':run.name,'parent_run':protocol['parent_run'],'protocol':protocol,
        'cv_training_only':cv,'historical_test_reused':final,'cv_selection':selection,
        'data_counts':{'original_train':6941,'historical_test':1736,'categories':101,'features':512},
        'four_algorithms_five_configurations':True,'integrity':counts,
        'test_caveat':'These 1736 test images were already evaluated in prior experiments. They are NOT a new independent validation.',
        'validation_caveat':'Five-fold internal CV is conditional on the already chosen ResNet descriptor; not an external or nested assessment of descriptor selection.'}
    save_json(run/'results.json',result)
    columns=['key','name','accuracy','balanced_accuracy','f1_macro','fit_s','fit_time_source','inference_ms','full_model_reused','test_role']
    with (run/'historical_test_comparison.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore'); writer.writeheader(); writer.writerows(final)
    with (run/'training_cv_comparison.csv').open('w',newline='',encoding='utf-8') as f:
        columns=['key','accuracy_mean','accuracy_std','balanced_accuracy_mean','balanced_accuracy_std','f1_macro_mean','f1_macro_std','oof_accuracy','oof_balanced_accuracy','oof_f1_macro']
        writer=csv.DictWriter(f,fieldnames=columns); writer.writeheader()
        for row in cv:
            writer.writerow({'key':row['key'],**{f'{m}_{s}':row[s][m] for m in ['accuracy','balanced_accuracy','f1_macro'] for s in ['mean','std']},
                             **{f'oof_{m}':row['oof'][m] for m in ['accuracy','balanced_accuracy','f1_macro']}})
    classes=load_json(parent/'classes.json')
    extractor=load_json(parent/'extraction.json')
    catalog={'active':False,'purpose':'Integration proposal inventory, not read by the current app.',
        'descriptor':DESCRIPTOR,'dimensions':512,'split_fingerprint':protocol['split_fingerprint'],
        'classes':classes,'weights':extractor['weights'],'checkpoint_sha256':extractor['checkpoint_sha256'],
        'source_embeddings':protocol['parent_run'],'comparison_run':run.name,
        'families':{'decision_tree':['gini','entropy'],'random_forest':['forest'],'xgboost':['xgboost'],'lightgbm':['lightgbm']},
        'models':{r['key']:{k:r[k] for k in ['model_path','model_sha256','full_model_reused','parameters']} for r in final},
        'candidate_from_training_cv':selection['candidate']}
    save_json(run/'model_catalog_proposal.json',catalog)
    gallery={}
    for row in load_json(parent/'split.json')['train']:
        bucket=gallery.setdefault(row['category'],[])
        if len(bucket)<6:
            bucket.append({'image_id':row['id'],'category':row['category'],'split':'train',
                           'path':row['path'],'url':f"/api/images/train/{row['id']}"})
    save_json(run/'gallery_examples_proposal.json',{'active':False,'source':'original training manifest',
        'purpose':'Real examples of the predicted category; not similarity results or proof of correctness',
        'categories':gallery})
    lines=['# ResNet-18: comparación de cuatro algoritmos y estabilidad interna','',
        f'Ejecución: `{run.name}`. Padre conservado: `{parent.name}`.',
        '**La aplicación principal no ha cambiado.** Se reutilizaron los embeddings de 512D y los modelos completos Random Forest/XGBoost; se añadieron Decision Tree (Gini y entropía) y LightGBM.',
        '', '## Validación interna: solo las 6.941 imágenes de entrenamiento',
        'Cinco particiones estratificadas compartidas por todos los modelos, shuffle y semilla 23. Estimador nuevo en cada partición; parámetros fijos, sin búsqueda ni early stopping. Ambas opciones de Decision Tree son criterios del mismo algoritmo.',
        '', '| Modelo | Accuracy media ± DE | Balanced accuracy media ± DE | F1 macro media ± DE |',
        '| --- | ---: | ---: | ---: |']
    for row in cv:
        lines.append(f"| {NAMES[row['key']]} | {row['mean']['accuracy']*100:.2f} ± {row['std']['accuracy']*100:.2f} % | {row['mean']['balanced_accuracy']*100:.2f} ± {row['std']['balanced_accuracy']*100:.2f} % | {row['mean']['f1_macro']:.4f} ± {row['std']['f1_macro']:.4f} |")
    lines += ['', 'DE: desviación estándar muestral entre cinco particiones con entrenamientos solapados; para accuracy/balanced accuracy se expresa en puntos porcentuales. No es un intervalo de confianza ni una prueba de superioridad estadística. Cada imagen de entrenamiento tiene exactamente una predicción out-of-fold guardada.',
        f"La regla fijada antes de evaluar recomendó **{NAMES[selection['candidate']]}** por mayor F1 macro medio de validación interna (desempate por balanced accuracy). La recomendación quedó congelada antes de evaluar los modelos nuevos en prueba. No se activó ningún modelo.",
        'Random Forest y XGBoost sostienen resultados altos con el descriptor congelado. Los árboles individuales son menos precisos. LightGBM con los parámetros heredados muestra baja precisión y alta dispersión entre folds: no debe promoverse con esta configuración sin un diagnóstico o ajuste posterior exclusivamente dentro de entrenamiento. No se modificaron parámetros para elevar su resultado después de observar la prueba histórica. Esta comparación no implica que ResNet beneficie por igual a todos los clasificadores.',
        '', '## Comparación descriptiva en las 1.736 imágenes de prueba YA UTILIZADAS',
        '**Esta tabla no es una validación independiente nueva.** Mantiene la separación original para comparar resultados de las cuatro familias bajo el mismo descriptor. Los resultados RF/XGBoost corresponden a sus modelos completos existentes, sin reentrenarlos.',
        '', '| Modelo | Accuracy | Balanced accuracy | F1 macro | Entrenamiento (s) | Inferencia (ms/imagen) |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for row in final:
        marker='¹' if row['full_model_reused'] else ''
        lines.append(f"| {row['name']} | {row['accuracy']*100:.2f} % | {row['balanced_accuracy']*100:.2f} % | {row['f1_macro']:.4f} | {row['fit_s']:.3f}{marker} | {row['inference_ms']:.3f} |")
    lines += ['', '¹ Tiempo de entrenamiento conservado del experimento padre. RF/XGBoost no volvieron a ajustarse con los 6.941 ejemplos completos. Sus clones de validación sí se entrenaron únicamente con las filas correspondientes de cada partición.',
        'La inferencia de los cinco modelos se volvió a medir en esta ejecución: 128 consultas unitarias a predict_proba después de calentamiento, dos hilos CPU, mismo orden de imágenes. Excluye extracción ResNet, lectura de fotos, HTTP y visualización.',
        '', '## Trazabilidad y comprobaciones',
        f"- {counts['application']} archivos protegidos de la aplicación y {counts['parent_experiment']} archivos del experimento padre conservan su SHA-256.",
        '- Los embeddings no se recalcularon, normalizaron ni mezclaron con HSV; se comprobó su checksum y alineación con rutas/etiquetas.',
        '- Modelos nuevos versionados, RF/XGBoost referenciados en lectura; ningún modelo HSV ni referencia activa se sobrescribió.',
        '- Folds e índices guardados, parámetros y versiones fijados en protocol.json, predicciones por fold/OOF y por prueba disponibles para recalcular métricas.',
        '- CNN congelada de ImageNet, sin fine tuning ni extracción dependiente de etiquetas. Precalcular sus embeddings no ajusta un transformador a las etiquetas del conjunto de validación.',
        '', '## Límites e integración posterior',
        'La validación interna evalúa estabilidad condicional al descriptor ResNet previamente elegido. Ese enfoque ya fue explorado con la prueba histórica; ni esta CV ni reutilizar la prueba eliminan ese antecedente. Antes de una afirmación de generalización final conviene una evaluación externa o un protocolo prospectivo independiente.',
        'El candidato de integración se justifica por CV, no por elegir el mayor accuracy de la tabla de prueba reutilizada. El plan propone selector HSV/ResNet, modelos asociados al descriptor y una galería de ejemplos reales de la clase predicha; aún no implementa esos cambios en la página.',
        '', 'Referencias: [StratifiedKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html), [validación y selección de modelos](https://scikit-learn.org/stable/modules/cross_validation.html).']
    (run/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    plot_cv(cv,run/'cv_stability.png')
    print('\n'.join(lines),flush=True)


def plot_cv(rows,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(14,4.2),layout='constrained')
    labels=['DT Gini','DT Entropía','Random Forest','XGBoost','LightGBM']
    x=np.arange(len(rows))
    for axis,metric,title in zip(axes,['accuracy','balanced_accuracy','f1_macro'],['Accuracy','Balanced accuracy','F1 macro']):
        means=[r['mean'][metric] for r in rows]; std=[r['std'][metric] for r in rows]
        axis.bar(x,means,yerr=std,capsize=4,color=['#8c9dd2','#a99bd8','#77b0a2','#7963c7','#bb985d'],alpha=.9)
        for index,row in enumerate(rows):
            values=[f[metric] for f in row['folds']]
            axis.scatter(np.full(5,index)+np.linspace(-.13,.13,5),values,s=13,color='#3a4155',zorder=4)
        axis.set_xticks(x,labels,rotation=25,ha='right',fontsize=8); axis.set_ylim(0,1)
        axis.set_title(title); axis.grid(axis='y',alpha=.15); axis.set_axisbelow(True); axis.spines[['top','right']].set_visible(False)
    fig.suptitle('ResNet-18 · 5-fold CV solo en entrenamiento · media, DE y valores por fold',fontsize=11)
    fig.savefig(path,dpi=180); plt.close(fig)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--run',type=Path,required=True)
    report(parser.parse_args().run)


if __name__=='__main__': main()
