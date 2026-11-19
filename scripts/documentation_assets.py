"""Generate figures from real manifests/results and copy curated real screenshots."""
import csv
import json
import shutil
from collections import Counter
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
from scripts.resource_io import ROOT,read_json

OUT=ROOT/'documentacion_entrega/assets'
def save(name):
    plt.savefig(OUT/name,dpi=190,bbox_inches='tight',facecolor='white'); plt.close()

def diagram(name,boxes,arrows,title):
    fig,ax=plt.subplots(figsize=(11,5)); ax.set(xlim=(0,11),ylim=(0,5)); ax.axis('off'); ax.set_title(title,fontsize=16,pad=14)
    for x,y,text in boxes:
        ax.add_patch(FancyBboxPatch((x,y),2.4,.75,boxstyle='round,pad=.12',fc='#eff2ff',ec='#7563c7'))
        ax.text(x+1.2,y+.375,text,ha='center',va='center',fontsize=10)
    for a,b in arrows:
        x,y,_=boxes[a]; xx,yy,_=boxes[b]
        start=(x+2.5,y+.375) if xx>x else (x+1.2,y-.08)
        end=(xx-.1,yy+.375) if xx>x else (xx+1.2,yy+.85)
        ax.add_patch(FancyArrowPatch(start,end,arrowstyle='-|>',mutation_scale=15,color='#65748b',connectionstyle='arc3,rad=0.04'))
    save(name)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    m=read_json(ROOT/'runtime/data/manifest.json'); registry=read_json(ROOT/'runtime/descriptor_registry.json')
    from backend.modeling import artifact_directory
    hsv=read_json(artifact_directory(ROOT/'runtime/data')/'experiments.json')
    keys=['gini','entropy','forest','xgboost','lightgbm']; summary={'scope':'historical_test_reused_not_new_validation','n_train':6941,'n_test':1736,'fingerprint':m['fingerprint'],'hsv':[],'resnet18':[]}
    scalar=['key','model','accuracy','balanced_accuracy','f1_macro','train_s','fit_s','inference_ms']
    for k in keys:
        row=next(r for r in hsv['metrics'] if r['key']==k); summary['hsv'].append({n:row[n] for n in scalar if n in row})
        r=read_json(ROOT/registry['models'][k]['metrics_path']); summary['resnet18'].append({n:r[n] for n in scalar if n in r})
    comparison=read_json(ROOT/registry['comparison_results']); summary['cv_original']=comparison['cv_training_only']
    from scripts.package_resources import write_json
    summary['lightgbm_selection']=read_json(ROOT/registry['tuning_selection'])
    confirmation=read_json(ROOT/registry['tuning_confirmation'])
    summary['lightgbm_confirmation']={k:{n:v[n] for n in ['accuracy','balanced_accuracy','f1_macro','predicted_classes','max_abs_raw_margin']} for k,v in confirmation.items()}
    summary['parameters_resnet']={k:e['metadata']['params'] for k,e in registry['models'].items()}
    summary['parameters_hsv']=read_json(artifact_directory(ROOT/'runtime/data')/'metadata.json')
    write_json(ROOT/'resources/experiment_summary.json',summary)
    counts={s:Counter(r['category'] for r in m[s]) for s in ['train','query']}; classes=sorted(counts['train'])
    with (ROOT/'resources/dataset_distribution.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.writer(f);w.writerow(['category','train','historical_test']);w.writerows((c,counts['train'][c],counts['query'][c]) for c in classes)
    top=sorted(classes,key=lambda c:-counts['train'][c])[:20]
    fig,ax=plt.subplots(figsize=(11,6)); y=np.arange(len(top)); ax.barh(y,[counts['train'][c] for c in top],label='Entrenamiento',color='#7765d0');ax.barh(y,[counts['query'][c] for c in top],left=[counts['train'][c] for c in top],label='Prueba histórica',color='#89a6e8');ax.set_yticks(y,top);ax.invert_yaxis();ax.legend();ax.set_xlabel('Fotografías');ax.set_title('Figura 2.1. Las 20 categorías más numerosas del manifiesto local');save('02_distribucion.png')
    fig,axes=plt.subplots(1,2,figsize=(11,4.4)); labels=['DT Gini','DT entropía','RF','XGB','LGBM'];x=np.arange(5)
    for ax,metric,title in [(axes[0],'accuracy','Accuracy (%)'),(axes[1],'f1_macro','F1 macro')]:
        factor=100 if metric=='accuracy' else 1
        ax.bar(x-.18,[r[metric]*factor for r in summary['hsv']],.36,label='HSV histórico',color='#87a4e5');ax.bar(x+.18,[r[metric]*factor for r in summary['resnet18']],.36,label='ResNet activado',color='#7765d0');ax.set_xticks(x,labels,rotation=20);ax.set_title(title);ax.legend(fontsize=8)
    fig.suptitle('Figura 6.1. Prueba histórica reutilizada; no es nueva validación');plt.tight_layout();save('06_resultados.png')
    r=read_json(ROOT/registry['models']['lightgbm']['metrics_path']);matrix=np.asarray(r['confusion']);normalized=matrix/matrix.sum(axis=1,keepdims=True)
    fig,ax=plt.subplots(figsize=(9,7));im=ax.imshow(normalized,vmin=0,vmax=1,cmap='Blues');fig.colorbar(im,ax=ax,label='Fracción dentro de cada categoría real');ticks=np.arange(0,101,10);ax.set_xticks(ticks,[classes[i] for i in ticks],rotation=70);ax.set_yticks(ticks,[classes[i] for i in ticks]);ax.set(xlabel='Predicción',ylabel='Categoría real',title='Figura 6.2. LightGBM ResNet: matriz normalizada por fila');save('06_matriz.png')
    from backend.core import describe_bytes
    row=m['query'][0]; vec=describe_bytes((ROOT/row['path']).read_bytes());fig,axes=plt.subplots(1,2,figsize=(11,3.5));axes[0].bar(range(128),vec[:128],color='#7765d0');axes[1].bar(range(16),vec[128:],color='#87a4e5');axes[0].set_title('128 bins HSV, suma ≈ 1');axes[1].set_title('16 orientaciones Sobel, suma ≈ 1');fig.suptitle('Figura 3.1. Descriptor real de accordion / image_0032.jpg');save('03_hsv_real.png')
    from PIL import Image
    fig,axes=plt.subplots(2,3,figsize=(10,6))
    for ax,c in zip(axes.flat,['accordion','airplanes','ant','Faces','Motorbikes','water_lilly']):
        row=next(r for r in m['train'] if r['category']==c);ax.imshow(Image.open(ROOT/row['path']));ax.set_title(c);ax.axis('off')
    fig.suptitle('Figura 2.2. Fotografías reales de entrenamiento; Caltech-101 / CC BY 4.0');save('02_ejemplos.png')
    diagram('01_flujo.png',[(.3,3.5,'Fotografía JPG/PNG'),(4.2,3.5,'HSV144 o ResNet512'),(8,3.5,'Modelo compatible'),(8,1,'101 probabilidades'),(4.2,1,'Categoría predicha'),(.3,1,'6 ejemplos de train')],[(0,1),(1,2),(2,3),(3,4),(4,5)],'Figura 1.1. Flujo de clasificación de MetricLab Visual')
    diagram('07_arquitectura.png',[(.2,3.5,'React / TypeScript'),(4.1,3.5,'FastAPI / validaciones'),(8,3.5,'Registro + modelos'),(8,.9,'Worker ResNet\nCPU / CUDA'),(4.1,.9,'Galería / manifiesto'),(.2,.9,'Motor C++\nLaboratorio HSV')],[(0,1),(1,2),(2,3),(1,4),(1,5)],'Figura 7.1. Componentes reales y responsabilidades')
    diagram('03_resnet.png',[(.2,3.5,'RGB / resize256\ncenter crop224'),(4.1,3.5,'Conv + bloques\nresiduales [2,2,2,2]'),(8,3.5,'512 mapas de 7×7'),(8,1,'Average pooling\n512×1×1'),(4.1,1,'Flatten / fc Identity\n512 números'),(.2,1,'Clasificador tabular\n101 clases')],[(0,1),(1,2),(2,3),(3,4),(4,5)],'Figura 3.2. Red preentrenada congelada como extractor')
    diagram('04_algoritmos.png',[(.2,3.5,'Árbol individual\nGini o entropía'),(4.1,3.5,'Bosque: árboles\nbootstrap independientes'),(8,3.5,'Promedio de\nprobabilidades'),(.2,1,'Boosting: etapa t'),(4.1,1,'Gradientes / Hessianos\nde la pérdida'),(8,1,'Nuevo árbol\nactualiza puntuaciones')],[(1,2),(3,4),(4,5)],'Figura 4.1. Bosque y boosting siguen mecanismos distintos')
    equations={
        '04_gini':r'$G=1-\sum_k p_k^2\qquad H=-\sum_k p_k\log_2 p_k$',
        '04_gain':r'$\Delta I=I(P)-\frac{n_L}{n_P}I(L)-\frac{n_R}{n_P}I(R)$',
        '04_boost':r'$\mathcal{L}^{(t)}=\sum_i l(y_i,\hat{y}^{(t-1)}_i+f_t(x_i))+\Omega(f_t)$',
        '04_leaf':r'$w_j^*=-\frac{\sum_{i\in I_j}g_i}{\sum_{i\in I_j}h_i+\lambda}$',
        '06_metrics':r'$Accuracy=\frac{\sum_k C_{kk}}{N}\quad BA=\frac{1}{K}\sum_k\frac{C_{kk}}{\sum_jC_{kj}}$',
        '06_f1':r'$P_k=\frac{TP_k}{TP_k+FP_k}\quad R_k=\frac{TP_k}{TP_k+FN_k}\quad F1_k=\frac{2P_kR_k}{P_k+R_k}$',
        '06_std':r'$\bar{x}=\frac{1}{m}\sum_i x_i\qquad s=\sqrt{\frac{\sum_i(x_i-\bar{x})^2}{m-1}}$',
        '03_pool':r'$z_c=\frac{1}{49}\sum_{u=1}^{7}\sum_{v=1}^{7}A_{cuv}\qquad c=1,\ldots,512$'}
    for name,formula in equations.items():
        fig,ax=plt.subplots(figsize=(11,1));ax.axis('off');ax.text(.5,.5,formula,fontsize=20,ha='center',va='center');save(name+'.png')
    for name in ['home-desktop.png','resnet-classification-desktop.png','classifier-explorer-desktop.png','classifier-evaluation-desktop.png','lab-desktop.png','resnet-classification-mobile.png']:
        shutil.copy2(ROOT/'runtime'/name,OUT/name)
    shutil.copy2(ROOT/'runtime/experiments/lightgbm_resnet/20261118T211652Z-600d0ec2/training_curves.png',OUT/'10_lightgbm_curvas.png')
    print('Figures and summaries generated from original artifacts; no model trained.')

if __name__=='__main__': main()
