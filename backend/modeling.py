"""Shared classifier metadata, artifact selection and real sklearn tree inspection."""
import json
from pathlib import Path

import numpy as np

MODEL_NAMES = {
    'gini': 'Decision Tree — Gini', 'entropy': 'Decision Tree — Entropía',
    'forest': 'Random Forest', 'xgboost': 'XGBoost', 'lightgbm': 'LightGBM',
    'knn': 'k-NN / VP-Tree',
}
PRIMARY_KEYS = ('gini', 'entropy', 'forest', 'xgboost', 'lightgbm')
FAMILIES = {'gini': 'decision_tree', 'entropy': 'decision_tree', 'forest': 'random_forest',
            'xgboost': 'xgboost', 'lightgbm': 'lightgbm', 'knn': 'secondary'}


def artifact_directory(data: Path) -> Path:
    pointer = data / 'active_classifiers.json'
    if not pointer.exists():
        return data  # Backwards compatible with the first delivery.
    selected = (data / json.loads(pointer.read_text(encoding='utf-8'))['directory']).resolve()
    if not selected.is_relative_to(data.resolve()) or not selected.is_dir():
        raise ValueError('La preparación activa de modelos no es válida.')
    return selected


def feature_name(index: int, descriptor_id='hsv128-edge16-v1') -> str:
    if index < 0:
        return 'Hoja'
    if descriptor_id != 'hsv128-edge16-v1':
        return f'ResNet · componente del embedding {index}'
    if index < 128:
        h, remainder = divmod(index, 16)
        s, v = divmod(remainder, 4)
        return f'HSV · tono {h}, saturación {s}, brillo {v}'
    edge = index - 128
    return f'Bordes · {edge * 11.25:g}°–{(edge + 1) * 11.25:g}°'


def estimator_for(model, key, tree_index=0):
    if key not in {'gini', 'entropy', 'forest'}:
        raise ValueError('Este visor de nodos admite Decision Tree y Random Forest. Para boosting consulta las importancias.')
    trees = model.estimators_ if key == 'forest' else [model]
    if not 0 <= tree_index < len(trees):
        raise ValueError('El árbol solicitado no existe.')
    return trees[tree_index], len(trees)


def node_details(estimator, node, classes, descriptor_id='hsv128-edge16-v1'):
    tree = estimator.tree_
    values = np.asarray(tree.value[node]).reshape(-1)
    total = float(values.sum())
    proportions = values / total if total else values
    distribution = [
        {'category': classes[i], 'proportion': float(p),
         'weight': float(p * tree.weighted_n_node_samples[node])}
        for i, p in enumerate(proportions) if p > 0
    ]
    distribution.sort(key=lambda row: -row['proportion'])
    feature = int(tree.feature[node])
    return {'id': int(node), 'left': int(tree.children_left[node]), 'right': int(tree.children_right[node]),
            'feature': feature, 'feature_name': feature_name(feature,descriptor_id),
            'threshold': float(tree.threshold[node]) if feature >= 0 else None,
            'impurity': float(tree.impurity[node]), 'samples': int(tree.n_node_samples[node]),
            'weighted_samples': float(tree.weighted_n_node_samples[node]),
            'leaf': int(tree.children_left[node]) == -1, 'classes': distribution}


def tree_structure(model, key, classes, tree_index=0, root=0, max_depth=3, descriptor_id='hsv128-edge16-v1'):
    estimator, tree_count = estimator_for(model, key, tree_index)
    if not 0 <= root < estimator.tree_.node_count:
        raise ValueError('El nodo solicitado no existe.')
    if not 1 <= max_depth <= 4:
        raise ValueError('El límite de profundidad debe estar entre 1 y 4.')
    nodes, queue = [], [(root, 0)]
    while queue:
        node, depth = queue.pop(0)
        item = node_details(estimator, node, classes, descriptor_id)
        item['display_depth'] = depth
        nodes.append(item)
        if depth < max_depth and not item['leaf']:
            queue.extend([(item['left'], depth+1), (item['right'], depth+1)])
    return {'key': key, 'criterion': estimator.criterion, 'tree_index': tree_index,
            'tree_count': tree_count, 'total_nodes': int(estimator.tree_.node_count),
            'total_depth': int(estimator.tree_.max_depth), 'root': root, 'nodes': nodes}


def decision_path(model, key, classes, vector, tree_index=0, descriptor_id='hsv128-edge16-v1'):
    estimator, _ = estimator_for(model, key, tree_index)
    nodes = estimator.decision_path(vector[None]).indices
    path = []
    for node in nodes:
        item = node_details(estimator, node, classes, descriptor_id)
        item['node'] = int(node)  # Existing API field retained.
        feature = item['feature']
        item['value'] = float(vector[feature]) if feature >= 0 else None
        item['direction'] = ('left' if vector[feature] <= item['threshold'] else 'right') if feature >= 0 else None
        path.append(item)
    return path
