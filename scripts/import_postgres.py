"""Importación transaccional e idempotente de artefactos locales a PostgreSQL."""
import json
import os
from backend.core import DATA, ROOT, load_vectors


def main():
    import psycopg
    from psycopg.types.json import Jsonb
    url = os.getenv('DATABASE_URL')
    if not url:
        raise SystemExit('Configura DATABASE_URL en el entorno. No incluyas la contraseña en comandos compartidos.')
    manifest = json.loads((DATA / 'manifest.json').read_text(encoding='utf-8'))
    fingerprint = manifest['fingerprint']
    with psycopg.connect(url) as conn:
        with conn.cursor() as cursor:
            cursor.execute((ROOT / 'docs/schema.sql').read_text(encoding='utf-8'))
            cursor.execute('INSERT INTO datasets(fingerprint,name,kind,descriptor,dimensions,metadata) VALUES (%s,%s,%s,%s,%s,%s) ON CONFLICT (fingerprint) DO NOTHING',
                           (fingerprint, manifest['dataset'], manifest['kind'], manifest['descriptor'], 144,
                            Jsonb({k: v for k, v in manifest.items() if k not in {'train', 'query'}})))
            for split in ['train', 'query']:
                vectors = load_vectors(DATA / f'{split}.bin')
                cursor.executemany('INSERT INTO images VALUES (%s,%s,%s,%s,%s,%s) ON CONFLICT (dataset_fingerprint,split,local_id) DO NOTHING',
                                   [(fingerprint, split, r['id'], r['category'], r['path'], vectors[i].tolist()) for i, r in enumerate(manifest[split])])
            path = DATA / 'experiments.json'
            if path.exists():
                experiments = json.loads(path.read_text(encoding='utf-8'))
                if experiments['fingerprint'] == fingerprint:
                    cursor.execute('INSERT INTO experiments(dataset_fingerprint,results) VALUES (%s,%s) ON CONFLICT (dataset_fingerprint) DO UPDATE SET results=EXCLUDED.results,imported_at=now()', (fingerprint, Jsonb(experiments)))
    print('Importación completada. La aplicación continúa usando los archivos locales.')


if __name__ == '__main__':
    main()
