-- PostgreSQL opcional: la búsqueda permanece en el motor C++.
CREATE TABLE IF NOT EXISTS datasets (
    fingerprint text PRIMARY KEY,
    name text NOT NULL,
    kind text NOT NULL CHECK (kind IN ('caltech101', 'synthetic')),
    descriptor text NOT NULL,
    dimensions integer NOT NULL CHECK (dimensions = 144),
    metadata jsonb NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS images (
    dataset_fingerprint text NOT NULL REFERENCES datasets(fingerprint),
    split text NOT NULL CHECK (split IN ('train', 'query')),
    local_id integer NOT NULL CHECK (local_id >= 0),
    category text NOT NULL,
    relative_path text NOT NULL,
    descriptor real[] NOT NULL CHECK (cardinality(descriptor) = 144),
    PRIMARY KEY (dataset_fingerprint, split, local_id)
);
CREATE INDEX IF NOT EXISTS images_category_idx ON images(dataset_fingerprint, category);
CREATE TABLE IF NOT EXISTS experiments (
    dataset_fingerprint text PRIMARY KEY REFERENCES datasets(fingerprint),
    results jsonb NOT NULL,
    imported_at timestamptz NOT NULL DEFAULT now()
);
