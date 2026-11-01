"""Validation for the local, train/validation-only CNN feature cache."""
import hashlib
import json
from pathlib import Path
from zipfile import BadZipFile

import numpy as np


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def feature_cache_metadata(frame, split_path, prepared_manifest_path,
                           preprocessing_config_path, encoder, transform):
    """Bind features to current image bytes and transforms, never reading test images."""
    if not frame.split.isin(['train', 'validation']).all():
        raise ValueError('Feature cache may contain only train/validation images.')
    images = hashlib.sha256()
    for row in frame.itertuples():
        record = [str(row.id_code), file_sha256(row.image_path)]
        images.update((json.dumps(record, separators=(',', ':')) + '\n').encode())
    return {
        'schema_version': 2,
        'manifest_sha256': file_sha256(split_path),
        'prepared_manifest_sha256': file_sha256(prepared_manifest_path),
        'preprocessing_config_sha256': file_sha256(preprocessing_config_path),
        'prepared_images_sha256': images.hexdigest(),
        'encoder': encoder,
        'transform': transform,
        'feature_shape_per_image': [49, 512],
        'included_splits': ['train', 'validation'],
    }


def load_feature_cache(path, expected_metadata, expected_ids, expected_labels):
    """Return verified arrays, or None to rebuild a missing/stale/corrupt cache."""
    path = Path(path)
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as cached:
            metadata = json.loads(str(cached['metadata'].item()))
            if metadata != expected_metadata:
                return None
            ids = cached['id_code'].astype(str)
            labels = cached['labels']
            features = cached['features']
            if not np.array_equal(ids, np.asarray(expected_ids, dtype=str)):
                return None
            if (not np.issubdtype(labels.dtype, np.integer)
                    or not np.array_equal(labels, np.asarray(expected_labels, dtype=np.int64))):
                return None
            if (features.shape != (len(ids), 49, 512)
                    or features.dtype != np.float32 or not np.isfinite(features).all()):
                return None
            return features, labels.astype(np.int64), ids
    except (OSError, ValueError, KeyError, TypeError, EOFError, BadZipFile):
        return None
