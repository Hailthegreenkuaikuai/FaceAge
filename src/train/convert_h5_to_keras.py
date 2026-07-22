# -----------------
# Convert legacy FaceAge .h5 model to Keras v3 .keras format
# (compatible with Python 3.13+)
# -----------------
#
# The original faceage_model.h5 was saved with TF 1.x / Python ~3.6.
# Three incompatibilities block loading on modern Keras 3 + Python 3.13:
#   1. Lambda layer bytecode (marshal) is Python-version-specific → EOFError
#   2. Sequential auto-build discovers shape incompatibilities
#   3. InceptionResNetV1 outputs [tensor] (list) instead of tensor
#
# This script patches around all three, loads weights, and saves as .keras.

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import json
import argparse
import numpy as np
import keras
import tensorflow as tf
import h5py

from keras.src.utils import python_utils
from keras.src.models.sequential import Sequential as KerasSequential
from keras.src.legacy.saving import legacy_h5_format, saving_utils, saving_options
from keras.src.saving import serialization_lib

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def apply_patches():
    """Monkey-patch Keras internals for Python 3.13 compat."""

    # ---- patch 1: marshal bytecode for Lambda layers ----
    _original_func_load = python_utils.func_load

    def _patched_func_load(code, defaults=None, closure=None, globs=None):
        try:
            return _original_func_load(code, defaults, closure, globs)
        except EOFError:
            # All Lambda layers in this model are ScaleSum:
            #   lambda inputs, scale: inputs[0] + inputs[1] * scale
            return lambda inputs, scale: inputs[0] + inputs[1] * scale

    python_utils.func_load = _patched_func_load

    # ---- patch 2: skip Sequential auto-rebuild during from_config ----
    _original_add = KerasSequential.add

    def patched_add(self, layer, rebuild=True):
        self._layers.append(layer)
        self.built = False
        self._functional = None

    KerasSequential.add = patched_add


def load_nested_weights(h5_group, model):
    """Recursively load weights from old-format H5 sub-groups into a Keras model."""
    layer_map = {layer.name: layer for layer in model.layers}
    count = 0

    for key in h5_group.keys():
        obj = h5_group[key]
        if not isinstance(obj, h5py.Group) or key not in layer_map:
            continue

        target = layer_map[key]

        # Collect weight arrays, matching by base name (kernel:0 → kernel)
        h5_weights = {}
        for wk in obj.keys():
            if isinstance(obj[wk], h5py.Dataset):
                base = wk.split(":")[0] if ":" in wk else wk
                h5_weights[base] = np.array(obj[wk])

        expected = [w.name.split("/")[-1].split(":")[0] for w in target.weights]
        matched = [h5_weights[n] for n in expected if n in h5_weights]

        if matched:
            target.set_weights(matched)
            count += 1

        # Recurse if the target has sub-layers
        if hasattr(target, "layers") and target.layers:
            load_nested_weights(obj, target)

    return count


def convert_h5_to_keras(h5_path, keras_path):
    """Load a legacy .h5 model and save as .keras format."""

    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"Model not found: {h5_path}")

    apply_patches()

    with h5py.File(h5_path, "r") as f:
        model_config = f.attrs["model_config"]
        if isinstance(model_config, bytes):
            model_config = model_config.decode("utf-8")
        model_config = json.loads(model_config)

        legacy_scope = saving_options.keras_option_scope(use_legacy_config=True)
        safe_mode_scope = serialization_lib.SafeModeScope(False)
        with legacy_scope, safe_mode_scope:
            model = saving_utils.model_from_config(model_config, custom_objects={})

        # ---- fix: InceptionResNetV1 returns [tensor], flatten it ----
        extract = keras.layers.Lambda(lambda x: x[0], name="extract_output")
        model._layers.insert(1, extract)

        # ---- build to initialize weight slots ----
        dummy = tf.zeros((1, 160, 160, 3))
        _ = model(dummy)
        print(f"Model built. Layers: {len(model.layers)}")

        # ---- load weights ----
        weights_group = f["model_weights"]

        irv1 = model.get_layer("inception_resnet_v1")
        irv1_group = weights_group["inception_resnet_v1"]
        irv1_count = load_nested_weights(irv1_group, irv1)
        print(f"InceptionResNetV1 weights: {irv1_count} sub-layers")

        for layer_name in ["dense_1", "classifier_1_BatchNorm", "dense_2"]:
            g = weights_group[layer_name]
            weight_names = list(g.attrs["weight_names"])
            weight_values = [np.array(g[wn]) for wn in weight_names]
            model.get_layer(layer_name).set_weights(weight_values)
        print("Outer layer weights loaded")

    # ---- verify ----
    result = model.predict(dummy, verbose=0)
    print(f"Sanity check - prediction on zeros: {result[0][0]:.4f}")

    # ---- save ----
    model.save(keras_path)
    size_mb = os.path.getsize(keras_path) / (1024 * 1024)
    print(f"Saved: {keras_path} ({size_mb:.1f} MB)")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert FaceAge .h5 to .keras")
    parser.add_argument(
        "--input", default="models/faceage_model.h5", help="Path to legacy .h5 model"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path for output .keras model (default: same dir, .keras extension)",
    )
    args = parser.parse_args()

    out = args.output
    if out is None:
        out = os.path.splitext(args.input)[0] + ".keras"

    convert_h5_to_keras(args.input, out)
