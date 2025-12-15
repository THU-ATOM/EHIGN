#!/bin/bash

# Prefer conda run to avoid activation issues in non-interactive shells
eval "$(conda shell.bash hook)" || true

# set PYTHONPATH
export PYTHONPATH=/app/ehign
export DGLBACKEND=pytorch

data_dir="/data_lmdb/data"
# Use checkpoints mount for model weights; Ehignmodel.pt is directly under /checkpoints
model_dir="/checkpoints"

# --- Check and copy model file if needed ---
if [ ! -f "$model_dir/Ehignmodel.pt" ]; then
    echo "[Setup] Ehignmodel.pt not found in $model_dir"
    if [ -f "/app/model/Ehignmodel.pt" ]; then
        echo "[Setup] Copying Ehignmodel.pt from /app/model/ to $model_dir/"
        mkdir -p "$model_dir"
        cp /app/model/Ehignmodel.pt "$model_dir/Ehignmodel.pt"
        echo "[Setup] Copy completed"
    else
        echo "[Setup] WARNING: Ehignmodel.pt not found in /app/model/ either!"
    fi
else
    echo "[Setup] Ehignmodel.pt already exists in $model_dir"
fi

# --- Mount diagnostics ---
echo "[Diag] Listing root and expected mounts:"
ls -la /
echo "[Diag] /data_lmdb:"; ls -la /data_lmdb || echo "(missing)"
echo "[Diag] /run_ehign:"; ls -la /run_ehign || echo "(missing)"
echo "[Diag] /checkpoints:"; ls -la /checkpoints || echo "(missing)"
echo "[Diag] model_dir ($model_dir):"; ls -la "$model_dir" || echo "(missing)"
# check if csv file existed
if [ -f "$data_dir/external_test.csv" ]; then
    echo "external_test.csv existed"
else
    echo "external_test.csv not existed"
fi

    # --- Diagnostics: verify conda and RDKit availability in both envs ---
    echo "[Diag] conda version:"
    /usr/local/bin/conda --version || true
    echo "[Diag] conda envs:"
    /usr/local/bin/conda info --envs || true
    echo "[Diag] checking rdkit in ehignfirst:"
    /usr/local/bin/conda run -n ehignfirst python -c "import sys; print('python=', sys.executable); import rdkit; print('rdkit=', rdkit.__version__)" || echo "[Diag] ehignfirst: rdkit NOT available"
    echo "[Diag] checking rdkit in ehignlasttwo:"
    /usr/local/bin/conda run -n ehignlasttwo python -c "import sys; print('python=', sys.executable); import rdkit; print('rdkit=', rdkit.__version__)" || echo "[Diag] ehignlasttwo: rdkit NOT available"

# run python with explicit environments using conda run
# /app copies are fine, but ensure the correct env provides dependencies
/usr/local/bin/conda run -n ehignfirst python /app/run_preprocess_complex.py --data_dir "$data_dir"
/usr/local/bin/conda run -n ehignlasttwo bash -lc 'export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH; python /app/run_graph_constructor.py --data_dir '"$data_dir"''
# Ensure LD paths inside the env when running the final test
output=$(/usr/local/bin/conda run -n ehignlasttwo bash -lc 'export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH; python /app/run_test.py --data_dir '"$data_dir"' --model_dir '"$model_dir" )
echo "$output"
