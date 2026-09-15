#!/bin/sh
set -eu

data_dir="${IMAGEPOOL_DATA_DIR:-/app/data}"
mkdir -p "$data_dir"
chown -R imagepool:imagepool "$data_dir"

exec gosu imagepool streamlit run app.py \
  --server.address=0.0.0.0 \
  --server.port=8503

