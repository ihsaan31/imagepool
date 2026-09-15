# Imagepool

A private, self-hosted visual asset library built with Streamlit, OpenCLIP, SQLite, and FAISS. Upload JPEG/PNG originals, search by a reference image, and download the original bytes.

## Folder imports

Choose **Upload asset → Folder** and select a directory. JPEG/PNG images from its subdirectories are included in one collection. Review the collection name and file list before importing. Limits are 200 images, 512 MB total, and 25 MB per image.

Importing the same collection again adds new images without deleting old ones. Identical bytes are stored only once and can belong to multiple collections. Failed files are reported individually; retrying is safe. Search and library filters support one or several collections, including **Tanpa koleksi** for older/single-file assets.

Original files are never compressed, resized, or re-encoded. Only separate preview thumbnails are optimized. Open any search result and use **Download original** to retrieve the exact uploaded bytes.

## Run locally

Python 3.11 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/hash_password.py
cp .env.example .env
```

Set `IMAGEPOOL_PASSWORD_HASH` in your shell or load `.env`, then run:

```bash
streamlit run app.py
```

The first local run downloads the configured OpenCLIP model unless it already exists in the model cache. Production Docker builds cache it into the image.

## Deploy with Coolify

1. Create a new Coolify application from this Git repository.
2. Select **Dockerfile** as the build pack. The Dockerfile is at the repository root.
3. Set the exposed/container port to `8501`.
4. Attach a persistent volume with destination path `/app/data`. Never deploy without this mount after storing assets.
5. Add these environment variables:

   - `IMAGEPOOL_USERNAME`
   - `IMAGEPOOL_PASSWORD_HASH`
   - `IMAGEPOOL_DATA_DIR=/app/data`
   - `IMAGEPOOL_MAX_UPLOAD_MB=25`

6. Generate the password hash locally with `python scripts/hash_password.py`. Paste the complete bcrypt string into Coolify; do not commit it.
7. Assign a domain in Coolify. Coolify's proxy provides HTTPS and forwards Streamlit's WebSocket connection.
8. Use `/_stcore/health` as the HTTP health-check path. The Docker image also includes an equivalent container health check.
9. Deploy with exactly one replica. SQLite and FAISS use the attached local volume and do not support multiple writers.

The Docker build downloads and caches the pinned OpenCLIP weights. This makes the image large and the first build slower, but production startup does not depend on an external model service.

## Back up and restore

All durable state lives under `/app/data`. Take a consistent snapshot of that mounted directory. For the safest backup, stop the application first. Restore the directory to the same mount and redeploy; the FAISS index is rebuilt automatically when it does not match SQLite.

## Tests

```bash
pip install -r requirements-dev.txt
pytest -q
```

Tests cover image validation and byte preservation, SQLite metadata, similarity ranking, filtering, and index persistence without loading the OpenCLIP model.
# imagepool
