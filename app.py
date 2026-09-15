from __future__ import annotations

import html
import time
from datetime import date
from pathlib import Path

import bcrypt
import streamlit as st

from imagepool.config import Settings
from imagepool.db import Asset, AssetDatabase
from imagepool.embeddings import ImageEmbedder
from imagepool.search import SearchIndex
from imagepool.storage import InvalidImage, persist_files, prepare_image
from imagepool.importer import import_files


st.set_page_config(
    page_title="Imagepool",
    page_icon="◫",
    layout="wide",
    initial_sidebar_state="collapsed",
)


STYLE = """
<style>
:root { --ink:#17243A; --muted:#64748B; --line:#CAD4E2; --paper:#F8FAFD; --blue:#3857D6; --amber:#E9A23B; }
html, body, [class*="css"] { font-family: "Avenir Next", "Segoe UI", Arial, sans-serif; color: var(--ink); }
.stApp { background: radial-gradient(circle at 8% 3%, #F7FAFF 0, #E9EEF5 42%, #E4EAF2 100%); }
[data-testid="stHeader"] { background: transparent; }
.block-container { max-width: 1480px; padding-top: 2.1rem; padding-bottom: 5rem; }
.ip-kicker { font-family:ui-monospace,"SFMono-Regular",Consolas,monospace; font-size:.72rem; letter-spacing:.14em; text-transform:uppercase; color:#5D6E87; }
.ip-title { margin:.15rem 0 0; font-size:clamp(2rem,4vw,4.2rem); line-height:.94; letter-spacing:-.065em; font-weight:700; color:#12203A; }
.ip-title span { color:var(--blue); }
.ip-meta { font-family:ui-monospace,"SFMono-Regular",Consolas,monospace; font-size:.78rem; color:var(--muted); margin-top:.85rem; }
.ip-search-frame { position:relative; border:1px solid #AEBBD0; background:rgba(248,250,253,.88); padding:1.2rem 1.25rem .3rem; margin:1.6rem 0 1rem; box-shadow:0 18px 50px rgba(45,66,102,.08); }
.ip-search-frame:before,.ip-search-frame:after { content:""; position:absolute; width:22px; height:22px; border-color:var(--blue); }
.ip-search-frame:before { left:-6px; top:-6px; border-left:2px solid var(--blue); border-top:2px solid var(--blue); }
.ip-search-frame:after { right:-6px; bottom:-6px; border-right:2px solid var(--blue); border-bottom:2px solid var(--blue); }
.ip-section { font-family:ui-monospace,"SFMono-Regular",Consolas,monospace; font-size:.72rem; letter-spacing:.1em; text-transform:uppercase; color:#53647E; border-bottom:1px solid var(--line); padding-bottom:.55rem; margin:1rem 0; }
.ip-card-meta { font-family:ui-monospace,"SFMono-Regular",Consolas,monospace; font-size:.7rem; line-height:1.6; color:#63738B; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.ip-card-name { font-size:.88rem; font-weight:650; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; margin-top:.55rem; }
.ip-login { max-width:440px; margin:10vh auto 1rem; padding:2rem; border:1px solid var(--line); background:rgba(248,250,253,.94); box-shadow:0 24px 70px rgba(40,59,90,.12); }
[data-testid="stFileUploaderDropzone"] { min-height:132px; background:#F4F7FC; border:1px dashed #8EA2C4; border-radius:2px; }
[data-testid="stImage"] img { border-radius:2px; border:1px solid #D4DCE8; background:#DDE4EE; aspect-ratio:4/3; object-fit:contain; }
.stButton > button, .stDownloadButton > button { border-radius:2px; font-weight:650; }
.stButton > button[kind="primary"] { background:var(--blue); border-color:var(--blue); }
div[data-baseweb="tab-list"] { gap:1.4rem; border-bottom:1px solid var(--line); }
button[data-baseweb="tab"] { font-family:ui-monospace,"SFMono-Regular",Consolas,monospace; font-size:.76rem; text-transform:uppercase; letter-spacing:.08em; }
*:focus-visible { outline:3px solid rgba(56,87,214,.35) !important; outline-offset:2px; }
@media (max-width:700px){ .block-container{padding:1rem 1rem 3rem}.ip-title{font-size:2.6rem}.ip-search-frame{padding:.8rem .8rem .2rem} }
</style>
"""
st.markdown(STYLE, unsafe_allow_html=True)


@st.cache_resource
def resources(settings: Settings):
    settings.prepare_storage()
    database = AssetDatabase(settings.database_path)
    embedder = ImageEmbedder(settings.model_name, settings.model_pretrained)
    search_index = SearchIndex(database, settings.index_path, embedder.version)
    return database, embedder, search_index


def auth_gate(settings: Settings) -> bool:
    if st.session_state.get("authenticated"):
        return True
    st.markdown('<div class="ip-login">', unsafe_allow_html=True)
    st.markdown('<div class="ip-kicker">Private collection</div>', unsafe_allow_html=True)
    st.markdown('<h1 class="ip-title">Image<span>pool</span></h1>', unsafe_allow_html=True)
    st.caption("Sign in to search and manage your image library.")
    if not settings.username or not settings.password_hash:
        st.error("Authentication is not configured. Set IMAGEPOOL_USERNAME and IMAGEPOOL_PASSWORD_HASH in Coolify.")
        st.markdown("</div>", unsafe_allow_html=True)
        return False
    locked_until = float(st.session_state.get("locked_until", 0))
    if time.time() < locked_until:
        st.error("Too many attempts. Try again in one minute.")
        st.markdown("</div>", unsafe_allow_html=True)
        return False
    with st.form("login", clear_on_submit=True):
        username = st.text_input("Username", autocomplete="username")
        password = st.text_input("Password", type="password", autocomplete="current-password")
        submitted = st.form_submit_button("Open library", type="primary", use_container_width=True)
    if submitted:
        try:
            password_ok = bcrypt.checkpw(password.encode(), settings.password_hash.encode())
        except ValueError:
            password_ok = False
        if username == settings.username and password_ok:
            st.session_state.authenticated = True
            st.session_state.auth_failures = 0
            st.rerun()
        failures = int(st.session_state.get("auth_failures", 0)) + 1
        st.session_state.auth_failures = failures
        if failures >= 5:
            st.session_state.locked_until = time.time() + 60
            st.session_state.auth_failures = 0
        st.error("Username or password is incorrect.")
    st.markdown("</div>", unsafe_allow_html=True)
    return False


def human_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} GB"


def render_assets(
    assets: list[tuple[Asset, float | None]], empty_message: str, scope: str
) -> None:
    if not assets:
        st.info(empty_message)
        return
    for start in range(0, len(assets), 4):
        columns = st.columns(4, gap="medium")
        for column, (asset, score) in zip(columns, assets[start : start + 4]):
            with column:
                thumbnail = Path(asset.thumbnail_path)
                if thumbnail.exists():
                    st.image(str(thumbnail), use_container_width=True)
                else:
                    st.warning("Thumbnail unavailable")
                safe_name = html.escape(asset.original_name)
                st.markdown(f'<div class="ip-card-name" title="{safe_name}">{safe_name}</div>', unsafe_allow_html=True)
                suffix = f" · {score * 100:.1f}% match" if score is not None else ""
                st.markdown(
                    f'<div class="ip-card-meta">{asset.mime_type.split("/")[-1].upper()} · {asset.width}×{asset.height} · {human_bytes(asset.size_bytes)}{suffix}</div>',
                    unsafe_allow_html=True,
                )
                if st.button(
                    "Open asset",
                    key=f"open-{scope}-{asset.id}",
                    use_container_width=True,
                ):
                    st.session_state.selected_asset_id = asset.id
                    st.rerun()


@st.dialog("Asset details", width="large")
def asset_dialog(asset: Asset) -> None:
    sources = database.asset_sources(asset.id)
    for collection_name, relative_path in sources:
        st.text(f"{collection_name} / {relative_path}")
    if not sources:
        st.caption("Tanpa koleksi")
    preview = Path(asset.original_path)
    if preview.exists():
        st.image(str(preview), use_container_width=True)
        st.markdown(f"**{html.escape(asset.original_name)}**")
        st.caption(
            f"{asset.mime_type.split('/')[-1].upper()} · "
            f"{asset.width}×{asset.height} · {human_bytes(asset.size_bytes)}"
        )
        st.download_button(
            "Download original",
            data=preview.read_bytes(),
            file_name=asset.original_name,
            mime=asset.mime_type,
            key=f"detail-download-{asset.id}",
            type="primary",
            use_container_width=True,
        )
    else:
        st.error("The original file is missing from persistent storage.")
    if st.button("Close", use_container_width=True):
        st.session_state.pop("selected_asset_id", None)
        st.rerun()


def filters(prefix: str):
    collection_options = {0: "Tanpa koleksi", **database.collections()}
    selected = st.multiselect("Collections", list(collection_options), format_func=collection_options.get, key=f"{prefix}-collections", help="Leave empty to search all collections.")
    one, two, three = st.columns([1.5, 1, 1.2])
    name = one.text_input("Filename", placeholder="Search filename", key=f"{prefix}-name")
    display_formats = two.multiselect(
        "Format", ["JPEG", "PNG"], default=["JPEG", "PNG"], key=f"{prefix}-formats"
    )
    use_dates = three.checkbox("Filter upload date", key=f"{prefix}-use-dates")
    start_date = end_date = None
    if use_dates:
        date_columns = st.columns(2)
        start_date = date_columns[0].date_input("From", value=date.today(), key=f"{prefix}-from")
        end_date = date_columns[1].date_input("To", value=date.today(), key=f"{prefix}-to")
    mime_map = {"JPEG": "image/jpeg", "PNG": "image/png"}
    formats = [mime_map[item] for item in display_formats]
    return name, formats, start_date, end_date, [item for item in selected if item], 0 in selected


settings = Settings.from_env()
if not auth_gate(settings):
    st.stop()

try:
    with st.spinner("Opening the visual index…"):
        database, embedder, search_index = resources(settings)
except Exception as error:
    st.error("Imagepool could not start. Check the persistent volume and model configuration in Coolify.")
    st.exception(error)
    st.stop()

header_main, header_actions = st.columns([5, 1])
with header_main:
    st.markdown('<div class="ip-kicker">Private visual archive</div>', unsafe_allow_html=True)
    st.markdown('<h1 class="ip-title">Image<span>pool</span></h1>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="ip-meta">{database.count():,} assets · {html.escape(embedder.model_name)} visual index</div>',
        unsafe_allow_html=True,
    )
with header_actions:
    if st.button("Sign out", use_container_width=True):
        st.session_state.clear()
        st.rerun()

selected_asset_id = st.session_state.get("selected_asset_id")
if selected_asset_id:
    selected_asset = database.get(int(selected_asset_id))
    if selected_asset:
        asset_dialog(selected_asset)
    else:
        st.session_state.pop("selected_asset_id", None)

search_tab, library_tab, upload_tab = st.tabs(["Search by image", "Library", "Upload asset"])

with search_tab:
    st.markdown('<div class="ip-search-frame">', unsafe_allow_html=True)
    query_file = st.file_uploader(
        "Drop a reference image",
        type=["jpg", "jpeg", "png"],
        key="query-image",
        help="The reference stays in memory and is never added to your library.",
    )
    search_name, search_formats, search_start, search_end, search_collections, search_uncollected = filters("search")
    run_search = st.button("Find similar assets", type="primary", disabled=query_file is None)
    st.markdown("</div>", unsafe_allow_html=True)
    if run_search and query_file:
        try:
            prepared_query = prepare_image(
                query_file.getvalue(), query_file.name, settings.max_upload_mb
            )
            with st.spinner("Reading visual features…"):
                query_embedding = embedder.encode(prepared_query.image)
                allowed = database.matching_ids(
                    name=search_name,
                    formats=search_formats,
                    start_date=search_start,
                    end_date=search_end,
                    collection_ids=search_collections,
                    uncollected=search_uncollected,
                )
                matches = search_index.search(query_embedding, allowed_ids=allowed)
                st.session_state.search_results = matches
        except InvalidImage as error:
            st.error(str(error))
        except Exception:
            st.error("The visual search failed. Try the image again.")
    result_pairs: list[tuple[Asset, float | None]] = []
    for asset_id, score in st.session_state.get("search_results", []):
        asset = database.get(asset_id)
        if asset:
            result_pairs.append((asset, score))
    if result_pairs:
        st.markdown(f'<div class="ip-section">Closest matches · {len(result_pairs)} results</div>', unsafe_allow_html=True)
    render_assets(result_pairs, "Drop an image above to search your visual archive.", "search")

with library_tab:
    st.markdown('<div class="ip-section">Browse the archive</div>', unsafe_allow_html=True)
    library_name, library_formats, library_start, library_end, library_collections, library_uncollected = filters("library")
    filter_signature = (library_name, tuple(library_formats), library_start, library_end, tuple(library_collections), library_uncollected)
    if st.session_state.get("library_filter_signature") != filter_signature:
        st.session_state.library_page = 0
        st.session_state.library_filter_signature = filter_signature
    if "library_page" not in st.session_state:
        st.session_state.library_page = 0
    page_size = 48
    assets, total = database.list_assets(
        name=library_name,
        formats=library_formats,
        start_date=library_start,
        end_date=library_end,
        limit=page_size,
        offset=st.session_state.library_page * page_size,
        collection_ids=library_collections,
        uncollected=library_uncollected,
    )
    st.caption(f"{total:,} matching assets")
    render_assets(
        [(asset, None) for asset in assets], "No assets match these filters.", "library"
    )
    previous, page_label, following = st.columns([1, 2, 1])
    if previous.button("Previous", disabled=st.session_state.library_page == 0, use_container_width=True):
        st.session_state.library_page -= 1
        st.rerun()
    pages = max(1, (total + page_size - 1) // page_size)
    page_label.markdown(
        f'<div class="ip-meta" style="text-align:center">Page {st.session_state.library_page + 1} of {pages}</div>',
        unsafe_allow_html=True,
    )
    if following.button(
        "Next",
        disabled=(st.session_state.library_page + 1) * page_size >= total,
        use_container_width=True,
    ):
        st.session_state.library_page += 1
        st.rerun()

with upload_tab:
    st.markdown('<div class="ip-section">Import originals · no compression</div>', unsafe_allow_html=True)
    mode = st.radio("Upload mode", ["Folder", "Single image"], horizontal=True)
    folder_mode = mode == "Folder"
    upload = st.file_uploader(
        "Choose a folder" if folder_mode else "Choose an image",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files="directory" if folder_mode else False,
        key=f"asset-upload-{mode}",
    )
    files = (upload or []) if folder_mode else ([upload] if upload else [])
    collection_name = ""
    if folder_mode and files:
        roots = {file.name.replace("\\", "/").split("/")[0] for file in files}
        root_signature = tuple(sorted(roots))
        if st.session_state.get("upload_roots") != root_signature:
            st.session_state.upload_collection_name = next(iter(roots)) if len(roots) == 1 else ""
            st.session_state.upload_roots = root_signature
        collection_name = st.text_input("Collection name", key="upload_collection_name")
        st.caption("Existing collection names are merged. Subfolders are recorded as source paths.")
    total_size = sum(file.size for file in files)
    if files:
        st.caption(f"{len(files)} images · {human_bytes(total_size)} · originals stay unchanged")
        with st.expander("Review files"):
            st.dataframe([{"File": file.name, "Size": human_bytes(file.size)} for file in files], hide_index=True)
    over_limit = len(files) > 200 or total_size > 512 * 1024 * 1024
    if over_limit:
        st.error("Choose at most 200 images and 512 MB total per import.")
    if st.button("Import folder" if folder_mode else "Add image", type="primary", disabled=not files or over_limit or (folder_mode and not collection_name.strip())):
        try:
            collection_id = database.collection(collection_name) if folder_mode else None
            progress = st.progress(0, text="Starting import")
            report = import_files(
                files, collection_id, settings, database, embedder, search_index,
                on_progress=lambda value, name: progress.progress(value, text=f"Processing {name}"),
            )
            counts = {status: sum(row["Status"] == status for row in report) for status in ("Imported", "Duplicate", "Failed")}
            st.success(f"Import complete: {counts['Imported']} added · {counts['Duplicate']} duplicates · {counts['Failed']} failed")
            st.dataframe(report, hide_index=True, use_container_width=True)
        except Exception as error:
            st.error(f"Import could not finish: {error}. Saved originals are preserved; retry is safe.")
