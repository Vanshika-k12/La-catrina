"""Local storage: photos in data/wardrobe/, metadata in data/wardrobe.json, feedback in data/feedback.json."""
import hashlib
import json
import os
import re
import time

# LACATRINA_DATA_DIR lets tests (or a demo) use a separate folder without touching data/.
DATA_DIR = os.environ.get("LACATRINA_DATA_DIR", "data")
PHOTO_DIR = os.path.join(DATA_DIR, "wardrobe")
WARDROBE_FILE = os.path.join(DATA_DIR, "wardrobe.json")
FEEDBACK_FILE = os.path.join(DATA_DIR, "feedback.json")


def _ensure_dirs():
    os.makedirs(PHOTO_DIR, exist_ok=True)


def _read_json(path, default):
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path, obj):
    _ensure_dirs()
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f)
    os.replace(tmp, path)  # atomic-ish: never leaves a half-written file


def load_wardrobe():
    return _read_json(WARDROBE_FILE, [])


def save_wardrobe(items):
    _write_json(WARDROBE_FILE, items)


def content_hash(data: bytes):
    return hashlib.sha1(data).hexdigest()


def unique_photo_name(original_name):
    """Safe .jpg filename in data/wardrobe/ that does not overwrite an existing photo."""
    _ensure_dirs()
    stem = os.path.splitext(os.path.basename(original_name or "photo"))[0]
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", stem).strip("_") or "photo"
    name, n = f"{stem}.jpg", 1
    while os.path.exists(os.path.join(PHOTO_DIR, name)):
        n += 1
        name = f"{stem}_{n}.jpg"
    return name


def save_photo(img, filename, max_side=800):
    """Store a resized JPEG copy (keeps the page fast on a phone)."""
    img = img.copy()
    img.thumbnail((max_side, max_side))
    img.save(os.path.join(PHOTO_DIR, filename), "JPEG", quality=90)


def photo_path(item):
    return os.path.join(PHOTO_DIR, item["filename"])


def outfit_key(item_ids):
    """Order-independent identity of an outfit."""
    return "|".join(sorted(item_ids))


def load_feedback():
    """One rating per outfit. Older files may hold duplicates (the first version appended
    every click); for those the most recent rating wins. File format is unchanged:
    [{"items": [...ids], "label": 0|1, "time": unix_seconds}, ...]"""
    latest = {}
    for f in sorted(_read_json(FEEDBACK_FILE, []), key=lambda f: f.get("time", 0)):
        latest[outfit_key(f["items"])] = f
    return list(latest.values())


def set_feedback(item_ids, label):
    """Save (or change) the rating for an outfit; never creates a duplicate record."""
    key = outfit_key(item_ids)
    fb = [f for f in load_feedback() if outfit_key(f["items"]) != key]
    fb.append({"items": sorted(item_ids), "label": int(label), "time": time.time()})
    _write_json(FEEDBACK_FILE, fb)
    return fb


def clear_feedback(item_ids):
    """Remove the rating for an outfit (used when the user un-selects their rating)."""
    key = outfit_key(item_ids)
    fb = [f for f in load_feedback() if outfit_key(f["items"]) != key]
    _write_json(FEEDBACK_FILE, fb)
    return fb


def feedback_lookup(feedback):
    """{outfit_key: label} for quick 'what did I rate this?' checks."""
    return {outfit_key(f["items"]): int(f["label"]) for f in feedback}
