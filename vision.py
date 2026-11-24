"""CLIP zero-shot category detection, CLIP image embeddings and K-Means dominant colours."""
import os
import random
import warnings

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")  # harmless Windows cache warning

import numpy as np
import torch
from PIL import Image, ImageOps
from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning
from transformers import CLIPModel, CLIPProcessor

from colours import rgb_to_name

MODEL_NAME = "openai/clip-vit-base-patch32"
SEED = 0

CATEGORIES = ["shirt", "T-shirt", "jeans", "trousers", "skirt",
              "dress", "jacket", "footwear", "accessory"]

# A couple of prompts per class; their text embeddings are averaged.
PROMPTS = {
    "shirt": ["a photo of a button-up shirt", "a photo of a collared shirt"],
    "T-shirt": ["a photo of a t-shirt", "a photo of a plain tee",
                "a photo of a tank top", "a photo of a camisole"],
    "jeans": ["a photo of jeans", "a photo of denim jeans"],
    "trousers": ["a photo of trousers", "a photo of formal pants"],
    "skirt": ["a photo of a skirt", "a photo of a women's skirt"],
    "dress": ["a photo of a dress", "a photo of a one-piece dress"],
    "jacket": ["a photo of a jacket", "a photo of a coat or blazer"],
    "footwear": ["a photo of shoes", "a photo of sneakers, boots or sandals"],
    "accessory": ["a photo of a fashion accessory", "a photo of a bag, belt, hat or scarf"],
}


def set_seeds(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _features(out):
    """transformers versions differ: get_*_features may return a tensor or an output object."""
    if isinstance(out, torch.Tensor):
        return out
    for name in ("image_embeds", "text_embeds", "pooler_output"):
        if getattr(out, name, None) is not None:
            return getattr(out, name)
    return out[0]


def load_clip():
    """Load model + processor once and pre-compute the class text embeddings."""
    set_seeds()
    torch.set_num_threads(max(1, torch.get_num_threads()))
    model = CLIPModel.from_pretrained(MODEL_NAME).eval()
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    clip = {"model": model, "processor": processor}
    clip["text_embs"] = text_embeddings(clip, PROMPTS)
    return clip


def text_embeddings(clip, prompts):
    """One unit vector per category: the average of its prompts' CLIP text embeddings.
    `prompts` maps each name in CATEGORIES to a list of prompt strings."""
    with torch.no_grad():
        class_embs = []
        for cat in CATEGORIES:
            tok = clip["processor"](text=prompts[cat], return_tensors="pt", padding=True)
            t = _features(clip["model"].get_text_features(**tok))
            t = t / t.norm(dim=-1, keepdim=True)
            m = t.mean(dim=0)
            class_embs.append(m / m.norm())
        return torch.stack(class_embs)


def prepare_image(img, max_side=448):
    """Fix phone rotation, force RGB and shrink before inference."""
    img = ImageOps.exif_transpose(img).convert("RGB")
    img.thumbnail((max_side, max_side))
    return img


def analyse(clip, img):
    """Return (category, {category: prob}, normalised 512-d embedding as list)."""
    img = prepare_image(img)
    with torch.no_grad():
        inputs = clip["processor"](images=img, return_tensors="pt")
        e = _features(clip["model"].get_image_features(**inputs))[0]
        e = e / e.norm()
        logits = 100.0 * (clip["text_embs"] @ e)
        probs = logits.softmax(dim=0).numpy()
    best = int(np.argmax(probs))
    return (CATEGORIES[best],
            {c: float(p) for c, p in zip(CATEGORIES, probs)},
            e.numpy().astype(float).round(5).tolist())


def dominant_colours(img, k=3):
    """K-Means on garment pixels -> up to k distinct named colours.

    Background handling (simple, no segmentation): find the 2 main colours of the photo's
    border (e.g. bed + floor) and drop pixels close to them; then use the centre 70%.
    If almost nothing is left (e.g. white shirt on white background) all centre pixels are used.
    K-Means uses 5 clusters; clusters with the same colour name (shadow/highlight of one
    garment) are merged, and the k largest distinct colours are returned.
    Returns a list of {name, rgb, share}, largest first.
    """
    centers, counts = colour_clusters(img)
    return name_clusters(centers, counts, k)


def name_clusters(centers, counts, k=3):
    """Name K-Means clusters, merge clusters with the same name, return the k largest."""
    merged = {}
    for centre_rgb, cnt in zip(centers, counts):
        name = rgb_to_name(centre_rgb)
        tot, rgb_sum = merged.get(name, (0, np.zeros(3)))
        merged[name] = (tot + cnt, rgb_sum + centre_rgb * cnt)
    ranked = sorted(merged.items(), key=lambda kv: -kv[1][0])[:k]
    total = counts.sum()
    return [{"name": name, "rgb": [int(round(v)) for v in rgb_sum / cnt],
             "share": round(float(cnt / total), 3)}
            for name, (cnt, rgb_sum) in ranked]


def colour_clusters(img):
    """Background removal + K-Means (up to 5 clusters) on garment pixels.
    Returns (cluster centre RGBs, pixel counts), unnamed - naming happens in dominant_colours."""
    # plain-background photos have few distinct colours; K-Means then warns harmlessly
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    img = prepare_image(img, max_side=128)
    arr = np.asarray(img, dtype=float)
    h, w = arr.shape[:2]
    border = np.concatenate([arr[:3].reshape(-1, 3), arr[-3:].reshape(-1, 3),
                             arr[:, :3].reshape(-1, 3), arr[:, -3:].reshape(-1, 3)])
    bg = KMeans(n_clusters=2, n_init=4, random_state=SEED).fit(border).cluster_centers_
    centre = arr[int(h * 0.15):int(h * 0.85), int(w * 0.15):int(w * 0.85)].reshape(-1, 3)
    dist_bg = np.min(np.linalg.norm(centre[:, None, :] - bg[None], axis=2), axis=1)
    fg = centre[dist_bg > 40]
    px = fg if len(fg) >= 0.1 * len(centre) else centre

    n = min(5, len(np.unique(px, axis=0)))
    km = KMeans(n_clusters=n, n_init=10, random_state=SEED).fit(px)
    return km.cluster_centers_, np.bincount(km.labels_, minlength=n)
