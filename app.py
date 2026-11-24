"""La Catrina - AI outfit planner prototype (Streamlit)."""
import base64
import html
import io
import os

import streamlit as st
from PIL import Image
from scipy.stats import spearmanr

import planner
import preference
import storage
import vision
from colours import COLOUR_NAMES, rgb_to_hex

st.set_page_config(page_title="La Catrina", page_icon="👗", layout="centered")

# Vintage Analog / Retro Film look. Colours, fonts and surfaces live in .streamlit/config.toml;
# this CSS only adds what the theme can't: film grain, polaroid prints, VHS hover, fade-ins.
st.html("""<style>
:root {
  --film-cream: #F5E6C8; --kodak-tan: #D4A574; --faded-teal: #4A7B7C; --teal-ink: #335859;
  --dusty-rose: #E8B4B8; --sepia-brown: #8B4513; --ink: #4E342E;
}
/* Film grain overlay (light enough to keep text above 4.5:1) */
body::before {
  content: ""; position: fixed; inset: 0; z-index: 10; pointer-events: none;
  opacity: 0.14; mix-blend-mode: multiply;
  background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 200 200'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.65' numOctaves='3' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E");
}
h1 { color: var(--sepia-brown) !important; }

/* Polaroid prints: each rests at its own tilt so the hover shake never jumps */
.photo {
  --tilt: -1.2deg; position: relative; overflow: hidden; display: inline-block;
  background: #FFFFFF; border: 8px solid #FFFFFF; border-bottom-width: 32px;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2); transform: rotate(var(--tilt));
}
.photo:nth-child(even) { --tilt: 1.4deg; }
.photo.small { border-width: 4px; border-bottom-width: 14px; margin: 0 6px 8px 0;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2); }
.photo img { display: block; width: 100%; filter: sepia(0.15) contrast(0.95) saturate(0.9); }
.photo:hover { animation: film-shake 400ms ease-in-out; }
.photo:hover img { animation: vhs-track 500ms steps(4) 1; }
.photo::after {  /* warm light leak from the corner */
  content: ""; position: absolute; inset: 0; pointer-events: none;
  background: radial-gradient(circle at 100% 0%, rgba(232, 180, 184, 0.45), transparent 55%);
  mix-blend-mode: screen;
}
@keyframes film-shake {
  0%, 100% { transform: rotate(var(--tilt)); }
  33% { transform: rotate(calc(var(--tilt) + 1.5deg)); }
  66% { transform: rotate(calc(var(--tilt) - 1.5deg)); }
}
@keyframes vhs-track {
  0%, 100% { transform: none; }
  25% { transform: translateX(-4px); filter: sepia(0.15) drop-shadow(3px 0 0 rgba(255, 0, 0, 0.5)) drop-shadow(-3px 0 0 rgba(0, 200, 255, 0.5)); }
  50% { transform: translateX(3px) skewX(-2deg); }
  75% { transform: translateX(-2px); }
}
@keyframes film-fade { from { opacity: 0; } }
.fade-in, [class*="st-key-day-"] { animation: film-fade 600ms ease-out both; }

/* Badges: palette colours as backgrounds, ink text on top (AA) */
.lc-badge { display: inline-block; color: #3E2723; background: var(--kodak-tan);
  font-size: 0.75rem; padding: 1px 8px; border-radius: 3px; margin-left: 6px; vertical-align: middle; }
.lc-badge.rose { background: var(--dusty-rose); }
.lc-day { color: var(--sepia-brown); font-size: 1.1rem; }
/* Small print: ink-soft instead of Streamlit's faded caption grey, which is below 4.5:1 on cream */
.lc-note { color: #6D4C41; font-size: 0.875rem; margin: 0 0 0.5rem; }

a:hover { text-decoration: underline; }
a:focus-visible, button:focus-visible { outline: 2px solid var(--teal-ink); outline-offset: 3px; }

@media (prefers-reduced-motion: reduce) {
  .fade-in, [class*="st-key-day-"], .photo, .photo:hover, .photo:hover img { animation: none; }
}
</style>""")


def note(text):
    """Small print in a colour that meets AA contrast on the cream background."""
    st.markdown(f"<p class='lc-note'>{html.escape(text)}</p>", unsafe_allow_html=True)


def badge(text, style=""):
    return f"<span class='lc-badge {style}'>{text}</span>"


def polaroid(b64, alt, small=False):
    return (f"<div class='photo{' small' if small else ''}'>"
            f"<img src='data:image/jpeg;base64,{b64}' alt='{alt}'></div>")
ss = st.session_state
ss.setdefault("up_n", 0)
ss.setdefault("add_msg", None)


@st.cache_resource(show_spinner="Loading CLIP model (first run downloads ~600 MB)...")
def get_clip():
    return vision.load_clip()


@st.cache_data(show_spinner=False)
def thumb_b64(path, mtime, size=160):
    img = Image.open(path)
    img.thumbnail((size, size))
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=80)
    return base64.b64encode(buf.getvalue()).decode()


def thumb(item, size=160):
    p = storage.photo_path(item)
    return thumb_b64(p, os.path.getmtime(p), size)


def swatches_html(colours):
    chips = "".join(
        f"<span style='display:inline-block;margin:2px 8px 2px 0;white-space:nowrap'>"
        f"<span style='display:inline-block;width:14px;height:14px;border-radius:3px;"
        f"vertical-align:middle;border:1px solid #888;background:{rgb_to_hex(c['rgb'])}'></span>"
        f" {c['name']} {int(c['share'] * 100)}%</span>"
        for c in colours if c["share"] >= 0.01)
    return f"<div style='font-size:0.85rem'>{chips}</div>"


def add_photos(uploads):
    clip = get_clip()
    items = storage.load_wardrobe()
    known = {it["sha1"] for it in items}
    added, skipped = [], []
    bar = st.progress(0.0, text="Analysing photos...")
    for k, f in enumerate(uploads):
        data = f.getvalue()
        h = storage.content_hash(data)
        if h in known:
            skipped.append(f"{f.name} (already in wardrobe)")
            continue
        try:
            img = vision.prepare_image(Image.open(io.BytesIO(data)), max_side=800)
        except Exception:
            skipped.append(f"{f.name} (not a readable image)")
            continue
        cat, probs, emb = vision.analyse(clip, img)
        colours = vision.dominant_colours(img)
        filename = storage.unique_photo_name(f.name)
        storage.save_photo(img, filename)
        items.append({"id": h[:12], "sha1": h, "filename": filename,
                      "category": cat, "predicted_category": cat, "category_probs": probs,
                      "colours": colours, "main_colour": colours[0]["name"],
                      "predicted_main_colour": colours[0]["name"],
                      "embedding": emb, "corrected": False})
        known.add(h)
        added.append(f"{filename}: {cat} ({probs[cat]:.0%}), {colours[0]['name']}")
        bar.progress((k + 1) / len(uploads), text=f"Analysed {k + 1}/{len(uploads)}")
    storage.save_wardrobe(items)
    return added, skipped


st.title("👗 La Catrina")
note("AI outfit planner - prototype. Compatibility scores are a baseline, not a trained model.")
get_clip()

tab_add, tab_wardrobe, tab_plan = st.tabs(["➕ Add", "👚 Wardrobe", "📅 Week plan"])

# ---------------------------------------------------------------- Add
with tab_add:
    files = st.file_uploader("Clothing photos (on a phone this can open the camera)",
                             type=["jpg", "jpeg", "png", "webp"],
                             accept_multiple_files=True, key=f"up_{ss.up_n}")
    shot = None
    if st.toggle("Optional: in-page camera (needs HTTPS or localhost)"):
        shot = st.camera_input("Take a photo", key=f"cam_{ss.up_n}")
    uploads = list(files or []) + ([shot] if shot else [])
    if st.button(f"Add {len(uploads)} photo(s) to wardrobe", type="primary", disabled=not uploads):
        ss.add_msg = add_photos(uploads)
        ss.up_n += 1  # resets the uploader so photos aren't processed twice
        st.rerun()
    if ss.add_msg:
        added, skipped = ss.add_msg
        if added:
            st.success("Added:\n\n" + "\n\n".join(f"- {a}" for a in added))
        for s in skipped:
            st.warning(f"Skipped {s}")

# ---------------------------------------------------------------- Wardrobe
with tab_wardrobe:
    items = storage.load_wardrobe()
    if not items:
        st.info("Your wardrobe is empty - add photos in the ➕ Add tab.")
    else:
        n_fixed = sum(it["corrected"] for it in items)
        note(f"{len(items)} items, {n_fixed} corrected by you. Fix anything CLIP got wrong.")
    for it in items:
        c1, c2 = st.columns([1, 2])
        c1.markdown(f"<div class='fade-in' style='max-width:160px'>"
                    f"{polaroid(thumb(it), it['category'] + ', ' + it['main_colour'])}</div>",
                    unsafe_allow_html=True)
        with c2.form(key=f"edit_{it['id']}", border=False):
            conf = it["category_probs"][it["predicted_category"]]
            st.markdown(f"**{it['filename']}** · CLIP: {it['predicted_category']} ({conf:.0%})"
                        + (badge("✏️ corrected") if it["corrected"] else ""), unsafe_allow_html=True)
            st.markdown(swatches_html(it["colours"]), unsafe_allow_html=True)
            cat = st.selectbox("Category", vision.CATEGORIES,
                               index=vision.CATEGORIES.index(it["category"]), key=f"cat_{it['id']}")
            col = st.selectbox("Main colour", COLOUR_NAMES,
                               index=COLOUR_NAMES.index(it["main_colour"]), key=f"col_{it['id']}")
            if st.form_submit_button("Save"):
                items = storage.load_wardrobe()
                for x in items:
                    if x["id"] == it["id"]:
                        x["category"], x["main_colour"] = cat, col
                        x["corrected"] = (cat != x["predicted_category"]
                                          or col != x["predicted_main_colour"])
                storage.save_wardrobe(items)
                st.toast(f"Saved {it['filename']}: {cat}, {col}")
                st.rerun()
        st.divider()

# ---------------------------------------------------------------- Week plan
def outfit_html(items_by_id, ids):
    imgs = "".join(
        f"<div style='width:76px'>"
        + polaroid(thumb(items_by_id[i], 120),
                   f"{items_by_id[i]['main_colour']} {items_by_id[i]['category']}", small=True)
        + "</div>"
        for i in ids)
    return f"<div class='fade-in' style='display:flex;flex-wrap:wrap;padding:4px 0'>{imgs}</div>"


def short_desc(items_by_id, ids):
    return " + ".join(f"{items_by_id[i]['main_colour']} {items_by_id[i]['category']}"
                      for i in ids if i in items_by_id)


def fb_signature(feedback):
    return tuple(sorted((storage.outfit_key(f["items"]), int(f["label"])) for f in feedback))


def make_plan(items, feedback, weight):
    scorer, model = planner.build_scorer(items, feedback, weight)
    disliked = [f["items"] for f in feedback if f["label"] == 0]
    week, notes, cands = planner.plan_week(scorer, disliked_ids=disliked)
    return {"week": week, "notes": notes, "n": len(cands), "n_feedback": model.n_used,
            "personalised": scorer.personalised, "weight": weight, "fb": fb_signature(feedback),
            "version": ss.get("plan_version", 0) + 1}


def style_preferences(model, scorer, cands, feedback, items_by_id):
    """'My style preferences': feedback counts, model status, learned preferences, evaluation."""
    with st.container(border=True, key="style-prefs"):
        st.subheader("My style preferences")
        c1, c2, c3 = st.columns(3)
        c1.metric("Liked outfits", sum(f["label"] == 1 for f in feedback))
        c2.metric("Disliked outfits", sum(f["label"] == 0 for f in feedback))
        c3.metric("Preference model", "Ready" if model.ready else "Not yet")
        note(model.reason + ("" if model.ready else
                             " Until then outfits are ranked on compatibility only."))
        if model.ready:
            likes, dislikes = planner.summarise_preferences(model)
            if likes or dislikes:
                if likes:
                    st.markdown("**You tend to like:** " + ", ".join(likes))
                if dislikes:
                    st.markdown("**You tend to avoid:** " + ", ".join(dislikes))
                note(f"Learned from {model.n_used} ratings - a small sample, so treat these as "
                     "tentative. They come straight from the model's largest weights.")
            else:
                note("No single feature stands out in your ratings yet.")
        st.slider("Personalisation weight", 0.0, 0.5, planner.PREF_WEIGHT, 0.05, key="pref_weight",
                  help="final score = (1 - weight) x compatibility + weight x predicted like. "
                       "0 = ignore ratings; higher = ratings matter more. Applies when you "
                       "(re-)plan.")
        with st.expander("How well does the model predict my ratings?"):
            evaluation_panel(model, scorer, cands, feedback, items_by_id)


def evaluation_panel(model, scorer, cands, feedback, items_by_id):
    X, y = planner.training_data(list(items_by_id.values()), feedback)
    ev = preference.evaluate(planner.FEATURE_NAMES, X, y)
    if not ev["ready"]:
        note(ev["reason"] + " No evaluation is shown for so few ratings.")
        return
    rows = [{"Metric": m.capitalize(), "Training (optimistic)": f"{ev['train'][m]:.2f}",
             "Cross-validated": f"{ev['cv'][m]:.2f}" if ev["cv"] else "-"}
            for m in ("accuracy", "precision", "recall", "f1")]
    st.dataframe(rows, hide_index=True, width="stretch")
    note(f"Majority-class baseline accuracy: {ev['majority_baseline_accuracy']:.2f}. "
         + (f"Cross-validation: {ev['cv_folds']}-fold stratified; scaler and model re-fitted in "
            "every fold, so held-out ratings never influence training."
            if ev["cv"] else ev["cv_reason"])
         + f" Based on {ev['n']} ratings - too few for reliable numbers.")
    if scorer.personalised and len(cands) > 2:
        rho = spearmanr([c["base_rank"] for c in cands], [c["rank"] for c in cands]).statistic
        note(f"Personalised vs compatibility-only ranking of all {len(cands)} possible outfits: "
             f"Spearman correlation {rho:.2f} (1.00 = identical order).")
        liked, disliked = [], []
        for f in feedback:
            if all(i in scorer.index for i in f["items"]):
                idx = [scorer.index[i] for i in f["items"]]
                base = scorer.base_score(idx)
                base_rank = 1 + sum(c["base"] > base + 1e-12 for c in cands)
                (liked if f["label"] else disliked).append(
                    (base_rank, planner.rank_of(scorer, cands, idx)))
        if liked and disliked:
            mean = lambda v, k: sum(x[k] for x in v) / len(v)  # noqa: E731
            st.dataframe([{"Rated outfits": "Liked", "Avg rank, compatibility only": f"{mean(liked, 0):.1f}",
                           "Avg rank, personalised": f"{mean(liked, 1):.1f}"},
                          {"Rated outfits": "Disliked", "Avg rank, compatibility only": f"{mean(disliked, 0):.1f}",
                           "Avg rank, personalised": f"{mean(disliked, 1):.1f}"}],
                         hide_index=True, width="stretch")
            note("These outfits are the training data, so this shows the model fitting your "
                 "ratings, not how well it generalises.")


with tab_plan:
    items = storage.load_wardrobe()
    items_by_id = {it["id"]: it for it in items}
    feedback = storage.load_feedback()
    ratings = storage.feedback_lookup(feedback)
    ss.setdefault("plan", None)
    ss.setdefault("prev_plan", None)
    weight = ss.get("pref_weight", planner.PREF_WEIGHT)

    # one trained model + candidate ranking per run, reflecting the latest saved ratings
    scorer_now = model_now = cands_now = None
    if items:
        scorer_now, model_now = planner.build_scorer(items, feedback, weight)
        cands_now = planner.candidates(scorer_now) if not planner.missing_parts(items) else []
        style_preferences(model_now, scorer_now, cands_now, feedback, items_by_id)

    c1, c2 = st.columns(2)
    if c1.button("📅 Plan my week", type="primary", disabled=not items, width="stretch"):
        ss.plan, ss.prev_plan = make_plan(items, feedback, weight), None
        ss.plan_version = ss.plan["version"]
    if c2.button("🔁 Re-plan with my feedback", disabled=ss.plan is None, width="stretch"):
        ss.prev_plan, ss.plan = ss.plan, make_plan(items, feedback, weight)
        ss.plan_version = ss.plan["version"]
    note("Rate outfits below. Ratings are saved straight away and the model retrains; "
         "re-plan to rebuild the week with the updated ranking.")

    plan = ss.plan
    if plan:
        for plan_note in plan["notes"]:
            st.warning(plan_note)
        note("This plan is " + (f"personalised (weight {plan['weight']:.2f}, "
                                f"{plan['n_feedback']} ratings)." if plan["personalised"]
                                else "ranked on compatibility only (personalisation not active)."))

        # Before / after comparison (shown after a re-plan)
        prev = ss.prev_plan
        if prev and prev["week"] and plan["week"]:
            rows = []
            for b, a in zip(prev["week"], plan["week"]):
                row = {"Day": b["day"][:3],
                       "Before score (rank)": f"{b['score']:.3f} ({b['rank']})",
                       "Old outfit now": "-"}
                if all(i in scorer_now.index for i in b["ids"]):
                    idx = [scorer_now.index[i] for i in b["ids"]]
                    row["Old outfit now"] = (f"{scorer_now.score(tuple(idx))['score']:.3f} "
                                             f"({planner.rank_of(scorer_now, cands_now, idx)})")
                row["After score (rank)"] = f"{a['score']:.3f} ({a['rank']})"
                row["Before outfit"] = short_desc(items_by_id, b["ids"])
                row["After outfit"] = short_desc(items_by_id, a["ids"])
                rows.append(row)
            st.markdown(f"**Before vs after feedback** (rank 1 = best of {plan['n']} possible outfits; "
                        f"ratings used before: {prev['n_feedback']}, after: {plan['n_feedback']})")
            st.dataframe(rows, hide_index=True, width="stretch")

        stale = plan["fb"] != fb_signature(feedback)
        for d in plan["week"] or []:
            if not all(i in items_by_id for i in d["ids"]):
                continue
            with st.container(border=True, key=f"day-{plan['version']}-{d['day']}"):
                st.markdown(f"<span class='lc-day'>{d['day']}</span> · score {d['score']:.3f}"
                            + badge(f"rank {d['rank']}/{plan['n']}", "rose"), unsafe_allow_html=True)
                st.markdown(outfit_html(items_by_id, d["ids"]), unsafe_allow_html=True)
                note(d["explanation"])
                note(d["personal"] or "Ranked on compatibility only (personalisation not active yet).")
                if stale and all(i in scorer_now.index for i in d["ids"]):
                    idx = [scorer_now.index[i] for i in d["ids"]]
                    now = scorer_now.score(tuple(idx))
                    note(f"With your latest ratings this outfit now scores {now['score']:.3f} "
                         f"(rank {planner.rank_of(scorer_now, cands_now, idx)}/{len(cands_now)}).")
                current = ratings.get(storage.outfit_key(d["ids"]))
                b1, b2, _ = st.columns([1, 1, 2])
                for col, label, text, icon in ((b1, 1, "Like", ":material/thumb_up:"),
                                               (b2, 0, "Dislike", ":material/thumb_down:")):
                    chosen = current == label
                    if col.button(text, icon=icon, key=f"fb_{label}_{plan['version']}_{d['day']}",
                                  type="primary" if chosen else "secondary", width="stretch",
                                  help="Click again to remove your rating" if chosen else None):
                        if chosen:
                            storage.clear_feedback(d["ids"])
                        else:
                            storage.set_feedback(d["ids"], label)
                        st.rerun()
                if current is not None:
                    note(f"You {'liked' if current else 'disliked'} this outfit (saved).")
