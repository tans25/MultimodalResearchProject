import torch 
import clip 
import argparse 
import os 
import re 
import shutil 
from collections import defaultdict
from pathlib import Path 
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from tqdm import tqdm 
import csv 
csv.field_size_limit(2**20)
import numpy as np
import ast
import requests
requests.packages.urllib3.disable_warnings()
import io 
import time 

from concurrent.futures import ThreadPoolExecutor, as_completed

MAX_DOWNLOAD_WORKERS = 8  # number of parallel download threads

NOISE_PROMPTS = [
    "a clean corporate logo centered on a white background",
    "a social media logo like facebook, instagram, linkedin, bluesky, twitter on a white blackground",
    "a person speaking into microphone",
    "a portrait photograph of a person with a white background",
    "a group of people smiling",
    "a person talking on the stage",
    "a brand logo with text and simple design",
    "a flat vector logo with solid colors",
    "a company wordmark logo",
    "a minimal icon logo",
    "one or multiple mobile app icons logo",
    "a website header with logo and branding",
]
MAX_RETRIES = 2
RETRY_DELAY = 1.0
SAVE_SIZE = (384,384)
ARTICLE_CHUNK_SIZE = 50
ALLOWED_FORMATS = {"JPEG", "WEBP"}  # PIL format strings — PNG="PNG", GIF="GIF" etc.
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".webp"}
NOISE_URL_KEYWORDS = {"logo", "icon", "avatar", "banner", "badge", "favicon"}


def write_result(results, scores_writer):
    for result in results:
        scores_writer.writerow(result)

def write_no_image(article_id, no_image_writer):
    no_image_writer.writerow({"article_id": article_id})

def open_output_files(scores_csv, no_image_csv):
    score_fields = [
        "article_id", "filename", "decision", "reason",
        "noise_score", "best_noise_prompt", "image_path"
    ]
    scores_file = open(scores_csv, "a", newline="", encoding="utf-8")
    no_image_file = open(no_image_csv, "a", newline="", encoding="utf-8")

    scores_writer = csv.DictWriter(scores_file, fieldnames=score_fields)
    no_image_writer = csv.DictWriter(no_image_file, fieldnames=["article_id"])

    if scores_csv.stat().st_size == 0:
        scores_writer.writeheader()
    if no_image_csv.stat().st_size == 0:
        no_image_writer.writeheader()
    
    return scores_file, no_image_file, scores_writer, no_image_writer

def close_output_files(scores_file, no_image_file):
    scores_file.close()
    no_image_file.close()
    print("Files closed safely")

def heuristic_filter(filepath, fmt):
    url = Path(filepath)
    ext = url.suffix.lower()
    if any(keyword in filepath.lower() for keyword in NOISE_URL_KEYWORDS):
        return False, f"url_contains_noise_keyword"
    if ext not in ALLOWED_EXTENSIONS:
        return False, "disallowed_extension"
    if fmt not in ALLOWED_FORMATS:
        return False, "disallowed_format"
    return True, "passed_heuristic"

def load_clip_model(device):
    print("Loading clip")
    model, preprocess = clip.load("ViT-L/14", device=device)
    model.eval()
    return model, preprocess 

def encode_prompts(model, prompts: list[str], device):
    tokens = clip.tokenize(prompts).to(device)
    with torch.no_grad():
        embeddings = model.encode_text(tokens)
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
    return embeddings 


def download_img(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/120.0.0.0 Safari/537.36",
    "Accept": "image/*,*/*;q=0.8",
    }
    for attempt in range(MAX_RETRIES + 1):
        try:
            response = requests.get(url, headers=headers, timeout=10, verify=False, stream=True)
            response.raise_for_status()
            img = Image.open(io.BytesIO(response.content))
            return img 
        except requests.exceptions.HTTPError as e:
            return None 
        except Exception as e:
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
            else:
                return None 


def load_articles(articles_csv: Path):
    articles = []
    with open(articles_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                urls = ast.literal_eval(row["images"])
                urls = [u.strip() for u in urls if u.strip()]
            except (ValueError, SyntaxError, KeyError):
                print(f" [skip] could not parse image urls article {row['id']}: {row['images']}")
                continue 
            if urls:
                articles.append({
                    "id": row["id"].strip(),
                    "images": urls
                })
    return articles 

CLIP_BATCH_SIZE = 32  # increase if you have a GPU with more VRAM

def clip_batch_score(survived, model, preprocess, noise_embeddings, device, args, output_dir):
    results = []
    embeddings_out = []

    for batch_start in range(0, len(survived), CLIP_BATCH_SIZE):
        batch = survived[batch_start: batch_start + CLIP_BATCH_SIZE]

        # Stack images into a single tensor
        tensors = torch.stack([
            preprocess(img) for _, _, _, img, _ in batch
        ]).to(device)

        with torch.no_grad():
            img_embs = model.encode_image(tensors)
            img_embs = img_embs / img_embs.norm(dim=-1, keepdim=True)

        noise_sims = (img_embs @ noise_embeddings.T).cpu().tolist()

        for i, (url, idx, article_id, img, fmt) in enumerate(batch):
            sims      = noise_sims[i]
            max_score = max(sims)
            best_prompt = NOISE_PROMPTS[sims.index(max_score)]

            if max_score >= args.noise_threshold:
                results.append({
                    "article_id": article_id,
                    "image_path":  url,
                    "filename":   None,
                    "decision":   "discard",
                    "reason":     f"noise_match ({best_prompt})",
                    "noise_score": round(max_score, 4),
                    "best_noise_prompt": best_prompt,
                })
            else:
                # Save image
                save_ext      = ".jpg" if fmt in {"JPEG", "MPO"} else ".webp"
                save_filename = f"article_{article_id}_img_{idx}{save_ext}"
                save_path     = output_dir / save_filename
                img.resize((384, 384), Image.LANCZOS).save(save_path, quality=90)

                # Store embedding
                embeddings_out.append({
                    "article_id": article_id,
                    "filename":   save_filename,
                    "embedding":  img_embs[i].cpu().numpy()
                })

                results.append({
                    "article_id": article_id,
                    "image_path":  url,
                    "filename":   save_filename,
                    "decision":   "keep",
                    "reason":     "passed_noise_filter",
                    "noise_score": round(max_score, 4),
                    "best_noise_prompt": best_prompt,
                })

    return results, embeddings_out

def download_and_heuristic(args):
    """
    Worker function: download one image and run heuristic filter.
    Returns (url, idx, img, fmt, keep, reason) 
    """
    url, idx, article_id = args
    img = download_img(url)
    
    if img is None:
        return url, idx, article_id, None, None, False, "download_failed"
    
    fmt = img.format or ""
    img = img.convert("RGB")
    keep, reason = heuristic_filter(url, fmt)
    
    return url, idx, article_id, img, fmt, keep, reason


def process_article_batch(articles, model, preprocess, noise_embeddings, device, args, output_dir):
    # ── Step 1: Collect all download tasks ────────────────────────────────
    tasks = []
    for article in articles:
        for idx, url in enumerate(article["images"]):
            tasks.append((url, idx, article["id"]))

    # ── Step 2: Download in parallel ──────────────────────────────────────
    survived = []   # (url, idx, article_id, img, fmt) — passed heuristic
    results  = []   # score rows for CSV
    all_embeddings_out = []
    total_downloaded = 0
    with ThreadPoolExecutor(max_workers=MAX_DOWNLOAD_WORKERS) as executor:
        futures = {executor.submit(download_and_heuristic, t): t for t in tasks}
        for future in as_completed(futures):
            url, idx, article_id, img, fmt, keep, reason = future.result()
            if keep:
                survived.append((url, idx, article_id, img, fmt))
                total_downloaded += 1 
            else:
                if reason != "download_failed":
                    total_downloaded += 1 
                results.append({
                    "article_id": article_id,
                    "image_path":  url,
                    "filename":   None,
                    "decision":   "discard",
                    "reason":     reason,
                    "noise_score": None,
                    "best_noise_prompt": None,
                })

    # ── Step 3: CLIP batch inference on survived images ────────────────────
    if survived:
        clip_results, clip_embeddings = clip_batch_score(
            survived, model, preprocess, noise_embeddings, device, args, output_dir
        )
        results.extend(clip_results)
        all_embeddings_out.extend(clip_embeddings)

    return results, all_embeddings_out, total_downloaded


def main(args):
    output_dir = Path(args.output_dir)
    scores_csv = Path(args.scores_csv)
    no_image_csv = Path(args.no_image_csv)
 
    output_dir.mkdir(parents=True, exist_ok=True)
    scores_csv.parent.mkdir(parents=True, exist_ok=True)
    no_image_csv.parent.mkdir(parents=True, exist_ok=True)
 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
 
    # Load model and encode prompts once — reused for all images
    model, preprocess = load_clip_model(device)
    noise_embeddings = encode_prompts(model, NOISE_PROMPTS, device)
    # content_embeddings = encode_prompts(model, AI_CONTENT_PROMPTS, device)
 
    # Group images by article
    print(f"\nScanning {args.articles_csv}...")
    articles = load_articles(Path(args.articles_csv))
    print(f"Found {len(articles)} articles\n")
    
    embeddings_map = {}

    total_downloaded = 0
    total_kept = 0
    total_discarded = 0
    total_failed = 0 
    scores_file, no_image_file, scores_writer, no_image_writer = open_output_files(scores_csv, no_image_csv)
    try:
        for chunk_start in range(0, len(articles), ARTICLE_CHUNK_SIZE):
            chunk = articles[chunk_start: chunk_start + ARTICLE_CHUNK_SIZE]
            results, embeddings, total_downloaded_batch = process_article_batch(chunk, model, preprocess, noise_embeddings, device, args, output_dir)
            # all_scores.extend(results)
            write_result(results, scores_writer)
            total_downloaded += total_downloaded_batch
            for e in embeddings:
                aid = e["article_id"]
                if aid not in embeddings_map:
                    embeddings_map[aid] = []
                embeddings_map[aid].append((e["filename"], e["embedding"]))
            kept_per_article = {}
            for r in results:
                aid = r["article_id"]
                if aid not in kept_per_article:
                    kept_per_article[aid] = 0
                if r["decision"] == "keep":
                    kept_per_article[aid] += 1
            no_image_count = 0
            for article in chunk:
                if kept_per_article.get(article["id"], 0) == 0:
                    write_no_image(article["id"], no_image_writer)
                    no_image_count += 1

            # Update totals
            for r in results:
                if r["decision"] == "keep":
                    total_kept += 1
                elif r["reason"] == "download_failed":
                    total_failed += 1
                else:
                    total_discarded += 1
            scores_file.flush()
            no_image_file.flush()    
            print(f"  Processed articles {chunk_start}–{chunk_start + len(chunk)} | "
                f"Kept: {total_kept} | Discarded: {total_discarded} | Failed: {total_failed}")
    finally:
        close_output_files(scores_file, no_image_file)

    print("\nSaving embeddings...")
    emb_output_dir = output_dir.parent / "embeddings"
    emb_output_dir.mkdir(parents=True, exist_ok=True)
 
    all_embeddings = []
    all_filenames  = []
 
    for article_id, items in embeddings_map.items():
        for filename, emb in items:
            all_filenames.append({"article_id": article_id, "filename": filename})
            all_embeddings.append(emb)
 
    if all_embeddings:
        np.save(emb_output_dir / "image_embeddings.npy", np.stack(all_embeddings))
        with open(emb_output_dir / "image_ids.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["article_id", "filename"])
            writer.writeheader()
            writer.writerows(all_filenames)
        print(f"  Saved {len(all_embeddings)} embeddings → {emb_output_dir}")

 
    # Summary
    total_attempted = total_downloaded + total_failed
    print(f"\n{'='*50}")
    print(f"  Articles processed        : {len(articles)}")
    print(f"  Total image URLs attempted: {total_attempted}")
    print(f"  Successfully downloaded   : {total_downloaded}")
    print(f"  Download failures         : {total_failed}")
    print(f"  Kept (passed filter)      : {total_kept} ({100*total_kept/max(total_downloaded,1):.1f}%)")
    print(f"  Discarded (noise/heuristic): {total_discarded} ({100*total_discarded/max(total_downloaded,1):.1f}%)")
    print(f"  Images saved to           : {output_dir}")
    print(f"{'='*50}\n")
 
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download article images and filter noise inline using CLIP"
    )
    parser.add_argument("--articles_csv",     required=True,
                        help="CSV with columns: article_id, image_urls (pipe-separated)")
    parser.add_argument("--output_dir",       required=True,
                        help="Directory to save kept images")
    parser.add_argument("--scores_csv",       required=True,
                        help="Path to write per-image decision CSV")
    parser.add_argument("--no_image_csv",     required=True,
                        help="Path to write article_ids with no surviving images")
    parser.add_argument("--noise_threshold",  type=float, default=0.25,
                        help="Discard image if max noise score >= this value (default: 0.25)")
    args = parser.parse_args()
    main(args)
