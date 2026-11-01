#!/usr/bin/env python3
"""Build Dataset V4 by preserving Dataset V2 and adding grouped AI candidates.

Expected Dataset V2 layout:
  <v2_root>/train/real, train/ai_generated
  <v2_root>/validation/real, validation/ai_generated
  <v2_root>/test/real, test/ai_generated

The original root is never modified. New data is copied into a fresh output root.
Exact duplicate AI candidates already present anywhere in V2 are skipped. An exact
match against a real-labelled V2 image is treated as a hard error for review.
Perceptual near matches are written to a review CSV and are never auto-deleted.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, shutil, sys
from pathlib import Path
from collections import Counter, defaultdict
from PIL import Image, ImageOps

EXTS={'.jpg','.jpeg','.png','.bmp','.gif','.webp','.tif','.tiff'}
KERAS_EXTS={'.jpg','.jpeg','.png','.bmp','.gif'}

def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()

def image_files(folder: Path):
    return sorted(p for p in folder.rglob('*') if p.is_file() and p.suffix.lower() in EXTS and not p.name.startswith('.'))

def verify_image(path: Path):
    with Image.open(path) as im:
        im.verify()
    with Image.open(path) as im:
        im.seek(0); im.convert('RGB').load()

def convert_unsupported_webp(root: Path):
    converted=[]
    for p in image_files(root):
        if p.suffix.lower() not in KERAS_EXTS:
            try:
                with Image.open(p) as im:
                    im.seek(0); rgb=ImageOps.exif_transpose(im).convert('RGB')
                    dest=p.with_suffix('.png')
                    n=1
                    while dest.exists():
                        dest=p.with_name(f'{p.stem}_converted_{n}.png'); n+=1
                    rgb.save(dest,format='PNG',optimize=True)
                p.unlink(); converted.append({'from':str(p.relative_to(root)),'to':str(dest.relative_to(root))})
            except Exception as e:
                raise RuntimeError(f'Cannot convert image {p}: {e}') from e
    return converted

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--v2-root',required=True,type=Path,help='Dataset V2 split root')
    ap.add_argument('--candidate-root',type=Path,default=Path(__file__).resolve().parent/'ai_candidates')
    ap.add_argument('--output-root',type=Path,default=Path('data_v4'))
    ap.add_argument('--phash-distance',type=int,default=6)
    args=ap.parse_args()
    v2=args.v2_root.expanduser().resolve(); cand=args.candidate_root.expanduser().resolve(); out=args.output_root.expanduser().resolve()
    if not v2.is_dir(): raise SystemExit(f'Dataset V2 root not found: {v2}')
    if not cand.is_dir(): raise SystemExit(f'Candidate folder not found: {cand}')
    if out.exists() and any(out.iterdir()): raise SystemExit(f'Output already exists and is not empty: {out}. Choose a new --output-root; source data is not modified.')
    expected=[(s,c) for s in ('train','validation','test') for c in ('real','ai_generated')]
    missing=[str(v2/s/c) for s,c in expected if not (v2/s/c).is_dir()]
    if missing: raise SystemExit('Dataset V2 layout is missing these folders:\n'+'\n'.join(missing))
    manifest=cand/'v4_candidate_manifest.csv'
    if not manifest.is_file(): raise SystemExit(f'Missing manifest: {manifest}')
    with manifest.open(newline='',encoding='utf-8-sig') as f: candidates=list(csv.DictReader(f))
    if not candidates: raise SystemExit('Candidate manifest is empty.')
    # Hash existing V2 images; identify conflicts before copying anything.
    existing_hashes=defaultdict(list); v2_counts={}
    for split,cls in expected:
        folder=v2/split/cls; files=image_files(folder); v2_counts[f'{split}/{cls}']=len(files)
        if not files: raise SystemExit(f'No images found in {folder}')
        for p in files: existing_hashes[sha256(p)].append((cls,p))
    # Optional pHash comparison against the base dataset for manual screening.
    try:
        import imagehash
        phash_available=True
    except Exception:
        imagehash=None; phash_available=False
    v2_phashes=[]
    if phash_available:
        for split,cls in expected:
            for p in image_files(v2/split/cls):
                try:
                    with Image.open(p) as im: v2_phashes.append((str(p.relative_to(v2)),cls,imagehash.phash(im.convert('RGB'))))
                except Exception: pass
    added=[]; skipped=[]; conflicts=[]; cross_near=[]; pending_copy=[]
    for row in candidates:
        rel=Path(row['file']); src=cand/rel
        if not src.is_file(): raise SystemExit(f'Manifest image missing: {src}')
        verify_image(src)
        if row.get('label')!='ai_generated': raise SystemExit(f'Unexpected label for {rel}: {row.get("label")}')
        split=row['split_assignment']
        if split not in {'train','validation','test'}: raise SystemExit(f'Invalid split assignment for {rel}: {split}')
        digest=sha256(src)
        matches=existing_hashes.get(digest,[])
        if matches:
            if any(cls=='real' for cls,_ in matches):
                conflicts.append({'candidate':str(rel),'sha256':digest,'v2_matches':';'.join(f'{cls}:{p}' for cls,p in matches)})
            else:
                skipped.append({'candidate':str(rel),'reason':'exact duplicate already exists in V2 AI class','v2_matches':';'.join(str(p) for _,p in matches)})
            continue
        if phash_available:
            try:
                with Image.open(src) as im: ph=imagehash.phash(im.convert('RGB'))
                for v2_name,v2_cls,vph in v2_phashes:
                    dist=ph-vph
                    if dist<=args.phash_distance:
                        cross_near.append({'candidate':str(rel),'split':split,'v2_class':v2_cls,'v2_file':v2_name,'phash_hamming_distance':dist,'review_status':'MANUAL_REVIEW_NOT_AUTO_DELETED'})
            except Exception: pass
        gen=row['source_folder_label'].lower()
        dest_rel=Path(split)/'ai_generated'/f'{gen}_{src.name}'
        pending_copy.append((src,dest_rel,{'candidate':str(rel),'generator':gen,'split':split,'sha256':digest,'leakage_group_id':row['leakage_group_id']}))
    if conflicts:
        out_report=Path.cwd()/'v4_exact_label_conflicts.csv'
        with out_report.open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(conflicts[0]));w.writeheader();w.writerows(conflicts)
        raise SystemExit(f'ABORT: {len(conflicts)} new AI candidates exactly match V2 real-labelled images. Review {out_report}; no output dataset was created. Resolve these before retrying.')
    # Copy only after all candidate validation and exact-label-conflict checks pass.
    shutil.copytree(v2,out,dirs_exist_ok=True)
    converted=convert_unsupported_webp(out)
    for src,dest_rel,meta in pending_copy:
        dest=out/dest_rel
        dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.exists(): raise SystemExit(f'Destination collision: {dest}')
        shutil.copy2(src,dest)
        meta['destination']=str(dest.relative_to(out))
        added.append(meta)
    # Count final dataset and write provenance.
    final_counts={}
    for split,cls in expected: final_counts[f'{split}/{cls}']=len(image_files(out/split/cls))
    def write_csv(path, rows):
        if not rows: path.write_text('none\n',encoding='utf-8'); return
        with path.open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    write_csv(out/'v4_additions_manifest.csv',added)
    write_csv(out/'v4_exact_duplicates_skipped.csv',skipped)
    write_csv(out/'v4_cross_dataset_near_duplicate_review.csv',cross_near)
    summary={'dataset_version':'V4','source_dataset_v2':str(v2),'new_candidate_root':str(cand),'original_v2_counts':v2_counts,'final_v4_counts':final_counts,'new_candidates_added':len(added),'exact_duplicates_skipped':len(skipped),'cross_dataset_near_duplicate_pairs_for_manual_review':len(cross_near),'webp_or_unsupported_images_converted_to_png':len(converted),'imagehash_available_for_cross_dataset_near_duplicate_scan':phash_available,'new_candidates_added_by_split':dict(Counter(x['split'] for x in added)),'new_candidates_added_by_generator':dict(Counter(x['generator'] for x in added)),'notes':['Original Dataset V2 folders are copied; the source root is never modified.','New AI candidates are the only additions; no real images are added or removed.','Perceptual matches are review flags, never automatic deletions.','Generator labels come from user folder names and are not independently authenticated.','The 17-image new-generator holdout is exploratory and too small for strong claims.','WebP conversion is performed only in the V4 copy to avoid Keras directory-loader skips.']}
    (out/'v4_preparation_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    (out/'v4_webp_conversion_log.json').write_text(json.dumps(converted,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))
    print('\nDataset V4 created at:',out)

if __name__=='__main__': main()
