#!/usr/bin/env python3
"""Fine-tune the saved Listing Lens V3 CNN to create a V4 candidate.

This script starts from existing EfficientNetB0 V3 weights. It never overwrites
V3 or the active app configuration. It uses the augmentation and preprocessing
layers already embedded in the saved model (raw RGB pixels 0..255).
"""
from __future__ import annotations
import argparse, hashlib, json, csv
from pathlib import Path
import numpy as np

def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()

def count_images(folder: Path):
    exts={'.jpg','.jpeg','.png','.bmp','.gif'}
    return sum(1 for p in folder.rglob('*') if p.is_file() and p.suffix.lower() in exts)

def metrics_at_threshold(y_true, scores, threshold):
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
    pred=(scores>=threshold).astype(int)
    return {
      'threshold':float(threshold), 'accuracy':float(accuracy_score(y_true,pred)),
      'balanced_accuracy':float(balanced_accuracy_score(y_true,pred)),
      'ai_precision':float(precision_score(y_true,pred,zero_division=0)),
      'ai_recall':float(recall_score(y_true,pred,zero_division=0)),
      'ai_f1':float(f1_score(y_true,pred,zero_division=0)),
      'roc_auc':float(roc_auc_score(y_true,scores)) if len(set(y_true))==2 else None,
      'confusion_matrix_actual_rows_predicted_columns':confusion_matrix(y_true,pred,labels=[0,1]).tolist()
    }

def get_labels(ds):
    labels=[]
    for _,y in ds: labels.extend(np.asarray(y).reshape(-1).astype(int).tolist())
    return np.asarray(labels,dtype=int)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--dataset-root',type=Path,default=Path('data_v4'))
    ap.add_argument('--v3-model',type=Path,default=Path('models_v3/car_cnn_model_v3.keras'))
    ap.add_argument('--v3-sha256',type=str,default='ea47dc1e9ded13944a2ac866ae968f3956a003d9faaf39ce4a0dfa8a8cad65b5',help='Expected SHA-256 for the historical V3 starting checkpoint')
    ap.add_argument('--v3-threshold',type=float,default=0.47,help='Historical V3 threshold for baseline comparison')
    ap.add_argument('--output-dir',type=Path,default=Path('outputs_v4'))
    ap.add_argument('--batch-size',type=int,default=8)
    ap.add_argument('--phase1-epochs',type=int,default=5)
    ap.add_argument('--phase2-epochs',type=int,default=12)
    args=ap.parse_args()
    try:
        import tensorflow as tf
    except Exception as e:
        raise SystemExit('TensorFlow is required. Activate the Listing Lens Python 3.11 environment and install requirements.txt.') from e
    from sklearn.utils.class_weight import compute_class_weight
    from sklearn.metrics import balanced_accuracy_score

    dataset_root=args.dataset_root.expanduser().resolve(); model_path=args.v3_model.expanduser().resolve(); out=args.output_dir.expanduser().resolve()
    if not dataset_root.is_dir(): raise SystemExit(f'Dataset V4 not found: {dataset_root}. Run prepare_v4_dataset.py first.')
    for split in ('train','validation','test'):
        for cls in ('real','ai_generated'):
            d=dataset_root/split/cls
            if not d.is_dir() or count_images(d)==0: raise SystemExit(f'Missing/empty folder: {d}')
    if not model_path.is_file(): raise SystemExit(f'V3 model not found: {model_path}')
    expected=args.v3_sha256.strip().lower(); actual=sha256(model_path)
    if expected and actual!=expected: raise SystemExit(f'V3 model hash mismatch. Expected {expected}, got {actual}. Do not train from an unverified checkpoint.')
    out.mkdir(parents=True,exist_ok=True)
    tf.keras.utils.set_random_seed(42)
    AUTOTUNE=tf.data.AUTOTUNE
    CLASS_NAMES=['real','ai_generated']  # label 0=real, 1=AI; do not rely on alphabetical ordering.
    def load_split(name,shuffle):
        return tf.keras.utils.image_dataset_from_directory(
            dataset_root/name, labels='inferred', label_mode='binary', class_names=CLASS_NAMES,
            color_mode='rgb', batch_size=args.batch_size, image_size=(256,256),
            interpolation='bilinear', shuffle=shuffle, seed=42 if shuffle else None
        ).prefetch(AUTOTUNE)
    train_raw=load_split('train',True); val_ds=load_split('validation',False); test_ds=load_split('test',False)
    counts={s:{c:count_images(dataset_root/s/c) for c in CLASS_NAMES} for s in ('train','validation','test')}
    y_train=np.array([0]*counts['train']['real']+[1]*counts['train']['ai_generated'])
    classes=np.array([0,1]); weights=compute_class_weight(class_weight='balanced',classes=classes,y=y_train)
    class_weights={int(c):float(w) for c,w in zip(classes,weights)}
    def add_sample_weights(x,y):
        sw=tf.where(tf.reshape(y,[-1])>=0.5,class_weights[1],class_weights[0])
        return x,y,tf.cast(sw,tf.float32)
    train_ds=train_raw.map(add_sample_weights,num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)

    print('Dataset counts:',json.dumps(counts,indent=2))
    print('Balanced sample weights:',class_weights)
    print('Verified V3 checkpoint SHA-256:',actual)
    model=tf.keras.models.load_model(model_path,compile=False)
    if tuple(model.input_shape[-3:])!=(256,256,3) or model.output_shape[-1]!=1:
        raise SystemExit(f'Unexpected model shape: {model.input_shape} -> {model.output_shape}')
    try: backbone=model.get_layer('efficientnetb0')
    except ValueError as e: raise SystemExit("Could not find embedded layer 'efficientnetb0' in V3 checkpoint; inspect model before training.") from e

    def compile_model(lr):
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                      loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.05),
                      metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy'),tf.keras.metrics.AUC(name='auc'),tf.keras.metrics.Precision(name='precision'),tf.keras.metrics.Recall(name='recall')])
    def callbacks(prefix):
        return [
          tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=3,restore_best_weights=True,verbose=1),
          tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',factor=0.5,patience=2,min_lr=1e-7,verbose=1),
          tf.keras.callbacks.CSVLogger(str(out/f'{prefix}_history.csv'))
        ]

    # Phase 1: retain the V3 feature extractor; adapt the classifier head gently.
    model.trainable=True
    backbone.trainable=False
    compile_model(1e-4)
    h1=model.fit(train_ds,validation_data=val_ds,epochs=args.phase1_epochs,callbacks=callbacks('phase1'),verbose=1)
    p1=out/'phase1_best.keras'; model.save(p1)

    # Phase 2: unfreeze only the last 40 backbone layers, leaving all BN layers frozen.
    model=tf.keras.models.load_model(p1,compile=False)
    backbone=model.get_layer('efficientnetb0')
    backbone.trainable=True
    for layer in backbone.layers: layer.trainable=False
    for layer in backbone.layers[-40:]:
        if not isinstance(layer,tf.keras.layers.BatchNormalization): layer.trainable=True
    compile_model(1e-5)
    best_path=out/'car_cnn_model_v4_candidate.keras'
    cb=callbacks('phase2')+[tf.keras.callbacks.ModelCheckpoint(str(best_path),monitor='val_loss',save_best_only=True,verbose=1)]
    h2=model.fit(train_ds,validation_data=val_ds,epochs=args.phase2_epochs,callbacks=cb,verbose=1)
    if not best_path.exists(): model.save(best_path)
    candidate=tf.keras.models.load_model(best_path,compile=False)

    # Select a threshold using validation balanced accuracy only; test remains untouched.
    def collect(ds):
        ys=get_labels(ds)
        xs=ds.map(lambda x,y:x)
        scores=candidate.predict(xs,verbose=0).reshape(-1).astype(float)
        if len(ys)!=len(scores): raise RuntimeError('Validation label/prediction count mismatch.')
        return ys,scores
    y_val,s_val=collect(val_ds)
    choices=[]
    for t in np.arange(0.20,0.801,0.01):
        pred=(s_val>=t).astype(int)
        choices.append((balanced_accuracy_score(y_val,pred),float(t)))
    best_bal=max(x[0] for x in choices)
    threshold=min(t for score,t in choices if abs(score-best_bal)<1e-12)
    y_test,s_test=collect(test_ds)
    test_metrics=metrics_at_threshold(y_test,s_test,threshold)
    val_metrics=metrics_at_threshold(y_val,s_val,threshold)

    # Report the new-generator challenge examples separately: these are AI-only groups,
    # so report detection rate/recall, not precision or balanced accuracy.
    add_manifest=dataset_root/'v4_additions_manifest.csv'
    generator_results={}
    if add_manifest.exists():
        with add_manifest.open(newline='',encoding='utf-8-sig') as f: additions=list(csv.DictReader(f))
        test_additions=[r for r in additions if r['split']=='test']
        grouped={}
        for r in test_additions: grouped.setdefault(r['generator'],[]).append(r)
        for gen,items in grouped.items():
            arrays=[]; names=[]
            for r in items:
                p=dataset_root/r['destination']
                im=tf.keras.utils.load_img(p,target_size=(256,256),color_mode='rgb',interpolation='bilinear')
                arrays.append(tf.keras.utils.img_to_array(im)); names.append(p.name)
            probs=candidate.predict(np.asarray(arrays,dtype=np.float32),verbose=0).reshape(-1)
            generator_results[gen]={'n':len(items),'ai_detected':int(np.sum(probs>=threshold)),'ai_detection_rate':float(np.mean(probs>=threshold)),'mean_ai_score':float(np.mean(probs)),'files':names,'interpretation':'small exploratory AI-only holdout; not a robust estimate'}

    # Baseline V3 evaluation on the exact same untouched V4 test split.
    # This comparison uses V3's recorded 0.47 threshold, while V4 uses the
    # threshold chosen on V4 validation data.
    v3_model=tf.keras.models.load_model(model_path,compile=False)
    y_val_v3=get_labels(val_ds)
    s_val_v3=v3_model.predict(val_ds.map(lambda x,y:x),verbose=0).reshape(-1).astype(float)
    y_test_v3=get_labels(test_ds)
    s_test_v3=v3_model.predict(test_ds.map(lambda x,y:x),verbose=0).reshape(-1).astype(float)
    v3_threshold=float(args.v3_threshold)
    v3_val_metrics=metrics_at_threshold(y_val_v3,s_val_v3,v3_threshold)
    v3_test_metrics=metrics_at_threshold(y_test_v3,s_test_v3,v3_threshold)

    final_hash=sha256(best_path)
    report={
      'model':'Listing Lens V4 candidate, fine-tuned from EfficientNetB0 V3',
      'v3_source_model':'user-supplied historical V3 starting checkpoint','v3_source_sha256':actual,
      'v3_source_hash_unchanged_check':sha256(model_path)==actual,
      'v4_candidate_model':str(best_path),'v4_candidate_sha256':final_hash,
      'input':{'shape':[256,256,3],'color':'RGB','range':'0..255 float32','resize':'bilinear direct resize','normalization':'kept inside saved EfficientNetB0 model; no external /255'},
      'training':{'phase1_lr':1e-4,'phase1_max_epochs':args.phase1_epochs,'phase1_backbone':'frozen','phase2_lr':1e-5,'phase2_max_epochs':args.phase2_epochs,'phase2_backbone':'top 40 layers trainable except BatchNormalization','loss':'BinaryCrossentropy(label_smoothing=0.05)','optimizer':'Adam','batch_size':args.batch_size,'class_weights':class_weights,'seed':42},
      'dataset_counts':counts,'selected_threshold_from_validation_balanced_accuracy':threshold,
      'validation_metrics_at_selected_threshold':val_metrics,
      'test_metrics_at_selected_threshold':test_metrics,
      'v3_baseline_on_same_v4_validation_and_test_splits':{'threshold':v3_threshold,'validation_metrics':v3_val_metrics,'test_metrics':v3_test_metrics},
      'new_generator_challenge_test':generator_results,
      'caveats':['The supplied new-generator challenge set is only 17 images total and is too small for strong per-generator claims.','The user-provided folder labels GPT/GEMINI/META are recorded source labels, not independently authenticated.','This script requires the historical V3 starting checkpoint, supplied separately; V3 weights are not distributed in the final application repository.','Do not use the test split for training or threshold tuning.']
    }
    (out/'v4_training_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('\nV4 candidate saved:',best_path)
    print('Validation metrics:',json.dumps(val_metrics,indent=2))
    print('Test metrics:',json.dumps(test_metrics,indent=2))
    print('New-generator challenge:',json.dumps(generator_results,indent=2))
    print('\nV3 source model remained unchanged:',sha256(model_path)==actual)

if __name__=='__main__': main()
