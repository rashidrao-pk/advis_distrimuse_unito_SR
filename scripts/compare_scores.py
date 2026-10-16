#!/usr/bin/env python3
"""Compare score CSVs from original ADVIS and R2AD after supplying labels explicitly."""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score,precision_recall_fscore_support,confusion_matrix

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--scores',required=True,help='R2AD scores.csv');p.add_argument('--labels',required=True,help='CSV with file and label (0 normal, 1 anomaly)')
    p.add_argument('--threshold-json',required=True);p.add_argument('--out',required=True)
    args=p.parse_args();import json
    tau=json.loads(Path(args.threshold_json).read_text())['threshold']
    scores=pd.read_csv(args.scores);labels=pd.read_csv(args.labels)
    if labels['file'].duplicated().any():raise ValueError('Duplicate file names in labels')
    df=scores.merge(labels[['file','label']],on='file',how='inner',validate='one_to_one')
    if len(df)!=len(scores):raise ValueError('Some score files lack labels: use exact matching file paths')
    y=df['label'].to_numpy(dtype=int);s=df['score'].to_numpy();pred=(s>tau).astype(int)
    if not set(y).issubset({0,1}):raise ValueError('Labels must be binary 0/1')
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    precision,recall,f1,_=precision_recall_fscore_support(y,pred,average='binary',zero_division=0)
    result={'n':len(df),'tp':int(tp),'fp':int(fp),'tn':int(tn),'fn':int(fn),
            'precision':float(precision),'recall':float(recall),'f1':float(f1),
            'fpr':float(fp/max(fp+tn,1)),'balanced_accuracy':float((tp/max(tp+fn,1)+tn/max(tn+fp,1))/2),
            'auroc':float(roc_auc_score(y,s)) if len(np.unique(y))==2 else None,
            'aupr':float(average_precision_score(y,s)) if len(np.unique(y))==2 else None}
    out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
