"""Small official NeuralForecast architectures with an explicit causal Torch loop.

Networks are the library MLP/NBEATSx/TFT implementations; windowing, scaling,
early stopping and metrics are explicit here for auditable origin-by-origin fits.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import os
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR",str(Path(".mplconfig").resolve()))
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("lightning_fabric.utilities.seed").setLevel(logging.ERROR)

import numpy as np
import pandas as pd
import torch
from neuralforecast.models import MLP,NBEATSx,TFT
from neuralforecast.losses.pytorch import MSE
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from src.evaluate import summarize_predictions
from src.features import TARGET,validate_monthly_frame
from src.hypotheses import build_hypotheses


INPUTS={"price":[],"fundamentals":["production_level","storage_seasonal_z","consumption_z"],
        "market":["production_level","storage_seasonal_z","consumption_z","BCOMNG_INDX_return"]}
MODELS=("MLP","NBEATSx","TFT")
SEEDS=(11,42,77)


def split_origins(origin,lookback=24,val_count=12):
    # A three-output training label must be completely observed by origin.
    candidates=np.arange(lookback-1,origin-3+1)
    validation=candidates[-val_count:]
    if len(validation)!=val_count:
        raise ValueError("Insufficient validation history")
    train=candidates[candidates+3<=validation[0]]
    if len(train)<12:
        raise ValueError("Insufficient purged training history")
    return train,validation


def prepare(price,features,origin,columns,lookback=24):
    train,val=split_origins(origin,lookback)
    cutoff=int(val[0])  # last training target; never use later validation prices for scaling
    logprice=np.log(np.asarray(price[:origin+1],dtype=float))
    mean=float(logprice[:cutoff+1].mean())
    scale=max(float(logprice[:cutoff+1].std()),1e-8)
    y=(logprice-mean)/scale
    x=None
    names=[]
    if columns:
        raw=features.loc[:origin,columns]
        imputer=SimpleImputer(strategy="median",add_indicator=True,keep_empty_features=True)
        fitted=imputer.fit_transform(raw.loc[:cutoff])
        scaler=StandardScaler().fit(fitted)
        x=scaler.transform(imputer.transform(raw)).astype(np.float32)
        names=list(imputer.get_feature_names_out(columns))
    def windows(origins,labels=True):
        batch={"insample_y":torch.tensor(np.stack([y[o-lookback+1:o+1,None] for o in origins]),dtype=torch.float32),
               "insample_mask":torch.ones((len(origins),lookback,1)),
               "hist_exog":None if x is None else torch.tensor(np.stack([x[o-lookback+1:o+1] for o in origins])),
               "futr_exog":None,"stat_exog":None}
        target=torch.tensor(np.stack([y[o+1:o+4] for o in origins]),dtype=torch.float32) if labels else None
        return batch,target
    return windows(train),windows(val),windows([origin],False)[0],names,mean,scale,train,val,cutoff


def network(name,names,seed,lookback=24):
    torch.manual_seed(seed)
    common=dict(h=3,input_size=lookback,hist_exog_list=names or None,
                loss=MSE(),scaler_type="identity",random_seed=seed)
    if name=="MLP":
        return MLP(**common,hidden_size=32,num_layers=2)
    if name=="NBEATSx":
        return NBEATSx(**common,stack_types=["identity","identity"],n_blocks=[1,1],
                       mlp_units=[[32,32],[32,32]],dropout_prob_theta=0.)
    if name=="TFT":
        return TFT(**common,hidden_size=8,n_head=2,dropout=.1,attn_dropout=0.)
    raise ValueError(name)


def train_one(prepared,name,seed,max_steps=80):
    (train_batch,train_y),(val_batch,val_y),now,names,mean,scale,tr,va,cutoff=prepared
    start=time.perf_counter()
    model=network(name,names,seed)
    optimizer=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=.0001)
    best=float("inf")
    best_state=None
    best_step=0
    stale=0
    # Full batch is small (<200 windows). No random split, no outer-origin label.
    for step in range(1,max_steps+1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        predicted=model(train_batch).squeeze(-1)
        loss=torch.mean((predicted-train_y)**2)
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        optimizer.step()
        if step%10==0 or step==max_steps:
            model.eval()
            with torch.no_grad():
                pred=model(val_batch).squeeze(-1)
                # Actual-price MAPE computed directly from the log-price difference.
                score=float(torch.abs(torch.expm1((pred-val_y)*scale)).mean()*100)
            if not np.isfinite(score):
                raise ValueError("Nonfinite validation score")
            if score<best:
                best=score
                best_step=step
                best_state=copy.deepcopy(model.state_dict())
                stale=0
            else:
                stale+=1
            if stale>=2:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        prediction=np.exp(model(now).squeeze().numpy().astype(float)*scale+mean)
    if prediction.shape!=(3,) or not np.isfinite(prediction).all() or (prediction<=0).any():
        raise ValueError("Invalid forecast")
    return prediction,{"best_step":best_step,"steps_run":step,"inner_mape_pct":best,
                       "parameter_count":sum(p.numel() for p in model.parameters()),
                       "train_last_origin":int(tr[-1]),"train_last_target":int(tr[-1]+3),
                       "validation_first_origin":int(va[0]),"validation_last_target":int(va[-1]+3),
                       "scaler_last_observation":cutoff,"fit_seconds":time.perf_counter()-start}


def run(args):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    data=validate_monthly_frame(pd.read_csv(args.train))
    features,definitions,_=build_hypotheses(data,"conservative")
    output=Path(args.output_dir)
    output.mkdir(parents=True,exist_ok=True)
    plan={"train_sha256":hashlib.sha256(args.train.read_bytes()).hexdigest(),
          "implementation_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          "models":args.models,"seeds":args.seeds,"inputs":INPUTS,"lookback":24,
          "first_origin":args.first_origin,"last_origin":min(args.last_origin,len(data)-2),
          "max_steps":args.max_steps,"validation_origins":12,"patience_checks":2,"check_interval":10,
          "target":"joint three-output standardized log prices; exp inverse; no recursive forecasts",
          "normalization":"fit through last training target, before validation outcomes",
          "optimizer":"Adam lr=.001 weight_decay=.0001 full-batch, grad clip=1",
          "device":"CPU single thread","torch_version":torch.__version__,
          "architecture":"official NeuralForecast MLP(32x2), NBEATSx generic identity 2 blocks (32x2 each), TFT(hidden8 heads2)",
          "training":"custom explicit Torch loop using official forward; NOT library Trainer; no refit after early stop",
          "limitations":"fixed 24-month lookback and small budget, not tuned; three input groups proposed after earlier exploration; not untouched holdout; training prefix excludes validation labels; differs from separate-horizon ML log-return models"}
    manifest_path=output/"manifest.json"
    if manifest_path.exists():
        previous=json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous!=plan:
            raise ValueError("Output configuration differs; use a separate output directory")
    manifest_path.write_text(json.dumps(plan,ensure_ascii=False,indent=2),encoding="utf-8")
    definitions.loc[definitions.feature.isin(INPUTS["market"])].to_csv(output/"feature_definitions.csv",index=False,encoding="utf-8-sig")
    price=data[TARGET].to_numpy()
    for name in args.models:
        for group,columns in INPUTS.items():
            for seed in args.seeds:
                path=output/f"{name}_{group}_{seed}.csv"
                diag_path=output/f"{name}_{group}_{seed}_fits.csv"
                rows=pd.read_csv(path).to_dict("records") if path.exists() else []
                diagnostics=pd.read_csv(diag_path).to_dict("records") if diag_path.exists() else []
                done={int(r["origin"]) for r in diagnostics}
                rows=[r for r in rows if int(r["origin"]) in done]
                for origin in range(args.first_origin,min(args.last_origin,len(data)-2)+1):
                    if origin in done:
                        continue
                    prepared=prepare(price,features,origin,columns)
                    prediction,diag=train_one(prepared,name,seed,args.max_steps)
                    for h in (1,2,3):
                        if origin+h>=len(data):
                            continue
                        rows.append({"date":str(data.loc[origin,"date"].date()),"origin":origin,"horizon":h,
                                     "model":name,"input_group":group,"seed":seed,"actual":price[origin+h],
                                     "predicted":prediction[h-1],"current_price":price[origin],"previous_price":price[origin-1]})
                    diagnostics.append({"origin":origin,"model":name,"input_group":group,"seed":seed,**diag})
                    if (origin-args.first_origin+1)%10==0:
                        print(f"{name}/{group}/seed{seed}: origin {origin} complete",flush=True)
                    pd.DataFrame(rows).to_csv(path,index=False)
                    pd.DataFrame(diagnostics).to_csv(diag_path,index=False)
                print(f"DONE {name}/{group}/seed{seed}",flush=True)
    summarize(output)


def summarize(output):
    plan=json.loads((output/"manifest.json").read_text(encoding="utf-8"))
    parts=[pd.read_csv(output/f"{m}_{g}_{s}.csv",parse_dates=["date"])
           for m in plan["models"] for g in INPUTS for s in plan["seeds"]]
    predictions=pd.concat(parts,ignore_index=True)
    rows=[]
    for (m,g,s,h),part in predictions.groupby(["model","input_group","seed","horizon"]):
        for year,sub in [("all",part),*part.groupby(part.date.dt.year)]:
            rows.append({"model":m,"input_group":g,"seed":s,"horizon":h,"year":year,**summarize_predictions(sub)})
    metrics=pd.DataFrame(rows)
    stability=metrics.loc[metrics.year=="all"].groupby(["model","input_group","horizon"]).agg(
        mape_mean=("mape_pct","mean"),mape_std=("mape_pct","std"),mape_min=("mape_pct","min"),
        mape_max=("mape_pct","max"),seeds=("seed","nunique"),turning_f1_mean=("turning_f1_pct","mean")).reset_index()
    predictions.to_csv(output/"predictions.csv",index=False)
    metrics.to_csv(output/"metrics.csv",index=False)
    stability.to_csv(output/"seed_stability.csv",index=False)
    print(stability.to_string(index=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train",type=Path)
    parser.add_argument("--output-dir",default="outputs_dl")
    parser.add_argument("--models",nargs="+",choices=MODELS,default=list(MODELS))
    parser.add_argument("--seeds",nargs="+",type=int,default=list(SEEDS))
    parser.add_argument("--first-origin",type=int,default=120)
    parser.add_argument("--last-origin",type=int,default=202)
    parser.add_argument("--max-steps",type=int,default=80)
    args=parser.parse_args()
    if args.max_steps<1 or args.first_origin>args.last_origin:
        raise ValueError("Invalid step count or origin range")
    if args.train is None:
        paths=list(Path('.').glob('*data_train.csv'))
        if len(paths)!=1:
            raise ValueError("Specify --train")
        args.train=paths[0]
    run(args)


if __name__=="__main__":
    main()
