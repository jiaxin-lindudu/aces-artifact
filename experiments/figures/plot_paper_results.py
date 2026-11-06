"""Publication figures from existing CSVs and explicitly labeled interface formulas."""
from pathlib import Path
import hashlib
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "experiments/results/experiment_summary_all.csv"
OUTPUT = ROOT / "paper/icde2027/figures"
DATASETS = ("digits", "usps", "coil20", "mnist_10000", "har")
NAMES = ("Digits", "USPS", "COIL20", "MNIST-10k", "HAR")
COLORS = ("#235C7E", "#CB7846", "#279989", "#8C6BA5", "#B05A7A")
MARKERS = ("o", "s", "^", "D", "v")
METHODS = {"ours": ("ACES", "#235C7E", "o"),
           "ffcm": ("FFCM", "#279989", "s"),
           "fedsc": ("FedSC", "#CB7846", "^"),
           "fkm": ("FKM", "#7C818A", "D")}

def finish(fig, name, source_hash, source_label="Summary"):
    for ax in fig.axes:
        if ax.get_label() == "<colorbar>" or not ax.axison:
            continue
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(length=2.5, width=.6, pad=3)
        ax.set_axisbelow(True)
    fig.savefig(OUTPUT / name, facecolor="white",
                metadata={"Subject": source_label+" SHA256="+source_hash,
                          "Creator": "ACES / Matplotlib", "CreationDate": None, "ModDate": None})
    plt.close(fig)
    print(name)

def curve(ax, xs, rows, metric, color, marker, label, scale=1):
    y=rows[metric+"_mean"].to_numpy(float)*scale
    sd=rows[metric+"_split_std"].to_numpy(float)*scale
    ax.plot(xs,y,color=color,marker=marker,linewidth=1.6,markersize=4,label=label)
    ax.fill_between(xs,y-sd,y+sd,color=color,alpha=.10,linewidth=0)
    ax.grid(axis="y",color="#E5E9EE",linewidth=.6)
    ax.margins(x=.06)

def plot_payload_interfaces():
    """Analytical counts, not measured algorithms, bytes, or training results."""
    plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Arial","DejaVu Sans"],
        "font.size":9,"axes.labelsize":9,"axes.titlesize":9,"legend.fontsize":8.5,
        "xtick.labelsize":8.5,"ytick.labelsize":8.5,"pdf.fonttype":42})
    OUTPUT.mkdir(parents=True,exist_ok=True)
    k,d,p,slots=10,64,4,4096
    n=np.unique(np.geomspace(10,100000,400).astype(int))
    interfaces=[(np.full_like(n,k*(d+p+1)),"Dual-space statistics","#235C7E","-"),
                (np.full_like(n,k*(d+1)),"Euclidean statistics","#279989","--"),
                (n*p,"Sample coordinates","#CB7846","-")]
    fig,axes=plt.subplots(1,2,figsize=(7.16,2.35))
    for values,label,color,style in interfaces:
        axes[0].plot(n,values,label=label,color=color,linestyle=style,lw=1.7)
        axes[1].step(n,np.ceil(values/slots),where="post",color=color,linestyle=style,lw=1.7)
    for ax,title,ylabel in zip(axes,["(a) Statistical payload","(b) Ideal packed chunks"],
                              ["Scalars / client / round","Chunks / client / round"]):
        ax.set_xscale("log");ax.set_yscale("log");ax.set_title(title)
        ax.set_xlabel(r"Local records $n_i$");ax.set_ylabel(ylabel)
        ax.grid(axis="y",color="#E5E9EE",lw=.6)
        ax.set_xticks([10,100,1000,10000,100000],["10","100","1k","10k","100k"])
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,ncol=3,loc="upper center",bbox_to_anchor=(.52,1.01),frameon=False)
    fig.subplots_adjust(left=.09,right=.99,bottom=.25,top=.73,wspace=.35)
    specification=json.dumps(dict(K=k,d=d,p=p,slots=slots,kind="analytical interface counts"),sort_keys=True)
    finish(fig,"payload_interfaces.pdf",hashlib.sha256(specification.encode()).hexdigest(),"Model parameters")


def main():
    plot_payload_interfaces()
    frame=pd.read_csv(SOURCE)
    frame=frame[frame.dataset.isin(DATASETS)].copy()
    frame=frame[~(frame.study.eq("main_quality") & frame.dataset.eq("usps")
                  & frame.clients.ne(50) & ~frame.method.str.startswith("central_"))]
    # HAR uses the same statistics, read directly from its completed raw CSVs.
    # The historical summary stays unchanged; no intermediate summary is saved.
    har_root = ROOT / "experiments/results/har"
    specifications = [
        ("mechanism", "ours", "fixed_best", "grid_ours_ablation_beta0.5_clients50", "variant"),
        ("anchors", "ours", "fixed_best", "grid_ours_anchors_beta0.5_clients50", "variant"),
        ("space", "ours", "fixed_best", "grid_ours_space_beta0.5_clients50", "variant"),
        ("privacy", "ours", "fixed_best", "grid_ours_privacy_beta0.5_clients50", "condition"),
    ]
    for method in ("ours", "ffcm", "fedsc"):
        for study in ("quantity", "feature"):
            specifications.append((study, method, "heterogeneity",
                f"grid_{method}_{study}_sourcebeta0.5_clients50", "condition"))
    for method in ("ours", "fedsc", "fkm"):
        specifications.append(("participation", method, "fixed_best",
            f"grid_{method}_participation_beta0.5_clients50", "condition"))
    for study in ("samples", "clients", "clusters"):
        specifications.append(("scale_" + study, "ours", "scale",
            f"grid_ours_scale_{study}_beta0.5_clients50", "condition"))
    for method in ("ours", "ffcm", "fedsc"):
        for beta in (.1, .5, 1.):
            specifications.append(("main_quality" if beta == .5 else "label_skew_transfer",
                method, "fixed_transfer", f"grid_{method}_fixed_beta{beta}_clients50", "condition"))
    additions, har_sources = [], []
    for study, method, suffix, stem, key in specifications:
        paths = [har_root / f"federated_{seed}_clients50_{suffix}" / (stem + ".csv")
                 for seed in range(5)]
        raw = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
        if not raw.status.eq("completed").all():
            raise ValueError(f"Incomplete HAR source: {stem}")
        har_sources.extend(paths)
        numeric = [c for c in raw.select_dtypes(include=["number", "bool"]).columns
                   if c not in ("data_seed", "init_seed")]
        for condition, block in raw.groupby(key, sort=False):
            if (len(block) != 25 or block.duplicated(["data_seed", "init_seed"]).any()
                    or not block.groupby("data_seed").init_seed.apply(
                        lambda values: sorted(values) == list(range(40, 45))).all()):
                raise ValueError(f"Incomplete HAR condition: {stem}/{condition}")
            split = block.groupby("data_seed")[numeric].mean()
            label = ("beta=0.5" if study == "main_quality" else
                     f"partition_beta_{float(block.beta.iloc[0]):g}"
                     if study == "label_skew_transfer" else condition)
            record = dict(study=study, dataset="har", method=method, condition=label,
                          n_splits=5, n_runs=25,
                          source_csvs=json.dumps([str(path.resolve()) for path in paths]))
            record.update({c + "_mean": split[c].mean() for c in numeric})
            record.update({c + "_split_std": split[c].std(ddof=1) for c in numeric})
            additions.append(record)
    frame = pd.concat([frame, pd.DataFrame(additions)], ignore_index=True)
    keys=["study","dataset","method","condition"]
    if frame.duplicated(keys).any() or not frame.n_splits.eq(5).all():
        raise ValueError("Expected one row per study/dataset/method/condition and five splits")
    data=frame.set_index(keys).sort_index()
    checksum = hashlib.sha256(SOURCE.read_bytes())
    for path in sorted(set(har_sources)):
        checksum.update(str(path.relative_to(ROOT)).encode())
        checksum.update(path.read_bytes())
    digest = checksum.hexdigest()
    OUTPUT.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Arial","DejaVu Sans"],
        "font.size":8.5,"axes.labelsize":9,"axes.titlesize":9.2,"axes.titleweight":"medium",
        "xtick.labelsize":8.5,"ytick.labelsize":8.5,"legend.fontsize":8.2,
        "axes.linewidth":.65,"text.color":"#253343","axes.labelcolor":"#253343",
        "xtick.color":"#536171","ytick.color":"#536171","pdf.fonttype":42,"ps.fonttype":42})

    # Two prediction bases remain in separate panels.
    fig=plt.figure(figsize=(7.16,3.0))
    grid=fig.add_gridspec(1,2,width_ratios=[1.58,1],left=.115,right=.985,bottom=.22,top=.73,wspace=.62)
    ax=fig.add_subplot(grid[0,0])
    # Paired effects avoid hiding small mechanism differences behind lambda=0.
    specs=[("no_topology","No topology","#235C7E","o"),
           ("no_procrustes","No alignment","#279989","s"),
           ("original_membership","Original distance","#CB7846","^")]
    for offset,(variant,label,color,marker) in zip((-.22,0,.22),specs):
        for j,d in enumerate(DATASETS):
            differences=[]
            for file in json.loads(data.loc[("mechanism",d,"ours","full"),"source_csvs"]):
                raw=pd.read_csv(file)
                full=raw[raw.variant.eq("full")].set_index("init_seed").test_acc
                removed=raw[raw.variant.eq(variant)].set_index("init_seed").test_acc
                differences.append(100*(full-removed).mean())
            ax.errorbar(np.mean(differences),j+offset,xerr=np.std(differences,ddof=1),
                        fmt=marker,color=color,markersize=4,capsize=2,elinewidth=.8,
                        label=label if j==0 else None)
    ax.axvline(0,color="#A8B4C0",linewidth=.8,linestyle="--")
    ax.set_yticks(range(5),NAMES);ax.invert_yaxis()
    ax.set_xlabel("Full minus intervention (ACC pp)")
    ax.set_title("(a) Paired mechanism effects",pad=12)
    ax.grid(axis="x",color="#E5E9EE",linewidth=.6)
    ax.legend(frameon=False,loc="lower left",bbox_to_anchor=(-.2,1.16),
              ncol=2,fontsize=8,columnspacing=.8)
    ax=fig.add_subplot(grid[0,1])
    for i,d in enumerate(DATASETS):
        full=data.loc[("anchors",d,"ours","full_matched")]
        no=data.loc[("anchors",d,"ours","no_anchors")]
        vals=np.array([no.test_acc_mean,full.test_acc_mean])*100
        ax.plot(vals,[i,i],color="#B6C5CE",linewidth=2)
        for row,c,m,label in [(no,"#CB7846","o","No anchors"),(full,"#235C7E","D","Matched full")]:
            ax.errorbar(row.test_acc_mean*100,i,xerr=row.test_acc_split_std*100,
                        fmt=m,color=c,markersize=4,capsize=2,elinewidth=.75,label=label if i==0 else None)
    ax.set_yticks(range(5),NAMES); ax.invert_yaxis()
    ax.set_xlabel("Matched TEST ACC (%)");ax.set_title("(b) Anchor intervention",pad=10)
    ax.grid(axis="x",color="#E5E9EE",linewidth=.6)
    ax.legend(frameon=False,loc="lower center",bbox_to_anchor=(.45,1.25),ncol=2,fontsize=8)
    finish(fig,"mechanism_ablation.pdf",digest)

    fig,axes=plt.subplots(1,5,figsize=(7.16,1.95),sharey=True)
    for j,(d,name,ax) in enumerate(zip(DATASETS,NAMES,axes)):
        for method in ("ours","ffcm","fedsc"):
            label,color,marker=METHODS[method]; rows=[]
            for beta in (.1,.5,1.):
                key=("label_skew_tuned",d,method,f"beta={beta:.1f}") if j<2 else (
                    ("main_quality",d,method,"beta=0.5") if beta==.5 else
                    ("label_skew_transfer",d,method,f"partition_beta_{beta:g}"))
                rows.append(data.loc[key])
            curve(ax,[.1,.5,1],pd.DataFrame(rows),"test_acc",color,marker,label,100)
        ax.set_title(name+("\nRetuned" if j<2 else "\nTransfer"),fontsize=8.2,pad=7)
        ax.set_xticks([.1,.5,1],[".1",".5","1"])
        ax.set_xlabel(r"Dirichlet $\beta$")
        ax.set_ylim(40,85)
    axes[0].set_ylabel("TEST ACC (%)")
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,ncol=3,frameon=False,loc="upper center",bbox_to_anchor=(.5,1.02))
    fig.subplots_adjust(left=.075,right=.99,bottom=.25,top=.70,wspace=.18)
    finish(fig,"label_skew.pdf",digest)

    fig,axes=plt.subplots(2,5,figsize=(7.16,3.35),sharey="row")
    for j,(d,name) in enumerate(zip(DATASETS,NAMES)):
        for row,study,xs,xlabel in [(0,"quantity",[1,2,5,10],"Target size ratio"),
                                    (1,"feature",[0,.25,.5,1],"Feature shift")]:
            ax=axes[row,j]
            for method in ("ours","ffcm","fedsc"):
                label,color,marker=METHODS[method]
                rows=data.loc[(study,d,method)].loc[[study+"_"+f"{v:g}" for v in xs]]
                curve(ax,xs,rows,"test_acc",color,marker,label,100)
            ax.set_xticks(xs,[str(v).replace("0."," .").strip() for v in xs])
            ax.set_xlabel(xlabel)
            ax.set_ylim((40,85) if row==0 else (15,85))
            if row==0:
                ax.set_xscale("log",base=2);ax.set_xticks(xs,[str(v) for v in xs]);ax.minorticks_off()
                ax.set_title(name,pad=8)
    axes[0,0].set_ylabel("TEST ACC (%)");axes[1,0].set_ylabel("TEST ACC (%)")
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,ncol=3,frameon=False,loc="upper center",bbox_to_anchor=(.5,1.01))
    fig.subplots_adjust(left=.075,right=.99,bottom=.15,top=.82,wspace=.18,hspace=.62)
    finish(fig,"heterogeneity.pdf",digest)

    # Paired split-level differences: SD includes the covariance with active=1.
    cache={}
    def split_scores(row):
        values={}
        for file in json.loads(row.source_csvs):
            if file not in cache: cache[file]=pd.read_csv(file)
            raw=cache[file]
            match=raw[raw.condition.eq(row.name[-1])]
            values[int(match.data_seed.iloc[0])]=100*match.test_acc.mean()
        return pd.Series(values).sort_index()
    fig,axes=plt.subplots(1,5,figsize=(7.16,1.95),sharey=True)
    for d,name,ax in zip(DATASETS,NAMES,axes):
        for method in ("ours","fedsc","fkm"):
            label,color,marker=METHODS[method]
            base=split_scores(data.loc[("participation",d,method,"active_1")])
            deltas=pd.DataFrame({v:split_scores(data.loc[("participation",d,method,f"active_{v:g}")])-base
                                 for v in (.25,.5,.75,1)})
            xs=np.array(deltas.columns,float);y=deltas.mean().to_numpy();sd=deltas.std(ddof=1).to_numpy()
            ax.plot(xs,y,color=color,marker=marker,linewidth=1.6,markersize=4,label=label)
            ax.fill_between(xs,y-sd,y+sd,color=color,alpha=.10,linewidth=0)
        ax.axhline(0,color="#A8B4C0",linewidth=.7,linestyle=(0,(3,3)))
        ax.set_xticks([.25,.5,.75,1],[".25",".5",".75","1"]);ax.set_xlabel("Active ratio")
        ax.set_title(name,pad=8);ax.grid(axis="y",color="#E5E9EE",linewidth=.6)
    axes[0].set_ylabel("ACC change (pp)")
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,ncol=3,frameon=False,loc="upper center",bbox_to_anchor=(.5,1.01))
    fig.subplots_adjust(left=.085,right=.99,bottom=.25,top=.74,wspace=.18)
    finish(fig,"participation.pdf",digest)

    fig,axes=plt.subplots(1,3,figsize=(7.16,2.0))
    for col,study,xlabel in [(0,"scale_samples","Local sample fraction"),(1,"scale_clients","Clients"),
                             (2,"scale_clusters","Model K")]:
        for d,name,color,marker in zip(DATASETS,NAMES,COLORS,MARKERS):
            rows=data.loc[(study,d,"ours")].copy()
            xs=np.array([float(c.rsplit("_",1)[-1]) for c in rows.index])
            order=np.argsort(xs);xs=xs[order];rows=rows.iloc[order]
            curve(axes[col],xs,rows,"bytes_per_round",color,marker,name,1/2**20)
        ax=axes[col]
        ax.set_ylim(bottom=0);ax.set_xlabel(xlabel)
        if col==0: ax.set_xticks([.25,.5,.75,1],[".25",".5",".75","1"])
        elif col==1: ax.set_xticks([10,20,50,100])
        else:
            ax.set_xscale("log",base=2)
            ax.set_xticks([3,6,10,20,40,80], ["3","6","10","20","40","80"])
            ax.minorticks_off()
    axes[0].set_ylabel("Traffic (MiB / round)")
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,ncol=5,frameon=False,loc="upper center",bbox_to_anchor=(.5,1.01))
    fig.subplots_adjust(left=.085,right=.99,bottom=.25,top=.73,wspace=.34)
    finish(fig,"ckks_scaling.pdf",digest)

    fig,axes=plt.subplots(1,2,figsize=(7.16,2.5),sharey=True)
    for i,d in enumerate(DATASETS):
        a=data.loc[("privacy",d,"ours","plain")];b=data.loc[("privacy",d,"ours","ckks")]
        for ax,metric,scale in [(axes[0],"trainer_seconds",1),(axes[1],"total_bytes",1/2**20)]:
            ax.plot([a[metric+"_mean"]*scale,b[metric+"_mean"]*scale],[i,i],color="#B8C7D0",linewidth=1.5)
            for row,label,color,marker in [(a,"Plaintext","#86929D","o"),(b,"CKKS","#235C7E","D")]:
                ax.errorbar(row[metric+"_mean"]*scale,i,xerr=row[metric+"_split_std"]*scale,
                            fmt=marker,color=color,markersize=4.5,elinewidth=.8,capsize=2,label=label if i==0 else None)
    axes[0].set_yticks(range(5),NAMES);axes[0].invert_yaxis();axes[0].set_xlim(left=0)
    axes[0].set_xlabel("Execution time (s)");axes[0].set_title("(a) Execution",pad=8)
    axes[1].set_xscale("log");axes[1].set_xlabel("Total communication (MiB)")
    axes[1].set_title("(b) Communication",pad=8)
    for ax in axes: ax.grid(axis="x",color="#E5E9EE",linewidth=.6)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,ncol=2,frameon=False,loc="upper center",bbox_to_anchor=(.5,1.02))
    fig.subplots_adjust(left=.11,right=.98,bottom=.24,top=.73,wspace=.28)
    finish(fig,"privacy_cost.pdf",digest)

    fig,axes=plt.subplots(1,2,figsize=(7.16,2.3),gridspec_kw={"width_ratios":[1.25,1]})
    collapse=[]
    for d,name,color,marker in zip(DATASETS[:4],NAMES[:4],COLORS,MARKERS):
        xs=[.5,1,3,10]
        rows=data.loc[("budget",d,"fastlloyd")].loc[[f"epsilon={v:g}" for v in xs]]
        curve(axes[0],xs,rows,"test_acc",color,marker,name,100)
        collapse.append(rows.single_cluster_runs.to_numpy()/rows.n_runs.to_numpy()*100)
    ax=axes[0];ax.set_xscale("log");ax.set_xticks(xs,[.5,1,3,10]);ax.minorticks_off()
    ax.set_ylim(0,65);ax.set_xlabel(r"Privacy budget $\epsilon$");ax.set_ylabel("TEST ACC (%)")
    ax.set_title("(a) FastLloyd utility",pad=8)
    ax=axes[1];a=np.array(collapse)
    ax.pcolormesh(np.arange(5)-.5,np.arange(5)-.5,a,cmap=LinearSegmentedColormap.from_list("collapse",["#F2F5F8","#CB7846"]),vmin=0,vmax=45,shading="flat",edgecolors="white",linewidth=.7)
    ax.set_xlim(-.5,3.5);ax.set_ylim(3.5,-.5)
    ax.set_xticks(range(4),[.5,1,3,10]);ax.set_yticks(range(4),NAMES[:4])
    for i in range(4):
        for j in range(4): ax.text(j,i,f"{a[i,j]:.0f}%",ha="center",va="center",fontsize=8,
                                  color="white" if a[i,j]>25 else "#253343")
    ax.tick_params(length=0);ax.set_xlabel(r"Privacy budget $\epsilon$")
    ax.set_title("(b) Single-cluster runs",pad=8)
    for spine in ax.spines.values(): spine.set_visible(False)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,ncol=4,frameon=False,loc="upper center",bbox_to_anchor=(.5,1.02))
    fig.subplots_adjust(left=.075,right=.98,bottom=.23,top=.73,wspace=.48)
    finish(fig,"dp_budget.pdf",digest)

if __name__=="__main__":
    main()
