"""Architecture exploration and exact compiler optimization, with frozen models."""
from dataclasses import replace
import json
import numpy as np
import pandas as pd
from qiskit import qpy
from sklearn.model_selection import train_test_split
from .optimization import compile_optimized
from .routing import compile_overlap, validate_compilation
from .processors import load_processor, reference_b
from .problem import load_dataset, make_scaler, states, kernel
from .kernels import simulate_pairs, assemble, sample_values
from .metrics import fit_kernel, psd_extension
from .selection import select_quantum, classical_baselines


def explore(root,spec,C,shots,repeats,batch_size):
    from .experiment import export_table,write_json
    configs = [load_processor(root/f"processors/processor_{name}.json")[0] for name in ["A","B"]]
    variants = [configs[0],
        {**configs[0],"name":"A_tree","edges":[[0,1],[1,2],[1,3],[3,4]],"rationale":"Guide tree reference."},
        {**configs[0],"name":"A_cycle","edges":[[0,1],[1,2],[2,3],[3,0],[3,4]],"rationale":"Four-cycle plus tail isolates connectivity from qubit budget."},
        configs[1],{**configs[1],**reference_b(),"rationale":"Guide reference B graph; matched synthetic channels."},
        {**configs[1],"name":"B_tree","edges":[[0,1],[0,2],[1,3],[1,4],[2,5],[2,6],[5,7],[6,8]],"rationale":"Nine-node tree tests whether extra qubits alone reduce routing."}]
    (root/'processors/variants').mkdir(exist_ok=True)
    grid = []
    for config in variants:
        write_json(root/f"processors/variants/{config['name']}.json",config)
        for topology in ["linear","ring","full"]:
            for strategy in ["basic","sabre"]:
                _,_,_,meta = compile_overlap(config,replace(spec,topology=topology),strategy)
                import networkx as nx
                graph = nx.Graph(config['edges'])
                grid.append({**meta,"interaction":topology,"physical_edges":graph.number_of_edges(),
                             "graph_density":nx.density(graph),"maximum_degree":max(dict(graph.degree()).values())})
    export_table(root,"topology_grid",grid)
    X,y,tr,te = load_dataset()
    from .design_controls import training_controls
    training_controls(root,X[tr],y[tr],spec,configs)
    scaler = make_scaler().fit(X[tr])
    points = scaler.transform(np.vstack([X[tr],X[te]]))
    ref = np.load(root/'data/processed/dataset_and_reference.npz')
    exact_train,exact_cross = ref['exact_train'],ref['exact_cross']
    pairs = [tuple(p) for p in ref['pair_indices']]
    rows,trials = [],[]
    validations,timings = {},{}
    for config,strategy in [(configs[0],"baseline"),(configs[1],"baseline"),
                             (configs[0],"fused_sabre"),(configs[1],"fused_sabre"),
                             (configs[0],"fused_parity"),(configs[1],"fused_parity"),
                             (variants[1],"baseline"),(variants[-1],"baseline")]:
        name = config['name']
        key = f"{name}_{strategy}"
        compiler = compile_optimized if strategy == "fused_parity" else compile_overlap
        circuit,xp,yp,meta = compiler(config,spec) if strategy != "fused_sabre" else compile_overlap(config,spec,strategy)
        validations[key] = validate_compilation(circuit,xp,yp,points[:80],exact_train)
        print(f"Architecture/optimization acquisition: {key}",flush=True)
        simulation = simulate_pairs(circuit,xp,yp,points,pairs,config,root,shots=shots,batch_size=batch_size)
        timings[key] = simulation['timing']
        raw_train,raw_cross = assemble(simulation['sampled_values'],pairs)
        train,cross,correction = psd_extension(raw_train,raw_cross)
        _,stats,pred = fit_kernel(train,cross,y[tr],y[te],C)
        rows.append({**meta,"compiler":strategy,"accuracy":stats['accuracy'],"f1":stats['f1'],
                     "kernel_rmse":float(np.sqrt(np.mean((raw_train-exact_train)**2))),
                     "cross_rmse":float(np.sqrt(np.mean((raw_cross-exact_cross)**2))),
                     "evaluated_pairs":len(pairs),"shots":shots,
                     "aer_seconds":simulation['timing']['aer_original_seconds'],
                     "test_predictions":pred.tolist(),**correction})
        if strategy != "baseline" or name.endswith('tree'):
            with (root/f"assets/circuits/{key}.qpy").open('wb') as handle:
                qpy.dump(circuit,handle)
            np.savez_compressed(root/f"data/processed/{key}_kernels.npz",
                raw_train=raw_train,raw_cross=raw_cross,psd_train=train,psd_cross=cross,
                histograms=simulation['histograms'],
                observed_probabilities=simulation['observed_probabilities'],pair_indices=pairs,
                predictions=pred,test_labels=y[te],C=C)
        for repeat in range(repeats):
            sampled = sample_values(simulation['expected_values'],shots,1000+repeat)
            a,b = assemble(sampled,pairs)
            _,score,_ = fit_kernel(a,b,y[tr],y[te],C,repair=True)
            trials.append({"processor":name,"compiler":strategy,"repeat":repeat,"accuracy":score['accuracy']})
    export_table(root,"compiler_and_topology_measurements",rows)
    export_table(root,"compiler_resampling",trials)
    trial_frame = pd.DataFrame(trials)
    summary = trial_frame.groupby(['processor','compiler']).accuracy.agg(['mean','std','min','max']).reset_index()
    export_table(root,"compiler_resampling_summary",summary)
    # Predeclared split seeds: nested training-only tuning in every split.
    # Exact simulation diagnoses split sensitivity independently of device shots.
    robustness = []
    for seed in range(10):
        print(f"Nested split robustness {seed+1}/10",flush=True)
        train_idx,test_idx = train_test_split(np.arange(len(y)),test_size=.2,stratify=y,random_state=seed)
        selected,penalty,selection = select_quantum(X[train_idx],y[train_idx])
        scale = make_scaler().fit(X[train_idx])
        ts = states(scale.transform(X[train_idx]),selected)
        vs = states(scale.transform(X[test_idx]),selected)
        _,stats,_ = fit_kernel(kernel(ts),kernel(vs,ts),y[train_idx],y[test_idx],penalty)
        baseline,_ = classical_baselines(X[train_idx],y[train_idx],X[test_idx],y[test_idx])
        robustness.append({"split_seed":seed,"model":"Exact QSVM",**selected.to_dict(),"C":penalty,
                           "cv_mean":float(selection.iloc[0].cv_mean),"accuracy":stats['accuracy']})
        for row in baseline:
            robustness.append({"split_seed":seed,"model":row['model'],"accuracy":row['accuracy'],"cv_mean":row['cv_mean']})
    export_table(root,"nested_split_robustness",robustness)
    robustness_frame = pd.DataFrame(robustness)
    export_table(root,'nested_split_robustness_summary',robustness_frame.groupby('model').accuracy.agg(['mean','std','min','max','count']).reset_index())
    import matplotlib.pyplot as plt
    fig,axes = plt.subplots(1,3,figsize=(15,4))
    for ax,metric in zip(axes,['cx_count','depth','routing_swaps']):
        table = pd.DataFrame(grid)
        table = table[table.strategy=='sabre'].pivot(index='processor',columns='interaction',values=metric)
        table.plot.bar(ax=ax,rot=25,title=metric.replace('_',' '))
        ax.get_legend().remove()
    handles,labels = axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',ncol=3,frameon=False)
    fig.tight_layout(rect=[0,0,1,.92]); fig.savefig(root/'results/figures/topology_grid.png',dpi=180); plt.close(fig)
    fig,axes = plt.subplots(1,2,figsize=(11,4))
    frame = pd.DataFrame(rows)
    labels = frame.processor+' / '+frame.compiler
    axes[0].bar(labels,frame.cx_count); axes[0].set_ylabel('Native CX per pair')
    axes[1].bar(labels,frame.kernel_rmse); axes[1].set_ylabel('Raw training kernel RMSE')
    for ax in axes: ax.tick_params(axis='x',rotation=40)
    fig.tight_layout(); fig.savefig(root/'results/figures/compiler_optimization.png',dpi=180); plt.close(fig)
    return {"compilation_errors":validations,"timings":timings,"topology_grid_rows":len(grid),
            "additional_measured_kernels":6,"nested_split_seeds":list(range(10)),
            "robustness_scope":"Exact kernels only; repeated splits share the same fixed dataset."}
