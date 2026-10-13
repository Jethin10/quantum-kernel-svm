"""Paired noisy architecture/compiler comparisons across predeclared splits.

Uses the exact training-only model choice already exported for each split.
No held-out labels select feature maps, C, or compiler.
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from .problem import FeatureSpec,load_dataset,make_scaler,states,kernel
from .processors import load_processor
from .routing import compile_overlap,validate_compilation
from .optimization import compile_optimized
from .kernels import evaluation_pairs,simulate_pairs,assemble,sample_values
from .metrics import fit_kernel,psd_extension


SPLIT_SEEDS = (0,1,2,3,4)


def run_noisy_robustness(root,shots=4096,repeats=20,batch_size=128):
    from .experiment import export_table
    root = Path(root)
    choices = pd.read_csv(root/'results/tables/nested_split_robustness.csv')
    choices = choices[choices.model=='Exact QSVM'].set_index('split_seed')
    configs = [load_processor(root/f'processors/processor_{name}.json')[0] for name in ['A','B']]
    assert configs[0]['noise']==configs[1]['noise']
    X,y,_,_ = load_dataset()
    pairs = evaluation_pairs(80,20)
    rows,trials = [],[]
    validations,timings = {},{}
    directory = root/'data/processed/noisy_robustness'
    directory.mkdir(parents=True,exist_ok=True)
    for seed in SPLIT_SEEDS:
        choice = choices.loc[seed]
        if str(choice.topology) != 'ring':
            raise ValueError('Native parity study requires a ring feature map')
        spec = FeatureSpec(int(choice.reps),float(choice.scale),'ring')
        C = float(choice.C)
        tr,te = train_test_split(np.arange(len(y)),test_size=.2,stratify=y,random_state=seed)
        scaler = make_scaler().fit(X[tr])
        points = scaler.transform(np.vstack([X[tr],X[te]]))
        s = states(points,spec)
        exact_train,exact_cross = kernel(s[:80]),kernel(s[80:],s[:80])
        for config in configs:
            for compiler in ['baseline','fused_parity']:
                key = f"split_{seed}_{config['name']}_{compiler}"
                print(f'Noisy split study: {key}, {spec}, C={C}',flush=True)
                compile_fn = compile_overlap if compiler=='baseline' else compile_optimized
                circuit,xp,yp,meta = compile_fn(config,spec)
                validations[key] = validate_compilation(circuit,xp,yp,points[:80],exact_train)
                simulation = simulate_pairs(circuit,xp,yp,points,pairs,config,root,shots=shots,batch_size=batch_size)
                timings[key] = simulation['timing']
                raw_train,raw_cross = assemble(simulation['sampled_values'],pairs)
                train,cross,correction = psd_extension(raw_train,raw_cross)
                _,score,pred = fit_kernel(train,cross,y[tr],y[te],C)
                rmse = float(np.sqrt(np.mean((raw_train-exact_train)**2)))
                rows.append({'split_seed':seed,'processor':config['name'],'compiler':compiler,
                    **spec.to_dict(),'C':C,'cv_mean':float(choice.cv_mean),
                    'accuracy':score['accuracy'],'balanced_accuracy':score['balanced_accuracy'],'f1':score['f1'],
                    'depth':meta['depth'],'cx_count':meta['cx_count'],'routing_swaps':meta['routing_swaps'],
                    'kernel_rmse':rmse,'cross_kernel_rmse':float(np.sqrt(np.mean((raw_cross-exact_cross)**2))),
                    'aer_seconds':simulation['timing']['aer_original_seconds'],
                    'cache_hit_batches':simulation['timing']['cache_hit_batches'],
                    'evaluated_pairs':len(pairs),'shots':shots,'test_predictions':pred.tolist(),**correction})
                np.savez_compressed(directory/f'{key}.npz',X=X,y=y,train_idx=tr,test_idx=te,
                    scaled_points=points,exact_train=exact_train,exact_cross=exact_cross,
                    raw_train=raw_train,raw_cross=raw_cross,psd_train=train,psd_cross=cross,
                    histograms=simulation['histograms'],observed_probabilities=simulation['observed_probabilities'],
                    pair_indices=pairs,predictions=pred,C=C,shots=shots)
                for repeat in range(repeats):
                    a,b = assemble(sample_values(simulation['expected_values'],shots,1000+repeat),pairs)
                    _,stats,_ = fit_kernel(a,b,y[tr],y[te],C,repair=True)
                    trials.append({'split_seed':seed,'processor':config['name'],'compiler':compiler,
                                   'repeat':repeat,'accuracy':stats['accuracy']})
    table = export_table(root,'noisy_split_robustness',rows)
    export_table(root,'noisy_split_resampling',trials)
    summary = table.groupby(['processor','compiler']).agg(
        accuracy_mean=('accuracy','mean'),accuracy_std=('accuracy','std'),accuracy_min=('accuracy','min'),
        accuracy_max=('accuracy','max'),kernel_rmse_mean=('kernel_rmse','mean'),
        cx_mean=('cx_count','mean'),depth_mean=('depth','mean'),splits=('split_seed','count')).reset_index()
    export_table(root,'noisy_split_robustness_summary',summary)
    paired = []
    for processor in ['A','B']:
        t = table[table.processor==processor].set_index(['split_seed','compiler'])
        for seed in SPLIT_SEEDS:
            before,after = t.loc[(seed,'baseline')],t.loc[(seed,'fused_parity')]
            paired.append({'processor':processor,'split_seed':seed,
                           'accuracy_difference':after.accuracy-before.accuracy,
                           'kernel_rmse_reduction':before.kernel_rmse-after.kernel_rmse,
                           'cx_reduction':before.cx_count-after.cx_count,
                           'cx_reduction_percent':100*(before.cx_count-after.cx_count)/before.cx_count})
    export_table(root,'noisy_split_paired_differences',paired)
    import matplotlib.pyplot as plt
    fig,axes = plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    colors = {'A':'#2864ad','B':'#df782c'}
    for processor in ['A','B']:
        for compiler in ['baseline','fused_parity']:
            t = table[(table.processor==processor)&(table.compiler==compiler)]
            for ax,metric in zip(axes,['accuracy','kernel_rmse']):
                ax.plot(t.split_seed,t[metric],marker='o',color=colors[processor],
                        linestyle='--' if compiler=='baseline' else '-',label=f'{processor}: {compiler}')
    axes[0].set_ylabel('Seeded noisy held-out accuracy'); axes[0].set_ylim(0,1.05)
    axes[1].set_ylabel('Raw kernel RMSE versus own exact reference')
    for ax in axes: ax.set_xlabel('Predeclared split seed'); ax.set_xticks(SPLIT_SEEDS); ax.grid(alpha=.2)
    axes[1].legend(fontsize=8)
    fig.savefig(root/'results/figures/noisy_split_robustness.png',dpi=180); plt.close(fig)
    return {'split_seeds':list(SPLIT_SEEDS),'measured_kernels':len(rows),'shots':shots,
            'resampling_repeats':repeats,'compiled_overlap_max_errors':validations,'aer_timings':timings,
            'selection':'Per-split training-only five-fold exact model selection, fold-fitted scaling, C frozen across compiler/device conditions',
            'scope':'Same prescribed dataset, overlapping train/test partitions, matched synthetic noise. Not independent datasets or physical hardware.'}


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shots',type=int,default=4096)
    args = parser.parse_args()
    run_noisy_robustness(Path(__file__).resolve().parents[1],shots=args.shots)
