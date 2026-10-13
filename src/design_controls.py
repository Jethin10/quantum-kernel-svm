"""Training-only cost/quality and coordinate-encoding diagnostics."""
from dataclasses import replace
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from .problem import states,kernel,make_scaler,interaction_edges
from .routing import compile_overlap
from .selection import C_VALUES


def derived_states(points,spec):
    p = ParameterVector('z',2)
    theta = [spec.scale*t for t in [p[0],p[1],(p[0]+p[1])/2,(p[0]-p[1])/2]]
    circuit = QuantumCircuit(4)
    for _ in range(spec.reps):
        circuit.h(range(4))
        for i in range(4): circuit.rz(2*theta[i],i)
        for i,j in interaction_edges(spec.topology): circuit.rzz(2*theta[i]*theta[j],i,j)
    return np.asarray([Statevector.from_instruction(circuit.assign_parameters(dict(zip(p,x)))).data for x in points])


def training_controls(root,X,y,spec,configs):
    from .experiment import export_table
    folds = list(StratifiedKFold(5,shuffle=True,random_state=42).split(X,y))
    rows = []
    for topology in ['linear','ring','full']:
        for reps in [1,2,3]:
            candidate = replace(spec,topology=topology,reps=reps)
            scores = {C:[] for C in C_VALUES}
            for tr,va in folds:
                scale = make_scaler().fit(X[tr])
                s = states(scale.transform(X),candidate)
                for C in C_VALUES:
                    model = SVC(kernel='precomputed',C=C).fit(kernel(s[tr]),y[tr])
                    scores[C].append(model.score(kernel(s[va],s[tr]),y[va]))
            best = min(C_VALUES,key=lambda C:(-np.mean(scores[C]),np.std(scores[C],ddof=1),C))
            for config in configs:
                _,_,_,meta = compile_overlap(config,candidate)
                rows.append({'processor':config['name'],'interaction':topology,'reps':reps,'scale':spec.scale,
                             'C':best,'cv_mean':np.mean(scores[best]),'cv_std':np.std(scores[best],ddof=1),
                             'cx_count':meta['cx_count'],'depth':meta['depth']})
    for row in rows:
        row['pareto'] = not any(other['processor']==row['processor'] and
             other['cx_count']<=row['cx_count'] and other['cv_mean']>=row['cv_mean'] and
             (other['cx_count']<row['cx_count'] or other['cv_mean']>row['cv_mean']) for other in rows)
    export_table(root,'training_cost_quality',rows)
    encoding = []
    for name,state_function in [('repeated_coordinates',states),('mean_difference_coordinates',derived_states)]:
        scores = {C:[] for C in C_VALUES}
        for tr,va in folds:
            scale = make_scaler().fit(X[tr])
            s = state_function(scale.transform(X),spec)
            for C in C_VALUES:
                model = SVC(kernel='precomputed',C=C).fit(kernel(s[tr]),y[tr])
                scores[C].append(model.score(kernel(s[va],s[tr]),y[va]))
        for C in C_VALUES:
            encoding.append({'encoding':name,'reps':spec.reps,'scale':spec.scale,'C':C,
                             'cv_mean':np.mean(scores[C]),'cv_std':np.std(scores[C],ddof=1),
                             'fold_scores':scores[C]})
    export_table(root,'encoding_training_control',encoding)
    import matplotlib.pyplot as plt
    fig,axes = plt.subplots(1,2,figsize=(11,4),sharey=True)
    for ax,config in zip(axes,configs):
        for row in rows:
            if row['processor']!=config['name']: continue
            color = {'linear':'#ee7733','ring':'#228833','full':'#4477aa'}[row['interaction']]
            ax.scatter(row['cx_count'],row['cv_mean'],marker='*' if row['pareto'] else 'o',s=100,color=color)
            ax.annotate(f"{row['interaction'][0].upper()}{row['reps']}",(row['cx_count'],row['cv_mean']),
                        xytext=(3,6),textcoords='offset points',fontsize=8)
        ax.set_title(config['name']); ax.set_xlabel('Baseline native CX per overlap'); ax.grid(alpha=.2)
    axes[0].set_ylabel('Training-only 5-fold CV accuracy')
    fig.suptitle('L: linear · R: ring · F: full K4; stars: evaluated Pareto frontier',fontsize=10)
    fig.tight_layout(rect=[0,0,1,.94]); fig.savefig(root/'results/figures/training_cost_quality.png',dpi=180); plt.close(fig)


def evaluate_cost_choice(root,shots,repeats,batch_size):
    """Secondary model selected by training CV, then minimax baseline CX cost."""
    import pandas as pd
    from .experiment import export_table
    from .problem import FeatureSpec,load_dataset
    from .processors import load_processor
    from .routing import validate_compilation
    from .kernels import simulate_pairs,evaluation_pairs,assemble,sample_values
    from .metrics import psd_extension,fit_kernel
    candidates = pd.read_csv(root/'results/tables/training_cost_quality.csv')
    choices = candidates.groupby(['interaction','reps','scale','C','cv_mean','cv_std'],as_index=False).cx_count.max()
    winner = choices.sort_values(['cv_mean','cx_count','cv_std','reps','interaction'],ascending=[False,True,True,True,True]).iloc[0]
    spec = FeatureSpec(int(winner.reps),float(winner.scale),winner.interaction)
    C = float(winner.C)
    X,y,tr,te = load_dataset()
    scaler = make_scaler().fit(X[tr])
    points = scaler.transform(np.vstack([X[tr],X[te]]))
    all_states = states(points,spec)
    exact_train,exact_cross = kernel(all_states[:80]),kernel(all_states[80:],all_states[:80])
    pairs = evaluation_pairs(80,20)
    rows,trials = [],[]
    for name in ['A','B']:
        config = load_processor(root/f'processors/processor_{name}.json')[0]
        template,xp,yp,meta = compile_overlap(config,spec)
        error = validate_compilation(template,xp,yp,points[:80],exact_train)
        print(f'Training-selected cost-aware acquisition {name}: {spec}',flush=True)
        simulation = simulate_pairs(template,xp,yp,points,pairs,config,root,shots=shots,batch_size=batch_size)
        raw_train,raw_cross = assemble(simulation['sampled_values'],pairs)
        train,cross,_ = psd_extension(raw_train,raw_cross)
        _,stats,pred = fit_kernel(train,cross,y[tr],y[te],C)
        rows.append({'processor':name,**spec.to_dict(),'C':C,'cv_mean':winner.cv_mean,
                     'depth':meta['depth'],'cx_count':meta['cx_count'],'routing_swaps':meta['routing_swaps'],
                     'accuracy':stats['accuracy'],'kernel_rmse_vs_own_exact':np.sqrt(np.mean((raw_train-exact_train)**2)),
                     'compiled_error':error,'aer_seconds':simulation['timing']['aer_original_seconds'],
                     'cache_hit_batches':simulation['timing']['cache_hit_batches']})
        np.savez_compressed(root/f'data/processed/{name}_cost_aware_kernels.npz',
                            raw_train=raw_train,raw_cross=raw_cross,psd_train=train,psd_cross=cross,
                            histograms=simulation['histograms'],pair_indices=pairs,
                            observed_probabilities=simulation['observed_probabilities'],
                            pre_readout_probabilities=simulation['pre_readout_probabilities'],
                            exact_train=exact_train,exact_cross=exact_cross,predictions=pred,test_labels=y[te],C=C)
        for repeat in range(repeats):
            a,b = assemble(sample_values(simulation['expected_values'],shots,1000+repeat),pairs)
            _,s,_ = fit_kernel(a,b,y[tr],y[te],C,repair=True)
            trials.append({'processor':name,'repeat':repeat,'accuracy':s['accuracy']})
    export_table(root,'cost_aware_measurements',rows)
    export_table(root,'cost_aware_resampling',trials)
    export_table(root,'cost_aware_resampling_summary',pd.DataFrame(trials).groupby('processor').accuracy.agg(['mean','std','min','max']).reset_index())
    return {'selection_rule':'Maximum training CV mean; minimum worst-processor baseline CX; CV std; repetitions; interaction name',
            'feature_map':spec.to_dict(),'C':C,'scope':'Secondary model, conditional on primary training-selected scale; not a change to the frozen ring comparison',
            'measured_kernels':2}
