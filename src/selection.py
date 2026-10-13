"""Training-only model selection with a fresh scaler in every validation fold."""
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from .metrics import classification
from .problem import FeatureSpec, kernel, make_scaler, states


C_VALUES = [0.1, 1.0, 10.0, 100.0]


def select_quantum(Xtrain, ytrain):
    folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(Xtrain, ytrain))
    rows = []
    for reps in [1, 2]:
        for scale in [0.25, 0.5, 1.0, 1.5]:
            spec = FeatureSpec(reps=reps, scale=scale, topology="ring")
            scores = {C: [] for C in C_VALUES}
            for tr, va in folds:
                scaler = make_scaler().fit(Xtrain[tr])
                all_states = states(scaler.transform(Xtrain), spec)
                train = kernel(all_states[tr])
                cross = kernel(all_states[va], all_states[tr])
                for C in C_VALUES:
                    model = SVC(kernel="precomputed", C=C).fit(train, ytrain[tr])
                    scores[C].append(float(model.score(cross, ytrain[va])))
            for C, values in scores.items():
                rows.append({**spec.to_dict(), "C": C, "cv_mean": np.mean(values),
                             "cv_std": np.std(values, ddof=1), "fold_scores": values})
            print(f"  Training-only CV: reps={reps}, scale={scale}", flush=True)
    # Deterministic tie-break: lower variation, shallower circuit, lower scale/C.
    table = pd.DataFrame(rows).sort_values(
        ["cv_mean", "cv_std", "reps", "scale", "C"],
        ascending=[False, True, True, True, True], kind="stable",
    ).reset_index(drop=True)
    winner = table.iloc[0]
    return FeatureSpec(int(winner.reps), float(winner.scale), "ring"), float(winner.C), table


def classical_baselines(Xtrain, ytrain, Xtest, ytest):
    rows, models = [], {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for name in ["linear", "rbf"]:
        pipeline = Pipeline([("scale", make_scaler()), ("svc", SVC(kernel=name))])
        grid = {"svc__C": C_VALUES}
        if name == "rbf":
            grid["svc__gamma"] = ["scale", 0.1, 1.0, 10.0]
        t0 = time.perf_counter()
        search = GridSearchCV(pipeline, grid, cv=cv, n_jobs=1).fit(Xtrain, ytrain)
        prediction = search.predict(Xtest)
        rows.append({"model": f"Classical {name.upper()}", **classification(ytest, prediction),
                     "cv_mean": float(search.best_score_),
                     "parameters": search.best_params_,
                     "selection_and_fit_seconds": time.perf_counter()-t0})
        models[name] = search.best_estimator_
    return rows, models
