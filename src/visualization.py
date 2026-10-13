"""Publication-ready plots; decision boundaries use the exact reference only."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from .problem import kernel, states


BLUE, ORANGE, INK = "#2864ad", "#df782c", "#243247"


def configure():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.labelcolor": INK, "text.color": INK,
        "figure.facecolor": "white", "savefig.facecolor": "white",
        "axes.titleweight": "bold", "figure.dpi": 120, "savefig.dpi": 240,
    })


def save(fig, path):
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def architectures(configs, layouts, path):
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
    for ax, config in zip(axs, configs):
        graph = nx.Graph()
        graph.add_nodes_from(range(config["num_qubits"]))
        graph.add_edges_from(config["edges"])
        if config["name"] == "A":
            positions = {i: (i, 0) for i in graph.nodes}
        else:
            positions = {i: (i % 3, -(i // 3)) for i in graph.nodes}
        used = layouts[config["name"]]
        nx.draw_networkx(graph, positions, ax=ax, node_size=620,
                         node_color=[BLUE if i in used else "#d8e0e9" for i in graph.nodes],
                         with_labels=False, width=2, edge_color="#93a1b2")
        nx.draw_networkx_labels(graph,positions,ax=ax,
                                labels={i:i for i in graph.nodes if i in used},font_color="white")
        nx.draw_networkx_labels(graph,positions,ax=ax,
                                labels={i:i for i in graph.nodes if i not in used},font_color=INK)
        ax.set_title(f"Processor {config['name']} · {config['num_qubits']} physical qubits")
        ax.text(0.5, -0.08, f"Logical 0–3 placement: {used}", transform=ax.transAxes,
                ha="center", fontsize=9)
        ax.axis("off")
    fig.suptitle("Same four logical qubits, different interaction graphs", fontsize=14)
    save(fig, path)


def feature_circuit(spec,path):
    from matplotlib.patches import FancyBboxPatch
    fig,axs=plt.subplots(2,1,figsize=(11,4.5),height_ratios=[3,1],constrained_layout=True)
    ax=axs[0]
    for q in range(4):
        yy=3-q
        ax.plot([.8,7.2],[yy,yy],color="#93a1b2",lw=1.1,zorder=0)
        coord="a" if q%2==0 else "b"
        ax.text(.7,yy,f"q{q}: θ{q} = s·{coord}",ha="right",va="center",fontsize=10)
        for xx,label in [(1.3,"H"),(2.4,f"RZ(2θ{q})")]:
            box=FancyBboxPatch((xx-.4,yy-.22),.8,.44,boxstyle="round,pad=.02",
                               facecolor="#e0ebf8",edgecolor=BLUE)
            ax.add_patch(box)
            ax.text(xx,yy,label,ha="center",va="center",fontsize=9)
    for xx,(i,j) in zip([3.5,4.5,5.5,6.5],[(0,1),(1,2),(2,3),(3,0)]):
        ax.plot([xx,xx],[3-i,3-j],color=ORANGE,lw=2)
        ax.scatter([xx,xx],[3-i,3-j],s=60,color=ORANGE,zorder=3)
        ax.text(xx,3.55,f"RZZ\n2θ{i}θ{j}",ha="center",va="center",fontsize=9,color=INK)
    ax.set_xlim(-.3,7.5)
    ax.set_ylim(-.6,4)
    ax.set_title(f"Four-qubit feature map · repeat this layer {spec.reps} times · s={spec.scale}")
    ax.axis("off")
    ax=axs[1]
    for xx,label,color in [(1.4,"Prepare U(x)",BLUE),(4.2,"Apply U(x′) inverse",ORANGE),(7.2,"Measure four qubits",INK)]:
        box=FancyBboxPatch((xx-1.1,.15),2.2,.55,boxstyle="round,pad=.03",
                           facecolor="white",edgecolor=color,lw=1.5)
        ax.add_patch(box)
        ax.text(xx,.425,label,ha="center",va="center",fontsize=10)
    for start,end in [(2.55,3.05),(5.35,6.05)]:
        ax.annotate("",xy=(end,.425),xytext=(start,.425),arrowprops={"arrowstyle":"->","color":INK})
    ax.text(4.2,-.15,"K(x, x′) = frequency of 0000 · 4096 shots per pair",ha="center",fontsize=11)
    ax.set_xlim(0,8.5)
    ax.set_ylim(-.35,.85)
    ax.axis("off")
    save(fig,path)


def boundaries(X, y, train_idx, test_idx, scaler, spec, qmodel, train_states, baselines, path):
    gx, gy = np.meshgrid(np.linspace(X[:, 0].min()-.35, X[:, 0].max()+.35, 70),
                         np.linspace(X[:, 1].min()-.35, X[:, 1].max()+.35, 70))
    grid = np.c_[gx.ravel(), gy.ravel()]
    decisions = [baselines["linear"].decision_function(grid),
                 baselines["rbf"].decision_function(grid),
                 qmodel.decision_function(kernel(states(scaler.transform(grid), spec), train_states))]
    fig, axs = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for ax, scores, title in zip(axs, decisions, ["Classical linear SVM", "Classical RBF SVM", "Exact four-qubit QSVM"]):
        ax.contourf(gx, gy, scores.reshape(gx.shape) > 0, levels=[-.5,.5,1.5],
                    colors=["#e0ebf8", "#fae8d8"], alpha=.9)
        ax.contour(gx, gy, scores.reshape(gx.shape), levels=[0], colors=INK, linewidths=1.2)
        for label, color in [(0, BLUE), (1, ORANGE)]:
            tr = train_idx[y[train_idx] == label]
            te = test_idx[y[test_idx] == label]
            ax.scatter(X[tr,0], X[tr,1], s=21, c=color, alpha=.8)
            ax.scatter(X[te,0], X[te,1], s=55, c=color, marker="s", edgecolor=INK,
                       label=f"Test class {label}")
        ax.set(title=title, xlabel="Coordinate 1", ylabel="Coordinate 2")
    axs[-1].legend(loc="upper right", fontsize=8)
    fig.suptitle("Training-selected models · squares mark the untouched test set", fontsize=14)
    save(fig, path)


def kernel_comparison(ideal, raw_kernels, path):
    fig, axs = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    matrices = [ideal, raw_kernels["A"], raw_kernels["B"]]
    for ax, matrix, title in zip(axs[0], matrices, ["Exact reference", "A · noisy, 4096 shots", "B · noisy, 4096 shots"]):
        im = ax.imshow(matrix, vmin=0, vmax=1, cmap="viridis", interpolation="nearest")
        ax.set(title=title, xlabel="Training sample", ylabel="Training sample")
    fig.colorbar(im, ax=list(axs[0]), shrink=.8, label="Fidelity kernel estimate")
    values = np.linalg.eigvalsh(ideal)
    axs[1,0].plot(values, color=INK, label="Exact")
    for name, color, ax in [("A", ORANGE, axs[1,1]), ("B", BLUE, axs[1,2])]:
        axs[1,0].plot(np.linalg.eigvalsh(raw_kernels[name]), color=color, label=name)
        error = raw_kernels[name] - ideal
        im2 = ax.imshow(error, cmap="RdBu_r", vmin=-.25, vmax=.25, interpolation="nearest")
        ax.set(title=f"{name} minus exact", xlabel="Training sample", ylabel="Training sample")
    axs[1,0].axhline(0, color="grey", lw=.7)
    axs[1,0].set_yscale("symlog", linthresh=.01)
    axs[1,0].set(title="Raw eigenvalues · symmetric log scale", xlabel="Eigenvalue rank", ylabel="Eigenvalue")
    axs[1,0].legend()
    fig.colorbar(im2, ax=list(axs[1,1:]), shrink=.8, label="Kernel error")
    save(fig, path)


def cost_comparison(resource_rows, path):
    primary = [row for row in resource_rows if row["experiment"] == "primary"]
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.5), constrained_layout=True)
    for ax, column, title in zip(axs, ["depth", "cx_count", "routing_swaps"],
                                ["Compiled depth", "CX gates per pair", "Inserted routing SWAPs"]):
        values = [r[column] for r in primary]
        bars = ax.bar([r["processor"] for r in primary], values, color=[ORANGE, BLUE], width=.55)
        ax.bar_label(bars, padding=3)
        ax.set(title=title, ylabel="Count", ylim=(0, max(values + [1])*1.2))
    fig.suptitle("Identical feature map · optimization level 1 · seed 42", fontsize=13)
    save(fig, path)


def shot_sensitivity(table, path):
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    for name, color in [("A", ORANGE), ("B", BLUE)]:
        subset = table[table.processor == name]
        for ax, metric, title in zip(axs, ["accuracy", "kernel_rmse"],
                                    ["Classification accuracy", "Kernel RMSE versus exact"]):
            groups = subset.groupby("shots")[metric]
            means, std = groups.mean(), groups.std()
            ax.errorbar(means.index, means.values, yerr=std.values, marker="o",
                        color=color, capsize=4, label=name)
            ax.set_xscale("log", base=2)
            ax.set(title=title, xlabel="Shots per pair", ylabel=metric.replace("_", " "))
            ax.legend()
    fig.suptitle("20 Monte Carlo shot realizations per setting · bars show one standard deviation",
                 fontsize=11)
    save(fig, path)


def routing_controls(rows,path):
    fig,axs=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    topologies=["linear","ring","full"]
    names=["A","B","B_reference","A_cycle_control"]
    labels=["A: 5q chain","B: 9q lattice","B: guide reference","Control: 5q cycle"]
    colors=[ORANGE,BLUE,"#7d69a7","#289080"]
    for ax,column,title in zip(axs,["cx_count","depth"],["CX count","Compiled depth"]):
        for k,(name,label,color) in enumerate(zip(names,labels,colors)):
            selected=[next(r for r in rows if r["experiment"]=="routing_control"
                           and r["strategy"]=="sabre" and r["processor"]==name
                           and r["feature_map"]["topology"]==topology) for topology in topologies]
            ax.bar(np.arange(3)+(k-1.5)*.18,[r[column] for r in selected],
                   width=.18,color=color,label=label)
        ax.set_xticks(range(3),["Linear","Ring","All-to-all"])
        ax.set(title=title,xlabel="Logical interaction pattern",ylabel="Per-pair count")
    axs[1].legend(fontsize=8,loc="upper left")
    fig.suptitle("Connectivity controls · same repetitions, compiler, seed, and layout rule",fontsize=12)
    save(fig,path)


def accuracy_plot(rows, path):
    selected = [r for r in rows if r["model"] in {
        "Classical LINEAR", "Classical RBF", "Exact QSVM", "Ideal shot QSVM",
        "A noisy PSD", "B noisy PSD", "A readout-mitigated PSD", "B readout-mitigated PSD"}]
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    values = [100*r["accuracy"] for r in selected]
    labels = [r["model"] for r in selected]
    bars = ax.barh(labels[::-1], values[::-1], color=[
        ORANGE if r["model"].startswith("A ") else BLUE for r in selected[::-1]
    ])
    ax.bar_label(bars, fmt="%.0f%%", padding=4)
    ax.set(xlim=(0,110), xlabel="Held-out accuracy (%)", title="20 held-out samples · one error changes accuracy by 5 points")
    save(fig, path)
