# Website design and evidence

The GitHub Pages lab uses HTML, CSS and JavaScript. Its live view computes exact 16-amplitude states for the four-qubit feature map, fidelity overlaps, seeded ideal shot samples, and inference from the exported, training-fitted exact SVM. It never launches hardware jobs or retrains a classifier in the browser. Saved Aer measurements remain separate and labeled.

The visual direction uses compact tabs, navy framing, white plot surfaces and blue/orange class colors. The main interaction is the draggable moons plot. Explanations sit beside their controls or inside expandable details. System fonts and local scientific figures avoid third-party asset dependencies.

Reference research:

- [Dribbble science dashboards](https://dribbble.com/search/science-dashboard), including Protera by Fulcrum Rocks, informed restrained chart framing and navigation.
- [Pinterest dashboard reference](https://www.pinterest.com/pin/907334656178708572/) was found in search but required sign-in. Its artwork was unavailable and is not used.

Every plotted number comes from this repository. No reference artwork is copied. Six views cover live encoding, saved measured kernels, all six architectures and 36 compiled controls, five paired noisy splits, ten chart-driven study questions, and all 13 original scientific figures.

The live feature map uses H, RZ and ZZ operations with Qiskit's qubit ordering. Changing its settings hides the frozen classifier's decision regions. `node scripts/test_browser_engine.cjs` checks 27 state/fidelity cases against Qiskit and all 20 held-out SVM predictions against sklearn. Browser sampling is ideal Bernoulli sampling, distinct from saved noisy channels.

Run `python scripts/export_frontend.py`, then `python scripts/build_site.py`. The build verifies scientific source digests and copies only public assets into `frontend/dist`. Preview with `python -m http.server 4174 --directory frontend/dist`. GitHub Pages checks the browser engine before publishing this directory.
