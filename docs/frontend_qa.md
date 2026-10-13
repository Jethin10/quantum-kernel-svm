# Explorer verification

Checked the static site in the browser before publication:

- Switched processor A/B and baseline/fused/native-parity compilers; metric panels, measured matrix, graph placement and saved predictions updated together.
- Inspected training pair row 79 / column 78 using keyboard-accessible numeric controls. Native-parity A displayed 2,052 zero outcomes out of 4,096, matching the exported evidence. Fixed numeric controls to update on input as well as change.
- Selected full K4 with SABRE: chain A displayed 86 CX / depth 118, square B 68 CX / depth 91. All six topology diagrams remained visible.
- Switched the five-split robustness chart to raw kernel RMSE and expanded its uncertainty explanation.
- Checked the compact mobile layout: navigation, stacked plots and controls remained usable without horizontal page overflow.
- Temporarily removed the served data file to simulate a failed fetch. The site displayed a recovery message. Restored the file and pressed Retry; the experiment loaded successfully.

`node --check frontend/app.js` passes. Scientific integration tests verify browser metrics and counts against the source artifacts; the static build verifies every embedded evidence digest. No frontend backend, cloud credentials or live quantum execution is required.
