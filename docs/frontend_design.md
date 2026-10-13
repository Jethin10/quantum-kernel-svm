# Website design and evidence

The GitHub Pages explorer presents the saved PS-Q2 experiments. It serves ordinary HTML, CSS, JavaScript and an exported JSON dataset. No quantum jobs or SVM training run in the browser.

Visual reference research:

- [Dribbble science dashboards](https://dribbble.com/search/science-dashboard), including the Protera dashboard by Fulcrum Rocks: compact navigation, restrained chart framing and readable tables.
- [Pinterest analytics dashboard reference](https://www.pinterest.com/pin/907334656178708572/): found in public search, but Pinterest required sign-in and did not expose the pin artwork. No visual claims are based on that inaccessible image.

These references inform layout; their artwork is not copied into the project. Every plot and topology diagram is drawn from this repository's actual evidence. The design uses an ink sidebar, white plotting surfaces, blue/coral class colors and a large kernel matrix alongside the measured moons predictions. System fonts keep the site usable without third-party requests.

The interface lets readers switch processor/compiler conditions, inspect an individual measured training pair, explore all six connectivity graphs and compare paired noisy split results. Exact, seeded noisy and conditional shot-resampling outputs have distinct labels. All metrics identify their experiment and preserve unfavorable outcomes.

Run `python scripts/export_frontend.py` after the scientific pipeline, then `python scripts/build_site.py`. The build verifies the source digests embedded in the exported data. Preview with `python -m http.server 4173 --directory frontend/dist`. The Pages workflow publishes only the curated `frontend/dist` directory.
