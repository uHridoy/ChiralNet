# ChiralNet

**Symmetry-reasoning AI agents for detecting chiral charge order in STM/STS data**

ChiralNet determines whether a scanning tunneling microscopy (STM) topograph, optionally accompanied by a d*I*/d*V* map, shows a charge-density wave (CDW), and whether that CDW is chiral. It combines a deterministic symmetry-analysis layer with three specialist vision-language agents and requires no task-specific training.

This repository contains the code and data accompanying:

> H. Hridoy, T. Chowdhury, P. Chen, C.-T. Lien, M. S. Hossain. *Harnessing symmetry-reasoning AI agents for chiral charge-order discovery in quantum materials.* (2026). [Journal reference and DOI to be added.]

---

## Why ChiralNet

In STM, chirality is usually inferred from an ordering of the intensities of the three CDW Fourier peaks. That ordering can also arise from achiral anisotropy, finite domains, and instrumental artifacts such as tip shape, drift, and strain. ChiralNet therefore does not treat an intensity ordering as evidence of handedness.

- **Measurement is separated from judgment.** Every quantity on which a decision depends is computed deterministically before any agent runs. The agents interpret these measurements but never produce or alter them.
- **Handedness comes from symmetry tests.** A handedness is assigned only by tests that do not rely on the intensity ordering: a lattice-referenced registry-phase test of mirror symmetry and a structural test of superlattice rotation.
- **Uncertainty is reported, not hidden.** When no robust test can decide, ChiralNet withholds a handedness and says why, rather than guessing.

## How it works

<!-- Add the architecture schematic (Fig. 1 of the paper), e.g. docs/fig1.png -->
<!-- ![ChiralNet architecture](docs/fig1.png) -->

1. **Deterministic layer.** Loads the data, computes the Fourier transforms, detects Bragg and superlattice peaks, and identifies the lattice and candidate superlattice cells. It then runs the chirality-related measurements:
   - **Intensity anisotropy.** The CDW intensity anisotropy α is screened against a threshold calibrated by forward modelling (α<sub>crit</sub> = 0.310).
   - **Hierarchy diagnostics.** Intensity-hierarchy diagnostics characterize the peak ordering but never assign a handedness.
   - **Registry-phase test.** Tests mirror symmetry of the CDW relative to the atomic lattice on all six mirror axes.
   - **Structural test.** Tests whether the superlattice is rotated against the lattice.
2. **Specialist agents, run in sequence.**
   - **Moiré agent:** decides between moiré interference and charge order. It is the only agent with tools, which return the measured periodicities, the lattice identification, and the peak table.
   - **Topographic agent:** decides from the topograph's Fourier transform whether a CDW is present.
   - **Spectroscopy agent:** judges the organization of the CDW in the d*I*/d*V* transform. It runs only when a d*I*/d*V* map is supplied and never infers handedness from peak amplitudes.
3. **Label reconciliation and judge.** The spectroscopy label is checked against the registry-phase verdict. A rule-based judge then assigns the final label: `Non-CDW`, `CDW`, `Chance of Chiral CDW`, `Chiral CDW`, or `Inconclusive`.
4. **Chiral-CDW metrology.** A deterministic 0–100 score summarizes the evidence behind each result. It combines an anisotropy criterion (40 points), a handedness criterion (20 points), and a judge criterion (40 points). A measurement-only score that excludes the agents is also reported.

## Repository structure

| Path | Contents |
|---|---|
| `ui.py` | Streamlit web application (entry point) |
| `core_pipeline.py` | LangGraph workflow: deterministic nodes, the three agents, label reconciliation, the judge, and the metrology step |
| `stm_data_io.py` | Data loaders for images and native Nanonis files, with channel and bias selection |
| `fft_core.py` | Preprocessing, windowed Fourier transforms, and the figures shown to the agents |
| `fft_peak_analysis.py` | Peak detection, hexagonal-family grouping, Bragg identification, and superlattice models |
| `didv_anisotropy.py` | CDW intensity anisotropy α, integration apertures, and the calibrated screen |
| `didv_alpha_crit.py` | Calibrated thresholds and the operating-point selection rule |
| `didv_handedness.py` | Intensity-hierarchy diagnostics |
| `didv_phase_chirality.py` | Registry-phase test of mirror symmetry |
| `chiral_metrology.py` | Chiral-CDW metrology score |
| `Benchmark Data/` | Data supporting the benchmark reported in the paper |

## Installation

Python 3.10 or later is recommended.

```bash
git clone https://github.com/uHridoy/ChiralNet.git
cd ChiralNet
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install numpy scipy matplotlib pillow streamlit langgraph anthropic
```

The agents call the Anthropic API, so you need an Anthropic API key. See the [Claude API documentation](https://docs.claude.com/en/api/overview) for how to obtain one. API usage is billed to your account.

## Quick start

```bash
streamlit run ui.py
```

Then, in the browser:

1. **API key:** paste your Anthropic API key.
2. **Upload data:** upload a topograph (required) and, optionally, a d*I*/d*V* map.
3. **Calibrate:**
   - For native Nanonis files, choose the channel and scan direction and, for `.3ds` grids, the bias of the d*I*/d*V* slice. The scan size is read from the file header.
   - For image files, enter the topograph width and height in nm to calibrate the Fourier transform. Without them, results are reported in cycles per pixel.
4. **Superlattice:** select the CDW superlattice. Choose **Auto-detect**, or pin a known cell (2×2, 3×3, √13×√13, √3×√3, 4×4, or a custom |q<sub>CDW</sub>|/|q<sub>Bragg</sub>| ratio with rotation). Pinning is recommended whenever the structure is known; auto-detection can choose the wrong shell when the Bragg peaks are weak.
5. **Run:** click **Run ChiralNet**.

The results page shows the final label, the chiral-CDW metrology score, the anisotropy and handedness measurements, the Fourier transform with the peak apertures used, the individual agent reports, and the numerical FFT analysis.

## Supported inputs

| Type | Formats | Notes |
|---|---|---|
| Native instrument files | Nanonis `.sxm` (scan), `.3ds` (grid spectroscopy) | Floating-point data in physical units; calibrated automatically |
| Rendered images | `.png`, `.jpg`/`.jpeg`, `.tif`/`.tiff` | Converted to a scalar field by inverting the colormap |

Native files are preferred. Rendered images are quantized to 8 bits, can be saturated at the colour-bar limits, and may have been resampled. They therefore lie outside the domain in which the anisotropy screen was calibrated, and every result records whether its input was native or rendered.

For the registry-phase test, constant-height d*I*/d*V* maps, or constant-current maps normalized as (d*I*/d*V*)/(*I*/*V*), are recommended.

## Model and reproducibility

All three agents run on Claude Fable 5.1 (`claude-fable-5-1`) at temperature 1.0 with medium reasoning effort. The model assignment is set in `AGENT_MODELS` in `core_pipeline.py`.

The deterministic layer returns identical measurements in every run. Agent labels can vary between runs: in the paper's benchmark, 85.4% of maps returned the same label in all five repeated runs. For borderline maps, run ChiralNet several times and report the distribution of labels.

## Interpreting the results

- **`Chance of Chiral CDW` means undetermined handedness.** It denotes a resolved three-*Q* CDW whose handedness has not been established. It is not evidence of chirality.
- **The metrology score is not a probability.** Its weights and band edges (above 75: chiral CDW; 50–75: chance of chiral CDW; below 50: not chiral CDW) are a reporting convention. Read the score together with the channel on which any handedness assignment rests.
- **Anisotropy alone is not chirality.** An α above α<sub>crit</sub> shows anisotropy beyond the modelled artifacts. Strain, unequal domain populations, and bias-dependent matrix elements can also produce it in an achiral state.
- **The results describe the measured surface.** The tests measure the registry of one terminating layer in one field of view at one bias. Chirality that resides in interlayer stacking is beyond what a single surface map can establish.

## Benchmark

On 137 STM/STS maps (125 reproduced from 93 publications and 12 recorded in our laboratory), ChiralNet classified 94.9% correctly. Seven multimodal models queried zero-shot reached 36.5–55.5%; this includes the model that runs ChiralNet's own agents, which reached 36.5%. Per-map sources, reference labels, and outputs are listed in Extended Data Table 1 of the paper. Maps reproduced from published studies are available from the cited sources.

## Citation

If you use ChiralNet, please cite:

```bibtex
@article{hridoy2026chiralnet,
  title   = {Harnessing symmetry-reasoning AI agents for chiral charge-order discovery in quantum materials},
  author  = {Hridoy, Hossain and Chowdhury, Tahiya and Chen, Pochang and Lien, Chun-Tung and Hossain, Md Shafayat},
  journal = {[to be added]},
  year    = {2026},
  doi     = {[to be added]}
}
```

## License

[Add a license, e.g. MIT or BSD-3-Clause, and include a `LICENSE` file.]

## Contact

Correspondence: Md Shafayat Hossain (mshossain@g.ucla.edu). For bugs and feature requests, please open a GitHub issue.
