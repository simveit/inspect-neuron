# Citation and attribution

## The essay

This project follows the blind-test idea in:

> Nick Cammarata. *Neural Networks Are Already Clean*. Goodfire.
> <https://already-clean.goodfire.com/> (retrieved 2026-10-27)

The essay is **not** included in this repository: no article text, HTML snapshot or figure
images are redistributed here. Read and cite the original at the link above. This repository
implements an independent pipeline inspired by its protocol; it is not an official
implementation and is not endorsed by the author or Goodfire.

If you write about results produced with this repository, cite the essay and link this
repository. A suggested sentence:

> This experiment uses the blind neuron-explanation protocol inspired by Nick Cammarata's
> *Neural Networks Are Already Clean* (Goodfire, <https://already-clean.goodfire.com/>,
> retrieved 2026-10-27), implemented in inspect-neuron
> (<https://github.com/simveit/inspect-neuron>).

A machine-readable citation is in [CITATION.cff](../CITATION.cff).

## Third-party artifacts

The code is a client of public models and datasets. Their terms apply to their weights, data
and outputs:

| Artifact | Used for | Where |
|---|---|---|
| `Qwen/Qwen3-0.6B` | the default example model | <https://huggingface.co/Qwen/Qwen3-0.6B> |
| `HuggingFaceFW/fineweb` (`sample-10BT`) | the default streaming corpus | <https://huggingface.co/datasets/HuggingFaceFW/fineweb> |
| `PKU-Alignment/BeaverTails` | an example intent/benign dataset (not bundled; CC BY-NC 4.0, non-commercial use only) | <https://huggingface.co/datasets/PKU-Alignment/BeaverTails> |

`examples/tiny-corpus.jsonl` is synthetic text written for this repository and contains no
third-party corpus material. The HTML templates are part of this repository; the CSS uses local
system font stacks, and no font files or CDN links are bundled.

## What is not included

- The essay text, its figures, or any HTML mirror of it.
- Datasets, model weights, tokenizers or generated stores. They are streamed or downloaded by
  the user and stored outside the tracked tree.
- Experiment reports, runs, stores, neuron pages or logs. They are gitignored.

## Contact

Questions, bug reports and corrections: open an issue at
<https://github.com/simveit/inspect-neuron/issues>.

## License

The code and documentation in this repository are released under the MIT License; see
[LICENSE](../LICENSE) for the full text. Copyright © 2026 Simon Veitner.

The third-party artifacts above keep their own licenses, and nothing here relicenses them.
