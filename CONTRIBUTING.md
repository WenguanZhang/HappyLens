# Contributing to HappyLens

Thank you for your interest in improving HappyLens. Contributions to the core
optics framework, documentation, examples, and tests are welcome.

By participating in this project, you agree to follow the
[Code of Conduct](CODE_OF_CONDUCT.md).

## Before You Start

* Search the existing issues and pull requests before opening a new one.
* Use an issue to discuss substantial API changes, new optical models, or large
  workflow additions before investing in an implementation.
* Do not use a public issue to report conduct concerns. Follow the private
  reporting instructions in the Code of Conduct.

## Reporting a Bug

Include enough information for another person to reproduce the problem:

* the HappyLens commit or version;
* operating system, Python version, PyTorch version, CUDA version, GPU, and
  precision setting where relevant;
* the smallest script and configuration that reproduce the behavior;
* the complete error message or an explanation of the incorrect result; and
* any required input shapes, lens configuration, and expected behavior.

Remove private paths, credentials, proprietary prescriptions, datasets, and
other information you do not have permission to publish.

## Development Setup

HappyLens targets Python 3.12. Install the PyTorch build appropriate for your
CUDA environment first, then install the remaining dependencies:

```bash
git clone https://github.com/WenguanZhang/HappyLens.git
cd HappyLens
python -m pip install -r requirements.txt
```

See the [Quickstart](doc/quickstart.md) for device and precision conventions.

## Making Changes

Keep each pull request focused on one coherent change. Preserve the conventions
documented in [`doc/conventions.md`](doc/conventions.md), including tensor
shapes, units, configuration dimensions, dtype, and device behavior. New public
behavior should include clear docstrings and corresponding documentation.

The `test_*` directories contain research workflows, not a lightweight unit-test
suite. Many workflows require CUDA, external datasets, generated prescriptions,
or local path configuration. Do not assume that they can all be run in every
development environment.

When changing a workflow or configuration:

* run the smallest relevant example that exercises the change;
* avoid committing generated outputs, datasets, model checkpoints, caches, or
  machine-specific paths; and
* document any hardware, data, or configuration prerequisites needed to verify
  the result.

For documentation changes, install the documentation dependencies and build the
site locally:

```bash
python -m pip install -r doc/requirements.txt
python -m sphinx -W -b html doc doc/_build/html
```

## Third-Party Code and Data

Only submit code, data, weights, and other material that you have the right to
contribute under terms compatible with this repository. Do not copy or adapt an
implementation merely because its source is publicly visible.

If a contribution includes permitted third-party material:

* identify its source and license in the pull request;
* retain all required copyright and attribution notices;
* describe substantial modifications; and
* update [`NOTICE`](NOTICE),
  [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md), and
  [`third_party_licenses/`](third_party_licenses) when applicable.

Consult [Third-Party Network Implementations](doc/third_party.md) before changing
code under `nets/`. Implementations that HappyLens does not have permission to
redistribute must remain excluded; a pull request must not restore them without
documented authorization.

## Pull Requests

A pull request should:

* explain the problem and the chosen solution;
* link related issues or publications where useful;
* list the checks or workflows that were run and their environment;
* update documentation and examples when public behavior changes; and
* avoid unrelated formatting or generated-file changes.

Maintainers may request revisions or decline changes that are out of scope,
insufficiently documented, not reproducible, or incompatible with the project's
licensing obligations.

## Licensing Contributions

HappyLens does not currently require a Contributor License Agreement (CLA) or a
Developer Certificate of Origin (DCO) sign-off. By submitting a contribution,
you represent that you have the right to submit it and agree that it will be
licensed under the [Apache License 2.0](LICENSE), except for clearly identified
third-party material that retains a compatible upstream license.
