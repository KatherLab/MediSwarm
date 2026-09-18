# Contributing to MediSwarm

Thanks for your interest in improving MediSwarm — a privacy-preserving swarm /
federated learning framework for medical imaging, built on
[NVFlare](https://github.com/NVIDIA/NVFlare) and developed with the ODELIA
consortium.

## Ways to contribute

- Report bugs and request features via [Issues](https://github.com/KatherLab/MediSwarm/issues).
- Improve documentation.
- Submit code via pull requests (see below).

For anything security-related, **do not open a public issue** — follow
[SECURITY.md](SECURITY.md).

## Development setup

MediSwarm runs entirely in Docker; you do not install the training stack on the
host. Start with the [Developer guide](assets/readme/README.developer.md), which
covers building the images, running a job, and the local test project.

Unit tests run on GitHub-hosted runners and are quick to run locally:

```bash
python -m pytest tests/unit_tests
```

The heavier GPU integration / swarm-validation tests run in CI on self-hosted
runners; you do not need a GPU to contribute most changes.

## Pull requests

1. **Fork** the repository and create a topic branch off `main`
   (e.g. `fix/short-description` or `feat/short-description`).
2. Keep each PR focused on a single change; write a clear description of **what**
   and **why**.
3. Make sure the checks pass:
   - **Unit Tests** (`ubuntu-latest`) run on every PR.
   - **PR Validation** (`pr-test`) runs a single-GPU swarm validation on the
     self-hosted runner; it is **skipped for docs-only PRs** (`**/*.md`,
     `docs/**`).
   - Secret scanning / push protection is enabled — never commit credentials,
     `.ovpn` files, or `deploy_sites*.conf` (these are git-ignored for a reason).
4. Update or add tests under `tests/` when you change behaviour.
5. Update relevant docs when you change user-facing behaviour.

### A note for external contributors

MediSwarm's CI uses **self-hosted runners on data-holding nodes**. For safety,
workflows on pull requests from forks require **maintainer approval** before they
run, and the workflow token is read-only by default. This is expected — a
maintainer will review and approve CI for your PR.

## Commit and review

- Reference the issue a PR addresses (e.g. `Fixes #123`).
- A maintainer will review; please be responsive to review comments.
- By contributing, you agree your contributions are licensed under the
  repository's [MIT License](LICENSE).
