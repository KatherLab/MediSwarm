# Security Policy

MediSwarm is a swarm/federated learning framework used by the ODELIA and DECADE
medical-imaging consortia. Because it is deployed across hospital sites, we take
security reports seriously.

## Reporting a vulnerability

**Please do not open a public issue for security problems.**

Report privately through GitHub's **"Report a vulnerability"** button on the
[Security tab](https://github.com/KatherLab/MediSwarm/security/advisories/new)
(Security → Advisories → Report a vulnerability). This opens a private advisory
visible only to you and the maintainers.

Please include:

- a description of the issue and its impact,
- steps to reproduce (a minimal proof-of-concept if possible),
- affected version / commit and pipeline (ODELIA or STAMP), and
- any suggested remediation.

We aim to acknowledge a report within **5 working days** and to agree a
disclosure timeline with you. Please give us a reasonable window to release a
fix before any public disclosure.

## Scope

In scope:

- the framework code in this repository (`application/`, `scripts/`,
  `docker_config/`, provisioning, and the startup-kit tooling),
- the container images built from this repository, and
- the CI/CD workflows.

Out of scope:

- vulnerabilities in upstream dependencies (report those upstream — e.g.
  [NVFlare](https://github.com/NVIDIA/NVFlare)), though we are glad to be told,
- issues that require an already-compromised site host or valid VPN
  credentials, and
- site-specific data, credentials, or network configuration (those live outside
  this repository and are managed by each participating institution).

MediSwarm is designed so that **raw patient data never leaves a site** — only
model updates are exchanged. Reports demonstrating a way to exfiltrate local
data or training inputs through the framework are especially welcome.

## Supported versions

Security fixes target the latest released version on `main`. Older releases are
not maintained; please upgrade to the current release before reporting.
