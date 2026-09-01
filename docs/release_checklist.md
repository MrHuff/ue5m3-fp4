# Public-release checklist

The clean repository must not be made public until all blocking items are
closed.

## Credentials and history

- Obtain security approval for the clean extraction and complete any required
  source-repository credential remediation out of band.
- Keep this repository on fresh history; never import the source Git objects.
- Run gitleaks or TruffleHog over the complete standalone history before push.
- Confirm that CI and examples run without credentials.

## Rights and attribution

- Confirm organizational approval for the Apache-2.0 code release.
- Review `NOTICE` against the exact TransformerEngine source revision.
- Confirm author list and paper metadata before adding `CITATION.cff`.
- Keep corporate logos and internal report templates out unless trademark and
  template publication permission is explicit.

## Technical release gates

- Build and install a wheel in a fresh environment.
- Run CPU unit tests and GPU numerical tests on the documented hardware.
- Freeze a supported Python/PyTorch matrix.
- Add a tiny end-to-end checkpoint reload example.
- Add optional TorchTitan/Nemotron-H integration without vendored submodules.
- Document that the probe-matched software path is not native UE5M3 hardware.
- Publish only sanitized metric tables or public artifact identifiers.
