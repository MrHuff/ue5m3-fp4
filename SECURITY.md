# Security policy

## Release status

This repository is an alpha release candidate and does not yet have a public
security-support commitment. It must not be made public until a tested private
reporting channel is configured for the destination repository.

## Reporting a vulnerability

Once the public GitHub repository exists, use its **Security → Advisories →
Report a vulnerability** workflow. The release owner must enable and test
GitHub private vulnerability reporting before publication.

Until that channel exists, report issues through the organization's approved
private security process or email the release owner at
`robert.stats.hu@gmail.com`. Do not file a public issue containing credentials,
exploit details, private checkpoint locations, internal bucket names, or
cluster manifests. Revoke any exposed credential immediately.

The reference implementation does not require network credentials.
