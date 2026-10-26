# Reporting security problems

Do not attach credentials, private CSI recordings or participant information
to public issues. For a security problem, use GitHub's private vulnerability
reporting for this repository when enabled; otherwise contact the maintainer
through their public profile before sharing sensitive details.

The replay service binds only to IPv4 loopback, serves an exact static-file
allowlist, snapshots the selected replay and has no server upload or arbitrary
file-reading endpoint. Browser replay files are parsed as data, not executed.
The acquisition tool accepts only bounded public ZIP ranges and verifies paths,
size, archive checksums and content hashes.

These boundaries do not make an untrusted model or private recording safe to
redistribute. Use only trusted checkpoints and permissioned data. Model loading
uses PyTorch's weights-only loader; no arbitrary Python checkpoint code is
required.
