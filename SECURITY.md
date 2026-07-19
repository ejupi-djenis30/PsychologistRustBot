# Safety and privacy model

ELIZA Lab is an educational dialogue-system demonstration. It is not a psychologist, therapist,
medical device, crisis service, or source of diagnosis.

## Data handling

- The Rust engine stores only a numeric turn counter in memory.
- The command-line application does not write transcripts or analytics.
- The GitHub Pages demo makes no network requests after its static assets load.
- There are no accounts, cookies, databases, API keys, or remote models.

## Safety boundary

The engine detects a small set of urgent-safety phrases only to stop the simulation and direct the
visitor toward immediate human help. It does not assess risk and must not be used as a safety tool.

If someone may be in immediate danger, contact local emergency services or a trusted person.

## Reporting

Please use GitHub's private vulnerability reporting feature when available. Do not include real
conversation transcripts, credentials, or personal data in a report.
