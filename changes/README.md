# Release-note fragments

Every pull request with a user-visible change adds one file here, named
`<PR number>.<type>.md`, holding its release note: one line for the release
reader, optionally followed by `* ` sub-bullets. At release time
`towncrier build` turns the fragments into a section of
[`docs/releases.md`](../docs/releases.md) and deletes them.

| Type          | Section in the release notes |
|---------------|------------------------------|
| `breaking`    | Breaking Changes             |
| `deprecation` | Deprecations                 |
| `feature`     | New Features                 |
| `bugfix`      | Bug Fixes                    |
| `improvement` | Improvements                 |
| `misc`        | Miscellaneous                |

The [contributor guide](../docs/contributing.rst) has the details and examples.
