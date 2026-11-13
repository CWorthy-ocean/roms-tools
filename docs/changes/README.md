# Release-note fragments

Every pull request with a user-visible change adds a file here holding its
release note, named `+<name>.<type>.md` (any short `<name>`, for example
`+bottom-fill.bugfix.md`) and committed with the code. The note is one line for
the release reader, optionally followed by `* ` sub-bullets. At release time
each `+<name>` file is renamed after the pull request that added it, then
`towncrier build` turns the fragments into a section of
[`docs/releases.md`](../releases.md) and deletes them.

| Type          | Section in the release notes |
|---------------|------------------------------|
| `breaking`    | Breaking Changes             |
| `output`      | Changes to Output            |
| `deprecation` | Deprecations                 |
| `feature`     | New Features                 |
| `bugfix`      | Bug Fixes                    |
| `improvement` | Improvements                 |
| `misc`        | Miscellaneous                |

The [contributor guide](../contributing.rst) has the details and examples.
