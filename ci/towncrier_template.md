{#-
  towncrier template for docs/releases.md (see [tool.towncrier] in pyproject.toml).
  towncrier writes the "## <version>" heading itself. This renders a
  "### <type name>" block for each fragment type that has fragments, and one
  "* " bullet per fragment: the PR link goes at the end of the fragment's lead
  text, before any "* " sub-bullets, as in the releases already in the file.
  A fragment written as a bullet ("- Fixed X") loses its marker, so it does not
  render as a list nested in the bullet.
-#}
{% for section, _ in sections.items() %}
{% if sections[section] %}
{% for category, val in definitions.items() if category in sections[section] %}

### {{ definitions[category]["name"] }}

{% for text, values in sections[section][category].items() %}
{% set entry = namespace(lead=[], subs=[]) %}
{% for line in text.split("\n") %}
{% if loop.first %}
{% set entry.lead = [line[2:] if line[:2] in ("* ", "- ") else line] %}
{% elif entry.subs or line.lstrip()[:2] in ("* ", "- ") %}
{% set entry.subs = entry.subs + [line] %}
{% else %}
{% set entry.lead = entry.lead + [line] %}
{% endif %}
{% endfor %}
* {{ entry.lead | join("\n") }}{% if values %} ({{ values | join(", ") }}){% endif %}

{% if entry.subs %}
{{ entry.subs | join("\n") }}
{% endif %}
{% endfor %}
{% endfor %}
{% else %}

No significant changes.
{% endif %}
{% endfor %}
{{ "\n" }}
