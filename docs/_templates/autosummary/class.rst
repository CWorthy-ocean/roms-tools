{{ objname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

   {% block methods %}
   {% if methods %}
   .. rubric:: Methods
   .. autosummary::
      :toctree:
   {% for item in all_methods %}
      {%- if not item.startswith('_') or item in ['__call__',
                                                  ] %}
        {%- if objname == 'ChildGrid' and item != 'from_file' %}  {# Exclude 'from_file' only in ChildGrid #}
        {{ name }}.{{ item }}
       . {% endif %}
      {% endif %}
   {% endfor %}
   {% endif %}
   {% endblock %}

   {% block attributes %}
   {% if attributes %}
   .. rubric:: Attributes
   .. autosummary::
   {% for item in attributes %}
      {{ name }}.{{ item }}
   {% endfor %}
   {% endif %}
   {% endblock %}
