{{ objname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

   {% block methods %}
   {% if methods %}
   .. rubric:: Methods
   .. autosummary::
      :toctree:

   {% for item in methods %}
        {%- if not item in inherited_members and not item.startswith('_') %}
            {{ name }}.{{ item }}
        {%- endif %}
   {%- endfor %}
   {%- endif %}
   {% endblock %}

   {% block attributes %}
   {% if attributes %}
   .. rubric:: Attributes
   .. autosummary::
   {% for item in attributes %}
    {%- if not item in inherited_members and not item == 'model_config' %}
      {{ name }}.{{ item }}
    {%- endif %}
   {%- endfor %}
   {%- endif %}
   {% endblock %}
