{{ objname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

   {% block methods %}
   {% if methods %}
   .. rubric:: Methods
   .. autosummary::
      :toctree:

   {% for item in all_methods %}
      {% if item not in [
          'copy', 'dict', 'json', 'schema', 'schema_json',
          'validate', 'from_orm', 'from_json', 'construct',
          'update_forward_refs', 'parse_obj', 'parse_file',
          'model_construct', 'model_parametrized_name', 'parse_raw',
          'model_dump', 'model_dump_json', 'model_validate',
          'model_validate_json', 'model_validate_strings',
          'model_post_init', 'model_copy', 'model_json_schema',
          'model_rebuild', 'model_fields_set', 'model_computed_fields'
      ] and not item.startswith('_') %}
         {{ name }}.{{ item }}
      {% endif %}
   {% endfor %}
   {% endif %}
   {% endblock %}

   {% block attributes %}
   {% if attributes %}
   .. rubric:: Attributes
   .. autosummary::
   {% for item in attributes %}
      {% if item not in [
          'model_computed_fields', 'model_fields', 'model_config',
          'model_extra', 'model_fields_set', '__private_attributes__'
      ] and not item.startswith('_') %}
         {{ name }}.{{ item }}
      {% endif %}
   {% endfor %}
   {% endif %}
   {% endblock %}
