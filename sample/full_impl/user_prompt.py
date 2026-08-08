from jinja2 import Template

USER_PROMPT_TEMPLATE = """
Context:
{{ context_block }}

Previous conversation:
{% for role, msg in history %}
- {{ role }}: {{ msg }}
{% endfor %}

User Question:
{{ query }}
"""


def render_user_prompt(context_block: str, query: str, history: list) -> str:
    template = Template(USER_PROMPT_TEMPLATE)
    return template.render(
        context_block=context_block, query=query, history=history or []
    )
