from Scripts.plot_output_eskf3d_offline_ablation import (
    add_two_dimensional_filter_logic,
    replace_filter_controls,
)
from Scripts.plot_static63_gt_macvo import HTML_TEMPLATE


def customized_template() -> str:
    template = replace_filter_controls(HTML_TEMPLATE)
    template = template.replace("{{", "{").replace("}}", "}")
    return add_two_dimensional_filter_logic(template)


def test_two_dropdowns_replace_individual_fusion_checkboxes():
    template = customized_template()
    assert "data-method-filter" in template
    assert "data-config-filter" in template
    assert template.count('value="all"') >= 2
    assert "data-show-fusion" not in template
    assert "data-show-gt" in template
    assert "data-show-macvo" in template
    assert "MACVO_Pure" in template


def test_optimizer_and_ekf_filters_form_an_intersection():
    template = customized_template()
    assert 'item.optimizer === state.methodFilter' in template
    assert 'item.config === state.configFilter' in template
    assert "return methodMatches && ekfMatches" in template
    assert 'data-legend-method="${item.optimizer}"' in template
