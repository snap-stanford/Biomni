# Source Generated with Decompyle++
# File: integrated_ui_variants.cpython-312.pyc (Python 3.12)

'''Variant registry for OmicsHorizon integrated analysis UI layouts.'''
from __future__ import annotations
from typing import Any, Callable, Dict, List
import streamlit as st
VariantContext = Dict[(str, Any)]
VariantRenderer = Callable[([
    VariantContext], None)]
_VARIANTS: 'Dict[str, VariantRenderer]' = { }

def register_variant(name = None, renderer = None):
    '''Register a new integrated analysis UI variant.'''
    _VARIANTS[name] = renderer


def list_variants():
    '''Return the list of registered variant names.'''
    if 'default' not in _VARIANTS and _VARIANTS:
        return sorted(_VARIANTS.keys())
# WARNING: Decompyle incomplete


def render_integrated_analysis_ui(ctx = None, variant = None):
    '''Render the integrated analysis UI using a registered variant.'''
    if not _VARIANTS:
        raise RuntimeError('No integrated analysis UI variants registered')
# WARNING: Decompyle incomplete


def _default_variant(ctx = None):
    '''Default OmicsHorizon integrated analysis UI.'''
    t = ctx['t']
    parse_analysis_steps = ctx['parse_analysis_steps']
    render_sequential_interactive_mode = ctx['render_sequential_interactive_mode']
    is_workflow_completed = ctx['is_workflow_completed']
    apply_analysis_refinement = ctx['apply_analysis_refinement']
    execute_refinement_action = ctx['execute_refinement_action']
    answer_qa_question = ctx['answer_qa_question']
    handle_chat_input = ctx['handle_chat_input']
    add_chat_message = ctx['add_chat_message']
    llm_model = ctx.get('llm_model', 'unknown')
    session_state = ctx.get('session_state', st.session_state)
    st.markdown(f'''<div class="panel-header">🎮 {t('panel3_title')} - Interactive Mode</div>''', unsafe_allow_html = True)
# WARNING: Decompyle incomplete


def _minimal_variant(ctx = None):
    '''Minimal variant that renders a batch summary for quick inspection.'''
    t = ctx['t']
    parse_analysis_steps = ctx['parse_analysis_steps']
    render_batch_interactive_mode = ctx['render_batch_interactive_mode']
    session_state = ctx.get('session_state', st.session_state)
    st.markdown(f'''<div class="panel-header">🧪 {t('panel3_title')} - Minimal Preview</div>''', unsafe_allow_html = True)
    if session_state.data_files and session_state.analysis_method:
        analysis_steps = parse_analysis_steps(session_state.analysis_method)
        if analysis_steps:
            render_batch_interactive_mode(analysis_steps)
        else:
            st.warning('⚠️ Could not parse analysis steps from Panel 2. Please check the format.')
    elif not session_state.data_files:
        st.warning('⚠️ Please upload data files in Panel 1')
    elif not session_state.analysis_method:
        st.warning('⚠️ Please upload a paper or define analysis method in Panel 2')
# WARNING: Decompyle incomplete

register_variant('default', _default_variant)
register_variant('minimal', _minimal_variant)
