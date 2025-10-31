"""
Integrated Laboratory Information Management System (LIMS) with Analysis Apps

This is the main entry point for the unified analysis platform that integrates:
- LIMS (Laboratory Information Management System) - data collection hub
- OmicsHorizon - Transcriptome analysis app
- Future analysis apps (Proteomics, Metabolomics, etc.)

Author: JHJeon
Date: 2025
"""

import streamlit as st
import os
import sys
from datetime import datetime
import tempfile
import shutil
from pathlib import Path


def _ensure_streamlit_shim():
    """Provide no-op fallbacks when running outside Streamlit."""
    def _noop(*args, **kwargs):
        return None

    fallback_attrs = ['set_page_config', 'markdown', 'error', 'warning', 'info', 'success']
    for attr in fallback_attrs:
        if not hasattr(st, attr):
            setattr(st, attr, _noop)

_ensure_streamlit_shim()

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Import analysis apps
try:
    from streamlit_app import run_omicshorizon_app
except ImportError:
    st.error("Failed to import OmicsHorizon app. Please check streamlit_app.py")
    run_omicshorizon_app = None

# Configuration
APP_VERSION = "1.0.0"
APP_TITLE = "Integrated LIMS & Analysis Platform"
DATA_STORAGE_PATH = "/workdir_efs/jhjeon/Biomni/data"
WORKSPACE_PATH = "/workdir_efs/jhjeon/Biomni/workspace"

# Analysis Apps Registry
ANALYSIS_APPS = {
    'omics_horizon': {
        'name': 'OmicsHorizon™',
        'description': 'AI-Powered Transcriptomic Analysis Platform',
        'icon': '🧬',
        'function': run_omicshorizon_app,
        'data_types': ['.csv', '.xlsx', '.xls', '.tsv', '.txt', '.json', '.gz', '.csv.gz', '.tsv.gz', '.txt.gz'],
        'category': 'Genomics',
        'enabled': True
    },
    # Future apps can be added here
    # 'proteomics_app': {
    #     'name': 'Proteomics Analyzer',
    #     'description': 'Protein expression analysis platform',
    #     'icon': '🧪',
    #     'function': None,  # To be implemented
    #     'data_types': ['mzML', 'mzXML', 'raw', 'csv'],
    #     'category': 'Proteomics',
    #     'enabled': False
    # },
    # 'metabolomics_app': {
    #     'name': 'Metabolomics Analyzer',
    #     'description': 'Metabolite profiling and analysis',
    #     'icon': '🧫',
    #     'function': None,  # To be implemented
    #     'data_types': ['mzML', 'mzXML', 'csv', 'txt'],
    #     'category': 'Metabolomics',
    #     'enabled': False
    # }
}

# Page configuration
st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Global CSS
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 1rem;
    }
    .subtitle {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .app-card {
        border: 2px solid #e0e0e0;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        background-color: #fafafa;
        transition: all 0.3s ease;
    }
    .app-card:hover {
        border-color: #3498db;
        box-shadow: 0 4px 12px rgba(52, 152, 219, 0.15);
    }
    .app-card.selected {
        border-color: #27ae60;
        background-color: #f0f9f0;
    }
    .app-icon {
        font-size: 2rem;
        margin-bottom: 10px;
    }
    .app-name {
        font-size: 1.3rem;
        font-weight: bold;
        color: #2c3e50;
        margin-bottom: 5px;
    }
    .app-description {
        color: #666;
        font-size: 0.9rem;
        margin-bottom: 15px;
    }
    .nav-button {
        background-color: #3498db;
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
        cursor: pointer;
        font-size: 1rem;
        margin: 10px 5px;
        transition: background-color 0.3s ease;
    }
    .nav-button:hover {
        background-color: #2980b9;
    }
    .nav-button.back {
        background-color: #95a5a6;
    }
    .nav-button.back:hover {
        background-color: #7f8c8d;
    }
    .status-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 5px;
    }
    .status-online {
        background-color: #27ae60;
    }
    .status-offline {
        background-color: #e74c3c;
    }
    .data-item {
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 10px;
        margin: 5px 0;
        background-color: #fff;
    }
    .data-item.selected {
        border-color: #3498db;
        background-color: #ecf0f1;
    }
</style>
""", unsafe_allow_html=True)


class AppRouter:
    """Manages navigation between LIMS and analysis apps"""

    def __init__(self):
        # Initialize session state for navigation
        if 'current_view' not in st.session_state:
            st.session_state.current_view = 'lims'  # 'lims', 'app_selection', or app_id

        if 'selected_app' not in st.session_state:
            st.session_state.selected_app = None

        if 'selected_data_files' not in st.session_state:
            st.session_state.selected_data_files = []

        if 'app_workspace' not in st.session_state:
            # Create unique workspace for this session
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            workspace_path = os.path.join(WORKSPACE_PATH, f"session_{timestamp}")
            os.makedirs(workspace_path, exist_ok=True)
            st.session_state.app_workspace = workspace_path

    def navigate_to(self, view, app_id=None):
        """Navigate to a specific view"""
        st.session_state.current_view = view
        if app_id:
            st.session_state.selected_app = app_id
        st.rerun()

    def go_back_to_lims(self):
        """Return to LIMS main screen"""
        st.session_state.current_view = 'lims'
        st.session_state.selected_app = None
        st.rerun()

    def get_current_view(self):
        """Get current view"""
        return st.session_state.current_view

    def get_selected_app(self):
        """Get currently selected app"""
        return st.session_state.selected_app


class DataManager:
    """Manages data files for LIMS"""

    def __init__(self):
        self.data_path = DATA_STORAGE_PATH
        os.makedirs(self.data_path, exist_ok=True)

    def list_data_files(self):
        """List all available data files in LIMS"""
        files = []
        if os.path.exists(self.data_path):
            for file in os.listdir(self.data_path):
                file_path = os.path.join(self.data_path, file)
                if os.path.isfile(file_path):
                    # Extract extension with better handling for compressed files
                    suffix = Path(file).suffix.lower()
                    if suffix == '.gz':
                        # For .gz files, try to get the actual file type
                        # e.g., 'data.txt.gz' -> '.txt.gz'
                        stem = Path(file).stem  # remove .gz
                        if '.' in stem:
                            # Has another extension
                            inner_suffix = Path(stem).suffix.lower()
                            extension = inner_suffix + '.gz'
                        else:
                            extension = '.gz'
                    else:
                        extension = suffix

                    file_info = {
                        'name': file,
                        'path': file_path,
                        'size': os.path.getsize(file_path),
                        'modified': datetime.fromtimestamp(os.path.getmtime(file_path)),
                        'extension': extension
                    }
                    files.append(file_info)
        return sorted(files, key=lambda x: x['modified'], reverse=True)

    def copy_files_to_workspace(self, file_paths, workspace_path):
        """Copy selected files to analysis app workspace"""
        copied_files = []
        for file_path in file_paths:
            if os.path.exists(file_path):
                filename = os.path.basename(file_path)
                dest_path = os.path.join(workspace_path, filename)
                shutil.copy2(file_path, dest_path)
                copied_files.append(dest_path)
        return copied_files


def render_sidebar(router, data_manager):
    """Render the sidebar with navigation and info"""
    with st.sidebar:
        st.markdown("## 🔬 Navigation")

        # Current view indicator
        current_view = router.get_current_view()
        if current_view == 'lims':
            st.markdown("📊 **Current: LIMS Dashboard**")
        elif current_view == 'app_selection':
            st.markdown("🎯 **Current: App Selection**")
        else:
            selected_app = router.get_selected_app()
            if selected_app and selected_app in ANALYSIS_APPS:
                app_info = ANALYSIS_APPS[selected_app]
                st.markdown(f"{app_info['icon']} **Current: {app_info['name']}**")

        st.markdown("---")

        # Navigation buttons
        if current_view != 'lims':
            if st.button("🏠 Back to LIMS", key="back_to_lims", use_container_width=True):
                router.go_back_to_lims()

        st.markdown("---")

        # Session Info
        st.markdown("## 📊 Session Info")
        workspace = st.session_state.get('app_workspace', 'Not initialized')
        workspace_short = workspace.replace('/workdir_efs/jhjeon/Biomni/', '')

        st.info(f"""
        **Version:** {APP_VERSION}
        **Workspace:** `{workspace_short}`
        **Data Files:** {len(data_manager.list_data_files())}
        """)

        st.markdown("---")

        # Quick Actions
        st.markdown("## ⚡ Quick Actions")

        if st.button("🗑️ Clear Session", key="clear_session", use_container_width=True):
            # Clear all session state
            for key in list(st.session_state.keys()):
                if key != 'app_workspace':  # Keep workspace
                    del st.session_state[key]
            st.success("✅ Session cleared!")
            st.rerun()


def render_lims_dashboard(router, data_manager):
    """Render the main LIMS dashboard"""
    st.markdown('<h1 class="main-title">🔬 Integrated LIMS & Analysis Platform</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Centralized data management and specialized analysis applications</p>', unsafe_allow_html=True)

    # Overview cards
    col1, col2, col3 = st.columns(3)

    with col1:
        total_files = len(data_manager.list_data_files())
        st.metric("📁 Data Files", total_files)

    with col2:
        active_apps = sum(1 for app in ANALYSIS_APPS.values() if app['enabled'])
        st.metric("🔬 Active Apps", active_apps)

    with col3:
        st.metric("💾 Storage", "Available")

    st.markdown("---")

    # Data Files Section
    st.markdown("## 📊 Laboratory Data Repository")

    data_files = data_manager.list_data_files()

    if not data_files:
        st.info("No data files found in the repository. Add files to get started.")
    else:
        st.markdown(f"**{len(data_files)} files available**")

        # File selection
        selected_files = []
        for file_info in data_files:
            size_mb = file_info['size'] / (1024 * 1024)
            modified_str = file_info['modified'].strftime("%Y-%m-%d %H:%M")

            col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
            with col1:
                is_selected = st.checkbox(
                    f"**{file_info['name']}**",
                    key=f"select_{file_info['name']}",
                    value=file_info['name'] in [f['name'] for f in st.session_state.selected_data_files]
                )
                if is_selected:
                    selected_files.append(file_info)
            with col2:
                st.caption(f"{size_mb:.1f} MB")
            with col3:
                st.caption(file_info['extension'].upper())
            with col4:
                st.caption(modified_str)

        # Update selected files in session state
        st.session_state.selected_data_files = selected_files

        if selected_files:
            st.success(f"✅ {len(selected_files)} file(s) selected for analysis")

            # Show selected files summary
            with st.expander("📋 Selected Files Summary", expanded=False):
                for file_info in selected_files:
                    st.markdown(f"- **{file_info['name']}** ({file_info['extension']})")

        st.markdown("---")

    # Analysis Apps Section
    st.markdown("## 🎯 Specialized Analysis Applications")

    # Filter enabled apps
    enabled_apps = {k: v for k, v in ANALYSIS_APPS.items() if v['enabled']}

    if not enabled_apps:
        st.warning("No analysis applications are currently available.")
    else:
        st.markdown(f"**{len(enabled_apps)} application(s) ready for use**")

        # App cards in a grid
        cols = st.columns(min(3, len(enabled_apps)))
        for idx, (app_id, app_info) in enumerate(enabled_apps.items()):
            with cols[idx % 3]:
                # App card
                selected_class = "selected" if st.session_state.selected_data_files else ""
                st.markdown(f"""
                <div class="app-card {selected_class}">
                    <div class="app-icon">{app_info['icon']}</div>
                    <div class="app-name">{app_info['name']}</div>
                    <div class="app-description">{app_info['description']}</div>
                </div>
                """, unsafe_allow_html=True)

                # Check if app can handle selected data types
                if st.session_state.selected_data_files:
                    selected_extensions = [f['extension'] for f in st.session_state.selected_data_files]
                    selected_filenames = [f['name'] for f in st.session_state.selected_data_files]

                    # Check compatibility with more flexible logic
                    compatible = False
                    for ext, filename in zip(selected_extensions, selected_filenames):
                        # Direct extension match
                        if ext in app_info['data_types']:
                            compatible = True
                            break

                        # Check for compressed files (.gz, .zip, etc.)
                        if ext == '.gz':
                            # Extract the actual file type from compressed files
                            # e.g., 'data.txt.gz' -> '.txt'
                            if '.' in filename:
                                parts = filename.split('.')
                                if len(parts) >= 3:  # filename.ext.gz
                                    actual_ext = '.' + parts[-2]  # second to last part
                                    if actual_ext in app_info['data_types']:
                                        compatible = True
                                        break

                        # Check for compound extensions
                        # e.g., '.tsv.gz' should match if '.tsv' or '.gz' is supported
                        for data_type in app_info['data_types']:
                            if data_type in filename or data_type in ext:
                                compatible = True
                                break

                        if compatible:
                            break

                    if compatible:
                        if st.button(f"🚀 Launch {app_info['name']}",
                                   key=f"launch_{app_id}",
                                   use_container_width=True,
                                   type="primary"):
                            # Copy selected files to app workspace
                            file_paths = [f['path'] for f in st.session_state.selected_data_files]
                            copied_files = data_manager.copy_files_to_workspace(
                                file_paths, st.session_state.app_workspace)

                            # Navigate to app
                            router.navigate_to(app_id, app_id)
                    else:
                        st.button(f"❌ Incompatible Data Types",
                                key=f"incompatible_{app_id}",
                                use_container_width=True,
                                disabled=True)
                        st.caption("Selected files are not compatible with this analysis app.")
                else:
                    st.button(f"📁 Select Data First",
                            key=f"select_data_{app_id}",
                            use_container_width=True,
                            disabled=True)
                    st.caption("Please select data files above to enable analysis.")


def render_analysis_app(app_id, app_info):
    """Render a specific analysis application"""
    if app_info['function'] is None:
        st.error(f"❌ {app_info['name']} is not yet implemented.")
        return

    # Render the analysis app with LIMS integration
    try:
        # Pass from_lims=True and workspace path to indicate this is launched from LIMS
        app_info['function'](from_lims=True, workspace_path=st.session_state.app_workspace)
    except Exception as e:
        st.error(f"Error running {app_info['name']}: {str(e)}")
        st.exception(e)


def main():
    """Main application entry point"""
    # Initialize components
    router = AppRouter()
    data_manager = DataManager()

    # Render sidebar
    render_sidebar(router, data_manager)

    # Render main content based on current view
    current_view = router.get_current_view()

    if current_view == 'lims':
        render_lims_dashboard(router, data_manager)
    else:
        # It's an analysis app
        selected_app = router.get_selected_app()
        if selected_app and selected_app in ANALYSIS_APPS:
            app_info = ANALYSIS_APPS[selected_app]
            render_analysis_app(selected_app, app_info)
        else:
            st.error("Unknown application selected.")
            if st.button("🏠 Return to LIMS"):
                router.go_back_to_lims()


if __name__ == "__main__":
    main()
