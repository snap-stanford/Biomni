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
import pandas as pd
import os
import yaml
import sys
from datetime import datetime
import tempfile
import shutil
from pathlib import Path
from collections import defaultdict

os.chdir(os.path.dirname(os.path.abspath(__file__)))


def _ensure_streamlit_shim():
    """Provide no-op fallbacks when running outside Streamlit."""

    def _noop(*args, **kwargs):
        return None

    fallback_attrs = [
        "set_page_config",
        "markdown",
        "error",
        "warning",
        "info",
        "success",
    ]
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


def _load_config_paths():
    """Load path settings from project-level config.yaml if present."""
    try:
        # project root: .../Biomni_HITS/streamlit/ -> .../Biomni_HITS -> repo root
        repo_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        config_path = os.path.join(repo_root, "config.yaml")
        if os.path.isfile(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
        else:
            cfg = {}
    except Exception:
        cfg = {}
    # Fallback to existing defaults if keys are missing
    data_storage_default = "/workdir_efs/jhjeon/Biomni/data"
    workspace_default = "/workdir_efs/jhjeon/Biomni/workspace"
    return (
        cfg.get("DATA_STORAGE_PATH", data_storage_default),
        cfg.get("WORKSPACE_PATH", workspace_default),
    )


DATA_STORAGE_PATH, WORKSPACE_PATH = _load_config_paths()

# Analysis Apps Registry
ANALYSIS_APPS = {
    "omics_horizon": {
        "name": "OmicsHorizon™",
        "description": "AI-Powered Transcriptomic Analysis Platform",
        "icon": "🧬",
        "function": run_omicshorizon_app,
        "data_types": [
            ".csv",
            ".xlsx",
            ".xls",
            ".tsv",
            ".txt",
            ".json",
            ".gz",
            ".csv.gz",
            ".tsv.gz",
            ".txt.gz",
        ],
        "category": "Genomics",
        "enabled": True,
    },
    "microbiome_analyzer": {
        "name": "Microbiome Analyzer",
        "description": "Microbial community analysis and diversity profiling",
        "icon": "🦠",
        "function": None,  # To be implemented
        "data_types": [
            ".csv",
            ".tsv",
            ".txt",
            ".xlsx",
            ".xls",
            ".fasta",
            ".fastq",
            ".fa",
            ".fq",
            ".fa.gz",
            ".fastq.gz",
        ],
        "category": "Microbiome",
        "enabled": True,
    },
    "metabolic_simulator": {
        "name": "Metabolic Simulator",
        "description": "Metabolic network simulation and flux analysis",
        "icon": "⚛️",
        "function": None,  # To be implemented
        "data_types": [
            ".csv",
            ".tsv",
            ".txt",
            ".xlsx",
            ".xls",
            ".json",
            ".xml",
            ".sbml",
            ".gz",
            ".csv.gz",
            ".tsv.gz",
        ],
        "category": "Metabolomics",
        "enabled": True,
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
    initial_sidebar_state="expanded",
)

# Global CSS
st.markdown(
    """
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
    /* Reset sidebar width to platform default when not in OmicsHorizon */
    [data-testid="stSidebar"] {
        min-width: 300px !important;
        max-width: 300px !important;
    }
</style>
""",
    unsafe_allow_html=True,
)


class AppRouter:
    """Manages navigation between LIMS and analysis apps"""

    def __init__(self):
        # Initialize session state for navigation
        if "current_view" not in st.session_state:
            st.session_state.current_view = "lims"  # 'lims', 'app_selection', or app_id

        if "selected_app" not in st.session_state:
            st.session_state.selected_app = None

        if "selected_data_files" not in st.session_state:
            st.session_state.selected_data_files = []

        if "app_workspace" not in st.session_state:
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
        st.session_state.current_view = "lims"
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
        if not os.path.exists(self.data_path):
            return files

        root_path = Path(self.data_path)
        for dirpath, dirnames, filenames in os.walk(self.data_path):
            # Skip hidden directories
            dirnames[:] = [d for d in dirnames if not d.startswith(".")]

            for filename in filenames:
                if filename.startswith("."):
                    continue

                file_path = Path(dirpath) / filename
                if not file_path.is_file():
                    continue

                try:
                    stat = file_path.stat()
                except OSError:
                    continue

                relative_path = file_path.relative_to(root_path)
                parts = relative_path.parts

                if len(parts) > 1:
                    instrument = parts[0]
                    remainder = (
                        Path(*parts[1:-1]) if len(parts) > 2 else Path()
                    )
                else:
                    instrument = "General"
                    remainder = Path()

                relative_folder = remainder.as_posix() if remainder.parts else ""

                suffixes = [s.lower() for s in file_path.suffixes]
                if suffixes and suffixes[-1] == ".gz" and len(suffixes) > 1:
                    extension = "".join(suffixes[-2:])
                else:
                    extension = suffixes[-1] if suffixes else ""

                file_info = {
                    "name": filename,
                    "path": str(file_path),
                    "size": stat.st_size,
                    "modified": datetime.fromtimestamp(stat.st_mtime),
                    "extension": extension,
                    "instrument": instrument,
                    "relative_path": relative_path.as_posix(),
                    "relative_folder": relative_folder,
                }
                files.append(file_info)

        return sorted(files, key=lambda x: x["modified"], reverse=True)

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
        if current_view == "lims":
            st.markdown("📊 **Current: LIMS Dashboard**")
        elif current_view == "app_selection":
            st.markdown("🎯 **Current: App Selection**")
        else:
            selected_app = router.get_selected_app()
            if selected_app and selected_app in ANALYSIS_APPS:
                app_info = ANALYSIS_APPS[selected_app]
                st.markdown(f"{app_info['icon']} **Current: {app_info['name']}**")

        st.markdown("---")

        # Navigation buttons
        if current_view != "lims":
            if st.button(
                "🏠 Back to LIMS", key="back_to_lims", use_container_width=True
            ):
                router.go_back_to_lims()

        st.markdown("---")

        # Session Info
        st.markdown("## 📊 Session Info")
        workspace = st.session_state.get("app_workspace", "Not initialized")
        workspace_short = workspace.replace("/workdir_efs/jhjeon/Biomni/", "")

        st.info(
            f"""
        **Version:** {APP_VERSION}
        **Workspace:** `{workspace_short}`
        **Data Files:** {len(data_manager.list_data_files())}
        """
        )

        st.markdown("---")

        # Quick Actions
        st.markdown("## ⚡ Quick Actions")

        if st.button("🗑️ Clear Session", key="clear_session", use_container_width=True):
            # Clear all session state
            for key in list(st.session_state.keys()):
                if key != "app_workspace":  # Keep workspace
                    del st.session_state[key]
            st.success("✅ Session cleared!")
            st.rerun()


def render_lims_dashboard(router, data_manager):
    """Render the main LIMS dashboard"""
    st.markdown(
        '<h1 class="main-title">🔬 Integrated LIMS & Analysis Platform</h1>',
        unsafe_allow_html=True,
    )
    # st.markdown(
    #     '<p class="subtitle">Centralized data management and specialized analysis applications</p>',
    #     unsafe_allow_html=True,
    # )

    # Overview cards
    # col1, col2, col3 = st.columns(3)

    # with col1:
    #     total_files = len(data_manager.list_data_files())
    #     st.metric("📁 Data Files", total_files)

    # with col2:
    #     active_apps = sum(1 for app in ANALYSIS_APPS.values() if app["enabled"])
    #     st.metric("🔬 Active Apps", active_apps)

    # with col3:
    #     st.metric("💾 Storage", "Available")

    st.markdown("---")

    # Data Files Section
    st.markdown("## 📊 LIMS Data Repository")

    data_files = data_manager.list_data_files()

    if not data_files:
        st.info("No data files found in the repository. Add files to get started.")
    else:
        grouped_files: dict[str, list[dict]] = defaultdict(list)
        for file_info in data_files:
            grouped_files[file_info["instrument"]].append(file_info)

        st.markdown(
            f"**{len(data_files)} files available across {len(grouped_files)} instrument folders**"
        )

        selected_keys = {
            f["relative_path"] for f in st.session_state.selected_data_files
        }
        selected_files: list[dict] = []

        for instrument in sorted(grouped_files.keys()):
            instrument_files = grouped_files[instrument]
            instrument_label = (
                f"{instrument}" if instrument != "General" else "General Files"
            )

            with st.expander(
                f"{instrument_label} ({len(instrument_files)} files)", expanded=True
            ):
                rows = []
                for info in instrument_files:
                    display_folder = info["relative_folder"] or "-"
                    rows.append(
                        {
                            "Select": info["relative_path"] in selected_keys,
                            "Folder": display_folder,
                            "File": info["name"],
                            "Type": info["extension"].upper(),
                            "Size (MB)": round(info["size"] / (1024 * 1024), 1),
                            "Modified": info["modified"].strftime("%Y-%m-%d %H:%M"),
                            "Path": info["relative_path"],
                        }
                    )

                df = pd.DataFrame(rows)
                editor_key = f"data_editor_{instrument}"
                edited_df = st.data_editor(
                    df,
                    hide_index=True,
                    height=min(320, 100 + 44 * len(rows)),
                    use_container_width=True,
                    key=editor_key,
                    column_config={
                        "Select": st.column_config.CheckboxColumn("Select"),
                        "Folder": st.column_config.TextColumn("Folder", disabled=True),
                        "File": st.column_config.TextColumn("File", disabled=True),
                        "Size (MB)": st.column_config.NumberColumn(
                            "Size (MB)", disabled=True, format="%.1f"
                        ),
                        "Type": st.column_config.TextColumn("Type", disabled=True),
                        "Modified": st.column_config.TextColumn(
                            "Modified", disabled=True
                        ),
                        "Path": st.column_config.TextColumn(
                            "Relative Path", disabled=True, width="medium"
                        ),
                    },
                )

                for _, row in edited_df.iterrows():
                    if bool(row.get("Select")):
                        row_path = row.get("Path")
                        match = next(
                            (f for f in instrument_files if f["relative_path"] == row_path),
                            None,
                        )
                        if match:
                            selected_files.append(match)

        # Update selected files in session state
        st.session_state.selected_data_files = selected_files

        if selected_files:
            st.success(f"✅ {len(selected_files)} file(s) selected for analysis")

            # Show selected files summary
            with st.expander("📋 Selected Files Summary", expanded=False):
                for file_info in selected_files:
                    instrument_label = (
                        f"{file_info['instrument']} / " if file_info["instrument"] != "General" else ""
                    )
                    st.markdown(
                        f"- **{instrument_label}{file_info['relative_path']}** ({file_info['extension']})"
                    )

        st.markdown("---")

    # Analysis Apps Section
    st.markdown("## 🎯 Specialized Analysis Applications")

    # Filter enabled apps
    enabled_apps = {k: v for k, v in ANALYSIS_APPS.items() if v["enabled"]}

    if not enabled_apps:
        st.warning("No analysis applications are currently available.")
    else:
        st.markdown(f"**{len(enabled_apps)} application(s) ready for use**")

        # App cards in a grid
        cols = st.columns(min(3, len(enabled_apps)))
        for idx, (app_id, app_info) in enumerate(enabled_apps.items()):
            with cols[idx % 3]:
                # App card
                selected_class = (
                    "selected" if st.session_state.selected_data_files else ""
                )
                st.markdown(
                    f"""
                <div class="app-card {selected_class}">
                    <div class="app-icon">{app_info['icon']}</div>
                    <div class="app-name">{app_info['name']}</div>
                    <div class="app-description">{app_info['description']}</div>
                </div>
                """,
                    unsafe_allow_html=True,
                )

                # Check if app can handle selected data types
                if st.session_state.selected_data_files:
                    selected_extensions = [
                        f["extension"] for f in st.session_state.selected_data_files
                    ]
                    selected_filenames = [
                        f["name"] for f in st.session_state.selected_data_files
                    ]

                    # Check compatibility with more flexible logic
                    compatible = False
                    for ext, filename in zip(selected_extensions, selected_filenames):
                        # Direct extension match
                        if ext in app_info["data_types"]:
                            compatible = True
                            break

                        # Check for compressed files (.gz, .zip, etc.)
                        if ext == ".gz":
                            # Extract the actual file type from compressed files
                            # e.g., 'data.txt.gz' -> '.txt'
                            if "." in filename:
                                parts = filename.split(".")
                                if len(parts) >= 3:  # filename.ext.gz
                                    actual_ext = "." + parts[-2]  # second to last part
                                    if actual_ext in app_info["data_types"]:
                                        compatible = True
                                        break

                        # Check for compound extensions
                        # e.g., '.tsv.gz' should match if '.tsv' or '.gz' is supported
                        for data_type in app_info["data_types"]:
                            if data_type in filename or data_type in ext:
                                compatible = True
                                break

                        if compatible:
                            break

                    if compatible:
                        if st.button(
                            f"🚀 Launch {app_info['name']}",
                            key=f"launch_{app_id}",
                            use_container_width=True,
                            type="primary",
                        ):
                            # Copy selected files to app workspace
                            file_paths = [
                                f["path"] for f in st.session_state.selected_data_files
                            ]
                            copied_files = data_manager.copy_files_to_workspace(
                                file_paths, st.session_state.app_workspace
                            )

                            # Navigate to app
                            router.navigate_to(app_id, app_id)
                    else:
                        st.button(
                            f"❌ Incompatible Data Types",
                            key=f"incompatible_{app_id}",
                            use_container_width=True,
                            disabled=True,
                        )
                        st.caption(
                            "Selected files are not compatible with this analysis app."
                        )
                else:
                    st.button(
                        f"📁 Select Data First",
                        key=f"select_data_{app_id}",
                        use_container_width=True,
                        disabled=True,
                    )
                    st.caption("Please select data files above to enable analysis.")


def render_analysis_app(app_id, app_info):
    """Render a specific analysis application"""
    if app_info["function"] is None:
        st.error(f"❌ {app_info['name']} is not yet implemented.")
        return

    # Render the analysis app with LIMS integration
    try:
        # Pass from_lims=True and workspace path to indicate this is launched from LIMS
        app_info["function"](
            from_lims=True, workspace_path=st.session_state.app_workspace
        )
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

    if current_view == "lims":
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
