"""
uv run python3 src_visualize/show_3d_plot.py
"""

import os
import webbrowser

project_root = os.path.join(os.path.dirname(__file__), "..") #

path = os.path.join(project_root, "src_visualize", "output", "hidden_state_pca_plots", "google_gemma-3-4b-it_layer9_pca_3d.html")
webbrowser.open(f"file://{os.path.abspath(path)}")