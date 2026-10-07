"""Start the interactive explorer (Level-1 pose/clearance + planner, Level-2 mission simulator) in your browser.

    python scripts/gui.py                 # http://127.0.0.1:7860 opens automatically
    python scripts/gui.py --no-browser --port 7861

Needs the extra packages gradio and plotly (pip install gradio plotly). Windows: python scripts\\gui.py
"""
from spacearm.gui.app import main

if __name__ == "__main__":
    main()
