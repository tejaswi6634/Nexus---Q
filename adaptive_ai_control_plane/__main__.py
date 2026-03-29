"""``python -m adaptive_ai_control_plane`` runs the CLI simulation.

``python -m adaptive_ai_control_plane web`` starts the interactive cockpit
(FastAPI on http://127.0.0.1:8765 by default). Override with NEXUS_Q_WEB_PORT.
"""

import sys

import adaptive_ai_control_plane.settings  # noqa: F401  # load .env first

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1].lower() in ("web", "ui", "cockpit"):
        from adaptive_ai_control_plane.web_ui import main as web_main

        web_main()
    else:
        from adaptive_ai_control_plane.main import run_simulation

        run_simulation()
