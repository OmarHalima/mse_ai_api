"""
PythonAnywhere WSGI entry point.

In PythonAnywhere Web tab:
  - Source code: /home/YOUR_USERNAME/mse_ai_api
  - WSGI file: point to this file
  - Python version: 3.10
"""
import sys, os

project_home = os.path.dirname(os.path.abspath(__file__))
if project_home not in sys.path:
    sys.path.insert(0, project_home)

# Optional: set env vars here if not using the Web tab's env section
# os.environ["API_SECRET_KEY"] = "your-secret-key"

from main import app as application  # noqa
