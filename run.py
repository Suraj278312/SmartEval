"""
SmartEval - Entry point script.
Runs the development server.
"""

import os
from dotenv import load_dotenv
from app import create_app

load_dotenv()

env = os.environ.get("FLASK_ENV", "development")
app = create_app(env)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "True").lower() in ("true", "1", "yes")
    use_reloader = os.environ.get("FLASK_USE_RELOADER", "False").lower() in ("true", "1", "yes")
    print(f"Starting SmartEval on http://127.0.0.1:{port} (Environment: {env}, Debug: {debug}, Reloader: {use_reloader})")
    app.run(host="0.0.0.0", port=port, debug=debug, use_reloader=use_reloader)
