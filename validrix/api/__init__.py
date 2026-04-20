"""
api — FastAPI application for the Validrix Web Agent.

Entry point::

    uvicorn validrix.api:app --reload
"""

from validrix.api.routes import create_app

app = create_app()

__all__ = ["app"]
