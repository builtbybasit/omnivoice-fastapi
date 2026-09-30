"""Entry point: ``uvicorn main:app``. The server lives in the omnivoice_api package."""

from omnivoice_api.app import create_app

app = create_app()
