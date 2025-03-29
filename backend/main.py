import uvicorn
from app import app  # Importing the FastAPI instance from app.py

if __name__ == "__main__":
    # Run the FastAPI app with Uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
