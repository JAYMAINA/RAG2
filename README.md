# RAG Project

The project consists of a **FastAPI** backend and a **React (Vite)** frontend.

## Prerequisites
Ensure you have the following installed on your system:
- Python (>=3.8)
- Node.js (>=16)
- npm or yarn

## Setting Up the Backend (FastAPI)

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate 
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the FastAPI server:
   ```bash
   python main.py
   ```
   The backend will start on `http://127.0.0.1:8000`

## Setting Up the Frontend (React + Vite)

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Start the development server:
   ```bash
   npm run dev
   ```
   The frontend will start on `http://localhost:5173`

## Testing the Connection
Once both servers are running:
- Open `http://localhost:5173` in your browser.
- The frontend should be able to make API requests to `http://127.0.0.1:8000`.
- You can test backend API endpoints via `http://127.0.0.1:8000/docs` (FastAPI Swagger UI).


