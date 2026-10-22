"""
Unified launcher to start both FastAPI Backend and Streamlit Frontend concurrently.
"""
import subprocess
import sys
import time

def main():
    print("=" * 60)
    print("STARTING POLYMER RESIN CLASSIFIER PROJECT")
    print("=" * 60)
    print("1. Launching FastAPI Backend on http://localhost:8000 ...")
    api_proc = subprocess.Popen([
        sys.executable, "-m", "uvicorn", "api:app",
        "--host", "0.0.0.0",
        "--port", "8000"
    ])

    time.sleep(2)

    print("2. Launching Streamlit Frontend on http://localhost:8501 ...")
    st_proc = subprocess.Popen([
        sys.executable, "-m", "streamlit", "run", "app.py",
        "--server.port", "8501"
    ])

    print("\nProject services are running:")
    print(" - Streamlit Web UI:     http://localhost:8501")
    print(" - FastAPI API Service:  http://localhost:8000")
    print(" - Interactive API Docs: http://localhost:8000/docs")
    print("\nPress Ctrl+C to stop all services.\n")

    try:
        api_proc.wait()
        st_proc.wait()
    except KeyboardInterrupt:
        print("\nStopping services...")
        api_proc.terminate()
        st_proc.terminate()
        print("All services stopped.")

if __name__ == "__main__":
    main()
