import sys
import os
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent / 'src'))

from app import app

if __name__ == '__main__':
    print("🎙️ Starting Voice Clone Studio Web Server...")
    print("📍 Access the application at: http://localhost:5000")
    print("📋 Upload audio files or record live audio to train voice models")
    print("🎵 Select trained speakers to synthesize speech from text")
    print("-" * 60)
    
    app.run(host='0.0.0.0', port=5000, debug=True)
