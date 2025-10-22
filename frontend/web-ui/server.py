#!/usr/bin/env python3
"""
Simple HTTP server to serve the CryBaby frontend UI
"""

import http.server
import socketserver
import os
import sys
from pathlib import Path

PORT = 3000

class MyHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        # Add CORS headers for development
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

    def do_OPTIONS(self):
        # Handle preflight requests
        self.send_response(200)
        self.end_headers()

def main():
    # Change to the directory containing this script
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    print(f"🌐 Starting CryBaby Frontend Server...")
    print(f"📁 Serving from: {script_dir}")
    print(f"🚀 Server running at: http://localhost:{PORT}")
    print(f"🔗 Backend should be running at: http://localhost:8000")
    print(f"⏹️  Press Ctrl+C to stop")
    print("-" * 50)
    
    try:
        with socketserver.TCPServer(("", PORT), MyHTTPRequestHandler) as httpd:
            httpd.serve_forever()
    except KeyboardInterrupt:
        print(f"\n👋 Server stopped.")
        sys.exit(0)
    except OSError as e:
        if e.errno == 48:  # Address already in use
            print(f"❌ Port {PORT} is already in use.")
            print(f"💡 Try: lsof -ti:{PORT} | xargs kill")
            sys.exit(1)
        else:
            raise

if __name__ == "__main__":
    main()
