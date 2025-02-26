import websocket
import json
import time
import threading

# Global variables (kept from your code, though not used here)
current_detected_person = "Unknown"
current_detected_person_count = 0
recognized_person = "Unknown"
after_recognized_count = 0

# WebSocket connection
uri = "ws://localhost:12393/client-ws"
try:
    ws = websocket.create_connection(uri)
    print("Connected to WebSocket server")
except Exception as e:
    print(f"Failed to connect to WebSocket server: {e}")
    exit(1)

def connect_and_send(text="", type="text-input"):
    """Send a message via WebSocket (kept from your code, though not used here)"""
    print('msg sending')
    try:
        message = {
            "type": type,
            "text": text,
            "images": []
        }
        ws.send(json.dumps(message))
        print("Message sent to WebSocket server")
    except Exception as e:
        print(f"Error: {e}")

def on_message(ws, message):
    """Handle incoming WebSocket messages"""
    data = json.loads(message)
    text = data.get("text", "")
    print(f"Received from backend: {text}")

def on_error(ws, error):
    print(f"WebSocket error: {error}")

def on_close(ws, close_status_code, close_msg):
    print("WebSocket connection closed")

def websocket_listener():
    """Run WebSocket listener"""
    ws.on_message = on_message
    ws.on_error = on_error
    ws.on_close = on_close
    ws.run_forever()

def main():
    print("Starting WebSocket client...")
    
    # Start WebSocket listener in a separate thread
    threading.Thread(target=websocket_listener, daemon=True).start()

    # Keep the main thread alive to allow receiving messages
    try:
        while True:
            time.sleep(1)  # Keep the main thread running
    except KeyboardInterrupt:
        print("\nShutting down WebSocket client...")
        ws.close()

if __name__ == "__main__":
    main()