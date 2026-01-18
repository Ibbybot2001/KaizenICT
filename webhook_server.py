from flask import Flask, request, jsonify
import time
from datetime import datetime
import pytz
from dashboard_logger import DashboardLogger

# Config
PORT = 5000

# Initialize Flask
app = Flask(__name__)
gs_logger = DashboardLogger()

@app.route('/webhook', methods=['POST'])
@app.route('/', methods=['POST']) # Fallback for users who forget /webhook
def webhook():
    try:
        data = request.json
        print(f"📩 Webhook Received: {data}")
        
        # Parse standard TradingView JSON
        ny_tz = pytz.timezone('America/New_York')
        ts = datetime.now(ny_tz)
        
        # Defaults
        price = data.get('price', 0.0)
        action = data.get('action', 'WAIT')
        details = data.get('details', 'Webhook Signal')
        
        # Metrics 
        vol = data.get('vol', 0)
        wick = data.get('wick', 0)
        body = data.get('body', 0)
        dh = data.get('dh', 0)
        dl = data.get('dl', 0)
        dist_h = data.get('dist_h', 0)
        dist_l = data.get('dist_l', 0)
        
        # Log to Sheets
        gs_logger.log_update(ts, price, action, details, vol, wick, body, dh, dl, dist_h, dist_l)
        
        return jsonify({"status": "success", "msg": "Logged to Sheet"}), 200
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return jsonify({"status": "error", "msg": str(e)}), 500

@app.route('/', methods=['GET'])
def health():
    return "Bot Listener is Online 🟢", 200

if __name__ == '__main__':
    print(f"🚀 Starting Local Server on Port {PORT}...")
    app.run(port=PORT)
