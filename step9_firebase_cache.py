"""
STEP 9 — Optional: Firebase Cache for API Data
Prevents hammering Coinbase on every refresh.
Run: pip install firebase-admin
     python step9_firebase_cache.py

Setup:
  1. Go to https://console.firebase.google.com
  2. Create a project → Firestore Database → Start in test mode
  3. Project Settings → Service Accounts → Generate new private key
  4. Save the JSON as firebase_key.json in this folder
"""

import json
import os
from datetime import datetime, timedelta

try:
    import firebase_admin
    from firebase_admin import credentials, firestore
    FIREBASE_AVAILABLE = True
except ImportError:
    FIREBASE_AVAILABLE = False
    print("firebase-admin not installed. Run: pip install firebase-admin")


class CryptoCache:
    """Simple Firebase cache for OHLCV data. Falls back to direct API if unavailable."""

    def __init__(self, key_path='firebase_key.json'):
        self.db = None
        if FIREBASE_AVAILABLE and os.path.exists(key_path):
            try:
                cred = credentials.Certificate(key_path)
                if not firebase_admin._apps:
                    firebase_admin.initialize_app(cred)
                self.db = firestore.client()
                print("Firebase connected.")
            except Exception as e:
                print(f"Firebase init failed: {e}. Using direct API.")
        else:
            print("Firebase not configured. Using direct API (no caching).")

    def get(self, symbol: str, days: int):
        """Return cached data if fresh (< 1 hour old), else None."""
        if not self.db:
            return None
        key = f"{symbol.replace('/', '_')}_{days}d"
        doc = self.db.collection('ohlcv_cache').document(key).get()
        if not doc.exists:
            return None
        data = doc.to_dict()
        cached_at = data.get('cached_at')
        if cached_at and (datetime.utcnow() - cached_at.replace(tzinfo=None)) < timedelta(hours=1):
            print(f"Cache hit for {symbol} ({days}d)")
            return json.loads(data['payload'])
        return None  # stale

    def set(self, symbol: str, days: int, df_json: str):
        """Store data in Firestore."""
        if not self.db:
            return
        key = f"{symbol.replace('/', '_')}_{days}d"
        self.db.collection('ohlcv_cache').document(key).set({
            'payload':   df_json,
            'cached_at': datetime.utcnow(),
            'symbol':    symbol,
            'days':      days
        })
        print(f"Cached {symbol} ({days}d) to Firestore.")


# ── How to use in streamlit_app.py ──────────────────────────────────────────
# from step9_firebase_cache import CryptoCache
# cache = CryptoCache()
#
# cached = cache.get('BTC/USD', 30)
# if cached:
#     df = pd.read_json(cached)
# else:
#     df, _ = fetch_data('BTC/USD', 30)
#     cache.set('BTC/USD', 30, df.to_json())

if __name__ == '__main__':
    cache = CryptoCache()
    print("\nCryptoCache ready.")
    print("See comments at bottom of file for integration instructions.")
