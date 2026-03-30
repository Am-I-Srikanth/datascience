# STEP 8 — Deploy to Streamlit Community Cloud (Free, 10 minutes)
# Follow these commands in order. Copy-paste each block.

# ── Prerequisites ────────────────────────────────────────────────────────────
# 1. Create a free account at https://github.com
# 2. Create a free account at https://share.streamlit.io (sign in with GitHub)

# ── One-time setup ───────────────────────────────────────────────────────────

# Install git if you don't have it:
#   Windows: https://git-scm.com/download/win
#   Mac:     brew install git

# ── Step-by-step terminal commands ──────────────────────────────────────────

# 1. Go into your project folder
cd crypto_project

# 2. Make sure you have a requirements.txt (already created — check it exists)
cat requirements.txt

# 3. Create .streamlit/config.toml so the app looks clean on the web
mkdir -p .streamlit
cat > .streamlit/config.toml << 'EOF'
[theme]
primaryColor = "#378ADD"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F4F3EF"
textColor = "#1A1A18"
font = "sans serif"
EOF

# 4. Initialise git and push to GitHub
git init
git add .
git commit -m "Initial commit: crypto volatility forecasting app"

# Replace YOUR_USERNAME and YOUR_REPO_NAME below:
git remote add origin https://github.com/YOUR_USERNAME/crypto-volatility-app.git
git branch -M main
git push -u origin main

# ── Deploy on Streamlit Community Cloud ────────────────────────────────────
# 1. Go to https://share.streamlit.io
# 2. Click "New app"
# 3. Select your GitHub repo
# 4. Set "Main file path" to:  app/streamlit_app.py
# 5. Click "Deploy!"
# → You get a live URL like: https://your-app.streamlit.app
# → Put THIS URL on your resume under the project link.

# ── Resume bullet (copy this) ───────────────────────────────────────────────
# Deployed a live crypto volatility forecasting web app on Streamlit Community
# Cloud; accessible at https://your-app.streamlit.app
