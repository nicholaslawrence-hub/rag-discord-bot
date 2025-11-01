# Discord Bot Setup Guide

## Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- A Discord account with administrative permissions on your server

## Step 1: Create a Discord Application and Bot

1. Go to the [Discord Developer Portal](https://discord.com/developers/applications)
2. Click "New Application" and give it a name
3. Go to the "Bot" tab and click "Add Bot"
4. Under the bot settings, enable the following permissions:
   - Presence Intent
   - Server Members Intent
   - Message Content Intent
5. Click "Reset Token" and copy your bot token
6. Under "OAuth2" > "URL Generator", select:
   - Scopes: `bot`
   - Bot Permissions: `Send Messages`, `Embed Links`, `Attach Files`, `Read Message History`, `Use External Emojis`
7. Copy the generated URL and open it in your browser to invite the bot to your server

## Step 2: Set Up the Environment

1. Create a new directory for your bot
2. Create a virtual environment:
   ```bash
   python -m venv venv
   venv\Scripts\activate (Windows)
   ```
3. Install required packages:
   ```bash
   pip install discord.py requests matplotlib python-dotenv
   ```
4. Create a `.env` file with the following content:
   ```
   DISCORD_TOKEN=your_bot_token_here
   DASHBOARD_URL=http://your_server_ip:5000
   PNW_API_KEY=your_pnw_api_key_here
   ```

## Step 3: Deploy the Bot

1. Save the Discord bot code (from the other artifact)
2. Start the bot:
   ```bash
   python yourbot.py
   ```

## Step 4: Integration with Web Dashboard

1. Make sure your Flask dashboard is running and accessible
2. The bot communicates with the dashboard through HTTP requests, so ensure the dashboard is reachable from where the bot is running
3. Update the `DASHBOARD_URL` in your `.env` file to match your dashboard's actual URL