# Discord Bot Setup Guide

This bot can be used with a variety of LLMs, currently factored for Google Gemini Embedding RAG functionality, and Grok 4 through xAI API for Response Generation. 

## Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- A Discord account with administrative permissions on your server
- Ability to call API (Internet Access)

## Step 1: Obtain OpenAI API Token, xAI Token, and Google OAuth2/AI Studio Token

This can be done through the respective websites.
https://openai.com/api/
https://x.ai/api
Guide for OAuth2: https://developers.google.com/identity/protocols/oauth2
https://aistudio.google.com/

## Step 2: Create a Discord Application and Bot

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

## Step 3: Set Up the Environment

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
4. Create a `credentials.json` file through going to Google Cloud Console, and creating new OAuth Client ID, then downloading JSON and saving as `credentials.json`
    
6. Create a `.env` file with the following content:
   ```
   DISCORD_TOKEN=your_bot_token_here
   GEMINI_API_KEY=your_api_token_here
   GROK_API_KEY=your_api_token_here
   DEEPSEEK_API_KEY=your_api_token_here #Optional
   PNW_API_KEY=your_pnw_api_key_here
   ```

## Step 3: Deploy the Bot

1. Save the Discord bot code (from the other artifact)
2. Start the bot:
   ```bash
   python yourbot.py
   ``