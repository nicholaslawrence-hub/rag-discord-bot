from __future__ import absolute_import
import os
import discord
from discord.ext import commands
import concurrent.futures
import functools
import json
import math
import logging
import datetime
import asyncio
from typing import List, Dict, Any, Union, Counter
import time
import pickle 
import google.generativeai as genai
import numpy as np
import sqlite3
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from openai import OpenAI
import re
from dotenv import load_dotenv
import aiohttp 
from bs4 import BeautifulSoup 
from urllib.parse import quote_plus, urljoin, unquote 

KEYWORD_LIST = [
    "a", "an", "the",
    "about", "above", "across", "after", "against", "along", "amid", "among", 
    "around", "as", "at", "before", "behind", "below", "beneath", "beside", 
    "between", "beyond", "by", "despite", "down", "during", "except", "for", 
    "from", "in", "inside", "into", "like", "near", "of", "off", "on", "onto", 
    "out", "outside", "over", "past", "regarding", "round", "since", "through", 
    "throughout", "to", "toward", "towards", "under", "underneath", "until", 
    "unto", "up", "upon", "with", "within", "without", "about", "against", "along",
    "among", "around", "at", "before", "behind", "between", "by", "for", "from",
    "in", "into", "near", "of", "off", "on", "onto", "out", "over", "through",
    "to", "toward", "under", "until", "up", "with", "without", 'for', 'in', 'on',
    'at', 'by', 'to', 'from', 'about', 'against', 'between', 'among', 'through',
    'during', 'before', 'after', 'since', 'until', 'within', 'without', 'like',
    "and", "but", "or", "nor", "so", "yet", "because", "although", "though",
    "while", "unless", "until", "how", "that", "than",
    "as", "whether", "both", "either", "neither", "not only", "but also",
    "whether", "unless", "as long as", "in case", "provided that",
    "even if", "in order that", "so that", "as if", "as though",
    "i", "me", "my", "mine", "myself", "you", "your", "yours", "yourself", 
    "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", 
    "its", "itself", "we", "us", "our", "ours", "ourselves", "they", "them", 
    "their", "theirs", "themselves", "who", "whom", "whose", "which", "what",
    "this", "that", "these", "those", "such", "some", "any", "each", "either",
    "neither", "every", "everyone", "everybody", "everything", "nobody", "nothing",
    "someone", "somebody", "something", "anyone", "anybody", "anything",
    "one", "ones", "another", "other", "others", "both", "few", "many", "several",
    "am", "is", "are", "was", "were", "be", "being", "been", "have", "has", 
    "had", "having", "do", "does", "did", "doing", "can", "could", "shall", 
    "should", "will", "would", "may", "might", "must",
    "ought", "need", "dare", "used to", "be going to", "have to", "has to",
    "had to", "will be", "would be", "can be", "could be", "may be", "might be",
    "very", "really", "quite", "rather", "somewhat", "too", "enough", "just",
    "almost", "nearly", "hardly", "barely", "scarcely", "actually", "certainly",
    "definitely", "probably", "possibly", "perhaps", "maybe", "not", "never",
    "ever", "always", "often", "frequently", "sometimes", "occasionally", 
    "seldom", "rarely", "usually", "generally", "normally", "naturally", 
    "especially", "particularly", "specifically", "mainly", "mostly", "largely",
    "primarily", "chiefly", "principally", "essentially", "basically", "virtually",
    "approximately", "roughly", "about", "almost", "nearly", "only", "just",
    "even", "still", "already", "yet", "now", "then", "soon", "later", "today",
    "tomorrow", "yesterday", "here", "there", "where", "anywhere", "everywhere",
    "somewhere", "nowhere", "however", "moreover", "furthermore", "consequently",
    "therefore", "thus", "hence", "accordingly", "instead", "nevertheless",
    "nonetheless", "meanwhile", "afterward", "afterwards", "subsequently",
    "previously", "initially", "originally", "eventually", "finally", "ultimately",
    "know", "think", "say", "said", "says", "tell", "told", "go", "goes", "went",
    "gone", "come", "comes", "came", "get", "gets", "got", "gotten", "make", 
    "makes", "made", "take", "takes", "took", "taken", "see", "sees", "saw", 
    "seen", "look", "looks", "looked", "seem", "seems", "seemed", "appear", 
    "appears", "appeared", "show", "shows", "showed", "shown", "find", "finds", 
    "found", "give", "gives", "gave", "given", "put", "puts", "turn", "turns", 
    "turned", "call", "calls", "called", "use", "uses", "used", "work", "works", 
    "worked", "try", "tries", "tried", "ask", "asks", "asked", "need", "needs", 
    "needed", "feel", "feels", "felt", "become", "becomes", "became", "leave", 
    "leaves", "left", "good", "better", "best", "bad", "worse", "worst", "big", 
    "bigger", "biggest", "small", "smaller", "smallest", "high", "higher", 
    "highest", "low", "lower", "lowest", "long", "longer", "longest", "short", 
    "shorter", "shortest", "great", "greater", "greatest", "little", "less", 
    "least", "much", "more", "most", "many", "few", "fewer", "fewest", "lot", 
    "lots", "plenty", "several", "various", "diverse", "different", "similar", 
    "same", "other", "another", "else", "own", "self", "thing", "things", "stuff",
    "matter", "issue", "problem", "situation", "case", "point", "example", "fact",
    "way", "means", "kind", "sort", "type", "form", "part", "place", "time", 
    "reason", "question", "answer", "theory", "idea", "thought", "term", "word", 
    "name", "title", "description", "definition", "well", "okay", "ok", "right", 
    "sure", "yes", "no", "yeah", "nope", "etc", "ie", "eg", "vs", "via",
    "let", "lets", "let's", "going", "gonna", "want", "wants", "wanted",
    "wanna", "got", "gotta", "getting", "done", "doing", "did", "does",
    'against', 'along', 'among', 'around', 'as', 'at', 'before', 'behind',
]

DB_PATH = "pnw_guides.db"

SCOPES = ['https://www.googleapis.com/auth/documents.readonly',
          'https://www.googleapis.com/auth/drive.readonly']

PNW_FANDOM_WIKI_DOMAIN = "politicsandwar.fandom.com" 
PNW_FANDOM_WIKI_BASE_URL = f"https://{PNW_FANDOM_WIKI_DOMAIN}/wiki/"
PNW_FANDOM_API_URL = f"https://{PNW_FANDOM_WIKI_DOMAIN}/api.php"

FORUM_BASE_URL = "https://forum.politicsandwar.com"
ALLIANCE_AFFAIRS_URL = f"{FORUM_BASE_URL}/index.php?/forum/42-alliance-affairs/"

CONVERSATION_TIMEOUT_SECONDS = 900
MAX_HISTORY_MESSAGES = 5
MAX = 5
user_message_history = {}
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(
    command_prefix=commands.when_mentioned, 
    heartbeat_timeout=150, 
    intents=intents, 
    help_command=None
    )
_EMBEDDING_MANAGER = None
EMBED_MODEL = None
AIOHTTP_SESSION = None

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("discord_bot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('discord')

thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)

async def monitor_heartbeat():
    """Monitor the WebSocket heartbeat and log issues"""
    while True:
        try:
            if bot.ws is not None:
                latency = bot.latency * 1000
                logger.info(f"WebSocket heartbeat latency: {latency:.2f}ms")
                
                if latency > 1000:
                    logger.warning(f"High latency detected: {latency:.2f}ms")
            
            await asyncio.sleep(3600)  
            
        except Exception as e:
            logger.error(f"Error in heartbeat monitor: {e}")
            await asyncio.sleep(60) 
            
def run_in_executor(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return asyncio.get_event_loop().run_in_executor(
            thread_pool, 
            functools.partial(func, *args, **kwargs)
        )
    return wrapper

load_dotenv("config.env")
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROK_API_KEY = os.getenv("GROK_API_KEY")
GROK_API_URL = "https://api.x.ai/v1"
async def get_aiohttp_session():
    global AIOHTTP_SESSION
    if AIOHTTP_SESSION is None or AIOHTTP_SESSION.closed:
        AIOHTTP_SESSION = aiohttp.ClientSession()
        print("DEBUG: New AIOHTTP_SESSION created.")
    return AIOHTTP_SESSION

def get_embedding_manager():
    """Get or create the embedding manager instance"""
    global _EMBEDDING_MANAGER
    
    if _EMBEDDING_MANAGER is None:
        _EMBEDDING_MANAGER = EnhancedEmbeddingManager()
        print("Initialized embedding manager")
    
    return _EMBEDDING_MANAGER

async def update_user_message_history(message):
    timestamp = time.time()
    user_id = str(message.author.id)
    if user_id not in user_message_history:
        user_message_history[user_id] = []
    
    if message.content and len(message.content.strip()) > 0:
        user_message_history[user_id].append({
            "content": message.content,
            "timestamp": timestamp,
            "channel_id": str(message.channel.id)
        })
        if len(user_message_history[user_id]) > MAX_HISTORY_MESSAGES:
            user_message_history[user_id] = user_message_history[user_id][-MAX_HISTORY_MESSAGES:]

def get_user_conversation_context(user_id):
    if user_id not in user_message_history or not user_message_history[user_id]:
        return ""
    current_time = time.time()

    recent_history = [
        msg for msg in user_message_history[user_id] 
        if (current_time - msg['timestamp']) <= CONVERSATION_TIMEOUT_SECONDS
    ]
    user_message_history[user_id] = recent_history
    
    if not recent_history:
        return "" 
    context_parts = ["Recent conversation history:"]
    for i, msg in enumerate(recent_history):
        seconds_ago = int(current_time - msg['timestamp'])
        if seconds_ago < 60:
            time_str = f"{seconds_ago} seconds ago"
        else:
            minutes_ago = seconds_ago // 60
            time_str = f"{minutes_ago} minute{'s' if minutes_ago != 1 else ''} ago"
        context_parts.append(f"Message {i+1} ({time_str}): {msg['content']}")
    return "\n".join(context_parts)

async def cleanup_expired_histories():
    while True:
        try:
            current_time = time.time()
            users_to_check = list(user_message_history.keys())
            for user_id in users_to_check:
                user_message_history[user_id] = [
                    msg for msg in user_message_history[user_id] 
                    if (current_time - msg['timestamp']) <= CONVERSATION_TIMEOUT_SECONDS
                ]
                if not user_message_history[user_id]:
                    del user_message_history[user_id]
            await asyncio.sleep(600)  
        except Exception as e:
            print(f"Error in cleanup_expired_histories: {e}")
            await asyncio.sleep(600)  

async def get_soup_from_url(url):
    session = await get_aiohttp_session()
    try:
        headers = {
            'User-Agent': 'KTDiscordBot/1.0 (Discord Bot for Knights Templar; https://github.com/YOURHUB)' 
        }
        async with session.get(url, timeout=20, headers=headers) as response:
            response.raise_for_status()
            html_content = await response.text()
            if html_content:
                return BeautifulSoup(html_content, 'html.parser')
            return None
    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
        print(f"Error fetching URL {url}: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error in get_soup_from_url for {url}: {e}")
        return None

def extract_fandom_article_text(soup):
    if not soup: return ""
    
    article_body = soup.select_one('div.mw-parser-output')
    if not article_body: 
        article_body = soup.select_one('#content #mw-content-text .mw-parser-output')
    if article_body:
        elements_to_remove_selectors = [
            '.toc', '.mw-editsection', '.catlinks', '.printfooter', '.nomobile',
            '.mobileonly', 'figure.article-thumb', 'aside.portable-infobox',
            '.navbox', '.infobox', 'div.gallery', 'div.wikia-gallery',
            'div#WikiaRail', 'div#WikiaArticleBottomAd',
            'div.reference', 'ol.references', 'div.video-thumbnail', '.wikia-slideshow',
            '.discord-widget', 'script', 'style', '.lightbox-caption'
        ]
        for selector in elements_to_remove_selectors:
            for unwanted_tag in article_body.select(selector):
                unwanted_tag.decompose()
        text_parts = []
        for element in article_body.children:
            if element.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                header_text = element.get_text(strip=True)
                if header_text:
                    level = int(element.name[1])
                    prefix = "\n" + ("#" * level) + " "
                    text_parts.append(prefix + header_text)
            elif element.name == 'p':
                text = element.get_text(separator=" ", strip=True)
                if len(text.split()) > 3 and not text.lower().startswith("this page refers to"):
                    text_parts.append(text)
            elif element.name == 'table':
                if 'wikitable' in element.get('class', []):
                    table_started = True
                    table_text = []
                    headers = element.select('tr th')
                    if headers:
                        header_row = '|'
                        for th in headers:
                            header_row += f" {th.get_text(strip=True)} |"
                        table_text.append(header_row)
                        table_text.append('|' + '---|' * len(headers) + '|')
                    for tr in element.select('tr'):
                        if not tr.select('th') or len(tr.select('th')) != len(headers):  
                            row_text = '|'
                            for td in tr.select('td'):
                                cell_text = td.get_text(separator=" ", strip=True)
                                row_text += f" {cell_text} |"
                            if row_text != '|':  
                                table_text.append(row_text)
                    
                    if table_text:
                        text_parts.append("\n".join(table_text))
            elif element.name in ['ul', 'ol']:
                list_items = []
                for i, li in enumerate(element.find_all('li', recursive=False)):
                    item_text = li.get_text(separator=" ", strip=True)
                    if len(item_text.split()) > 2:
                        prefix = "• " if element.name == 'ul' else f"{i+1}. "
                        list_items.append(prefix + item_text)
                
                if list_items:
                    text_parts.append("\n".join(list_items))
            elif element.name == 'div' and not any(cls in element.get('class', []) for cls in ['thumb', 'toc']):
                div_text = element.get_text(separator=" ", strip=True)
                if len(div_text.split()) > 10:  
                    text_parts.append(div_text)
        full_text = "\n\n".join(text_parts)
        full_text = re.sub(r'\n{3,}', '\n\n', full_text)
        return full_text[:4000]  

    return ""

def extract_fandom_last_modified_date(soup):
    if not soup: return None
    try:
        history_link_area = soup.select_one('a.page-header__history-button')
        if history_link_area and history_link_area.get('title'):
            match = re.search(r'last edited.*?on\s+([A-Za-z]+\s+\d{1,2},\s+\d{4})', history_link_area.get('title',''), re.IGNORECASE)
            if match: return datetime.datetime.strptime(match.group(1), '%B %d, %Y').date()

        history_button_with_timestamp = soup.select_one('a.page-header__history-button[data-last-edit-timestamp]')
        if history_button_with_timestamp and history_button_with_timestamp.get('data-last-edit-timestamp'):
            timestamp_str = history_button_with_timestamp.get('data-last-edit-timestamp')
            return datetime.datetime.fromtimestamp(int(timestamp_str), tz=datetime.timezone.utc).date()

        meta_modified_time = soup.find('meta', property='article:modified_time')
        if meta_modified_time and meta_modified_time.get('content'):
            return datetime.datetime.fromisoformat(meta_modified_time['content'].replace('Z', '+00:00')).date()
        
        footer_info = soup.select_one('#footer-info-lastmod, .page-footer__last-updated .page-footer__last-updated-text')
        if footer_info:
            date_text = footer_info.get_text()
            for fmt in ('%d %B %Y', '%B %d, %Y'):
                try:
                    match = re.search(r'(\d{1,2}\s+\w+\s+\d{4}|\w+\s+\d{1,2},\s+\d{4})', date_text)
                    if match: return datetime.datetime.strptime(match.group(1), fmt).date()
                except ValueError: continue
        print("DEBUG: Could not find a recognizable last modified date on Fandom page.")
    except Exception as e: print(f"Error parsing Fandom Pedia last modified date: {e}")
    return None

def get_age_disclaimer(last_modified_date_obj, source_name="Fandom Wiki"):
    if last_modified_date_obj:
        current_date_for_comparison = datetime.date.today() 
        six_years_ago = current_date_for_comparison - datetime.timedelta(days=6*365.25)
        if last_modified_date_obj < six_years_ago:
            return f"(Disclaimer: This {source_name} information was last updated on {last_modified_date_obj.strftime('%B %d, %Y')} and may be outdated.)"
    return ""

def get_title_from_url(url):
    try:
        path = url.split('/wiki/')[-1]
        title = unquote(path).replace('_', ' ')
        return title
    except:
        return "Unknown Fandom Page"

async def scrape_fandom_article(article_url, provided_title=None):
    global EMBED_MODEL
    if EMBED_MODEL is None: await load_embed_model()
    if EMBED_MODEL is None: print("CRITICAL: Embedding model not loaded, cannot process Fandom article."); return None
    
    article_soup = await get_soup_from_url(article_url)
    if not article_soup:
        print(f"DEBUG: Failed to get soup for Fandom article: {article_url}")
        return None
    title_tag = article_soup.select_one('h1.page-header__title, h1#firstHeading')
    actual_article_title = title_tag.get_text(strip=True) if title_tag else ""
    last_mod_date_obj = extract_fandom_last_modified_date(article_soup)
    content = extract_fandom_article_text(article_soup)
    if content:
        return {
            "url": article_url,
            "title": actual_article_title,
            "content": content,
            "fandom_last_modified": last_mod_date_obj.isoformat() if last_mod_date_obj else None,
            "embedding": None,
            "last_scraped": datetime.datetime.now(datetime.timezone.utc)
        }
    return None

async def search_guides_with_knn_bm25(
    query: str, 
    k: int = 10,
    similarity_threshold: float = 0.0, 
    use_keywords: bool = True, 
    bm25_weight: float = 0.2 
) -> List[Dict[str, Any]]:
    embedding_manager = get_embedding_manager()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute('''
        SELECT e.id, e.guide_id, e.chunk_text, e.embedding, 
               g.title, g.url 
        FROM guide_embeddings e 
        JOIN guide_documents g ON e.guide_id = g.id
    ''')
    # Create query embedding
    query_embed = await embedding_manager.create_enhanced_embedding([query], task_type="RETRIEVAL_QUERY")
    
    all_data = cursor.fetchall()
    conn.close()
    
    if not all_data:
        print("No embeddings in database.")
        return []
    
    vector_scores = []
    for r in all_data:
        emb_b, txt = r['embedding'], r['chunk_text']
        if emb_b is None or not txt.strip():
            continue
        try:
            emb = np.frombuffer(emb_b, dtype=np.float32)
            query_norm = np.linalg.norm(query_embed)
            emb_norm = np.linalg.norm(emb)
            if query_norm > 0 and emb_norm > 0:
                similarity = np.dot(query_embed, emb) / (np.linalg.norm(query_embed) * np.linalg.norm(emb))
            else:
                similarity = 0
            vector_scores.append({
                'id': r['id'],
                'guide_id': r['guide_id'],
                'title': r['title'],
                'url': r['url'],
                'chunk_text': txt,
                'vector_similarity': float(similarity),
                'doc_data': r
            })
        except Exception as e:
            print(f"Error computing vector similarity: {e}")
    vector_scores.sort(key=lambda x: x['vector_similarity'], reverse=True)
    knn_candidates = vector_scores[:k]
    
    results = []
    
    query_terms = [term.lower() for term in re.findall(r'\b[a-zA-Z]{3,}\b', query) 
                  if term.lower() not in KEYWORD_LIST]
    
    for candidate in knn_candidates:
        doc_text = candidate['chunk_text']
        doc_terms = [term.lower() for term in re.findall(r'\b[a-zA-Z]{3,}\b', doc_text) 
                    if term.lower() not in KEYWORD_LIST]
        
        bm25_score = embedding_manager.calculate_bm25(query_terms, doc_terms, 'guide')
        
        normalized_bm25 = min(1.0, bm25_score / 15.0)
        
        keyword_sim = 0
        if use_keywords:
            query_keywords = set(query_terms)
            doc_keywords = set(doc_terms)
            
            if query_keywords:
                matches = query_keywords.intersection(doc_keywords)
                keyword_sim = len(matches) / len(query_keywords)
                
                title_words = set([term.lower() for term in re.findall(r'\b[a-zA-Z]{3,}\b', candidate['title']) 
                              if term.lower() not in KEYWORD_LIST])
                title_matches = query_keywords.intersection(title_words)
                if title_matches:
                    keyword_sim += len(title_matches) / len(query_keywords)
        
        combined_sim = (
            0.8 * candidate['vector_similarity'] + 
            bm25_weight * normalized_bm25)
        
        results.append({
            'id': candidate['id'],
            'guide_id': candidate['guide_id'],
            'title': candidate['title'],
            'url': candidate['url'],
            'chunk_text': candidate['chunk_text'],
            'vector_similarity': float(candidate['vector_similarity']),
            'bm25_score': float(normalized_bm25),
            'keyword_similarity': float(keyword_sim),
            'similarity': float(combined_sim)
        })
    
    results.sort(key=lambda x: x['similarity'], reverse=True)
    
    results = [r for r in results if r["similarity"] >= similarity_threshold]
    
    if not results and knn_candidates:
        results.append({
            'id': knn_candidates[0]['id'],
            'guide_id': knn_candidates[0]['guide_id'],
            'title': knn_candidates[0]['title'],
            'url': knn_candidates[0]['url'],
            'chunk_text': knn_candidates[0]['chunk_text'],
            'vector_similarity': float(knn_candidates[0]['vector_similarity']),
            'bm25_score': 0.0,
            'keyword_similarity': 0.0,
            'similarity': float(knn_candidates[0]['vector_similarity'])
        })
    
    return results

class EnhancedEmbeddingManager:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self.corpus_stats = {} 
        self._init_corpus_stats()
    
    def _init_corpus_stats(self):
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("SELECT COUNT(*) FROM guide_embeddings")
            self.corpus_stats['guide_count'] = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM fandom_articles")
            self.corpus_stats['fandom_count'] = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM forum_threads")
            self.corpus_stats['forum_count'] = cursor.fetchone()[0]
            
            cursor.execute("SELECT AVG(LENGTH(chunk_text)) FROM guide_embeddings")
            self.corpus_stats['guide_avg_length'] = cursor.fetchone()[0]
            
            cursor.execute("SELECT AVG(LENGTH(content)) FROM fandom_articles")
            self.corpus_stats['fandom_avg_length'] = cursor.fetchone()[0]
            
            cursor.execute("SELECT AVG(LENGTH(content)) FROM forum_threads")
            self.corpus_stats['forum_avg_length'] = cursor.fetchone()[0]
            self.corpus_stats['term_freq'] = {}
        
            conn.close()
        except Exception as e:
            print(f"Error initializing corpus stats: {e}")
            self.corpus_stats = {
                'guide_count': 100,
                'fandom_count': 50,
                'forum_count': 50,
                'guide_avg_length': 1200,
                'fandom_avg_length': 1000,
                'forum_avg_length': 800,
                'term_freq': {}
            }
    
    def _get_term_frequency(self, term: str, content_type: str = 'guide') -> int:
        cache_key = f"{term}_{content_type}"
        if cache_key in self.corpus_stats['term_freq']:
            return self.corpus_stats['term_freq'][cache_key]
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            if content_type == 'guide':
                query = """
                    SELECT COUNT(*) FROM guide_embeddings 
                    WHERE chunk_text LIKE ?
                """
            elif content_type == 'fandom':
                query = """
                    SELECT COUNT(*) FROM fandom_articles 
                    WHERE content LIKE ?
                """
            elif content_type == 'forum':
                query = """
                    SELECT COUNT(*) FROM forum_threads 
                    WHERE content LIKE ?
                """
            else:
                return 1  
            
            cursor.execute(query, (f'%{term}%',))
            freq = cursor.fetchone()[0]
            
            conn.close()
            self.corpus_stats['term_freq'][cache_key] = freq
            return freq
        except Exception as e:
            print(f"Error getting term frequency: {e}")
            return 1
    
    def calculate_bm25(self, query_terms: List[str], 
                      doc_terms: List[str], 
                      content_type: str = 'guide') -> float:
        # BM25 parameters
        k1 = 1.3  
        b = 0.8 
        if content_type == 'guide':
            doc_count = self.corpus_stats['guide_count']
            avg_doc_len = self.corpus_stats['guide_avg_length']
        elif content_type == 'fandom':
            doc_count = self.corpus_stats['fandom_count']
            avg_doc_len = self.corpus_stats['fandom_avg_length']
        elif content_type == 'forum':
            doc_count = self.corpus_stats['forum_count']
            avg_doc_len = self.corpus_stats['forum_avg_length']
        else:
            doc_count = 100
            avg_doc_len = 500
        doc_len = len(" ".join(doc_terms))
        doc_term_freq = Counter(doc_terms)
        score = 0.0
        for term in query_terms:
            if term in doc_term_freq:
                tf = doc_term_freq[term]
                df = self._get_term_frequency(term, content_type)
                idf = max(0, math.log((doc_count - df + 0.5) / (df + 0.5)))
                # BM25 calculation for this term
                term_score = idf * ((tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_len / avg_doc_len))))
                score += term_score
        return score
    
    def _extract_context_features(self, text: str) -> str:
        headings = re.findall(r'(?:^|\n)#+\s+(.+?)(?:\n|$)', text)
        entities = re.findall(r'\b[A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)*\b', text)
        numbers = re.findall(r'\b\d+(?:\.\d+)?%?\b', text)
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
        word_counts = {}
        for word in words:
            if word not in KEYWORD_LIST:
                word_counts[word] = word_counts.get(word, 0) + 1
        top_keywords = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        keywords = [word for word, _ in top_keywords]
        
        context_parts = []
        if headings:
            context_parts.append("HEADINGS: " + ", ".join(headings))
        if entities:
            context_parts.append("ENTITIES: " + ", ".join(list(set(entities))[:15]))  
        if keywords:
            context_parts.append("KEYWORDS: " + ", ".join(keywords))
        if numbers:
            context_parts.append("VALUES: " + ", ".join(numbers[:8]))  
        context_text = " | ".join(context_parts)
        return context_text
    
    async def create_enhanced_embedding(self, 
                                      text_list: List[str], 
                                      task_type: str = None) -> Union[np.ndarray, List[np.ndarray]]:
        EXPECTED_DIM = 768
        is_query = task_type == "RETRIEVAL_QUERY" if task_type else len(text_list) == 1
        if not is_query and not text_list or all(not text or len(text.strip()) < 5 for text in text_list):
            if len(text_list) == 1:
                return np.zeros(768, dtype=np.float32)
            return [np.zeros(768, dtype=np.float32) for _ in range(len(text_list))]
        
        embeddings = []
        for text in text_list:
            if not is_query and not text or len(text.strip()) < 5:
                embeddings.append(np.zeros(768, dtype=np.float32))
                continue    
            try:
                if not is_query and len(text) > 200:
                    context_text = self._extract_context_features(text)
                    primary_embedding = await asyncio.to_thread(
                        genai.embed_content,
                        model="models/text-embedding-004",
                        content=text,
                        task_type="RETRIEVAL_DOCUMENT"
                    )
                    primary_vector = np.array(primary_embedding["embedding"], dtype=np.float32)
                    if primary_vector.shape[0] != EXPECTED_DIM:
                        print(f"Warning: Primary vector dimension {primary_vector.shape[0]} doesn't match expected {EXPECTED_DIM}")
                        if primary_vector.shape[0] > EXPECTED_DIM:
                            primary_vector = primary_vector[:EXPECTED_DIM]
                        else:
                            padding = np.zeros(EXPECTED_DIM - primary_vector.shape[0], dtype=np.float32)
                            primary_vector = np.concatenate([primary_vector, padding])

                    if context_text:
                        context_embedding = await asyncio.to_thread(
                            genai.embed_content,
                            model="models/text-embedding-004",
                            content=context_text,
                            task_type="RETRIEVAL_DOCUMENT"
                        )
                        context_vector = np.array(context_embedding["embedding"], dtype=np.float32)
                        combined_vector = 0.8 * primary_vector + 0.2 * context_vector
                        norm = np.linalg.norm(combined_vector)
                        if norm > 0:
                            combined_vector = combined_vector / norm
                        
                        embeddings.append(combined_vector)
                    else:
                        embeddings.append(primary_vector)
                else:
                    if is_query:
                        embedding = await asyncio.to_thread(
                            genai.embed_content,
                            model="models/text-embedding-004",
                            content=text,
                            task_type="RETRIEVAL_QUERY"
                        )
                        vector = np.array(embedding["embedding"], dtype=np.float32)
                    else:
                        embedding = await asyncio.to_thread(
                            genai.embed_content,
                            model="models/text-embedding-004",
                            content=text,
                            task_type="RETRIEVAL_DOCUMENT"
                        )
                        vector = np.array(embedding["embedding"], dtype=np.float32)
                    if vector.shape[0] > EXPECTED_DIM:
                        vector = vector[:EXPECTED_DIM]
                    else:
                        padding = np.zeros(EXPECTED_DIM - vector.shape[0], dtype=np.float32)
                        vector = np.concatenate([vector, padding])
                    embeddings.append(vector)
            except Exception as e:
                print(f"Error creating enhanced embedding: {e}")
                embeddings.append(np.zeros(768, dtype=np.float32))
        return embeddings[0] if len(embeddings) == 1 else embeddings

def create_embeddings_for_guides():
    """Create embeddings for guides using the enhanced embedding system"""
    embedding_manager = get_embedding_manager()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT g.id, g.content 
        FROM guide_documents g 
        WHERE NOT EXISTS (
            SELECT 1 FROM guide_embeddings e WHERE e.guide_id = g.id
        )
    ''')
    
    guides = cursor.fetchall()
    if not guides:
        print("No new/updated guides for embeddings.")
        conn.close()
        return True
    print(f"Found {len(guides)} guides for embeddings.")
    for gid, cont in guides:
        chunks = split_into_chunks(cont)
        if not chunks:
            print(f"Warning: No chunks for guide_id {gid}.")
            continue
        print(f"Processing guide_id {gid}: {len(chunks)} chunks.")
        try:
            embeds = asyncio.run(embedding_manager.create_enhanced_embedding(
                chunks, 
                task_type="RETRIEVAL_DOCUMENT"
            ))
            for txt, emb_arr in zip(chunks, embeds):
                if len(txt.strip()) < 10:
                    continue
                emb_bytes = emb_arr.tobytes()
                
                cursor.execute(
                    'INSERT INTO guide_embeddings (guide_id, chunk_text, embedding) VALUES (?, ?, ?)', 
                    (gid, txt, emb_bytes)
                )
            conn.commit()
        except Exception as e:
            print(f"Error creating embeddings for guide_id {gid}: {e}")
    conn.close()
    print("Enhanced embeddings creation complete.")
    return True

async def search_fandom_with_knn_bm25(
    query: str, 
    k: int = 10,
    similarity_threshold: float = 0.0, 
    use_keywords: bool = True,
    bm25_weight: float = 0.2
)->List[Dict[str, Any]]:
    """
    Enhanced search for Fandom articles using K-nearest neighbors, vector similarity, and BM25
    
    Args:
        query: The search query string
        k: Number of nearest neighbors to retrieve (default: 10)
        similarity_threshold: Minimum similarity score (default: 0.6)
        use_keywords: Whether to use keyword matching (default: True)
        bm25_weight: Weight of BM25 in final score (default: 0.15)
        
    Returns:
        List of search results
    """
    embedding_manager = get_embedding_manager()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute('''
        SELECT url, title, content, fandom_last_modified, embedding 
        FROM fandom_articles 
        WHERE embedding IS NOT NULL
    ''')
    articles_data = cursor.fetchall()
    query_embed = await embedding_manager.create_enhanced_embedding([query], task_type="RETRIEVAL_QUERY")
    conn.close()
    if not articles_data:
        print("No cached Fandom articles with embeddings found.")
        return []
    vector_scores = []
    for r in articles_data:
        emb_b, content = r['embedding'], r['content']
        if emb_b is None or not content.strip():
            continue
        try:
            emb = np.frombuffer(emb_b, dtype=np.float32)
            query_norm = np.linalg.norm(query_embed)
            emb_norm = np.linalg.norm(emb)
            if query_norm > 0 and emb_norm > 0:
                similarity = np.dot(query_embed, emb) / (np.linalg.norm(query_embed) * np.linalg.norm(emb))
            else:
                similarity = 0
            vector_scores.append({
                'url': r['url'],
                'title': r['title'],
                'content': r['content'],
                'fandom_last_modified': r['fandom_last_modified'],
                'vector_similarity': float(similarity),
                'doc_data': r
            })
        except Exception as e:
            print(f"Error computing vector similarity for fandom article: {e}")
    vector_scores.sort(key=lambda x: x['vector_similarity'], reverse=True)
    knn_candidates = vector_scores[:k]
    
    results = []
    query_terms = [term.lower() for term in re.findall(r'\b[a-zA-Z]{3,}\b', query) 
                  if term.lower() not in KEYWORD_LIST]
    for candidate in knn_candidates:
        doc_text = candidate['content']
        doc_terms = [term.lower() for term in re.findall(r'\b[a-zA-Z]{3,}\b', doc_text) 
                    if term.lower() not in KEYWORD_LIST]
        
        # Calculate BM25 score
        bm25_score = embedding_manager.calculate_bm25(query_terms, doc_terms, 'fandom')
        normalized_bm25 = min(1.0, bm25_score / 15.0)
        keyword_sim = 0
        if use_keywords:
            query_keywords = set(query_terms)
            doc_keywords = set(doc_terms)
            if query_keywords:
                matches = query_keywords.intersection(doc_keywords)
                keyword_sim = len(matches) / len(query_keywords)
                
                title_words = set([term.lower() for term in re.findall(r'\b[a-zA-Z]{3,}\b', candidate['title']) 
                              if term.lower() not in KEYWORD_LIST])
                title_matches = query_keywords.intersection(title_words)
                if title_matches:
                    keyword_sim += 0.2 * len(title_matches) / len(query_keywords)
        combined_sim = (
            0.8 * candidate['vector_similarity'] + 
            bm25_weight * normalized_bm25
        )
        results.append({
            'url': candidate['url'],
            'title': candidate['title'],
            'content': f"{candidate['content']}".strip(),
            'raw_content': candidate['content'],
            'fandom_last_modified_str': candidate['fandom_last_modified'],
            'vector_similarity': float(candidate['vector_similarity']),
            'bm25_score': float(normalized_bm25),
            'keyword_similarity': float(keyword_sim),
            'similarity': float(combined_sim)
        })
    results.sort(key=lambda x: x['similarity'], reverse=True)
    results = [r for r in results if r["similarity"] >= similarity_threshold]
    if not results and knn_candidates:
        disclaimer = ""
        if knn_candidates[0]['fandom_last_modified']:
            try:
                mod_date = datetime.date.fromisoformat(knn_candidates[0]['fandom_last_modified'])
                current_date = datetime.date(2025, 5, 12)
                years_old = (current_date - mod_date).days / 365.25
                if years_old > 6:
                    disclaimer = f"(Disclaimer: This P&W Fandom Wiki page '{knn_candidates[0]['title']}' was last updated on {mod_date.strftime('%B %d, %Y')} and may be outdated.)"
            except:
                pass
                
        results.append({
            'url': knn_candidates[0]['url'],
            'title': knn_candidates[0]['title'],
            'content': f"{knn_candidates[0]['content']}\n{disclaimer}".strip(),
            'raw_content': knn_candidates[0]['content'],
            'fandom_last_modified_str': knn_candidates[0]['fandom_last_modified'],
            'vector_similarity': float(knn_candidates[0]['vector_similarity']),
            'bm25_score': 0.0,
            'keyword_similarity': 0.0,
            'similarity': float(knn_candidates[0]['vector_similarity'])
        })
    return results

async def search_forum_with_knn_bm25(
    query: str, 
    k: int = 10,
    similarity_threshold: float = 0.0, 
    use_keywords: bool = True,
    bm25_weight: float = 0.2
)->List[Dict[str, Any]]:
    """
    Enhanced search for forum content using K-nearest neighbors, vector similarity, and BM25
    
    Args:
        query: The search query string
        k: Number of nearest neighbors to retrieve (default: 10)
        similarity_threshold: Minimum similarity score (default: 0.55)
        use_keywords: Whether to use keyword matching (default: True)
        bm25_weight: Weight of BM25 in final score (default: 0.15)
        
    Returns:
        List of search results
    """
    embedding_manager = EnhancedEmbeddingManager()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute('''
        SELECT thread_id, title, url, author, post_date, content, embedding 
        FROM forum_threads 
        WHERE embedding IS NOT NULL
    ''')
    threads = cursor.fetchall()
    conn.close()
    if not threads:
            print("No forum threads with embeddings found.")
            return []
    # Create query embedding
    query_embed = await embedding_manager.create_enhanced_embedding([query], task_type="RETRIEVAL_QUERY")
    try:
        vector_scores = []
        for r in threads:
            emb_b, content = r['embedding'], r['content']
            if emb_b is None or not content.strip():
                continue
            try:
                emb = np.frombuffer(emb_b, dtype=np.float32)
                query_norm = np.linalg.norm(query_embed)
                emb_norm = np.linalg.norm(emb)
                if query_norm > 0 and emb_norm > 0:
                    similarity = np.dot(query_embed, emb) / (np.linalg.norm(query_embed) * np.linalg.norm(emb))
                else:
                    similarity = 0
                vector_scores.append({
                    'thread_id': r['thread_id'],
                    'title': r['title'],
                    'url': r['url'],
                    'author': r['author'],
                    'post_date': r['post_date'],
                    'content': r['content'],
                    'vector_similarity': float(similarity),
                    'doc_data': r
                })
            except Exception as e:
                print(f"Error computing vector similarity for forum thread: {e}")
        vector_scores.sort(key=lambda x: x['vector_similarity'], reverse=True)
        knn_candidates = vector_scores[:k]
        results = []
        query_terms = [term.lower() for term in re.findall(r'\b[a-zA-Z]{3,}\b', query) 
                      if term.lower() not in KEYWORD_LIST]
        
        for candidate in knn_candidates:
            doc_text = candidate['content']
            doc_terms = [term.lower() for term in re.findall(r'\b[a-zA-Z]{3,}\b', doc_text) 
                        if term.lower() not in KEYWORD_LIST]
            # Calculate BM25 score
            bm25_score = embedding_manager.calculate_bm25(query_terms, doc_terms, 'forum')
            normalized_bm25 = min(1.0, bm25_score / 15.0)
            keyword_sim = 0
            if use_keywords:
                query_keywords = set(query_terms)
                doc_keywords = set(doc_terms)
                if query_keywords:
                    matches = query_keywords.intersection(doc_keywords)
                    keyword_sim = len(matches) / len(query_keywords)
                    title_words = set([term.lower() for term in re.findall(r'\b[a-zA-Z]{3,}\b', candidate['title']) 
                                  if term.lower() not in KEYWORD_LIST])
                    title_matches = query_keywords.intersection(title_words)
                    if title_matches:
                        keyword_sim += 0.4 * len(title_matches) / len(query_keywords)
            combined_sim = (
                (0.80 * candidate['vector_similarity'] + 
                bm25_weight * normalized_bm25)
            )
            results.append({
                'id': candidate['thread_id'],
                'title': candidate['title'],
                'url': candidate['url'],
                'author': candidate['author'],
                'post_date': candidate['post_date'],
                'content': candidate['content'],
                'vector_similarity': float(candidate['vector_similarity']),
                'bm25_score': float(normalized_bm25),
                'keyword_similarity': float(keyword_sim),
                'similarity': float(combined_sim),
                'type': 'thread'
            })
        results.sort(key=lambda x: x['similarity'], reverse=True)
        results = [r for r in results if r["similarity"] >= similarity_threshold]
        
        if not results and knn_candidates:
            results.append({
                'id': knn_candidates[0]['thread_id'],
                'title': knn_candidates[0]['title'],
                'url': knn_candidates[0]['url'],
                'author': knn_candidates[0]['author'],
                'post_date': knn_candidates[0]['post_date'],
                'content': knn_candidates[0]['content'],
                'vector_similarity': float(knn_candidates[0]['vector_similarity']),
                'bm25_score': 0.0,
                'keyword_similarity': 0.0,
                'similarity': float(knn_candidates[0]['vector_similarity']),
                'type': 'thread'
            })
        return results
    except Exception as e:
        print(f"Error in search_forum_with_knn_bm25: {e}")
        return []
    finally:
        conn.close()

async def fetch_all_fandom_page_urls(api_url, base_wiki_url, namespace="0", max_pages=None):
    session = await get_aiohttp_session()
    all_page_urls = []
    params = {
        "action": "query",
        "list": "allpages",
        "apnamespace": namespace,
        "aplimit": "max", 
        "format": "json"
    }
    apcontinue = None
    processed_pages = 0
    headers = {
        'User-Agent': 'KTDiscordBot/1.0 (Discord Bot for Knights Templar'
    }
    print(f"Starting to fetch all page URLs from namespace {namespace}...")
    while True:
        if apcontinue:
            params["apcontinue"] = apcontinue
        try:
            async with session.get(api_url, params=params, headers=headers, timeout=30) as response:
                response.raise_for_status()
                data = await response.json()

                if "query" in data and "allpages" in data["query"]:
                    for page in data["query"]["allpages"]:
                        page_title = page["title"]
                        full_url = urljoin(base_wiki_url, quote_plus(page_title.replace(" ", "_")))
                        all_page_urls.append(full_url)
                        processed_pages += 1
                        if max_pages and processed_pages >= max_pages:
                            print(f"Reached max_pages limit ({max_pages}). Stopping URL fetch.")
                            return list(set(all_page_urls)) 

                if "continue" in data and "apcontinue" in data["continue"]:
                    apcontinue = data["continue"]["apcontinue"]
                else:
                    break
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            print(f"API Error fetching page list: {e}. Returning what has been collected so far.")
            break 
        except Exception as e:
            print(f"Unexpected API error: {e}. Returning what has been collected so far.")
            break

    print(f"Finished fetching. Found {len(all_page_urls)} unique page URLs from namespace {namespace}.")
    return list(set(all_page_urls)) 

def setup_logs_database():
    conn = sqlite3.connect("bot_logs.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS interaction_logs (
            id INTEGER PRIMARY KEY,
            user_id TEXT,
            username TEXT,
            user_query TEXT,
            context_used TEXT,
            ai_response TEXT,
            rating INTEGER DEFAULT 0,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()
    print("Logs database initialized")

@bot.event
async def on_ready():
    global AIOHTTP_SESSION
    AIOHTTP_SESSION = aiohttp.ClientSession()
    print(f'{bot.user.name} has connected to Discord!')
    setup_database()
    setup_logs_database()  
    await load_embed_model()
    bot.loop.create_task(cleanup_expired_histories())
    bot.loop.create_task(monitor_heartbeat())
    print(f"Bot is ready. Prefix is mention (@{bot.user.name}).")

@bot.event
async def on_disconnect():
    global AIOHTTP_SESSION
    if AIOHTTP_SESSION and not AIOHTTP_SESSION.closed:
        await AIOHTTP_SESSION.close()
        AIOHTTP_SESSION = None
        print("AIOHTTP_SESSION closed on disconnect.")

@bot.event
async def on_closed():
    global AIOHTTP_SESSION
    if AIOHTTP_SESSION and not AIOHTTP_SESSION.closed:
        await AIOHTTP_SESSION.close()
        AIOHTTP_SESSION = None
        print("AIOHTTP_SESSION closed on bot closure.")

@bot.event
async def on_message(message):
    await bot.process_commands(message)
    if message.author.bot:
        return
    prefixes = await bot.get_prefix(message)
    prefix_list = list(prefixes) if isinstance(prefixes, (list, tuple)) else [prefixes]
    used_prefix = None
    for prefix in prefix_list:
        if message.content.startswith(prefix):
            used_prefix = prefix
            break
    if not used_prefix:
        return
    after_prefix = message.content[len(used_prefix):].strip()
    if not after_prefix:
        return
    parts = after_prefix.split(None, 1)
    potential_cmd = parts[0].lower() if parts else ""
    if bot.get_command(potential_cmd) is not None:
        return
    query = after_prefix
    if message.reference and message.reference.message_id:
        try:
            replied_msg = await message.channel.fetch_message(message.reference.message_id)
            if replied_msg and replied_msg.content:
                ref_txt = replied_msg.content.strip()
                query = f"{query}\n\n(In reference to: \"{ref_txt}\")"
        except Exception as e:
            print(f"DEBUG: Error processing replied message: {e}")
    
    ask_cmd = bot.get_command("ask")
    if not ask_cmd:
        print("ERROR: 'ask' command not found!")
        return

    ctx = await bot.get_context(message)
    
    try:
        await ask_cmd(ctx, question=query)
    except Exception as e:
        print(f"Error executing ask command: {e}")
        import traceback
        traceback.print_exc()
            
def setup_database():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS guide_documents (
        id INTEGER PRIMARY KEY, title TEXT NOT NULL, url TEXT NOT NULL, content TEXT NOT NULL,
        doc_id TEXT NOT NULL UNIQUE, last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )''')
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS guide_embeddings (
        id INTEGER PRIMARY KEY, guide_id INTEGER, chunk_text TEXT NOT NULL, embedding BLOB NOT NULL,
        FOREIGN KEY (guide_id) REFERENCES guide_documents (id) ON DELETE CASCADE
    )''')
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS fandom_articles (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        url TEXT NOT NULL UNIQUE,
        title TEXT NOT NULL,
        content TEXT NOT NULL,
        fandom_last_modified TEXT,
        last_scraped TIMESTAMP NOT NULL,
        embedding BLOB
    )''')
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS forum_threads (
        id INTEGER PRIMARY KEY,
        thread_id TEXT NOT NULL UNIQUE,
        title TEXT NOT NULL,
        url TEXT NOT NULL,
        author TEXT,
        post_date TIMESTAMP,
        last_updated TIMESTAMP,
        content TEXT NOT NULL,
        embedding BLOB,
        last_scraped TIMESTAMP NOT NULL
    )''')
    conn.commit(); conn.close(); print("DB setup complete (fandom_articles table includes embedding column).")

async def load_embed_model():
    global EMBED_MODEL, GEMINI_API_KEY
    if not GEMINI_API_KEY:
        print("ERROR: GEMINI_API_KEY not set. Cannot load embedding model.")
        return
    
    print("Configuring Gemini embedding model...")
    genai.configure(api_key=GEMINI_API_KEY)
    print("Gemini embedding model configured.")
    EMBED_MODEL = "READY"  

def get_google_service():
    creds = None
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token: creds = pickle.load(token)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token: creds.refresh(Request())
        else:
            if not os.path.exists('credentials.json'): raise FileNotFoundError("Google API 'credentials.json' not found.")
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES); creds = flow.run_local_server(port=0)
        with open('token.pickle', 'wb') as token: pickle.dump(creds, token)
    return build('docs', 'v1', credentials=creds), build('drive', 'v3', credentials=creds)

def list_beginner_guides_from_drive(drive_service, folder_id=None, query_filter=None):
    if not query_filter:
        base = "mimeType='application/vnd.google-apps.document'"
        query_filter = f"'{folder_id}' in parents and {base}" if folder_id else f"{base} and (name contains 'guide' or name contains 'beginner' or name contains 'tutorial' or name contains 'PnW')"
    return drive_service.files().list(q=query_filter, spaces='drive', fields='files(id, name, webViewLink, modifiedTime)').execute().get('files', [])

def add_google_docs_to_database(docs_list_content):
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        docs_added = 0
        docs_updated = 0
        docs_skipped = 0
        docs_errored = 0
        for doc_data in docs_list_content:
            try:
                if not all(k in doc_data for k in ['doc_id', 'title', 'content', 'url']):
                    print(f"Error: Missing required fields for document. Fields present: {list(doc_data.keys())}")
                    docs_errored += 1
                    continue
                if not doc_data['content'] or len(doc_data['content'].strip()) < 10:
                    print(f"Warning: Document '{doc_data['title']}' has little or no content. Skipping.")
                    docs_skipped += 1
                    continue
                cursor.execute("SELECT id, last_updated FROM guide_documents WHERE doc_id = ?", 
                              (doc_data['doc_id'],))
                existing = cursor.fetchone()
                mod_time_str = doc_data.get('modifiedTime')
                db_ts = None
                dt_obj = None
                if mod_time_str:
                    try:
                        if 'Z' in mod_time_str:
                            dt_obj = datetime.datetime.fromisoformat(mod_time_str.replace('Z', '+00:00'))
                        elif 'T' in mod_time_str and '+' not in mod_time_str and '-' not in mod_time_str[10:]:
                            dt_obj = datetime.datetime.fromisoformat(mod_time_str).replace(tzinfo=datetime.timezone.utc)
                        else:
                            dt_obj = datetime.datetime.fromisoformat(mod_time_str)
                        
                        db_ts = dt_obj.strftime('%Y-%m-%d %H:%M:%S')
                    except ValueError:
                        print(f"Warning: Invalid modification time '{mod_time_str}' for doc '{doc_data['title']}' (ID: {doc_data['doc_id']})")
                        dt_obj = datetime.datetime.now(datetime.timezone.utc)
                        db_ts = dt_obj.strftime('%Y-%m-%d %H:%M:%S')
                else:
                    dt_obj = datetime.datetime.now(datetime.timezone.utc)
                    db_ts = dt_obj.strftime('%Y-%m-%d %H:%M:%S')
                if existing:
                    db_doc_id, db_lu_str = existing
                    needs_update = True
                    
                    if dt_obj and db_lu_str:
                        try:
                            db_dt_obj = datetime.datetime.strptime(db_lu_str, '%Y-%m-%d %H:%M:%S')
                            if db_dt_obj.tzinfo is None:
                                db_dt_obj = db_dt_obj.replace(tzinfo=datetime.timezone.utc)
                            if dt_obj <= db_dt_obj:
                                print(f"Skipping update for '{doc_data['title']}' - not modified since last update")
                                needs_update = False
                                docs_skipped += 1
                        except ValueError:
                            print(f"Warning: Invalid database timestamp '{db_lu_str}' for document '{doc_data['title']}' (ID: {doc_data['doc_id']})")
                    if needs_update:
                        print(f"Updating guide '{doc_data['title']}' in database")
                        cursor.execute('''
                            UPDATE guide_documents 
                            SET title=?, url=?, content=?, last_updated=? 
                            WHERE doc_id=?
                        ''', (
                            doc_data['title'],
                            doc_data['url'],
                            doc_data['content'],
                            db_ts,
                            doc_data['doc_id']
                        ))
                        cursor.execute("DELETE FROM guide_embeddings WHERE guide_id=?", (db_doc_id,))
                        docs_updated += 1
                else:
                    print(f"Adding new guide '{doc_data['title']}' to database")
                    cursor.execute('''
                        INSERT INTO guide_documents 
                        (title, url, content, doc_id, last_updated) 
                        VALUES (?, ?, ?, ?, ?)
                    ''', (
                        doc_data['title'],
                        doc_data['url'],
                        doc_data['content'],
                        doc_data['doc_id'],
                        db_ts
                    ))
                    docs_added += 1
                conn.commit()
                
            except Exception as e:
                print(f"Error processing document '{doc_data.get('title', 'Unknown')}': {str(e)}")
                docs_errored += 1
        print(f"Database update complete: {docs_added} added, {docs_updated} updated, {docs_skipped} skipped, {docs_errored} errors")
        
    except sqlite3.Error as db_error:
        print(f"Database error: {str(db_error)}")
        if conn:
            conn.rollback()
    except Exception as e:
        print(f"Unexpected error in add_google_docs_to_database: {str(e)}")
        if conn:
            conn.rollback()
    finally:
        if conn:
            conn.close()
            
    return docs_added > 0 or docs_updated > 0  

def extract_text_from_google_doc(docs_service, doc_id):
    document = docs_service.documents().get(documentId=doc_id).execute()
    doc_content = document.get('body').get('content')
    text_content = ""
    
    def read_elements(elements, indent_level=0, is_list_item=False):
        text = ""
        for value in elements:
            if 'paragraph' in value:
                para = value.get('paragraph')
                if 'paragraphStyle' in para and 'namedStyleType' in para.get('paragraphStyle'):
                    style = para.get('paragraphStyle').get('namedStyleType')
                    if 'HEADING' in style:
                        heading_level = int(style[-1]) if style[-1].isdigit() else 1
                        prefix = '#' * heading_level + ' '
                    else:
                        prefix = ''
                else:
                    prefix = ''
                if 'bullet' in para:
                    bullet_info = para.get('bullet')
                    list_id = bullet_info.get('listId', '')
                    nesting_level = bullet_info.get('nestingLevel', 0)
                    indent = "  " * nesting_level
                    list_def = document.get('lists', {}).get(list_id, {})
                    list_type = list_def.get('listProperties', {}).get('nestingLevels', [{}])[0].get('glyphType', '')
                    
                    if 'NUMBER' in list_type:
                        prefix = f"{indent}{nesting_level+1}. "
                    else:
                        prefix = f"{indent}• "
                    
                    is_list_item = True
                elif not prefix:
                    prefix = ""
                para_text = ""
                for elem in para.get('elements'):
                    if 'textRun' in elem:
                        text_obj = elem.get('textRun')
                        content = text_obj.get('content', '')
                        text_style = text_obj.get('textStyle', {})
                        if text_style.get('bold'):
                            content = f"**{content}**"
                        if text_style.get('italic'):
                            content = f"*{content}*"
                        if text_style.get('underline'):
                            content = f"_{content}_"
                        para_text += content
                    elif 'pageBreak' in elem:
                        para_text += '\n'
                if para_text.strip():
                    text += prefix + para_text
                    if not text.endswith('\n'):
                        text += '\n'
            elif 'table' in value:
                table = value.get('table')
                rows = table.get('tableRows', [])
                has_header = table.get('tableStyle', {}).get('firstRowHeading', False)
                
                for row_idx, row in enumerate(rows):
                    cells = row.get('tableCells', [])
                    row_text = '|'
                    
                    for cell in cells:
                        cell_content = read_elements(cell.get('content', []), indent_level + 1)
                        cell_content = cell_content.replace('\n', ' ').strip()
                        row_text += f" {cell_content} |"
                    
                    text += row_text + '\n'
                    if row_idx == 0 and has_header:
                        text += '|' + '---|' * len(cells) + '\n'
                text += '\n'  
            elif 'sectionBreak' in value:
                text += '\n---\n'
            elif 'tableOfContents' in value:
                text += '\n### Table of Contents\n'
            elif 'content' in value:
                text += read_elements(value.get('content'), indent_level)
        return text
    if doc_content:
        text_content = read_elements(doc_content)
    
    text_content = re.sub(r'\n{3,}', '\n\n', text_content).strip()
    return {
        'title': document.get('title', 'Untitled'),
        'content': text_content,
        'url': f"https://docs.google.com/document/d/{doc_id}/edit"
    }

def split_into_chunks(content, max_chunk_size=360, overlap=75):
    special_sections = []
    toc_pattern = r'(^|\n)(Table of Contents|Contents|TOC).*?(?=\n\n|\n#|\Z)'
    table_pattern = r'(^|\n)(\|.*?\|.*?\n)+(?=\n|\Z)'
    for pattern, section_type in [(toc_pattern, 'toc'), (table_pattern, 'table')]:
        for match in re.finditer(pattern, content, re.DOTALL | re.IGNORECASE):
            start = match.start()
            if match.group(1) == '\n': 
                start += 1  
            end = match.end()
            text = content[start:end]
            special_sections.append({
                'type': section_type,
                'text': text,
                'start': start,
                'end': end
            })
    special_sections.sort(key=lambda x: x['start'])

    placeholder_content = content
    offset = 0  
    
    for i, section in enumerate(special_sections):
        placeholder = f"[[SPECIAL_SECTION_{i}]]"
        adj_start = section['start'] + offset
        adj_end = section['end'] + offset
        
        before = placeholder_content[:adj_start]
        after = placeholder_content[adj_end:]
        placeholder_content = before + placeholder + after
        
        offset += len(placeholder) - (adj_end - adj_start)
    paras = placeholder_content.split('\n\n')
    sentences = []
    for p in paras:
        p_trimmed = p.strip()
        if p_trimmed and (p_trimmed.startswith(('- ', '* ', '• ', '1. ', '2. ')) or 
                         p_trimmed.startswith('#')):
            sentences.append(p.replace('\n', ' '))
        else:
            p_parts = re.split(r'(?<=[.!?])\s+', p.replace('\n', ' '))
            sentences.extend([part for part in p_parts if part.strip()])
    chunks = []
    cur_chunk_toks = []
    cur_len = 0
    for s in sentences:
        s_trimmed = s.strip()
        if s_trimmed.startswith('[[SPECIAL_SECTION_') and s_trimmed.endswith(']]'):
            if cur_chunk_toks:
                chunks.append(" ".join(cur_chunk_toks))
                cur_chunk_toks = []
                cur_len = 0
            
            section_idx = int(re.search(r'SPECIAL_SECTION_(\d+)', s_trimmed).group(1))
            section_text = special_sections[section_idx]['text']
            chunks.append(section_text)
            continue

        if not s_trimmed:
            continue
            
        s_len = len(s_trimmed.split())
        if cur_len + s_len <= max_chunk_size:
            cur_chunk_toks.append(s_trimmed)
            cur_len += s_len
        else:
            if cur_chunk_toks:
                chunks.append(" ".join(cur_chunk_toks))
            
            if overlap > 0 and len(cur_chunk_toks) > overlap:
                cur_chunk_toks = cur_chunk_toks[-overlap:]
                cur_len = sum(len(x.split()) for x in cur_chunk_toks)
            else:
                cur_chunk_toks = [s_trimmed]
                cur_len = s_len

            if s_len > max_chunk_size:
                words = s_trimmed.split()
                for i in range(0, s_len, max_chunk_size):
                    chunks.append(" ".join(words[i:i+max_chunk_size]))
                cur_chunk_toks = []
                cur_len = 0
    if cur_chunk_toks:
        chunks.append(" ".join(cur_chunk_toks))
    return [c for c in chunks if c.strip()]

def create_embeddings_for_forum_threads():
    
    embedding_manager = get_embedding_manager()
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT id, thread_id, title, content 
        FROM forum_threads 
        WHERE embedding IS NULL
    ''')
    
    threads = cursor.fetchall()
    
    if not threads:
        print("No forum threads without embeddings.")
        conn.close()
        return True
    
    print(f"Found {len(threads)} forum threads for embeddings.")
    
    for row_id, thread_id, title, content in threads:
        if not content or len(content.strip()) < 10:
            print(f"Warning: Empty or short content for thread {thread_id}. Skipping.")
            continue      
        print(f"Processing forum thread {thread_id}")
        try:
            combined_text = f"{title}\n\n{content}"
            embedding = asyncio.run(embedding_manager.create_enhanced_embedding(
                [combined_text], 
                task_type="RETRIEVAL_DOCUMENT"
            ))
            emb_arr = embedding[0] if isinstance(embedding, list) else embedding
            emb_bytes = emb_arr.tobytes()
            cursor.execute(
                'UPDATE forum_threads SET embedding = ? WHERE id = ?', 
                (emb_bytes, row_id)
            )
            conn.commit()
            print(f"Successfully created embedding for thread {thread_id}")
        except Exception as e:
            print(f"Error creating embedding for thread {thread_id}: {e}")
    conn.close()
    print("Enhanced forum thread embeddings creation complete.")
    return True

def create_embeddings_for_fandom_articles():
    embedding_manager = get_embedding_manager()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT id, url, title, content 
        FROM fandom_articles 
    ''')
    
    articles = cursor.fetchall()
    if not articles:
        print("No fandom articles without embeddings.")
        conn.close()
        return True
    print(f"Found {len(articles)} fandom articles for embeddings.")
    for row_id, url, title, content in articles:
        if not content or len(content.strip()) < 10:
            print(f"Warning: Empty or short content for article {title}. Skipping.")
            continue
        print(f"Processing fandom article: {title}")
        try:
            embedding = asyncio.run(embedding_manager.create_enhanced_embedding(
                [content], 
                task_type="RETRIEVAL_DOCUMENT"
            ))
            emb_arr = embedding[0] if isinstance(embedding, list) else embedding
            emb_bytes = emb_arr.tobytes()
            cursor.execute(
                'UPDATE fandom_articles SET embedding = ? WHERE id = ?', 
                (emb_bytes, row_id)
            )
            conn.commit()
            print(f"Successfully created embedding for article {title}")
        except Exception as e:
            print(f"Error creating embedding for article {title}: {e}")
    conn.close()
    print("Enhanced fandom article embeddings creation complete.")
    return True

@bot.command(name="update_fandom_cache", help="Updates the local cache from P&W Fandom Wiki (Admin Only). Fetches all main pages.")
async def update_fandom_cache_command(ctx, max_articles: int = None):  
    await ctx.send(f"🔄 Starting P&W Fandom Wiki cache update. Fetching all main article URLs from API...")
    all_article_urls = await fetch_all_fandom_page_urls(
        PNW_FANDOM_API_URL,
        PNW_FANDOM_WIKI_BASE_URL,
        namespace="0", 
        max_pages=max_articles 
    )
    if not all_article_urls:
        await ctx.send("No Fandom URLs fetched from the API. Aborting Fandom cache update.")
        return
    total_to_scrape = len(all_article_urls)
    await ctx.send(f"Found {total_to_scrape} articles to process. Starting scraping and embedding... (This may take a very long time)")

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    successful_scrapes = 0
    failed_scrapes_titles = []
    for i, article_url in enumerate(all_article_urls):
        prelim_title = get_title_from_url(article_url) 
        if i % 100 == 0 or i == total_to_scrape -1 :
             await ctx.channel.send(f"Processing Fandom article {i+1}/{total_to_scrape}: {prelim_title} ...")
        scraped_data = await scrape_fandom_article(article_url, prelim_title)
        if scraped_data:
            try:
                cursor.execute("""
                    INSERT OR REPLACE INTO fandom_articles
                    (url, title, content, embedding, last_scraped)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    scraped_data["url"], scraped_data["title"], scraped_data["content"],
                    scraped_data["embedding"],
                    scraped_data["last_scraped"]
                ))
                conn.commit()
                successful_scrapes += 1
            except sqlite3.Error as e:
                print(f"DB Error caching Fandom article {scraped_data.get('title', prelim_title)}: {e}")
                failed_scrapes_titles.append(scraped_data.get('title', prelim_title))
            except Exception as e:
                print(f"Unexpected Error caching Fandom article {scraped_data.get('title', prelim_title)}: {e}")
                failed_scrapes_titles.append(scraped_data.get('title', prelim_title))
        else:
            print(f"Failed to scrape {prelim_title} from {article_url}. Skipping.")
            failed_scrapes_titles.append(prelim_title)

    if await asyncio.to_thread(create_embeddings_for_fandom_articles):
        print(f"Wiki scraping complete! Added {successful_scrapes}/{total_to_scrape} threads to the database with enhanced embeddings.")
    else:
        print(f"Forum scraping partially complete. Added {successful_scrapes}/{total_to_scrape} threads, but there was an error generating embeddings.")
    conn.close()
    final_message = f"P&W Fandom Wiki cache update complete. "
    if failed_scrapes_titles:
        final_message += f"\nFailed to scrape/cache {len(failed_scrapes_titles)} articles. Check logs for titles like: {', '.join(failed_scrapes_titles[:5])}{'...' if len(failed_scrapes_titles) > 5 else ''}"
    await ctx.send(final_message)

@bot.command(name="update_forums", help="Updates the local cache from P&W Forums (Admin Only).")
async def update_forum_command(ctx, max_pages: int = 50, max_threads: int = None):
    loading_msg = await ctx.send("Starting P&W Forum scraping...")
    try:
        await loading_msg.edit(content=f"Fetching thread URLs from up to {max_pages} pages of Alliance Affairs forum...")
        thread_urls = await scrape_forum_threads_simple(max_pages=max_pages)
        if not thread_urls:
            await loading_msg.edit(content="No thread URLs found or error fetching threads.")
            return
        if max_threads and max_threads < len(thread_urls):
            thread_urls = thread_urls[:max_threads]
        await loading_msg.edit(content=f"Found {len(thread_urls)} threads. Starting content scraping...")
        total = len(thread_urls)
        success = 0
        for i, thread_url in enumerate(thread_urls):
            if i % 10 == 0 or i == total - 1:
                await loading_msg.edit(content=f"Processing thread {i+1}/{total}...")
            thread_data = await extract_main_post_simple(thread_url)
            if thread_data:
                conn = sqlite3.connect(DB_PATH)
                cursor = conn.cursor()
                try:
                    cursor.execute("""
                        INSERT OR REPLACE INTO forum_threads
                        (thread_id, title, url, author, post_date, content, embedding, last_scraped)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        thread_data["thread_id"],
                        thread_data["title"],
                        thread_data["url"],
                        thread_data["author"],
                        thread_data["post_date"],
                        thread_data["content"],
                        thread_data['embedding'],
                        datetime.datetime.now().isoformat()
                    ))
                    conn.commit()
                    success += 1
                except Exception as e:
                    print(f"Error storing thread {thread_data['thread_id']}: {e}")
                    conn.rollback()
                    
                finally:
                    conn.close()
        
        await loading_msg.edit(content=f"Generating enhanced embeddings for {success} forum threads...")
        
        if await asyncio.to_thread(create_embeddings_for_forum_threads):
            await loading_msg.edit(content=f"Forum scraping complete! Added {success}/{total} threads to the database with enhanced embeddings.")
        else:
            await loading_msg.edit(content=f"Forum scraping partially complete. Added {success}/{total} threads, but there was an error generating embeddings.")
            
    except Exception as e:
        await loading_msg.edit(content=f"Error during forum scraping: {str(e)}")
        print(f"Error in update_forum_command: {e}")
        import traceback
        traceback.print_exc()

@bot.command(name="update_guides", help="Update guides from Google Docs (Admin Only)")
async def update_guides_command(ctx):
    setup_database()
    msg = await ctx.send("🔄 Starting guide update...")
    try:
        docs_s, drive_s = get_google_service()
        print("Google API services initialized.")
        meta = list_beginner_guides_from_drive(drive_s)
        if not meta: await msg.edit(content="No guides found in GDrive."); return
        await msg.edit(content=f"Found {len(meta)} guides. Fetching & updating DB...")
        processed = []
        for m_item in meta:
            print(f"Processing GDoc: {m_item['name']} (ID: {m_item['id']})")
            doc_d = extract_text_from_google_doc(docs_s, m_item['id'])
            doc_d['doc_id']=m_item['id']; 
            doc_d['modifiedTime']=m_item.get('modifiedTime')
            processed.append(doc_d)
        add_google_docs_to_database(processed)
        await msg.edit(content=f"DB updated with {len(processed)} guides. Creating/updating embeddings...")
        if await asyncio.to_thread(create_embeddings_for_guides):
            await msg.edit(content="Guide update complete! Embeddings done.")
        await msg.edit(content="Guide update complete! Embeddings done.")
    except FileNotFoundError as e: await msg.edit(content=f"Config Err: {str(e)}. Check 'credentials.json'.")
    except Exception as e:
        await msg.edit(content=f"Error occured during guide update: {str(e)}")
        print(f"Err in update_guides_cmd: {e}"); import traceback; traceback.print_exc()

    
@update_guides_command.error
async def update_guides_error_handler(ctx, error):
    if isinstance(error, commands.MissingPermissions): await ctx.send("Sorry, admin only.")
    else: await ctx.send(f"Unexpected err with update_guides: {error}"); print(f"Err in update_guides_cmd (handler): {error}")  

async def scrape_forum_threads_simple(max_pages=30):
    session = await get_aiohttp_session()
    all_thread_urls = []
    for page_num in range(1, max_pages + 1):
        try:
            if page_num == 1:
                page_url = ALLIANCE_AFFAIRS_URL
            else:
                page_url = f"{ALLIANCE_AFFAIRS_URL}page/{page_num}/"
            print(f"Fetching threads from page {page_num}/{max_pages}: {page_url}")
            async with session.get(page_url, timeout=30) as response:
                if response.status != 200:
                    print(f"Got status {response.status} for page {page_num}, stopping pagination")
                    break
                    
                html_content = await response.text()
                soup = BeautifulSoup(html_content, 'html.parser')
                
                if not soup.select_one('title') or "Page Not Found" in soup.select_one('title').get_text():
                    print(f"Page {page_num} not found, stopping pagination")
                    break
                
                page_thread_urls = []
                for link in soup.find_all('a'):
                    href = link.get('href', '')
                    
                    if '/topic/' in href or '?/topic/' in href:
                        base_url = href.split('/page/')[0] 
                        base_url = base_url.split('#')[0]   
                        
                        if not base_url.startswith('http'):
                            if base_url.startswith('/'):
                                base_url = f"https://forum.politicsandwar.com{base_url}"
                            else:
                                base_url = f"https://forum.politicsandwar.com/{base_url}"
                        
                        page_thread_urls.append(base_url)
                
                page_thread_urls = list(set(page_thread_urls))
                print(f"Found {len(page_thread_urls)} unique thread URLs on page {page_num}")
                
                # Add to total
                all_thread_urls.extend(page_thread_urls)
                
        except Exception as e:
            print(f"Error fetching forum page {page_num}: {e}")
            import traceback
            traceback.print_exc()
    
    all_thread_urls = list(set(all_thread_urls))
    print(f"Found a total of {len(all_thread_urls)} unique thread URLs across {max_pages} pages")
    
    return all_thread_urls

async def extract_main_post_simple(thread_url):
    session = await get_aiohttp_session()
    try:
        print(f"Fetching thread: {thread_url}")
        async with session.get(thread_url, timeout=30) as response:
            response.raise_for_status()
            html_content = await response.text()
            soup = BeautifulSoup(html_content, 'html.parser')
            
            thread_id_match = re.search(r'/topic/(\d+)', thread_url)
            if not thread_id_match:
                thread_id_match = re.search(r'\?/topic/(\d+)', thread_url)
            
            thread_id = thread_id_match.group(1) if thread_id_match else f"unknown_{hash(thread_url)}"
            
            title_elem = soup.select_one('h1.ipsType_pageTitle') or \
                         soup.select_one('title')
            thread_title = title_elem.get_text(strip=True) if title_elem else "Unknown Title"
            
            main_post_content = None
            
            content_selectors = [
                "article.ipsComment:first-child .ipsComment_content",
                ".cPost:first-of-type .cPost_contentWrap",
                ".ipsBox .ipsComment:first-child div[data-role='commentContent']",
                ".ipsComment:first-child div[itemprop='text']"
            ]
            
            for selector in content_selectors:
                content_elem = soup.select_one(selector)
                if content_elem:
                    for remove_elem in content_elem.select("blockquote, .ipsQuote, .ipsCode, script, iframe"):
                        remove_elem.decompose()
                    
                    main_post_content = content_elem.get_text(separator="\n", strip=True)
                    break
            if not main_post_content:
                any_post = soup.select_one(".cPost_contentWrap") or \
                          soup.select_one(".ipsComment_content") or \
                          soup.select_one("div[itemprop='text']")
                          
                if any_post:
                    for remove_elem in any_post.select("blockquote, .ipsQuote, .ipsCode, script, iframe"):
                        remove_elem.decompose()
                    
                    main_post_content = any_post.get_text(separator="\n", strip=True)
            
            author_elem = soup.select_one("a[itemprop='author']") or \
                         soup.select_one(".cAuthorPane_author") or \
                         soup.select_one(".ipsComment_author .ipsType_break")
                         
            author = author_elem.get_text(strip=True) if author_elem else "Unknown"
            
            if main_post_content:
                return {
                    "thread_id": thread_id,
                    "title": thread_title,
                    "url": thread_url,
                    "author": author,
                    "content": main_post_content,
                    "embedding": None,
                    "post_date": datetime.datetime.now().isoformat()  
                }
            else:
                print(f"Could not extract main post content from {thread_url}")
                return None
                
    except Exception as e:
        print(f"Error extracting main post from {thread_url}: {e}")
        import traceback
        traceback.print_exc()
        return None
    
async def search_all_sources_with_gemini_reranking(query: str, max_results: int = 12):

    guides_task = asyncio.create_task(search_guides_with_knn_bm25(query, k=3))
    fandom_task = asyncio.create_task(search_fandom_with_knn_bm25(query, k=3))
    forum_task = asyncio.create_task(search_forum_with_knn_bm25(query, k=3))
    
    guides_results = await guides_task
    fandom_results = await fandom_task
    forum_results = await forum_task
    
    print(f"Initial retrieval: {len(guides_results)} guides, {len(fandom_results)} fandom, {len(forum_results)} forum")
    
    rerank_candidates = []
    
    for i, result in enumerate(guides_results):
        rerank_candidates.append({
            "id": f"guide_{result['id']}",
            "title": result['title'],
            "content": result['chunk_text'],
            "url": result['url'],
            "source_type": "guide",
            "initial_score": result['similarity'],
            "initial_rank": i,
        })
    
    for i, result in enumerate(fandom_results):
        rerank_candidates.append({
            "id": f"fandom_{i}",
            "title": result['title'],
            "content": result['raw_content'], 
            "url": result['url'],
            "source_type": "fandom",
            "initial_score": result['similarity'],
            "initial_rank": i,
            "disclaimer": "" if "Disclaimer" not in result['content'] else 
                          result['content'].split("(Disclaimer:")[1].split(")", 1)[0] + ")"
        })
    
    for i, result in enumerate(forum_results):
        rerank_candidates.append({
            "id": f"forum_{result['id']}",
            "title": result['title'],
            "content": result['content'],
            "url": result['url'],
            "source_type": "forum",
            "author": result.get('author', 'Unknown'),
            "initial_score": result['similarity'],
            "initial_rank": i,
        })
    
    sources_info = []
    for res in rerank_candidates:
        if not any(s_url == res['url'] for _, s_url in sources_info):
            sources_info.append((res['title'], res['url']))
    
    context_parts = []
    
    for res in rerank_candidates:
        if res['source_type'] == 'guide':
            context_parts.append(f"From Guide '{res['title']}':\n{res['content']}")
        elif res['source_type'] == 'fandom':
            content = res['content']
            if res.get('disclaimer'):
                content = f"{content}\n(Disclaimer: {res['disclaimer']}"
            context_parts.append(f"From P&W Fandom Wiki page '{res['title']}':\n{content}")
        elif res['source_type'] == 'forum':
            context_parts.append(f"From Forum Thread '{res['title']}' by {res.get('author', 'Unknown')}:\n{res['content']}")
    
    combined_context = "\n\n---\n\n".join(context_parts)
    
    return {
        "query": query,
        "context": combined_context,
        "sources": sources_info,
        "reranked_results": rerank_candidates
    }

@bot.command(name="ask", help="Answer P&W questions using enhanced multi-stage retrieval")
async def pnw_question_enhanced(ctx, *, question: str):
    response_embed = discord.Embed(
        description="Thinking...",
        color=discord.Color.purple()
    )
    response_message = await ctx.send(embed=response_embed)
    
    bot.loop.create_task(
        process_question(ctx, question, response_message)
    )

async def process_question(ctx, question, response_message):
    try:
        await update_user_message_history(ctx.message)
        try:
            await response_message.edit(embed=discord.Embed(
                description="Retrieving information...",
                color=discord.Color.purple()
            ))
        except discord.errors.NotFound:
            logger.warning("Message not found when trying to update status - it may have been deleted")
            return
        except Exception as e:
            logger.error(f"Could not update message: {e}")
        
        retrieval_results = await search_all_sources_with_gemini_reranking(question)
        
        combined_context = retrieval_results["context"]
        
        response = await generate_response_with_grok(question, combined_context, user_history_context="")
        try:
            await response_message.edit(embed=discord.Embed(
                description=response,
                color=discord.Color.purple()    
            ))
        except Exception as e:
            logger.error(f"Could not update message: {e}")
        
        """
        if sources_info:
            source_links = []
            for title, url in sources_info[:5]:
                display_title = title.replace('_', ' ')
                if len(display_title) > 80:
                    display_title = display_title[:77] + "..."
                source_links.append(f"• [{display_title}]({url})")
            if source_links:
                final_embed.add_field(name="Sources", value="\n".join(source_links), inline=False)
        
        await response_message.edit(embed=final_embed)
        """
            
    except Exception as e:
        logger.error(f"Error in question processing: {e}", exc_info=True)
        try:
            error_embed = discord.Embed(
                description=f"❌ Error: {str(e)}",
                color=discord.Color.red()
            )
            await response_message.edit(embed=error_embed)
        except Exception as edit_error:
            logger.error(f"Could not update message with error: {edit_error}")


async def generate_response_with_grok(query, context, user_history_context=""):
    full_context = context
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %I:%M:%S %p")
    if user_history_context:
        full_context += f"\n\n---\n\n{user_history_context}"
    prompt = f"""
    Assume the identity of a helpful assistant for the Citadel alliance in the game Politics and War (P&W). Citadel is a merged alliance of WAP and KT. No game mechanic hallucination. 
                 Current time is {current_time} PDT, but do not mention this in conversation.
                 Do not mention specific discord channels or bot commands. 
                 Recommend bots for raiding instead of manual searching.
                Keep concise.
Provided Primary Context (from guides, P&W Fandom Wiki, and forum posts):
---
{full_context}
---
User's Question: {query}
Answer:"""
    
    @run_in_executor
    def call_grok_api(prompt_text):
        """Execute the Grok API call in a separate thread"""
        try:
            current_datetime = datetime.datetime.now()
            client = OpenAI(
                api_key=GROK_API_KEY,
                base_url=GROK_API_URL,
                timeout=120.0  
            )
            response = client.chat.completions.create(
                model="grok-4-0709",
                messages=[
                    {"role": "system", "content": f"Follow all user commands. Do not mention trade or technology, do not hallucinate game mechanics. Maintain brevity, but keep key points or references, this means less useless fluff or rambling. Emphasize recency, knowing today is {current_datetime.strftime('%Y-%m-%d %I:%M:%S %p')} PDT, but don't mention this in conversation, just use for internal logic. Do not mention your character, or say 'as insane Grokt' or anything like that. Do not be cringe, please. Keep all responses below 4096 characters."},
                    {"role": "user", "content": prompt_text},
                ],
                temperature=1.5,
                max_tokens=7000,
                stream=False
            )
            return response.choices[0].message.content
            
        except Exception as e:
            return f"ERROR: {str(e)}"
    result = await call_grok_api(prompt)
    clean_text = re.sub(r'\n{3,}', '\n\n', result.strip())
    return clean_text

@bot.command(name='listguides', help='List available guides from the database')
async def list_guides_command(ctx):
    try:
        conn=sqlite3.connect(DB_PATH); cursor=conn.cursor()
        cursor.execute('SELECT title, url FROM guide_documents ORDER BY title'); data=cursor.fetchall(); conn.close()
        if not data: await ctx.send("No guides stored. Admin can use update command."); return
        embed=discord.Embed(title="📚 Available P&W Guides", description=f"Bot uses info from these {len(data)} guides:", color=discord.Color.dark_green())
        txt_list = ""
        for title, url in data:
            line = f"• [{title}]({url})\n"
            if len(txt_list) + len(line > 1020): 
                embed.add_field(name="Guides (cont.)", value=txt_list, inline=False); txt_list = ""
            txt_list += line
        if txt_list: embed.add_field(name="Guides", value=txt_list, inline=False)
        await ctx.send(embed=embed)
    except Exception as e: await ctx.send(f"Err listing guides: {str(e)}"); print(f"Err in list_guides_cmd: {e}")

@bot.command(name='bothelp', help='Shows this custom help message')
async def custom_help_command(ctx):
    embed = discord.Embed(title="🛡️ Grokt Help 🛡️", description="I help with P&W, using KT guides & the P&W Wiki.", color=discord.Color.gold())
    embed.add_field(name=f"`@{bot.user.name} ask [question]`", value="Ask P&W questions. Reply to messages to add context.", inline=False)
    embed.add_field(name=f"`@{bot.user.name} list_guides`", value="Lists available Google Doc guides.", inline=False)
    embed.add_field(name=f"`@{bot.user.name} update_guides`", value="**(Admin only)** Updates guides from Google Docs.", inline=False)
    embed.add_field(name=f"`@{bot.user.name} update_fandom_cache`", value="**(Admin only)** Updates P&W Fandom Wiki cache.", inline=False)
    embed.add_field(name=f"`@{bot.user.name} update_forums`", value="**(Admin only)** Updates P&W Forum cache.", inline=False)
    await ctx.send(embed=embed)

def diagnose_embeddings():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute("SELECT id, guide_id, embedding FROM guide_embeddings")
    guide_embs = cursor.fetchall()
    
    cursor.execute("SELECT id, url, embedding FROM fandom_articles WHERE embedding IS NOT NULL")
    fandom_embs = cursor.fetchall()
    
    cursor.execute("SELECT id, thread_id, embedding FROM forum_threads WHERE embedding IS NOT NULL")
    forum_embs = cursor.fetchall()
    
    conn.close()
    
    issues = 0
    expected_dim = 768
    
    for emb in guide_embs:
        try:
            vector = np.frombuffer(emb['embedding'], dtype=np.float32)
            if len(vector) != expected_dim:
                print(f"Guide {emb['id']} (guide_id: {emb['guide_id']}): Wrong dimension {len(vector)}")
                issues += 1
        except Exception as e:
            print(f"Guide {emb['id']}: Error {e}")
            issues += 1
    for emb in fandom_embs:
        try:
            vector = np.frombuffer(emb['embedding'], dtype=np.float32)
            if len(vector) != expected_dim:
                print(f"Fandom {emb['id']} (Fandom: {emb['id']}): Wrong dimension {len(vector)}")
                issues += 1
        except Exception as e:
            print(f"Fandom {emb['id']}: Error {e}")
            issues += 1
    for emb in forum_embs:
        try:
            vector = np.frombuffer(emb['embedding'], dtype=np.float32)
            if len(vector) != expected_dim:
                print(f"Forum {emb['id']} (Forum: {emb['id']}): Wrong dimension {len(vector)}")
                issues += 1
        except Exception as e:
            print(f"Forum {emb['id']}: Error {e}")
            issues += 1
    
    print(f"Total issues found: {issues}")
    print("=== END DIAGNOSTICS ===")
    return issues

if __name__ == "__main__":
    if not DISCORD_TOKEN: print("Error: DISCORD_TOKEN not set.")
    elif not GEMINI_API_KEY: print("Error: GEMINI_API_KEY not set.")
    else:
        print("Starting bot...")
        diagnose_embeddings()
        try:
            bot.run(DISCORD_TOKEN)
        except Exception as e:
            print(f"Error running bot: {e}")