import os
import re
import requests
from bs4 import BeautifulSoup
from collections import Counter
import argparse
import sys

import json

# 停用词表，用于过滤无意义的词
STOP_WORDS = {
    'a', 'an', 'the', 'and', 'or', 'of', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 
    'is', 'are', 'was', 'were', 'be', 'been', 'being', 'it', 'this', 'that', 'these', 'those',
    'as', 'from', 'via', 'using', 'based', 'through', 'during', 'between', 'among',
    'paper', 'method', 'approach', 'framework', 'system', 'model', 'algorithm', 'proposed',
    'state', 'art', 'review', 'survey', 'analysis', 'study', 'towards', 'via', 'improving',
    'learning', 'deep', 'network', 'neural', 'networks', 'models', 'introduction', 'catalog'
}

def get_md_files(root_dir):
    md_files = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.md'):
                md_files.append(os.path.join(root, file))
    return md_files

def extract_titles_from_md(file_path):
    titles = []
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            matches = re.findall(r'\*\s*(?:\[[^\]]*\])*\[([^\]]+)\]\(([^)]+)\)', content)
            for match in matches:
                titles.append(match[0])
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    return titles

def get_all_history_titles(root_dir):
    md_files = get_md_files(root_dir)
    all_titles = []
    for md_file in md_files:
        all_titles.extend(extract_titles_from_md(md_file))
    return all_titles

import urllib.request
import urllib.error

def call_llm_filter(history_titles, candidate_papers, api_key, base_url, model, top_k=10):
    print(f"Calling LLM ({model})...", flush=True)
    
    # Send up to 300 candidates to LLM to find the best 10.
    # We strip links in the prompt to save tokens, but map them back later using ID.
    candidates_simple = [{"id": i, "title": p['title']} for i, p in enumerate(candidate_papers[:300])]
    
    # Use a larger sample of history to better represent interests
    # Random sample or just the first N is fine. 
    # Since these are likely chronological if read from file, maybe recent ones are better?
    # Or just random. For now, taking the first 100 found is a reasonable proxy.
    history_sample = history_titles[:1000] if len(history_titles) > 1000 else history_titles
    
    system_prompt = f"""You are an expert research assistant. 
Your task is to filter a list of new papers based on the user's historical reading list.
Select the most relevant papers (Top {top_k}).
Output ONLY a valid JSON object with a key "recommendations".
Each object must have: "id" (from input), "title", "reason" (concise explanation), "score" (0-100 relevance).
Do not output any markdown formatting like ```json ... ```. Just the raw JSON string.
"""
    
    user_prompt = f"""
User's Research Interests (Sample of read papers):
{json.dumps(history_sample, ensure_ascii=False)}

New Papers Candidates (Please select top {top_k}):
{json.dumps(candidates_simple, ensure_ascii=False)}

Recommend the best ones matching the user's interests.
"""

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "User-Agent": "Python-urllib/3.0"
    }
    
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.3
    }
    
    data = json.dumps(payload).encode('utf-8')
    # print(f"Payload size: {len(data)} bytes", flush=True)
    
    # Handle base_url normalization
    if not base_url.endswith('/v1'):
        if not base_url.endswith('/'):
            base_url += '/'
        base_url += 'v1'
        
    endpoint = f"{base_url}/chat/completions"
    # If user provided full path (e.g. including chat/completions), use it
    if 'chat/completions' in base_url.replace('/v1', ''):
         # If base_url was passed as .../v1/chat/completions, the logic above added another v1.
         # Let's simple check if 'chat/completions' is present in the original or modified url
         pass 

    # Robust endpoint construction
    # If base_url already contains 'chat/completions', use it directly
    # Otherwise append /chat/completions
    if 'chat/completions' in base_url:
        endpoint = base_url
    else:
        endpoint = f"{base_url}/chat/completions"
        
    # print(f"Endpoint: {endpoint}", flush=True)
    
    try:
        req = urllib.request.Request(endpoint, data=data, headers=headers, method='POST')
        
        with urllib.request.urlopen(req, timeout=120) as response:
            content = response.read().decode('utf-8')
            
            result = json.loads(content)
            content_str = result['choices'][0]['message']['content']
            
            # Clean up content
            content_str = content_str.strip()
            content_str = re.sub(r'^```json\s*', '', content_str)
            content_str = re.sub(r'^```\s*', '', content_str)
            content_str = re.sub(r'\s*```$', '', content_str)
            
            parsed = json.loads(content_str)
            recs = parsed.get("recommendations", [])
            
            final_recs = []
            for r in recs:
                idx = r.get('id')
                if idx is not None and isinstance(idx, int) and 0 <= idx < len(candidate_papers):
                     r['link'] = candidate_papers[idx]['link']
                     final_recs.append(r)
                else:
                     final_recs.append(r)
            return final_recs
            
    except urllib.error.HTTPError as e:
        print(f"HTTP Error: {e.code} {e.reason}", flush=True)
        try:
            print(e.read().decode('utf-8'), flush=True)
        except:
            pass
        return []
    except Exception as e:
        print(f"LLM API Error: {e}", flush=True)
        return []

def fetch_url_content(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    try:
        session = requests.Session()
        session.trust_env = False  # Disable proxy settings
        response = session.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        return response.text
    except Exception as e:
        print(f"Error fetching URL: {e}")
        return None

def extract_papers_from_html(html):
    soup = BeautifulSoup(html, 'html.parser')
    papers = []
    
    # 策略1：通用策略 - 寻找所有指向学术网站的链接
    links = soup.find_all('a', href=True)
    found_titles = set()
    
    for link in links:
        href = link['href']
        text = link.get_text().strip()
        
        is_academic = False
        if any(domain in href for domain in ['arxiv.org', 'openreview.net', 'ieee.org', 'cvf.com', 'acm.org']):
            is_academic = True
            
        if is_academic:
            title = text
            if len(title) < 10:
                parent = link.parent
                if parent and len(parent.get_text().strip()) > 10:
                    title = parent.get_text().strip()
                    # 尝试去掉末尾的 link 文本
                    if title.endswith(text):
                        title = title[:-len(text)].strip()
            
            title = re.sub(r'\s+', ' ', title).strip()
            if len(title) > 10 and title not in found_titles:
                papers.append({'title': title, 'link': href})
                found_titles.add(title)
            
    # 策略2：针对微信公众号格式 (【1】Title ... 标题：... 链接：url)
    text = soup.get_text()
    
    # 匹配模式：【数字】... (直到下一个【数字】)
    # 使用 split 分割每一段
    segments = re.split(r'(?=【\d+】)', text)
    
    for segment in segments:
        if not segment.strip().startswith('【'):
            continue
            
        # 提取标题：【\d+】(Title)
        title_match = re.match(r'【\d+】\s*(.*?)(?:\s*标题：|\s*链接：|\s*作者：|\s*摘要：|\n)', segment, re.DOTALL)
        if not title_match:
            continue
            
        raw_title = title_match.group(1).strip()
        clean_title = re.sub(r'\s+', ' ', raw_title)
        
        if len(clean_title) < 10:
            continue

        # 提取链接
        link = 'Link not detected'
        # 尝试匹配 "链接：http..."
        link_match = re.search(r'链接：\s*(https?://[^\s\u4e00-\u9fa5]+)', segment)
        if link_match:
            link = link_match.group(1)
        else:
            # 尝试匹配任意 url
            url_match = re.search(r'(https?://arxiv\.org[^\s\u4e00-\u9fa5]+)', segment)
            if url_match:
                link = url_match.group(1)

        is_duplicate = False
        for p in papers:
            if clean_title in p['title'] or p['title'] in clean_title:
                is_duplicate = True
                # 如果之前的没有链接，现在的有链接，更新它
                if p['link'].startswith('Link not') and not link.startswith('Link not'):
                    p['link'] = link
                break
        
        if not is_duplicate and clean_title not in found_titles:
             papers.append({'title': clean_title, 'link': link})
             found_titles.add(clean_title)

    return papers

def main():
    parser = argparse.ArgumentParser(description='Paper Recommender based on local markdown notes (LLM-powered).')
    parser.add_argument('url', help='The URL of the website to analyze')
    parser.add_argument('--dir', default='.', help='Root directory of markdown notes')
    parser.add_argument('--top_k', type=int, default=10, help='Number of papers to recommend')
    parser.add_argument('--api_key', default=os.environ.get('LLM_API_KEY', 'sk-863ce5e99f0c41059e1c1bbcb69bd340'), help='LLM API Key')
    parser.add_argument('--base_url', default=os.environ.get('LLM_BASE_URL', 'https://api.deepseek.com'), help='LLM Base URL')
    parser.add_argument('--model', default='deepseek-chat', help='Model name (e.g. gpt-4o, deepseek-chat, claude-3-5-sonnet-20240620)')
    
    args = parser.parse_args()
    
    if not args.api_key:
        print("Error: API Key is required. Please provide --api_key or set LLM_API_KEY environment variable.")
        print("Example: python paper_filter.py URL --api_key sk-xxxx --base_url https://api.deepseek.com")
        return

    print(f"Analyzing markdown files in {args.dir}...")
    history_titles = get_all_history_titles(args.dir)
    print(f"Found {len(history_titles)} papers in reading history.")
    #print(history_titles)
    
    print(f"\nFetching URL: {args.url}...")
    html = fetch_url_content(args.url)
    if not html:
        print("Failed to fetch content.")
        return
        
    print("Extracting papers from webpage...")
    candidates = extract_papers_from_html(html)
    print(f"Found {len(candidates)} candidate items.")
    
    if not candidates:
        print("No candidates found.")
        return

    print(f"\nRequesting LLM for Top {args.top_k} recommendations...")
    recommendations = call_llm_filter(history_titles, candidates, args.api_key, args.base_url, args.model, top_k=args.top_k)
    
    print(f"\n=== Top Recommended Papers (by LLM) ===\n")
    if not recommendations:
        print("No recommendations returned from LLM.")
    
    for i, paper in enumerate(recommendations):
        score = paper.get('score', 'N/A')
        print(f"{i+1}. [Score: {score}] {paper.get('title', 'Unknown Title')}")
        print(f"   Link: {paper.get('link', 'No Link')}")
        print(f"   Reason: {paper.get('reason', 'No reason provided')}\n")

if __name__ == '__main__':
    main()
