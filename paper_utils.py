import os
import requests
import fitz  # PyMuPDF
import re
import json
import hashlib

def get_cache_dir():
    cache_dir = os.path.join(os.getcwd(), "paper_cache")
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    return cache_dir

def normalize_arxiv_url(url):
    """
    Convert arxiv abstract/html URL to PDF URL.
    Example: https://arxiv.org/abs/2501.12948 -> https://arxiv.org/pdf/2501.12948.pdf
    """
    if "arxiv.org" in url:
        # Match arxiv ID
        match = re.search(r'(\d{4}\.\d{4,5})', url)
        if match:
            arxiv_id = match.group(1)
            return f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    return url

def download_pdf(url):
    """
    Download PDF from URL and cache it.
    Returns the path to the local file.
    """
    url = normalize_arxiv_url(url)
    
    # Create a filename based on the URL hash or content
    url_hash = hashlib.md5(url.encode('utf-8')).hexdigest()
    filename = f"{url_hash}.pdf"
    cache_dir = get_cache_dir()
    file_path = os.path.join(cache_dir, filename)
    
    if os.path.exists(file_path):
        return file_path
        
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers, stream=True, timeout=30)
        response.raise_for_status()
        
        with open(file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return file_path
    except Exception as e:
        print(f"Error downloading PDF: {e}")
        # Clean up partial file
        if os.path.exists(file_path):
            os.remove(file_path)
        return None

def extract_text_from_pdf(pdf_path, max_pages=10):
    """
    Extract text from PDF.
    Limiting pages to avoid too much context window usage.
    """
    try:
        doc = fitz.open(pdf_path)
        text = ""
        # Read first few pages (abstract + intro + conclusion usually enough for summary)
        # But for deep summary, maybe more. Let's take first 10 pages max.
        for i, page in enumerate(doc):
            if i >= max_pages:
                break
            text += page.get_text()
        return text
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return ""

def summarize_paper(text, api_key, base_url, model):
    """
    Call LLM to summarize the paper text.
    """
    if not text:
        return "Failed to extract text from the paper."

    # Truncate text if too long (approx 15k chars ~ 4k tokens)
    # DeepSeek supports 32k/128k context, so we can be generous.
    # But let's keep it reasonable for speed.
    truncated_text = text[:30000] 

    system_prompt = """你是一位专业的学术研究员，擅长深度解析计算机科学（特别是AI、自动驾驶、大模型）领域的论文。
请阅读以下论文内容，并按照提供的参考格式，输出一份详尽、深度且结构清晰的中文总结。

### 目标格式（请严格遵守）：

# [论文标题]（PDF深度总结）

## 一、研究背景与核心问题
1. **背景现状**：简述当前领域的现状。
2. **核心挑战**：列出当前面临的具体痛点或未解决的问题。
   - 挑战点1...
   - 挑战点2...

## 二、核心框架与方法
（请详细描述提出的框架/模型/方法，包含关键组件、设计目标和核心操作）
### 1. [关键组件/模块名称]
- **设计目标**：...
- **核心机制**：...
- **详细流程**：...

### 2. [关键组件/模块名称]
- ...

## 三、实验设计与结果
### 1. 实验设置
- **数据集**：...
- **基线模型**：...
- **评估指标**：...

### 2. 核心结果
（请尽可能提取关键数据，如果可能，请整理成 Markdown 表格展示对比）
- **整体性能**：...
- **消融实验**：...

## 四、创新点与贡献
1. ...
2. ...
3. ...

## 五、局限性与未来方向
1. **局限性**：...
2. **未来工作**：...

## 六、补充材料亮点（如有）
（如果有附录或补充材料的值得关注的内容）

---
**注意**：
- 语言必须流畅、专业的简体中文。
- 尽可能保留论文中的关键数据和技术细节，不要过于泛泛。
- 如果论文中有表格数据，请尝试用 Markdown 表格还原关键对比。
- 篇幅要足够详尽，能够替代粗读原文。
"""

    user_prompt = f"Paper Content:\n\n{truncated_text}\n\nPlease summarize this paper."

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.3
    }
    
    # URL handling similar to paper_filter.py
    endpoint = f"{base_url}/chat/completions"
    if 'chat/completions' in base_url:
        endpoint = base_url

    try:
        response = requests.post(endpoint, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        result = response.json()
        content = result['choices'][0]['message']['content']
        
        # Clean up markdown code blocks if present
        content = content.strip()
        if content.startswith("```markdown"):
            content = content.replace("```markdown", "", 1)
        elif content.startswith("```"):
            content = content.replace("```", "", 1)
            
        if content.endswith("```"):
            content = content[:-3]
            
        return content.strip()
    except Exception as e:
        return f"Error generating summary: {e}"
