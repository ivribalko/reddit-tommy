import os
import glob
from datetime import datetime, timedelta
from typing import List, Dict

import openai
import praw
import requests
from dotenv import load_dotenv
from openai.types.chat import ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam

SUBREDDIT_MAX_POSTS = 8
POST_MAX_COMMENTS = 21
OPENAI_MODEL="gpt-5-mini"
OPENAI_MAX_COMPLETION_TOKENS = 4000
OPENAI_MESSAGE_SYSTEM="""
You are a financial analyst. 
"""
OPENAI_MESSAGE_USER="""
Summarize this into numbered list with a few key items. No need to list everything. 
Add new lines between list items. Use notation like 1) for item start. 
Find best stock that will go up very soon. This is your main goal. 
Find main market sentiments.
Don't use company names, use stock names instead. 
For stock names use notation like $TSLA with a $. 
At the end of every list item append without new lines:
 - give related Reddit post links, format: '🗣️LINK' 
 - give Public stock links, format: '📈https://public.com/stocks/STOCK_NAME'.  
No markdown. 
DO NOT GO OVER 3000 characters:\n\n
"""


def send_telegram_message(text: str):
    url = f"https://api.telegram.org/bot{os.getenv('TELEGRAM_BOT_TOKEN')}/sendMessage"
    payload = {
        "chat_id": os.getenv("TELEGRAM_CHAT_ID"),
        "text": text,
        "disable_web_page_preview": True,
    }
    requests.post(url, json=payload, timeout=15)
    return None


class RedditSummarizer:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.reddit = praw.Reddit(
            client_id=os.getenv('REDDIT_CLIENT_ID'),
            client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
            user_agent='WSB-Summarizer/1.0'
        )
        self.openai_client = openai.OpenAI(
            api_key=os.getenv('OPENAI_API_KEY')
        )

    @staticmethod
    def get_submission_comments(submission) -> List[str]:
        comments = []
        try:
            submission.comments.replace_more(limit=POST_MAX_COMMENTS)
            # skip first, it's user report + ads
            for c in submission.comments.list()[1:]:
                body = getattr(c, 'body', None)
                if not body:
                    continue
                # remove all newlines
                b = " ".join(body.splitlines()).strip()
                comments.append(b)
                if len(comments) >= POST_MAX_COMMENTS:
                    break
        except Exception:
            pass
        return comments

    def get_today_posts(self, subreddit) -> List[Dict]:
        subreddit = self.reddit.subreddit(subreddit)
        cutoff_ts = (datetime.utcnow() - timedelta(hours=24)).timestamp()
        posts = []
        for submission in subreddit.new(limit=SUBREDDIT_MAX_POSTS):
            print(f"Fetching {subreddit} post #{len(posts) + 1}...")
            if submission.created_utc >= cutoff_ts:
                comments = self.get_submission_comments(submission)
                posts.append({
                    'title': submission.title,
                    'score': submission.score,
                    'num_comments': submission.num_comments,
                    'selftext': submission.selftext[:1000] if submission.selftext else '',
                    'shortlink': submission.shortlink,
                    'created_utc': submission.created_utc,
                    'comments': comments
                })
        return posts

    @staticmethod
    def prepare_posts_for_summary(subreddit, posts: List[Dict]) -> str:
        if not posts:
            return "No posts found for today."
        posts_sorted = sorted(posts, key=lambda x: x['score'], reverse=True)
        summary_text = f"Today's Reddit r/{subreddit} posts:\n\n"
        for i, post in enumerate(posts_sorted, 1):
            summary_text += f"~~~POST #{i} START\n"
            summary_text += f"Title: {post['title']}\n"
            summary_text += f"Score: {post['score']}\n"
            summary_text += f"Comments: {post['num_comments']}\n"
            summary_text += f"Link: {post['shortlink']}\n"
            if post.get('selftext'):
                summary_text += f"Post content: {post['selftext'][:1500]}...\n\n"
            # Incorporate comments: join all comments; to keep prompt manageable, trim very long combined text
            comments = post.get('comments') or []
            if comments:
                combined = " \n- ".join(comments)
                # Soft cap per-post comments text to avoid exceeding context; still uses all comments content insofar as possible
                max_chars = 3000
                if len(combined) > max_chars:
                    combined = combined[:max_chars] + "..."
                summary_text += f"Comments:\n- {combined}\n"
            summary_text += f"~~~POST #{i} END\n\n"
        return summary_text

    def summarize_with_openai(self, text: str) -> str:
        print("Generating summary with OpenAI...")
        try:
            response = self.openai_client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    ChatCompletionSystemMessageParam(role="system", content=OPENAI_MESSAGE_SYSTEM),
                    ChatCompletionUserMessageParam(role="user", content=OPENAI_MESSAGE_USER + text)
                ],
                max_completion_tokens=OPENAI_MAX_COMPLETION_TOKENS,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating summary: {str(e)}"

    def write_output_file(self, name=None, text=None):
        with open(os.path.join(self.output_dir, name), "w", encoding="utf-8") as f:
            f.write(text)

    @staticmethod
    def collect_summaries_in_folder(folder: str) -> str:
        paths = sorted(glob.glob(os.path.join(folder, "*-summary.txt")))
        chunks: List[str] = []
        for path in paths:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    chunks.append(f.read().strip())
            except Exception:
                continue
        return "\n\n".join([c for c in chunks if c])

    def run(self, subreddit: str) -> str:
        print(f"🚀 Starting r/{subreddit} Today's Summary")
        print("=" * 50)
        try:
            if True:
                posts = self.get_today_posts(subreddit)
                posts_text = self.prepare_posts_for_summary(subreddit, posts)
            else:
                with open(os.path.join(self.output_dir, f"{subreddit}.txt"), "r", encoding="utf-8") as f:
                    posts_text = f.read()
            summary = f"📊 TODAY'S r/{subreddit} SUMMARY\n\n"
            summary += self.summarize_with_openai(posts_text)
            self.write_output_file(f"{subreddit}.txt", posts_text)
            self.write_output_file(f"{subreddit}-summary.txt", summary)
            print("=" * 50)
            send_telegram_message(summary)
            return summary
        except Exception as e:
            print(f"Error: {str(e)}")
            return ""


def main():
    load_dotenv()
    required_vars = [
        'REDDIT_CLIENT_ID',
        'REDDIT_CLIENT_SECRET',
        'OPENAI_API_KEY',
        'TELEGRAM_BOT_TOKEN',
        'TELEGRAM_CHAT_ID',
    ]
    if [var for var in required_vars if not os.getenv(var)]:
        print("❌ Missing required environment variables")
        return
    base_output = os.path.join(os.getcwd(), 'output')
    date_str = datetime.now().strftime('%Y-%m-%d')
    day_output = os.path.join(base_output, date_str)
    os.makedirs(day_output, exist_ok=True)
    summarizer = RedditSummarizer(day_output)
    for sub in ["wallstreetbets", "stocks", "investing", "swingtrading", "StockMarket", "Economics"]:
        summarizer.run(sub)
    summaries = summarizer.collect_summaries_in_folder(day_output)
    summary = f"📊 TODAY'S SUMMARY\n\n"
    summary += summarizer.summarize_with_openai(summaries)
    summarizer.write_output_file(f"summary.txt", summary)
    send_telegram_message(summary)


if __name__ == "__main__":
    main()
