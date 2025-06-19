import praw
import pandas as pd
from typing import List, Dict, Optional, Union
from prawcore.exceptions import ResponseException, RequestException

class RedditUtils:
    """
    Simplified Reddit utilities for fetching posts and comments.
    """
    
    def __init__(self):
        """Initialize Reddit instance with credentials."""
        self.reddit = praw.Reddit(
            client_id="VdKRBCrVNscQ4LljxTyuaA",
            client_secret="sMc0PMHDEUWYXDsF770lcnLZaYge4A", 
            user_agent='creative-prefs'
        )
    
    def get_top_posts(self, subreddit_name: str, limit: int = 1000, time_filter: str = 'all') -> List[Dict]:
        """
        Get top posts from a subreddit.
        
        Args:
            subreddit_name (str): Name of the subreddit
            limit (int): Number of posts to retrieve (max 1000)
            time_filter (str): Time filter ('all', 'year', 'month', 'week', 'day')
        
        Returns:
            List[Dict]: List of post dictionaries
        """
        try:
            subreddit = self.reddit.subreddit(subreddit_name)
            posts = []
            
            print(f"Fetching top {limit} posts from r/{subreddit_name}...")
            
            for post in subreddit.top(time_filter=time_filter, limit=limit):
                post_data = {
                    'post_id': post.id,
                    'title': post.title,
                    'selftext': post.selftext,
                    'url': post.url,
                    'score': post.score,
                    'num_comments': post.num_comments,
                    'created_utc': post.created_utc,
                    'author': str(post.author),
                    'subreddit': str(post.subreddit),
                    'permalink': post.permalink,
                    'is_self': post.is_self,
                    'upvote_ratio': post.upvote_ratio
                }
                posts.append(post_data)
            
            print(f"Successfully fetched {len(posts)} posts")
            return posts
            
        except Exception as e:
            print(f"Error fetching posts from r/{subreddit_name}: {str(e)}")
            return []
    
    def get_post_by_id(self, post_id: str, include_comments: bool = True, get_all_comments: bool = True) -> Optional[Dict]:
        """
        Get a specific post by its ID.
        
        Args:
            post_id (str): Reddit post ID
            include_comments (bool): Whether to include comments
            get_all_comments (bool): Whether to fetch ALL comments (can be slow but comprehensive)
        
        Returns:
            Optional[Dict]: Post data with optional comments
        """
        try:
            submission = self.reddit.submission(id=post_id)
            
            post_data = {
                'post_id': submission.id,
                'title': submission.title,
                'selftext': submission.selftext,
                'url': submission.url,
                'score': submission.score,
                'num_comments': submission.num_comments,
                'created_utc': submission.created_utc,
                'author': str(submission.author),
                'subreddit': str(submission.subreddit),
                'permalink': submission.permalink,
                'is_self': submission.is_self,
                'upvote_ratio': submission.upvote_ratio
            }
            
            if include_comments:
                print(f"Fetching comments for post {post_id}...")
                
                if get_all_comments:
                    # Replace ALL MoreComments objects to get all comments
                    print("  Expanding all 'more comments' sections...")
                    submission.comments.replace_more(limit=None)
                    
                    # Get all comments as a flat list
                    all_comments = submission.comments.list()
                    print(f"  Found {len(all_comments)} total comments")
                else:
                    # Just remove MoreComments without fetching (faster but incomplete)
                    submission.comments.replace_more(limit=0)
                    all_comments = submission.comments.list()
                    print(f"  Found {len(all_comments)} comments (partial)")
                
                comments = []
                for comment in all_comments:
                    if hasattr(comment, 'body'):  # Make sure it's a comment, not a MoreComments object
                        comment_data = {
                            'comment_id': comment.id,
                            'body': comment.body,
                            'score': comment.score,
                            'created_utc': comment.created_utc,
                            'author': str(comment.author),
                            'is_top_level': comment.parent_id.startswith('t3_'),  # t3_ prefix indicates post
                            'parent_id': comment.parent_id
                        }
                        comments.append(comment_data)
                
                post_data['comments'] = comments
                print(f"Fetched {len(comments)} comments")
            
            return post_data
            
        except Exception as e:
            print(f"Error fetching post {post_id}: {str(e)}")
            return None
    
    def get_comment_by_id(self, comment_id: str, include_replies: bool = True, reply_limit: int = 50) -> Optional[Dict]:
        """
        Get a specific comment by its ID.
        
        Args:
            comment_id (str): Reddit comment ID
            include_replies (bool): Whether to include replies to this comment
            reply_limit (int): Maximum number of replies to fetch
        
        Returns:
            Optional[Dict]: Comment data with optional replies
        """
        try:
            comment = self.reddit.comment(id=comment_id)
            
            comment_data = {
                'comment_id': comment.id,
                'body': comment.body,
                'score': comment.score,
                'created_utc': comment.created_utc,
                'author': str(comment.author),
                'subreddit': str(comment.subreddit),
                'permalink': comment.permalink,
                'is_top_level': comment.parent_id.startswith('t3_'),  # t3_ prefix indicates post
                'parent_id': comment.parent_id,
                'post_id': comment.submission.id,
                'post_title': comment.submission.title,
                'distinguished': comment.distinguished,
                'is_submitter': comment.is_submitter,
                'stickied': comment.stickied
            }
            
            if include_replies:
                print(f"Fetching replies for comment {comment_id}...")
                comment.replies.replace_more(limit=0)  # Remove "more comments" objects
                replies = []
                
                for reply in comment.replies.list()[:reply_limit]:
                    if hasattr(reply, 'body'):  # Make sure it's a comment, not a MoreComments object
                        reply_data = {
                            'comment_id': reply.id,
                            'body': reply.body,
                            'score': reply.score,
                            'created_utc': reply.created_utc,
                            'author': str(reply.author),
                            'parent_id': reply.parent_id,
                            'depth': reply.depth
                        }
                        replies.append(reply_data)
                
                comment_data['replies'] = replies
                print(f"Fetched {len(replies)} replies")
            
            return comment_data
            
        except Exception as e:
            print(f"Error fetching comment {comment_id}: {str(e)}")
            return None
    
    def search_posts(self, subreddit_name: str, query: str, sort: str = 'relevance', limit: int = 100) -> List[Dict]:
        """
        Search for posts in a subreddit.
        
        Args:
            subreddit_name (str): Name of the subreddit
            query (str): Search query
            sort (str): Sort method ('relevance', 'hot', 'top', 'new')
            limit (int): Maximum number of results
        
        Returns:
            List[Dict]: List of matching posts
        """
        try:
            subreddit = self.reddit.subreddit(subreddit_name)
            posts = []
            
            print(f"Searching r/{subreddit_name} for: '{query}'...")
            
            for post in subreddit.search(query, sort=sort, limit=limit):
                post_data = {
                    'post_id': post.id,
                    'title': post.title,
                    'selftext': post.selftext,
                    'url': post.url,
                    'score': post.score,
                    'num_comments': post.num_comments,
                    'created_utc': post.created_utc,
                    'author': str(post.author),
                    'subreddit': str(post.subreddit),
                    'permalink': post.permalink,
                    'is_self': post.is_self,
                    'upvote_ratio': post.upvote_ratio
                }
                posts.append(post_data)
            
            print(f"Found {len(posts)} matching posts")
            return posts
            
        except Exception as e:
            print(f"Error searching r/{subreddit_name}: {str(e)}")
            return []
    
    def save_posts_to_csv(self, posts: List[Dict], filename: str) -> bool:
        """
        Save posts to a CSV file.
        
        Args:
            posts (List[Dict]): List of post dictionaries
            filename (str): Output filename
        
        Returns:
            bool: Success status
        """
        try:
            df = pd.DataFrame(posts)
            df.to_csv(filename, index=False)
            print(f"Saved {len(posts)} posts to {filename}")
            return True
        except Exception as e:
            print(f"Error saving to CSV: {str(e)}")
            return False


# Example usage functions
def main():
    """Example usage of RedditUtils."""
    reddit_utils = RedditUtils()
    
    # Example 1: Get top 1000 posts from WritingPrompts
    posts = reddit_utils.get_top_posts('WritingPrompts', limit=1000, time_filter='all')
    if posts:
        reddit_utils.save_posts_to_csv(posts, 'top_writing_prompts.csv')
    
    # Example 2: Get a specific post by ID (replace with actual post ID)
    # post = reddit_utils.get_post_by_id('abc123', include_comments=True)
    # if post:
    #     print(f"Post title: {post['title']}")
    #     print(f"Number of comments: {len(post.get('comments', []))}")
    
    # Example 3: Get a specific comment by ID (replace with actual comment ID)
    # comment = reddit_utils.get_comment_by_id('def456', include_replies=True)
    # if comment:
    #     print(f"Comment body: {comment['body'][:100]}...")
    #     print(f"Comment score: {comment['score']}")
    #     print(f"Number of replies: {len(comment.get('replies', []))}")
    #     print(f"From post: {comment['post_title']}")
    
    # Example 4: Search for specific posts
    # search_results = reddit_utils.search_posts('WritingPrompts', '[WP]', limit=50)
    # if search_results:
    #     reddit_utils.save_posts_to_csv(search_results, 'wp_search_results.csv')


if __name__ == "__main__":
    main() 