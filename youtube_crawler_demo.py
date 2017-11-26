
# coding: utf-8

# In[77]:

import unidecode
import sys
import os
import pandas as pd
from datetime import datetime
from apiclient.discovery import build


# In[4]:




# In[ ]:

class youtube_crawler(object):
    def __init__():
        self.DEVELOPER_KEY = "XXXXX"
        self.YOUTUBE_API_SERVICE_NAME = "youtube"
        self.YOUTUBE_API_VERSION = "v3"
        


# In[5]:

DEVELOPER_KEY = "XXXXX"
YOUTUBE_API_SERVICE_NAME = "youtube"
YOUTUBE_API_VERSION = "v3"
        


# In[6]:

youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=DEVELOPER_KEY)


# In[39]:

userName = "Pepsi"
user_info = youtube.channels().list(part="id,statistics", forUsername=userName).execute()


# In[40]:

user_info


# In[10]:

num_res = user_info["pageInfo"]["totalResults"]


# In[65]:

start_date = "2017-01-01"
end_date = "2017-11-01"


# In[41]:

user_id = {}


# In[44]:

users_list = [] 
if num_res == 1:
    print "Getting the ID for user name :- {}".format(userName)
    item_first = user_info["items"][0]
    user_id["id"] = item_first["id"]
    user_id["commentCount"] = item_first["statistics"]["commentCount"]
    user_id["subscriberCount"] = item_first["statistics"]["subscriberCount"]
    user_id["videoCount"] = item_first["statistics"]["videoCount"]
    user_id["viewCount"] = item_first["statistics"]["viewCount"]
    print "Exracted page info :- {} for user name :- {}".format(user_id, userName)
else:
    for users in user_info["items"]:
        if user_info["kind"] == "youtube#channel":
            users_list.append({"id" : users["id"], "commentCount" : users["statistics"]["commentCount"], 
                               "subscriberCount" : users["statistics"]["subscriberCount"],
                               "videoCount" : users["statistics"]["videoCount"], 
                               "viewCount" : users["statistics"]["viewCount"]})

if len(users_list) > 0:
    print "Found more than one user id...using the first one.."
    user_id = users_list[0]
    print "Exracted the ID :- {} for user name :- {}".format(user_id, userName)


# In[45]:

max_results = 50
search_response = youtube.search().list(channelId=user_id["id"], part="id,snippet", maxResults=max_results).execute()


# In[46]:

search_response.keys()


# In[47]:

search_response.get("items")[0]


# In[232]:

videos = []
next_page_token = ""
while next_page_token is not None:
    try:
        next_page_token = None if "nextPageToken" not in search_response.keys() else search_response.get("nextPageToken")
        for search_result in search_response.get("items", []):
            if search_result["id"]["kind"] == "youtube#video":
                title = search_result["snippet"]["title"]
                title = unidecode.unidecode(title)
                description = search_result["snippet"]["description"].encode("ascii", "ignore")
                created_at = datetime.strptime(search_result["snippet"]["publishedAt"], 
                                               "%Y-%m-%dT%H:%M:%S.000Z").strftime("%Y-%m-%d %H:%M:%S")
                date_cond = datetime.strptime(search_result["snippet"]["publishedAt"], 
                                               "%Y-%m-%dT%H:%M:%S.000Z").strftime("%Y-%m-%d")
                if date_cond >= start_date and date_cond <= end_date:
                    videoId = search_result["id"]["videoId"]
                    video_response = youtube.videos().list(id=videoId,part="statistics").execute()
                    for video_result in video_response.get("items",[]):
                        viewCount = video_result["statistics"]["viewCount"]
                        if 'likeCount' not in video_result["statistics"]:
                            likeCount = 0
                        else:
                            likeCount = video_result["statistics"]["likeCount"]
                        if 'dislikeCount' not in video_result["statistics"]:
                            dislikeCount = 0
                        else:
                            dislikeCount = video_result["statistics"]["dislikeCount"]
                        if 'commentCount' not in video_result["statistics"]:
                            commentCount = 0
                        else:
                            commentCount = video_result["statistics"]["commentCount"]
                        if 'favoriteCount' not in video_result["statistics"]:
                            favoriteCount = 0
                        else:
                            favoriteCount = video_result["statistics"]["favoriteCount"]
                    videos.append([user_id["id"], user_id["commentCount"], user_id["subscriberCount"], user_id["videoCount"], 
                                   user_id["viewCount"], title, description, created_at, videoId, viewCount, likeCount, dislikeCount, 
                                   commentCount, favoriteCount])
                else:
                    pass
        if len(videos) % 10 == 0 and len(videos) != 0:
            print "No. of Videos posts downloaded :- {}".format(len(videos))
        search_response = youtube.search().list(channelId=user_id["id"], part="id,snippet", maxResults=max_results, 
                                                pageToken = next_page_token).execute()
        next_page_token = None if "nextPageToken" not in search_response.keys() else search_response.get("nextPageToken")
    except Exception as Ex:
        print "Failed with ERROR :- {}".format(Ex)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        print(exc_type,exc_tb.tb_lineno)


# In[233]:

len(videos)


# In[280]:

final_output = []
final_labels = ["id", "commentCount","subscriberCount", "videoCount", "viewCount", "title","description", "created_at", 
             "videoId","viewCount","likeCount","dislikeCount","commentCount","favoriteCount", "comment_author_name", 
                "comment_publishedAt", "comment_text", "comment_replies", "comment_like_count", "comment_dislike_count", 
                "rep_author_name", "rep_publishedAt", "rep_text", "rep_like_count", "rep_dislike_count"]


# In[281]:

for counter, video_in in enumerate(videos):
    try:
        print "Starting processing for video No:- {}".format(counter + 1)
        comment_obj = youtube.commentThreads().list(part = "snippet,replies", videoId = video_in[8], 
                                                    maxResults=max_results).execute()
        next_comm_token = ""
        while next_comm_token is not None:
            next_comm_token = None if "nextPageToken" not in comment_obj.keys() else comment_obj.get("nextPageToken")
            comments_list = comment_obj.get("items")
            for temp_comment in comments_list:
                if "snippet" in temp_comment.keys():
                    in_snip = temp_comment["snippet"]["topLevelComment"]
                    comment_author_name = in_snip["snippet"]["authorDisplayName"]
                    comment_publishedAt = datetime.strptime(in_snip["snippet"]["publishedAt"], "%Y-%m-%dT%H:%M:%S.000Z").strftime("%Y-%m-%d %H:%M:%S")
                    comment_text = in_snip["snippet"]["textOriginal"].encode("ascii", "ignore")
                    comment_replies = 0 if 'totalReplyCount' not in temp_comment["snippet"] else temp_comment["snippet"]["totalReplyCount"]
                    comment_like_count = 0 if 'likeCount' not in temp_comment["snippet"] else temp_comment["snippet"]["likeCount"]
                    comment_dislike_count = 0 if 'dislikeCount' not in temp_comment["snippet"] else temp_comment["snippet"]["dislikeCount"]
                    if "replies" in temp_comment.keys():
                        for rep in temp_comment["replies"]["comments"]:
                            rep_snip = rep["snippet"]
                            rep_author_name = rep_snip["authorDisplayName"]
                            rep_publishedAt = datetime.strptime(rep_snip["publishedAt"],
                                                                    "%Y-%m-%dT%H:%M:%S.000Z").strftime("%Y-%m-%d %H:%M:%S")
                            rep_text = rep_snip["textOriginal"].encode("ascii", "ignore")
                            rep_like_count = rep_snip["likeCount"]
                            rep_dislike_count = 0 if 'dislikeCount' not in rep_snip else rep_snip["dislikeCount"]
                            final_output.append([video_in[0],video_in[1], video_in[2],video_in[3],video_in[4],
                                                        video_in[5],video_in[6],video_in[7],video_in[8],video_in[9],
                                                        video_in[10],video_in[11],video_in[12],video_in[13],
                                                        comment_author_name, comment_publishedAt, comment_text, 
                                                        comment_replies, comment_like_count, comment_dislike_count, 
                                                        rep_author_name, rep_publishedAt, rep_text, rep_like_count, 
                                                        rep_dislike_count])
                    else:
                        final_output.append([video_in[0],video_in[1], video_in[2],video_in[3],video_in[4],
                                                        video_in[5],video_in[6],video_in[7],video_in[8],video_in[9],
                                                        video_in[10],video_in[11],video_in[12],video_in[13],
                                                    comment_author_name, comment_publishedAt, comment_text, 
                                                    comment_replies, comment_like_count, comment_dislike_count, "NA", 
                                                   "NA", "NA", "NA", "NA"])
            if len(final_output) % 10 == 0 and len(final_output) != 0:
                print "No. of comments downloaded :- {}".format(len(final_output))
            comment_obj = youtube.commentThreads().list(part = "snippet,replies", videoId = video_in[8], 
                                                    maxResults=max_results, pageToken = next_comm_token).execute()
            next_comm_token = None if "nextPageToken" not in comment_obj.keys() else comment_obj.get("nextPageToken")
    except Exception as Ex:
        print "Failed with ERROR :- {}".format(Ex)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        print(exc_type,exc_tb.tb_lineno)


# In[282]:

final_df = pd.DataFrame.from_records(final_output, columns=final_labels)


# In[283]:

final_df


# In[ ]:




# In[ ]:




# In[ ]:




# In[168]:




# ### demo code for extracting Comments and sub comments 

# In[188]:


comments_subComment = []
if "snippet" in temp_comment.keys():
    in_snip = temp_comment["snippet"]["topLevelComment"]
    comment_author_name = in_snip["snippet"]["authorDisplayName"]
    comment_publishedAt = datetime.strptime(in_snip["snippet"]["publishedAt"], "%Y-%m-%dT%H:%M:%S.000Z").strftime("%Y-%m-%d %H:%M:%S")
    comment_text = in_snip["snippet"]["textOriginal"].encode("ascii", "ignore")
    comment_replies = 0 if 'totalReplyCount' not in temp_comment["snippet"] else temp_comment["snippet"]["totalReplyCount"]
    comment_like_count = 0 if 'likeCount' not in temp_comment["snippet"] else temp_comment["snippet"]["likeCount"]
    comment_dislike_count = 0 if 'dislikeCount' not in temp_comment["snippet"] else temp_comment["snippet"]["dislikeCount"]
    if "replies" in temp_comment.keys():
        for rep in temp_comment["replies"]["comments"]:
            rep_snip = rep["snippet"]
            rep_author_name = rep_snip["authorDisplayName"]
            rep_publishedAt = datetime.strptime(rep_snip["publishedAt"],
                                                    "%Y-%m-%dT%H:%M:%S.000Z").strftime("%Y-%m-%d %H:%M:%S")
            rep_text = rep_snip["textOriginal"].encode("ascii", "ignore")
            rep_like_count = rep_snip["likeCount"]
            rep_dislike_count = 0 if 'dislikeCount' not in rep_snip else rep_snip["dislikeCount"]
            comments_subComment.append([comment_author_name, comment_publishedAt, comment_text, comment_replies, 
                                       comment_like_count, comment_dislike_count, rep_author_name, 
                                       rep_publishedAt, rep_text, rep_like_count, rep_dislike_count])
    else:
        comments_subComment.append([comment_author_name, comment_publishedAt, comment_text, comment_replies, 
                                   comment_like_count, comment_dislike_count, "NA", 
                                   "NA", "NA", "NA", "NA"])
        
    


# In[189]:

comments_subComment


# In[178]:

temp_comment["snippet"]["topLevelComment"]


# In[118]:

temp_comment


# In[176]:

temp_comment["replies"]


# In[175]:

temp_comment.keys()


# In[ ]:



