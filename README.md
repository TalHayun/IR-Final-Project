# Final-Project

Intro
=====
In this project we built a search engine for 6 million (approximetely) of English Wikipedia.

Contents
==========

<!--ts-->
- [search_frontend](#search_frontend)
- [create_indexes](#create_indexes)
<!--te-->


## search_frontend
This file contains 5 main methods that process queries and return resultsfrom the entire corpus :

- `search` - The main method, returns up to a 100 of our best search results for the query. The search results ordered from best to worst where each element is a tuple (wiki_id, title).

- `search_body` - Method that return up to 100 search results for the query using TFIDIF and Cosine Similarity of the body of articles only.

- `search_title` - Method that returns all search results that contain a query word in the title of articles, ordered in descending order of the NUMBER OF DISTINCT QUERY WORDS that appear in the title.

- `search_anchor` - Method that returns all search results that contain a query word in the anchor text of articles, ordered in
descending order of the NUMBER OF QUERY WORDS that appear in anchor text linking to the page.

- `get_pagerank` - Method that returns PageRank values for a list of provided wiki article IDs.

- `get_pageview` - Method that returns the number of page views that each of the provide wiki articles had in August 2021.

## create_indexes

- `create_index` - The main method to create inverted index instance for title/body/anchor. In addition, addinng the posting locations and df to the inverted index.

- `create_index_for_5_func` - Method to create inverted index instance according to the required rules (without stremming and given regex).
