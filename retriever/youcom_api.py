import os
import re
import requests
import logging


YOUCOM_API_KEY=os.environ["YOUCOM_API_KEY"]


youcom_query_rewrite_prompt = """
Suggest You.com search queries to retrieve relevant information to answer the following question related to the most recent research. The search queries should\n
##\n
1. Identify the essential problem.\n
2. Think step by step to reason and describe what information could be relevant and helpful to address the questions in detail.\n
3. Draft an answer with as many thoughts as you have.\n
Now generate a new search query for this question:
Question: {question}\n
Search query:
"""

def get_snippets_for_query(query):
    
    headers = {"X-API-Key": YOUCOM_API_KEY}
    params = {"query": query}
    return requests.get(
        f"https://api.ydc-index.io/search",
        params=params,
        headers=headers,
    ).json()


def search_youcom(question, client=None, model_name=None, use_query_rewriting=False, cache=None):
    if use_query_rewriting:
        if cache is not None and question in cache.cot_query_cache:
            new_query = cache.cot_query_cache[question]
        else:
            new_query = query_rewrite(question, client, model_name=model_name)
            if cache is not None:
                cache.cot_query_cache[question] = new_query
        print(new_query)
    else:
        new_query = question
        if client is not None:
            logging.warning("We are not using client for query rewriting for YOU.com search.")
    
    if cache is not None and new_query in cache.search_cache:
        search_results = cache.search_cache[new_query]
    else:
        search_results = get_snippets_for_query(new_query)['hits']
        if cache is not None:
            cache.search_cache[new_query] = search_results
    
    return search_results, cache


def call_api(input_query, client, model_name="meta-llama/Llama-3-70b-chat-hf", max_tokens=1500, ):
    chat_completion = client.chat.completions.create(
    model=model_name,
    messages=[{"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": input_query}],
    temperature=0.1,
    max_tokens=max_tokens,
    )
    return chat_completion.choices[0].message.content


def query_rewrite(question, client, model_name):
    keywords = call_api(youcom_query_rewrite_prompt.format_map({"question": question}), client, model_name=model_name)
    # if "Search queries:" in keywords and len(keywords.split("\n\nSearch queries: ")) > 1:
    #     keywords = keywords.split("\n\nSearch queries: ")[1]
    return keywords.replace("Search queries: " , "")
    
    
def extract_youcom_relevant_info(results, topk=None):
    useful_info = []
    for hit in results[:topk]:
        info = {
            'title': hit['title'],
            'snippet': '\n'.join(hit['snippets']),
            'url': hit['url'],
        }
        useful_info.append(info)
    if len(useful_info) < topk:
        logging.warning(f"Not enough context retrieved: expecting {topk}, retrieved {len(useful_info)}.")
    return useful_info

def format_youcom_document_string(results, top_k, max_doc_len=None):
    relevant_info = extract_youcom_relevant_info(results, top_k)
    formatted_documents = ""
    for i, doc_info in enumerate(relevant_info):
        snippet = doc_info.get('snippet', "")
        if snippet is not None:
            clean_snippet = re.sub('<[^<]+?>', '', snippet)  # Removes HTML tags
        
            formatted_documents += f"**Document {i + 1}:**\n"
            formatted_documents += f"**Title:** {doc_info.get('title', '')}\n"
            formatted_documents += f"**Snippet:** {clean_snippet}\n"

    print(f"{len(formatted_documents.split(' '))} words in context.")
    if max_doc_len is not None:
        raw_doc_len = len(formatted_documents.split(' '))
        if raw_doc_len > max_doc_len:
            print(f"Documents exceeded max_doc_len, cutting from {raw_doc_len} to {max_doc_len} words.")
            formatted_documents = ' '.join(formatted_documents.split(' ')[:max_doc_len])
    
    return formatted_documents


if __name__ == '__main__':
    results = get_snippets_for_query("Two quantum states with energies E1 and E2 have a lifetime of 10^-9 sec and 10^-8 sec, respectively. We want to clearly distinguish these two energy levels. Which one of the following options could be their energy difference so that they can be clearly resolved?\n Choices:\n(A) 10^-9 eV\n(B) 10^-4 eV\n(C) 10^-8 eV\n\n(D) 10^-11 eV\n")
    import pdb; pdb.set_trace()