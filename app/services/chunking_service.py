import tiktoken

def chunk_text(text: str, chunk_size: int= 500, overlap:int=50) ->list[dict]:
    encoder = tiktoken.get_encoding("cl100k_base")
    tokens = encoder.encode(text)
    
    chunks = []
    start = 0
    chunk_index = 0
    
    while start  < len(tokens):
        end = start + chunk_size
        chunk_tokens = tokens[start:end]
        chunk_text_decoded = encoder.decode(chunk_tokens)
        
        chunks.append({
            "chunk_index": chunk_index,
            "text" : chunk_text_decoded,
            "token_count" : len(chunk_tokens),
            "start_token": start,
            "end_token": end
        })
        
        start = end - overlap
        chunk_index+=1
   
    return chunks