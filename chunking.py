import json
import os
with open("data.json", "r") as f:
    speech = json.load(f)

text = "\n".join(": ".join([item.get("speaker"), item.get("text")]) for item in speech["segment"])


from transformers import AutoTokenizer

HF_TOKEN = os.getenv("HF_TOKEN", "")
tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-12b-it", token=HF_TOKEN, cache_dir="./tokenizer/gemma-3-12b-it")

def chunking(text, tokenizer, max_tokens=500):
    lines = text.split("\n")
    chunks = []
    current_chunk = []
    chunk_token_count = 0

    for line in lines:
        tokenized = tokenizer.encode(line, add_special_tokens=True)
        token_count = len(tokenized)

        # 초과하면 이전 chunk를 저장하고 이 줄부터 새 chunk 시작
        if chunk_token_count + token_count > max_tokens:
            if current_chunk:
                chunks.append("\n".join(current_chunk))
            current_chunk = [line]
            chunk_token_count = token_count
        else:
            current_chunk.append(line)
            chunk_token_count += token_count

    if current_chunk:
        chunks.append("\n".join(current_chunk))

    return chunks

if __name__ == "__main__":
    chunks = chunking(text, tokenizer)
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1}:")
        print(chunk)

