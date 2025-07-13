import asyncio
import httpx

async def call_vllm_async(chunk, idx, url="http://localhost:8000/v1/completions", model="gemma-3-12b-it", is_chat=False):
    payload = {
        "model": model,
        "max_tokens": 1024,
        "temperature": 0.0,
    }

    if is_chat:
        payload["messages"] = [{"role": "user", "content": chunk}]
        endpoint = "/v1/chat/completions"
    else:
        payload["prompt"] = chunk
        endpoint = "/v1/completions"
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(url + endpoint, json=payload)
            result = response.json()

        if is_chat:
            corrected = result["choices"][0]["message"]["content"]
        else:
            corrected = result["choices"][0]["text"]
    
    except Exception as e:
        print(f"Error occurred: {e}")
        corrected = chunk

    return idx, corrected

async def async_process_chunks(chunks, max_concurrency=10):
    sem = asyncio.Semaphore(max_concurrency)

    async def sem_call(chunk, idx):
        async with sem:
            return await call_vllm_async(chunk, idx)

    tasks = [sem_call(chunk, idx) for idx, chunk in enumerate(chunks)]
    results = await asyncio.gather(*tasks)

    results.sort(key=lambda x: x[0])
    merged_text = "\n".join([text for _, text in results])
    return merged_text
