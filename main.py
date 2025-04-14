import openai
from retriever import retrieve_chunks
import json

def generate_answer(query, retrieved_chunks):
    context = "\n\n".join(retrieved_chunks)
    prompt = f"""Use the following context to answer the question.

    Context:
    {context}

    Question:
    {query}
    
    Answer: """

    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content

results = []

print("Do you have a question about your health plan?")
while True:
    query = input("> ")
    if query.strip().lower() in {"no", "exit", "quit"}:
        print("Okay, have a great day!")
        break
    top_chunks = retrieve_chunks(query)
    answer = generate_answer(query, top_chunks)
    print(answer)
    rating = input("\nRate this answer from 1 to 5: ")

    # Save interaction
    entry = {
        "question": query,
        "chunks": top_chunks,
        "answer": answer,
        "rating": rating,
    }
    results.append(entry)

    # Save to file incrementally
    with open("rag_feedback.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Can I help you with anything else?")
