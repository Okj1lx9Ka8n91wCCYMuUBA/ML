from sentence_transformers import SentenceTransformer, util


grant_embeddings = model.encode([grant['description'] for grant in grants], convert_to_tensor=True)

def find_top_grants_for_startup(startup, grant_embeddings):
    """
    This function finds the top 5 grants that are most similar to a given startup based on their descriptions.
    It uses the SentenceTransformer model to generate embeddings for both the startup and grants,
    and then calculates the cosine similarity between them.

    Parameters:
    startup (dict): A dictionary containing the startup's information, with 'Описание' as a key for the description.
    grants (list): A list of dictionaries, where each dictionary represents a grant and contains 'title', 'url', and 'description' keys.

    Returns:
    list: A list of dictionaries, where each dictionary represents a top grant and contains 'title', 'url', 'description', and 'similarity' keys.
    """
    model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

    startup_embedding = model.encode(startup['Описание'], convert_to_tensor=True)


    similarities = util.pytorch_cos_sim(startup_embedding, grant_embeddings).squeeze()

    top_k = similarities.topk(5)
    top_grants = []
    for idx, similarity in zip(top_k.indices, top_k.values):
        grant = grants[idx]
        top_grants.append({
            'title': grant['title'],
            'url': grant['url'],
            'description': grant['description'],
            'similarity': round(similarity.item(), 4)
        })

    return top_grants
