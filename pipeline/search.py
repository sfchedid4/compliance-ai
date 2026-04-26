import json
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def clean_text(text):
    text = str(text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def load_regulations(file_path="data/regulations.json"):
    with open(file_path, "r", encoding="utf-8") as file:
        regulations = json.load(file)

    return regulations


def regulation_to_text(regulation):
    text = ""

    if "title" in regulation:
        text += str(regulation["title"]) + " "

    if "business_type" in regulation:
        text += str(regulation["business_type"]) + " "

    if "location" in regulation:
        text += str(regulation["location"]) + " "

    if "agency" in regulation:
        text += str(regulation["agency"]) + " "

    if "permit" in regulation:
        text += str(regulation["permit"]) + " "

    if "license" in regulation:
        text += str(regulation["license"]) + " "

    if "features" in regulation:
        if type(regulation["features"]) == list:
            text += " ".join(regulation["features"]) + " "
        else:
            text += str(regulation["features"]) + " "

    if "requirements" in regulation:
        if type(regulation["requirements"]) == list:
            text += " ".join(regulation["requirements"]) + " "
        else:
            text += str(regulation["requirements"]) + " "

    if "text" in regulation:
        text += str(regulation["text"]) + " "

    if "content" in regulation:
        text += str(regulation["content"]) + " "

    return clean_text(text)


def regulation_matches_filters(regulation, filters):
    if filters is None:
        return True

    regulation_text = regulation_to_text(regulation)

    business_type = filters.get("business_type")
    location = filters.get("location")
    features = filters.get("features")
    capacity = filters.get("capacity")

    if business_type is not None:
        if clean_text(business_type) not in regulation_text:
            return False

    if location is not None:
        location_clean = clean_text(location)

        if location_clean not in regulation_text:
            if "los angeles" not in regulation_text and "la" not in regulation_text:
                return False

    if features is not None and len(features) > 0:
        feature_found = False

        for feature in features:
            if clean_text(feature) in regulation_text:
                feature_found = True

        if feature_found == False:
            return False

    if capacity is not None:
        capacity_words = ["capacity", "seats", "seating", "people", "guests", "occupancy"]

        capacity_related = False

        for word in capacity_words:
            if word in regulation_text:
                capacity_related = True

        if capacity_related == False:
            pass

    return True


def search_regulations(query, file_path="data/regulations.json", top_k=3, filters=None):
    regulations = load_regulations(file_path)

    filtered_regulations = []

    for regulation in regulations:
        if regulation_matches_filters(regulation, filters):
            filtered_regulations.append(regulation)

    if len(filtered_regulations) == 0:
        filtered_regulations = regulations

    documents = []

    for regulation in filtered_regulations:
        document_text = regulation_to_text(regulation)
        documents.append(document_text)

    model = SentenceTransformer("all-MiniLM-L6-v2")

    document_embeddings = model.encode(documents)
    query_embedding = model.encode([clean_text(query)])

    similarity_scores = cosine_similarity(query_embedding, document_embeddings)[0]

    sorted_indexes = np.argsort(similarity_scores)[::-1]

    results = []

    for index in sorted_indexes[:top_k]:
        regulation = filtered_regulations[index].copy()
        regulation["score"] = float(similarity_scores[index])
        regulation["matched_text"] = documents[index]
        results.append(regulation)

    return results


def retrieve_relevant_regulations(query, filters=None, file_path="data/regulations.json", top_k=3):
    results = search_regulations(
        query=query,
        file_path=file_path,
        top_k=top_k,
        filters=filters
    )

    return results


if __name__ == "__main__":
    user_query = "I'm opening a Korean BBQ restaurant with a full bar and rooftop patio in Koreatown, about 80 seats"

    example_filters = {
        "business_type": "restaurant",
        "features": ["liquor", "outdoor_dining"],
        "location": "Koreatown",
        "capacity": 80
    }

    results = retrieve_relevant_regulations(
        query=user_query,
        filters=example_filters,
        top_k=3
    )

    for result in results:
        print("Title:", result.get("title", "No title"))
        print("Score:", result["score"])
        print("Text:", result["matched_text"][:300])
        print()
