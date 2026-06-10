from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

faq_questions = [
    "What is Artificial Intelligence?",
    "What is Machine Learning?",
    "What is Python?",
    "What is Data Science?",
    "Who developed Python?"
]

faq_answers = [
    "Artificial Intelligence is the simulation of human intelligence by machines.",
    "Machine Learning is a subset of AI that learns from data.",
    "Python is a popular programming language.",
    "Data Science is the process of extracting insights from data.",
    "Python was developed by Guido van Rossum."
]

vectorizer = TfidfVectorizer()

while True:
    user_input = input("\nYou: ")

    if user_input.lower() == "exit":
        print("Bot: Goodbye!")
        break

    all_questions = faq_questions + [user_input]

    tfidf_matrix = vectorizer.fit_transform(all_questions)

    similarity = cosine_similarity(
        tfidf_matrix[-1],
        tfidf_matrix[:-1]
    )

    index = similarity.argmax()

    print("Bot:", faq_answers[index])
