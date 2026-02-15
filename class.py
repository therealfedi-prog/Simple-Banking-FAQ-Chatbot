import csv
import re
from collections import defaultdict
import math


class BankingFAQChatbot:
    def __init__(self, csv_file='banking_faqs.csv'):
        self.faqs = []
        self.questions = []
        self.answers = []
        self.categories = []
        self.load_faqs(csv_file)
        self.vocab = {}
        self.idf = {}
        self.question_vectors = []
        self.build_tfidf_model()
        self.confidence_threshold = 0.3

    def load_faqs(self, csv_file):
        try:
            with open(csv_file, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    self.faqs.append(row)
                    self.questions.append(row['question'])
                    self.answers.append(row['answer'])
                    self.categories.append(row['category'])
        except FileNotFoundError:
            print(f"✗ Error: Could not find {csv_file}")
            print("  Make sure banking_faqs.csv is in the same directory")
            exit(1)

    def preprocess_text(self, text):
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        tokens = text.split()
        stop_words = {'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves',
                      'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him',
                      'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its',
                      'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what',
                      'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am',
                      'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has',
                      'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the',
                      'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of',
                      'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
                      'through', 'during', 'before', 'after', 'above', 'below', 'to',
                      'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under',
                      'again', 'further', 'then', 'once'}
        tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
        return tokens

    def compute_term_frequency(self, tokens):
        tf = defaultdict(int)
        for token in tokens:
            tf[token] += 1
        max_freq = max(tf.values()) if tf else 1
        for token in tf:
            tf[token] = tf[token] / max_freq
        return dict(tf)

    def build_tfidf_model(self):
        tokenized_questions = [self.preprocess_text(q) for q in self.questions]

        doc_freq = defaultdict(int)
        for tokens in tokenized_questions:
            unique_tokens = set(tokens)
            for token in unique_tokens:
                doc_freq[token] += 1
        self.vocab = {word: idx for idx, word in enumerate(doc_freq.keys())}

        num_docs = len(tokenized_questions)
        for word in doc_freq:
            self.idf[word] = math.log(num_docs / (1 + doc_freq[word]))

        for tokens in tokenized_questions:
            tf = self.compute_term_frequency(tokens)
            vector = {}
            for word, tf_value in tf.items():
                if word in self.vocab:
                    vector[self.vocab[word]] = tf_value * self.idf[word]
            self.question_vectors.append(vector)

    def vectorize_query(self, query):
        tokens = self.preprocess_text(query)
        tf = self.compute_term_frequency(tokens)

        vector = {}
        for word, tf_value in tf.items():
            if word in self.vocab:
                vector[self.vocab[word]] = tf_value * self.idf[word]
        return vector

    def cosine_similarity(self, vec1, vec2):

        common_keys = set(vec1.keys()) & set(vec2.keys())

        if not common_keys:
            return 0.0

        dot_product = sum(vec1[key] * vec2[key] for key in common_keys)
        mag1 = math.sqrt(sum(val ** 2 for val in vec1.values()))
        mag2 = math.sqrt(sum(val ** 2 for val in vec2.values()))

        if mag1 == 0 or mag2 == 0:
            return 0.0

        return dot_product / (mag1 * mag2)

    def retrieve_similar_questions(self, query, top_k=3):
        query_vector = self.vectorize_query(query)
        similarities = []
        for idx, faq_vector in enumerate(self.question_vectors):
            similarity = self.cosine_similarity(query_vector, faq_vector)
            similarities.append((idx, similarity))

        similarities.sort(key=lambda x: x[1], reverse=True)

        # Get top-k results
        top_results = []
        for idx, score in similarities[:top_k]:
            top_results.append({
                'index': idx,
                'score': score,
                'question': self.questions[idx],
                'answer': self.answers[idx],
                'category': self.categories[idx]
            })

        return top_results

    def prompt_engineering_response(self, query, top_results):
        best_match = top_results[0]
        confidence = best_match['score']

        if confidence < self.confidence_threshold:
            print(f"\n❓ I'm not sure what you're asking about.")
            print(f"\n💡 Did you mean:")
            for i, result in enumerate(top_results, 1):
                print(f"    {i}. {result['question']}")
            print(f"\n📞 For more help, please call: 1-800-BANK-123")
            return None

        if confidence < 0.6:
            print(f"\n💡 Did you mean: \"{best_match['question']}\"?\n")
            print(f"{best_match['answer']}")
            return best_match['answer']

        print(f"\n{best_match['answer']}")

        return best_match['answer']

    def chat(self, query):
        top_results = self.retrieve_similar_questions(query, top_k=3)
        answer = self.prompt_engineering_response(query, top_results)
        return answer


def display_banner():

    print("\n" + "=" * 80)
    print(" " * 20 + "🏦 BANKING FAQ CHATBOT 🏦")
    print("=" * 80)
    print("\n💬 Ask me anything about banking services!")
    print("\n" + "-" * 80)
    print("Commands:")
    print("  • Type your question and press Enter")
    print("  • Type 'quit' or 'exit' to end the conversation")
    print("  • Type 'help' to see available topics")
    print("-" * 80 + "\n")


def display_help(chatbot):

    print("\n" + "=" * 80)
    print("📖 AVAILABLE TOPICS")
    print("=" * 80)


    categories = {}
    for faq in chatbot.faqs:
        cat = faq['category']
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(faq['question'])

    for category, questions in sorted(categories.items()):
        print(f"\n📁 {category} ({len(questions)} questions)")
        for i, q in enumerate(questions[:3], 1):
            print(f"   {i}. {q}")
        if len(questions) > 3:
            print(f"   ... and {len(questions) - 3} more")


def display_stats(chatbot):

    print("\n" + "=" * 80)
    print("📊 SYSTEM STATISTICS")
    print("=" * 80)
    print(f"\n✓ Total FAQs loaded: {len(chatbot.faqs)}")
    print(f"✓ Vocabulary size: {len(chatbot.vocab)} unique words")
    print(f"✓ Categories: {len(set(chatbot.categories))}")
    print(f"✓ Confidence threshold: {chatbot.confidence_threshold * 100}%")
    print(f"✓ RAG retrieval: Top 3 similar questions")


    category_counts = {}
    for cat in chatbot.categories:
        category_counts[cat] = category_counts.get(cat, 0) + 1

    print(f"\n📁 FAQs by Category:")
    for cat, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"   • {cat}: {count} questions")


def main():

    display_banner()


    chatbot = BankingFAQChatbot('banking_faqs.csv')

    print("✅ Ready! Ask me anything.\n")


    while True:
        try:

            user_input = input("You: ").strip()


            if user_input.lower() in ['quit', 'exit', 'q', 'bye']:
                print("\n👋 Thank you for using our banking chatbot! Have a great day!")
                break


            if user_input.lower() in ['help', 'h', '?']:
                display_help(chatbot)
                continue


            if user_input.lower() in ['stats', 'statistics', 'info']:
                display_stats(chatbot)
                continue


            if not user_input:
                continue

            chatbot.chat(user_input)
            print("\n" + "-" * 80 + "\n")

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Please try again or type 'help' for assistance.\n")


if __name__ == "__main__":
    main()