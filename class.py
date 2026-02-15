import csv
import ollama
import chromadb
from chromadb.config import Settings

class BankingFAQChatbot:
    def __init__(self, csv_file='banking_faqs.csv'):
        self.faqs = []
        self.questions = []
        self.answers = []
        self.categories = []
        self.load_faqs(csv_file)

        # Initialize ChromaDB vector database
        self.chroma_client = chromadb.Client(Settings(
            anonymized_telemetry=False
        ))

        # Create or get collection
        self.collection = self.chroma_client.get_or_create_collection(
            name="banking_faqs",
            metadata={"description": "Banking FAQ embeddings"}
        )

        # Build vector database
        self.build_vector_database()

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

    def build_vector_database(self):
        """Build vector database using Ollama embeddings"""
        print("🔧 Building vector database with Ollama embeddings...")

        # Clear existing data
        try:
            self.chroma_client.delete_collection("banking_faqs")
            self.collection = self.chroma_client.create_collection(
                name="banking_faqs",
                metadata={"description": "Banking FAQ embeddings"}
            )
        except:
            pass

        # Generate embeddings for all questions using Ollama
        for idx, question in enumerate(self.questions):
            # Get embedding from Ollama
            response = ollama.embeddings(
                model='llama2',
                prompt=question
            )
            embedding = response['embedding']

            # Add to ChromaDB
            self.collection.add(
                embeddings=[embedding],
                documents=[question],
                metadatas=[{
                    'answer': self.answers[idx],
                    'category': self.categories[idx]
                }],
                ids=[f"faq_{idx}"]
            )

        print(f"✓ Vector database built with {len(self.questions)} FAQs")

    def retrieve_similar_questions(self, query, top_k=3):
        """
        RAG: Retrieve top-k most similar FAQ questions using vector database
        """
        # Generate embedding for the query using Ollama
        response = ollama.embeddings(
            model='llama2',
            prompt=query
        )
        query_embedding = response['embedding']

        # Query ChromaDB for similar questions
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )

        # Format results
        top_results = []
        for i in range(len(results['documents'][0])):
            # Distance to similarity conversion (ChromaDB returns distances)
            distance = results['distances'][0][i]
            similarity = 1 / (1 + distance)  # Convert distance to similarity score

            metadata = results['metadatas'][0][i]
            top_results.append({
                'index': i,
                'score': similarity,
                'question': results['documents'][0][i],
                'answer': metadata['answer'],
                'category': metadata['category']
            })

        return top_results

    def prompt_engineering_response(self, query, top_results):
        best_match = top_results[0]
        confidence = best_match['score']

        # Low confidence - show suggestions
        if confidence < self.confidence_threshold:
            print(f"\n❓ I'm not sure what you're asking about.")
            print(f"\n💡 Did you mean:")
            for i, result in enumerate(top_results, 1):
                print(f"    {i}. {result['question']}")
            print(f"\n📞 For more help, please call: 1-800-BANK-123")
            return None

        # Medium confidence - show "did you mean"
        if confidence < 0.6:
            print(f"\n💡 Did you mean: \"{best_match['question']}\"?\n")
            print(f"{best_match['answer']}")
            return best_match['answer']

        # High confidence - show direct answer
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
    print(f"✓ Vector database: ChromaDB with Ollama embeddings")
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
