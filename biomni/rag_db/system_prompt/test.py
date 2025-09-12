from langchain_aws.embeddings.bedrock import BedrockEmbeddings
from langchain_community.vectorstores import FAISS
from typing import List, Tuple, Dict, Any


class RAGDatabaseTester:
    """Test class for evaluating RAG database retrieval performance."""

    def __init__(self):
        self.embeddings = BedrockEmbeddings(normalize=True)
        self.databases = self._load_databases()

    def _load_databases(self) -> Dict[str, FAISS]:
        """Load all FAISS databases."""
        return {
            "tool": FAISS.load_local(
                "tool_index", self.embeddings, allow_dangerous_deserialization=True
            ),
            "data_lake": FAISS.load_local(
                "data_lake_index", self.embeddings, allow_dangerous_deserialization=True
            ),
            "library": FAISS.load_local(
                "library_index", self.embeddings, allow_dangerous_deserialization=True
            ),
        }

    def _create_biomedical_query(self, user_query: str) -> str:
        """Create a structured query for biomedical research assistance."""
        return f"""
You are an expert biomedical research assistant. Your task is to select the relevant resources to help answer a user's query.

USER QUERY: {user_query}

Be generous in your selection - include resources that might be useful for the task, even if they're not explicitly mentioned in the query.
It's better to include slightly more resources than to miss potentially useful ones.

IMPORTANT GUIDELINES:
1. Be generous but not excessive - aim to include all potentially relevant resources
2. ALWAYS prioritize database tools for general queries - include as many database tools as possible
4. For wet lab sequence type of queries, ALWAYS include molecular biology tools
5. For data lake items, include datasets that could provide useful information
6. For libraries, include those that provide functions needed for analysis
7. Don't exclude resources just because they're not explicitly mentioned in the query
8. When in doubt about a database tool or molecular biology tool, include it rather than exclude it
"""

    def _search_database(
        self, db: FAISS, query: str, threshold: float, k: int = 30
    ) -> List[Tuple[Any, float]]:
        """Search a single database with given parameters."""
        return db.similarity_search_with_relevance_scores(
            query,
            score_threshold=threshold,
            k=k,
        )

    def _print_results(
        self, db_name: str, results: List[Tuple[Any, float]], total_docs: int
    ):
        """Print search results in a formatted way."""
        print(f"\n{db_name.upper()} DATABASE:")
        print(f"Found {len(results)} results out of {total_docs} total documents")

        for res, score in results:
            name = res.metadata["name"]
            description = res.metadata["description"]
            print(f"\t{name}\t{score:.3f}\t{description}")

    def test_retrieval(self, user_query: str = "KRAS에 대해서 조사해줘"):
        """Test retrieval across all databases with different thresholds."""
        query = self._create_biomedical_query(user_query)

        # Database-specific thresholds
        thresholds = {"tool": 0.2, "data_lake": 0.05, "library": 0.0}

        for db_name, db in self.databases.items():
            threshold = thresholds[db_name]
            total_docs = db.index.ntotal
            results = self._search_database(db, query, threshold)
            self._print_results(db_name, results, total_docs)


# Run the test
if __name__ == "__main__":
    tester = RAGDatabaseTester()
    tester.test_retrieval()
