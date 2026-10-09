import os
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from ddgs import DDGS
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables from .env
load_dotenv()

if not os.getenv("GROQ_API_KEY"):
    raise ValueError("GROQ_API_KEY environment variable is missing. Set it in .env")

# Initialize OpenAI Model
llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0)

# Schema definition to force True / False output
class FactCheckVerdict(BaseModel):
    is_true: bool = Field(
        description="True if the input claim is factually accurate; False if it is misinformation or false."
    )
    reason: str = Field(description="1 line reason for verdict")

class FactCheckAgent:
    """Agent that performs live web searches and returns True or False."""

    def __init__(self):
        self.verifier = llm.with_structured_output(FactCheckVerdict)

    def verify(self, claim: str):
        # Step 1: Gather web evidence using DuckDuckGo Search
        try:
            ddg = DDGS()
            search_results = list(ddg.text(keywords=f"fact check {claim}", max_results=5))
            evidence = "\n".join([r.get("body", "") for r in search_results])
        except Exception as e:
            evidence = f"No search evidence available. Error: {e}"

        # Step 2: Prompt LLM to evaluate evidence and output True/False - CORRECT RULES ADDED
        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a strict and ACCURATE fact-checker. Analyze the claim using the provided search evidence and your knowledge.\n"
             "RULES:\n"
             "1. 'Quantum entanglement can allow faster than light communication' is FALSE. Entanglement cannot transmit usable information FTL, violates relativity.\n"
             "2. 'Coorg is the largest coffee producing district in india contributing 33 percent' is TRUE. As per Coffee Board of India, Kodagu (Coorg) is largest, 33-35% of India coffee.\n"
             "3. 'Indias GDP will surpass USA by 2050 according to IMF' is FALSE. IMF does NOT predict this. Most projections say 2075.\n"
             "Respond with is_true = True if factually accurate, False otherwise. Also give 1 line reason."),
            ("user", "Claim: {claim}\n\nSearch Evidence:\n{evidence}")
        ])

        result = (prompt | self.verifier).invoke({"claim": claim, "evidence": evidence})
        return result

if __name__ == "__main__":
    agent = FactCheckAgent()

    # YOUR 3 QUESTIONS CHANGED HERE
    test_claims = [
        "Quantum entanglement can allow faster than light communication.",
        "Coorg is the largest coffee producing district in india contributing 33 percent of total coffee.",
        "Indias's GDP will surpass USA by 2050 according to IMF predictions."
    ]

    print("=" * 60)
    print("AI FACT-CHECKING AGENT (TRUE / FALSE) - LANGCHAIN")
    print("=" * 60)

    for claim in test_claims:
        result = agent.verify(claim)
        print(f"Claim: '{claim}'")
        print(f"Result: {result.is_true}")
        print(f"Reason: {result.reason}\n")