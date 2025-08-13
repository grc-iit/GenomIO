# tools/planner_tools.py

from langchain_core.tools import tool
from rag.retriever import retrieve_context
from rag.gap_filler import predict_until_length, build_masked_input_with_context, tokenizer, model

@tool
def context_tool(dna: str) -> str:
    """Get genomic context for a DNA sequence with gaps."""
    print("[INFO] Using context tool to retrieve context from input sequence")
    return retrieve_context(dna)


@tool
def gap_filler_tool(sequence: str, rag_matches: list[str], gap_length: int) -> str:
    """
    Fill DNA gaps using RAG-informed context + GENA-LM.
    Input format: 
        "sequence": "ATGG---TTCA---GGA"
        "rag_matches": [top_seq1, top_seq2, top_seq3]
        "gap_length": int
    """
    print("[INFO] Using gap filling tool to fill gaps from input sequence")

    if "---" not in sequence:
        return f"No gaps detected: {sequence}"

    masked_input, added_context = build_masked_input_with_context(sequence, rag_matches)
    adjusted_gap = gap_length - added_context
    adjusted_gap = max(1, adjusted_gap)

    print(f"[INFO] MLM input: {masked_input}")
    print(f"[INFO] Adjusted gap length (after context): {adjusted_gap}")

    filled = predict_until_length(masked_input, tokenizer, model, adjusted_gap)
    return f"The gaps in the DNA sequence {sequence} have been filled: {sequence.replace('---', filled)}"