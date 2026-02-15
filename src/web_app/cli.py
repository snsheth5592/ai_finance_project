# # src/web_app/cli.py
# from __future__ import annotations

# import json

# from src.core.config import load_settings
# from src.utils.logging import setup_logging
# from src.agents.finance_qa_agent import run_finance_qa_agent


# def main() -> None:
#     settings = load_settings("config.yaml")
#     setup_logging(settings.log_level)

#     print("AI Finance Assistant (Finance Q&A Agent) - CLI")
#     print("Type a question. Type 'exit' to quit.\n")

#     while True:
#         q = input("> ").strip()
#         if not q:
#             continue
#         if q.lower() in {"exit", "quit"}:
#             break

#         result = run_finance_qa_agent(q, rag_top_k=settings.rag_top_k)
#         print("\n" + json.dumps(result, indent=2))
#         print()


# if __name__ == "__main__":
#     main()