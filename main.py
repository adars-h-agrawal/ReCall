"""
main.py
────────
CLI entry point for RECALL — AI Meeting Intelligence.

Usage:
    python main.py
"""

from dotenv import load_dotenv
from utils.audio_processor import process_input
from core.pipeline import build_meeting
from core.extractor import _format_action_items, _format_decisions, _format_questions
from core.rag_engine import build_rag_chain_from_meeting, ask_question

load_dotenv()


def run_pipeline(source: str, language: str = "english") -> dict:
    """Run the full RECALL pipeline and return a result dictionary.

    Returns a dict with keys:
        title, transcript, summary, action_items (str), key_decisions (str),
        open_questions (str), rag_chain, meeting (Meeting object).
    """
    print("Starting RECALL — AI Meeting Intelligence")

    chunks, cleanup = process_input(source)

    try:
        meeting, summary = build_meeting(
            chunks=chunks,
            source=source,
            language=language,
        )
    finally:
        cleanup()

    transcript = meeting.plain_transcript()
    print(f"Transcription complete ({len(meeting.segments)} segments). "
          f"First 300 chars: {transcript[:300]}")

    rag_chain = build_rag_chain_from_meeting(meeting)

    return {
        "title": meeting.title,
        "transcript": transcript,
        "summary": summary,
        "action_items": _format_action_items(meeting.action_items),
        "key_decisions": _format_decisions(meeting.decisions),
        "open_questions": _format_questions(meeting.open_questions),
        "rag_chain": rag_chain,
        "meeting": meeting,
    }


if __name__ == "__main__":
    source = input("Enter YouTube URL or local file path: ").strip()
    language = input("Language (english/hinglish): ").strip() or "english"
    result = run_pipeline(source, language)

    print("\n" + "=" * 60)
    print(f"📌 Title: {result['title']}")
    print(f"\n📋 Summary:\n{result['summary']}")
    print(f"\n✅ Action Items:\n{result['action_items']}")
    print(f"\n🔑 Key Decisions:\n{result['key_decisions']}")
    print(f"\n❓ Open Questions:\n{result['open_questions']}")
    print("=" * 60)

    print("\n💬 Chat with your meeting (type 'exit' to quit)\n")
    rag_chain = result["rag_chain"]
    while True:
        question = input("You: ").strip()
        if question.lower() in ["exit", "quit", "q"]:
            print("👋 Goodbye!")
            break
        if not question:
            continue
        answer = ask_question(rag_chain, question)
        print(f"\n🤖 Assistant: {answer}\n")
