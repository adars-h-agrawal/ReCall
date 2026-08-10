# RECALL

### AI Meeting Intelligence

> **Meetings in. Intelligence out.**

RECALL transforms raw meeting recordings into a searchable, decision-ready knowledge workspace.

Upload a meeting recording or provide a YouTube URL. RECALL transcribes the conversation, generates an executive summary, extracts decisions, action items, and open questions, indexes the transcript for semantic retrieval, and lets you ask evidence-grounded questions about the meeting.

Instead of forcing you to rewatch an hour-long recording, RECALL turns the conversation into something you can **search, understand, and act on.**

---

## ✦ What is RECALL?

Meetings contain valuable information, but most of it disappears into recordings and transcripts.

RECALL is designed around a simple workflow:

```text
        RECORDING
            │
            ▼
      ┌─────────────┐
      │ Transcribe  │
      └──────┬──────┘
             │
             ▼
      ┌─────────────┐
      │ Understand  │
      │             │
      │ Summary     │
      │ Decisions   │
      │ Actions     │
      │ Questions   │
      └──────┬──────┘
             │
             ▼
      ┌─────────────┐
      │   Index     │
      │             │
      │ Embeddings  │
      │ + ChromaDB  │
      └──────┬──────┘
             │
             ▼
      ┌─────────────┐
      │    ASK      │
      │             │
      │ RAG + LLM   │
      │ + Evidence  │
      └──────┬──────┘
             │
             ▼
      ANSWERS YOU CAN TRUST
```

---

# ✨ Features

### 🎙️ Meeting ingestion

- Upload local meeting recordings
- Process YouTube meeting recordings
- Supports common media formats including:
  - MP4
  - MP3
  - WAV
  - M4A
  - WEBM
- Automatic audio preprocessing and chunking

### 📝 Intelligent transcription

- Local Whisper transcription for English meetings
- Sarvam AI transcription/translation support for Hindi/Hinglish workflows
- Timestamped transcript segments
- Structured meeting representation rather than treating the transcript as one giant string

### 🧠 Meeting intelligence

RECALL automatically extracts:

- Executive summary
- Key decisions
- Action items
- Open questions
- Meeting title
- Full searchable transcript

### 🔎 Evidence-grounded RAG

Ask questions such as:

> "How did the cart generation work?"

or:

> "What were the main decisions?"

RECALL retrieves relevant transcript segments and generates an answer using only the meeting context.

Every grounded answer can expose its supporting transcript evidence with timestamps.

```text
Question
   │
   ▼
Meeting-scoped Retriever
   │
   ▼
Relevant Transcript Segments
   │
   ▼
LLM
   │
   ▼
Answer + Source Evidence
```

### 🔐 Meeting-isolated retrieval

Every meeting receives its own identifier.

Transcript segments stored in ChromaDB carry provenance metadata such as:

- `meeting_id`
- `segment_id`
- `start`
- `end`
- `speaker`
- `language`
- `source`

Retrieval is filtered by the current `meeting_id`, preventing documents from unrelated meetings from entering the RAG context.

This isolation happens at the **retrieval layer**, rather than relying on the LLM to ignore unrelated context.

### 📚 Provenance-aware evidence

RECALL does not ask the LLM to invent timestamps or citations.

Evidence is derived directly from retrieved transcript documents.

Each evidence item preserves:

```text
segment_id
start timestamp
end timestamp
transcript text
```

Retrieved evidence is also deduplicated by transcript segment so the same segment does not appear repeatedly when multiple sub-chunks are retrieved.

### 💬 Ask RECALL

The meeting becomes an interactive knowledge base.

Example questions:

```text
What were the main decisions?

What action items were identified?

How did the cart generation work?

Who owns which tasks?

What budget constraint was discussed?

What did they say about nuclear reactors?
```

If the requested information is not present in the transcript, RECALL explicitly says so rather than fabricating an answer.

### 📄 Export

Generate downloadable meeting reports in:

- PDF
- TXT

### 🎨 Premium workspace

RECALL uses a cinematic dark workspace inspired by modern AI productivity products.

The interface is designed around:

- strong typography
- restrained gradients
- semantic color
- ambient lighting
- glass-like surfaces
- evidence-focused information hierarchy
- minimal interaction friction

The goal is not to make the application look like a generic dashboard.

It should feel like a **tool you would actually want to use.**

---

# 🏗️ Architecture

```text
                         ┌──────────────────────┐
                         │     Meeting Input    │
                         │                      │
                         │  YouTube / Upload    │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │   Audio Processing  │
                         │                      │
                         │  yt-dlp / pydub     │
                         │  FFmpeg / chunking   │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │    Transcription    │
                         │                      │
                         │ Whisper / Sarvam AI  │
                         └──────────┬───────────┘
                                    │
                                    ▼
                     ┌──────────────────────────────┐
                     │       Meeting Pipeline        │
                     │                              │
                     │  Title                       │
                     │  Summary                     │
                     │  Decisions                   │
                     │  Action Items                │
                     │  Open Questions              │
                     └──────────────┬───────────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │    Vector Store      │
                         │                      │
                         │ ChromaDB             │
                         │ HuggingFace Embeddings│
                         │ Provenance Metadata  │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │    Meeting-scoped    │
                         │      Retrieval       │
                         │                      │
                         │   meeting_id filter  │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │      RAG Engine      │
                         │                      │
                         │ LangChain LCEL       │
                         │ Mistral LLM          │
                         └──────────┬───────────┘
                                    │
                                    ▼
                    ┌────────────────────────────────┐
                    │          RECALL UI             │
                    │                                │
                    │ Summary • Decisions • Actions  │
                    │ Questions • Transcript         │
                    │                                │
                    │ Ask → Answer → Evidence        │
                    └────────────────────────────────┘
```

---

# 🧩 Core Pipeline

```text
Input
  ↓
Audio normalization
  ↓
Audio chunking
  ↓
Speech-to-text
  ↓
Structured transcript segments
  ↓
Meeting title generation
  ↓
Map-reduce summarization
  ↓
Decision / Action / Question extraction
  ↓
Embedding generation
  ↓
ChromaDB indexing
  ↓
Meeting-scoped retrieval
  ↓
Mistral-powered RAG
  ↓
Answer + transcript evidence
```

---

# 🧠 RAG Design

One of the main engineering goals of RECALL is to make meeting Q&A **grounded and traceable**.

The canonical retrieval path is:

```text
Meeting
   │
   ▼
add_meeting_segments(meeting.id, segments)
   │
   ▼
ChromaDB
   │
   ▼
Retriever
   │
   │  filter:
   │  meeting_id == current_meeting.id
   │
   ▼
Relevant transcript documents
   │
   ├───────────────┐
   ▼               ▼
RAG Answer       Evidence
   │               │
   └───────┬───────┘
           ▼
      RAGAnswer
```

The important distinction is that RECALL does not simply retrieve arbitrary transcript chunks and ask an LLM to answer.

The retrieval layer is explicitly scoped to the current meeting.

---

# 🛡️ Evidence Contract

RECALL treats evidence as a first-class object.

A grounded response has the conceptual structure:

```python
RAGAnswer(
    answer="...",
    evidence=[
        RAGEvidence(
            segment_id="...",
            start=74.0,
            end=77.0,
            text="..."
        )
    ]
)
```

This allows the UI to show:

```text
ANSWER

The cart is generated from products identified by the
shopping engine, with quantity controls for adjustment.

────────────────────────────────────

TRANSCRIPT EVIDENCE

01:14 → 01:17
"and instantly generates a complete cart."

01:59 → 02:09
"four luggage or cabin trolleys, two power banks, dry bags."

02:26 → 02:37
"The use of quantity controls to add or remove items as needed."
```

If the information is not found:

```text
I could not find this information in the meeting transcript.
```

and no unrelated evidence is shown.

---

# 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| Language | Python |
| Speech-to-text | OpenAI Whisper |
| Hindi/Hinglish STT | Sarvam AI |
| LLM | Mistral |
| LLM orchestration | LangChain / LCEL |
| Embeddings | Hugging Face Sentence Transformers |
| Vector database | ChromaDB |
| Audio processing | FFmpeg / pydub |
| YouTube ingestion | yt-dlp |
| Data validation | Pydantic |
| Testing | pytest |
| Reporting | ReportLab |

---

# 📁 Project Structure

```text
ReCall/
│
├── app.py
│   └── Streamlit application and UI
│
├── main.py
│   └── CLI entry point
│
├── core/
│   ├── config.py
│   ├── extractor.py
│   ├── llm.py
│   ├── pipeline.py
│   ├── rag_engine.py
│   ├── report_generator.py
│   ├── summarizer.py
│   ├── transcriber.py
│   ├── transcript_formatter.py
│   └── vector_store.py
│
├── utils/
│   └── audio_processor.py
│
├── tests/
│   ├── test_extraction.py
│   ├── test_foundation.py
│   ├── test_meeting_isolation.py
│   ├── test_models.py
│   ├── test_rag_evidence.py
│   ├── test_report_generation.py
│   └── test_vector_store.py
│
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

# 🚀 Getting Started

## 1. Clone the repository

```bash
git clone https://github.com/adars-h-agrawal/ReCall.git
cd ReCall
```

## 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

RECALL also requires **FFmpeg** to be installed and available on your system PATH.

Verify:

```bash
ffmpeg -version
```

## 4. Configure environment variables

Create a `.env` file:

```env
MISTRAL_API_KEY=your_mistral_api_key
SARVAM_API_KEY=your_sarvam_api_key

WHISPER_MODEL=small
SARVAM_STT_MODEL=saaras:v2.5
```

Never commit your `.env` file.

Use `.env.example` as the template.

---

# ▶️ Run RECALL

Start the Streamlit application:

```bash
streamlit run app.py
```

Then open:

```text
http://localhost:8501
```

---

# 🧪 Testing

Run the complete test suite:

```bash
pytest -q
```

The test suite covers areas including:

- extraction
- meeting models
- meeting isolation
- RAG evidence
- vector-store behavior
- report generation
- pipeline behavior

---

# 🔍 Example Workflow

### 01 — Add a meeting

Upload a recording or provide a YouTube URL.

### 02 — Let RECALL process it

The pipeline performs:

```text
Transcription
      ↓
Title generation
      ↓
Summarization
      ↓
Decision extraction
      ↓
Action extraction
      ↓
Question extraction
      ↓
Vector indexing
```

### 03 — Review the meeting

Navigate through:

```text
Overview
Decisions
Actions
Questions
Transcript
```

### 04 — Ask RECALL

Ask a natural-language question about the meeting.

### 05 — Inspect evidence

Relevant transcript segments are shown with their source timestamps.

### 06 — Export

Download the meeting intelligence as PDF or TXT.

---

# 🎯 Why RECALL?

Traditional meeting tools generally answer:

> "What was said?"

RECALL is designed around a different question:

> **"What should I know and what should I do next?"**

It converts an unstructured recording into structured meeting intelligence while preserving the ability to trace answers back to the underlying transcript.

---

# 🔬 Engineering Highlights

### Meeting-scoped RAG

Retrieval is filtered by the current meeting identifier before the LLM receives context.

This prevents cross-meeting retrieval contamination.

### Structured provenance

Transcript chunks preserve their relationship to the original transcript segment.

```text
Meeting
  └── TranscriptSegment
        ├── segment_id
        ├── start
        ├── end
        ├── speaker
        └── text
```

### Evidence deduplication

Multiple retrieved sub-chunks belonging to the same transcript segment are collapsed into a single evidence item.

### Explicit abstention

When relevant information is absent, the system is instructed to abstain instead of using general knowledge to fill the gap.

### Backward-compatible RAG API

The structured RAG path returns:

```text
RAGAnswer
├── answer
└── evidence[]
```

while the legacy answer-only interface remains available.

---

# 🖥️ Screenshots

> Add screenshots of the current RECALL workspace here.

Recommended screenshots:

```text
docs/screenshots/landing.png
docs/screenshots/meeting.png
docs/screenshots/chat.png
docs/screenshots/transcript.png
```

---

# 🗺️ Roadmap

Potential future improvements:

- [ ] Persistent multi-meeting history
- [ ] Meeting search across sessions
- [ ] Speaker identification / diarization
- [ ] Richer topic extraction
- [ ] Risk and blocker detection
- [ ] Conversation memory across questions
- [ ] Faster / cached model loading
- [ ] Background processing for long recordings
- [ ] Cloud deployment
- [ ] Authentication and user workspaces
- [ ] Team collaboration
- [ ] Calendar integrations
- [ ] Meeting comparison across sessions

---

# ⚠️ Current Considerations

RECALL currently performs local Whisper inference for supported workflows.

For longer recordings, processing time depends heavily on:

- CPU/GPU availability
- Whisper model size
- recording duration
- embedding generation
- LLM/API latency

For production deployment, model caching, asynchronous processing, persistent storage, authentication, and workload management would need to be added.

---

# 👨‍💻 Author

**Adarsh Agrawal**

B.Tech — Information Technology  
Manipal Institute of Technology

Interested in:

```text
AI / ML
LLMs
RAG Systems
Computer Vision
Full-Stack Engineering
Data-Driven Systems
```

---

# 📜 License

This project is currently intended as a personal / academic portfolio project.

Add an explicit license here if the repository is intended for redistribution.

---

<p align="center">

### RECALL

**Meetings in. Intelligence out.**

</p>
