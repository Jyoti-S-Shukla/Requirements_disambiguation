**CHiM-DSR: Context-Aware Hierarchical Multi-Stage Disambiguation of Software Requirements**
CHiM-DSR is an end-to-end automated framework for detecting, classifying, and resolving ambiguity in Software Requirements Specifications (SRS). Written in natural language, SRS documents are inherently prone to ambiguity — terms with multiple meanings, vague constraints, and structurally unclear sentences that lead to misinterpretation, implementation errors, and project delays. CHiM-DSR addresses this by combining LLM-based reasoning with document-level context retrieval in a lightweight, training-free pipeline.

**How it works**
The framework routes each requirement sentence through three sequential stages:
Stage I — Ambiguity Detection: Two LLMs (LLaMA 3.1-8B and Qwen2-7B) independently classify each requirement as ambiguous or unambiguous using few-shot prompting. A GPT-4o-mini judge selects the higher-quality output. Ambiguous requirements are forwarded with reasoning and targeted probing questions to Stage II.
Stage II — Document-Level Resolution (RAG): The source SRS document is indexed in a vector store (ChromaDB with BAAI/bge-large-en-v1.5 embeddings). Probing questions from Stage I are answered using retrieved document chunks, resolving ambiguities that are clarifiable from within the document itself. Requirements successfully resolved here exit the pipeline with a QA pair explanation.
Stage III — Multi-Label Classification and Interpretation: Requirements unresolved by RAG are classified into one or more ambiguity types — lexical, semantic, syntactic, or vagueness — and three ranked alternative interpretations are generated, from most to least plausible. This gives analysts a structured, actionable set of resolutions to work from.
Each stage is validated using an LLM-as-a-Judge paradigm, eliminating dependence on annotated training data.

**Key results**
Evaluated on 800 requirements from 56 SRS documents drawn from the PURE dataset
Over 15% improvement in ambiguity detection correctness versus direct prompting baselines
RAG-based resolution outperforms BM25 retrieval by more than 4x on document-dependent ambiguities
Strong human-LLM agreement: weighted κ = 0.78 (Stage I), 0.80 (Stage II), 0.66 (Stage III)
More than 75% of generated interpretations rated as useful by expert requirements analysts
End-to-end processing of 419 requirements in ~30 minutes versus ~3 days of manual effort

**Dataset**
This repository includes a publicly released dataset of 800 annotated software requirements extracted from 56 SRS documents, featuring:

Binary ambiguity labels
Multi-label ambiguity type annotations (lexical, semantic, syntactic, vagueness)
RAG-generated QA pairs for document-resolved requirements
Ranked alternative interpretations for Stage III requirements

This is one of the few publicly available datasets with multi-label ambiguity annotations for software requirements, going beyond the binary labeling common in prior work.
<img width="3012" height="906" alt="idea Diagram_5 drawio" src="https://github.com/user-attachments/assets/72df4de3-be5f-48ca-8e90-78c7176bf7ff" />

