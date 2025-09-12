RAG-based Medical FAQ Chatbot
Overview
This project implements a Retrieval-Augmented Generation (RAG) chatbot for answering medical FAQs, built for the AI/ML Engineer Assignment. It retrieves relevant contexts from a medical FAQ dataset using FAISS and generates answers with OpenRouter’s openai/gpt-3.5-turbo model (OpenAI-compatible API). The command-line interface (cli.py) processes queries like "What are the early symptoms of diabetes?" and saves results to query_results.csv.
Setup Instructions

Clone Repository:
git clone https://github.com/yourusername/rag-medical-faq-chatbot.git
cd rag-medical-faq-chatbot


Create Virtual Environment (recommended):
python -m venv myenv


Windows: myenv\Scripts\activate.bat (or myenv\Scripts\Activate.ps1 in PowerShell)
macOS/Linux: source myenv/bin/activate


Install Dependencies:
pip install -r requirements.txt


Set OpenRouter API Key:

Sign up at https://openrouter.ai, create an API key (free tier).
Create a .env file in the project root:OPENROUTER_API_KEY=sk-or-v1-xxx...




Download Dataset:

Preferred: Kaggle’s "Comprehensive Medical Q&A Dataset" from https://www.kaggle.com/datasets/thedevastator/comprehensive-medical-q-a-dataset (train.csv).
Alternative: MedQuAD from https://github.com/abachaa/MedQuAD (MedQuAD_v1.jsonl).
Convert to CSV:python -c "import pandas as pd; df = pd.read_json('MedQuAD_v1.jsonl', lines=True); df[['question', 'answer']].rename(columns={'question': 'Question', 'answer': 'Answer'}).to_csv('train.csv', index=False)"




Place as train.csv in the project root.


Build FAISS Index:
python build_index.py


Generates faiss_index.index and documents.pkl.


Run Chatbot:
python cli.py


Automatically answers:
"What are the early symptoms of diabetes?"
"Can children take paracetamol?"
"What foods are good for heart health?"


Enter additional queries interactively; type save to save results to query_results.csv, or exit to quit and save.



Design Choices

Dataset: Kaggle’s Medical Q&A or MedQuAD (~100 Q&A pairs after slicing). Chosen for coverage of diabetes symptoms (e.g., thirst, fatigue), pediatric drugs (e.g., paracetamol dosing), and nutrition (e.g., heart-healthy foods like oats, fish).
Embeddings: sentence-transformers/all-MiniLM-L6-v2 for lightweight, fast semantic search (free, open-source, 384-dimensional embeddings).
Vector Database: FAISS with IndexFlatL2 for efficient similarity search (free, local, no external dependencies). No chunking needed as FAQs are short (~1-2 sentences).
LLM: OpenRouter’s openai/gpt-3.5-turbo (free tier, OpenAI-compatible) for accurate, natural answers, meeting the assignment’s "OpenAI LLM" requirement. Chosen over other models for cost-free access and proven performance in conversational tasks.
Interface: Command-line tool (cli.py) for simplicity, processes required queries, and saves to CSV as specified. Preferred over Streamlit for faster development within 5-7 hour constraint.
Error Handling: Validates API key, logs errors (dataset loading, embedding, API calls), and handles missing files or empty contexts.
Modularity: Separated into build_index.py (preprocessing), rag.py (RAG pipeline), and cli.py (interface) for maintainability and clarity.

Expected Output (query_results.csv)
Question,Answer
"What are the early symptoms of diabetes?","Early symptoms include increased thirst, frequent urination, fatigue, blurred vision, and slow-healing sores."
"Can children take paracetamol?","Yes, paracetamol is safe for children in doses of 10-15 mg/kg every 4-6 hours, up to 5 doses daily. Consult a doctor."
"What foods are good for heart health?","Foods like oats, salmon, nuts, leafy greens, and berries support heart health by reducing cholesterol and inflammation."

Demo (Optional)
[Insert Loom video link here] demonstrating:

Running build_index.py to preprocess the dataset.
Running cli.py, answering the three required queries, and handling interactive input.
Displaying query_results.csv with results.

To create the video:

Record using Loom (free at https://www.loom.com).
Show terminal commands: python build_index.py, python cli.py, input queries, and open query_results.csv.
Upload to Loom and include the link in this README.

Troubleshooting

API Key Error:
Ensure OPENROUTER_API_KEY is set in .env (get from https://openrouter.ai/api-keys).
Regenerate key if invalid or check rate limits at https://openrouter.ai/usage.


Dataset Issues:
Verify train.csv has Question and Answer columns (case-insensitive).
Check content: python -c "import pandas as pd; print(pd.read_csv('train.csv').head())".
Use MedQuAD if Kaggle dataset lacks relevant FAQs.


No Relevant Answers:
Increase dataset slice in build_index.py (e.g., df.head(500)).
Verify dataset contains terms like diabetes|paracetamol|heart: python -c "import pandas as pd; print(pd.read_csv('train.csv')['Question'].str.contains('diabetes', case=False, na=False).sum())".


OpenRouter Errors:
Verify openai/gpt-3.5-turbo availability at https://openrouter.ai/models.
Fallback to openai/gpt-4o-mini if needed (also free on OpenRouter).



Notes

Free Tools: Uses FAISS, Sentence Transformers, and OpenRouter’s free tier (no cost for testing).
