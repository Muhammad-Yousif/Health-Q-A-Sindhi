# صحت بابت سوال ۽ جواب (Health Q&A Bot)

A Streamlit-powered web app that lets users ask health-related questions in Sindhi and returns answers sourced from your own PDF/DOCX “books” collection using semantic retrieval (TF-IDF + Chroma) and an OpenAI/Gemma LLM.

---

## 🚀 Features

- **Multi-format input**  
  - PDF & Word documents in the `books/` folder  
  - Automatic text extraction (PyPDF2 + python-docx)

- **Semantic Retrieval**  
  - TF-IDF embeddings (via `scikit-learn`)  
  - Chroma vector store for fast, persistent similarity search

- **Custom LLM Integration**  
  - Wraps OpenAI/Gemma API (`google/gemma-3-27b-it`) with retry logic  
  - All prompts & answers in Sindhi (سنڌي) with simple, user-friendly tone

- **Streamlit UI**  
  - Sidebar menu: “سوال جواب” (Ask) & “اسان جي باري ۾” (About)  
  - Real-time spinner while fetching answers  
  - Source attribution for returned answers

---

## 🛠️ Getting Started

### 1. Clone this repository

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
```

### 2. Install dependencies

Use `pip` and the provided requirements file:

```bash
pip install -r requirements.txt
```

### 3. Add your source documents

Create a folder named `books/` in the project root and drop in any `.pdf` or `.docx` files you want your chatbot to draw from.

### 4. Configure OpenAI/Gemma API

Streamlit will look for credentials in one of two places:

1. In `~/.streamlit/secrets.toml` under `[openai_gemma]`:  
   ```toml
   [openai_gemma]
   api_key = "YOUR_GEMMA_API_KEY"
   base_url = "https://api.your-gemma-endpoint.com"
   ```
2. Or via environment variables:  
   ```bash
   export GEMMA_API_KEY="YOUR_GEMMA_API_KEY"
   export GEMMA_BASE_URL="https://api.your-gemma-endpoint.com"
   ```

### 5. Run the app

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

---

## 📂 Project Structure

```
.
├── app.py               # Main Streamlit application
├── requirements.txt     # Python dependencies
├── books/               # Your PDF/DOCX sources (you add these)
├── chroma_db/           # Persisted Chroma vectorstore  
└── .streamlit/          # (Optional) Streamlit secrets/config
```

---

## 🎯 How It Works

1. **Load & Split Documents**  
   - Scans `books/` for PDFs & DOCX  
   - Extracts text (with caching)  
   - Splits into ~300-character chunks

2. **Embed & Index**  
   - Builds TF-IDF embeddings over all chunks  
   - Stores them in a persistent Chroma collection

3. **Retrieve & Answer**  
   - On user query, find top 3 similar chunks  
   - Feed them + the query into Gemma LLM  
   - Return answer + metadata sources

---

## 🤝 Contributing

1. Fork the repo  
2. Create a feature branch (`git checkout -b feat/awesome`)  
3. Commit your changes (`git commit -m "Add awesome feature"`)  
4. Push (`git push origin feat/awesome`)  
5. Open a pull request

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

---

*Built with ❤️ using [Streamlit](https://streamlit.io), [LangChain](https://langchain.com), and [Chroma](https://www.trychroma.com).*
