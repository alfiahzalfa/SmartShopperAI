# SmartShopper Assistant — Personalized AI Agent

> **Assignment: Personalized SmartShopper Assistant**  
> Implementasi AI Agent yang mampu menjawab pertanyaan produk maupun pertanyaan umum e-commerce secara otomatis menggunakan **Google ADK**, **Haystack RAG**, dan **MongoDB Atlas**.

---

## Daftar Isi

1. [Analisis Kebutuhan](#1-analisis-kebutuhan)
2. [Penyusunan Dataset Common Information](#2-penyusunan-dataset-common-information)
3. [Storing Data ke MongoDB Atlas](#3-storing-data-ke-mongodb-atlas)
4. [Tools Common Information (Google ADK FunctionTool)](#4-tools-common-information-google-adk-functiontool)
5. [Integrasi RAG](#5-integrasi-rag)
6. [Integrasi dengan AI Agent & Routing](#6-integrasi-dengan-ai-agent--routing)
7. [Pengujian dan Validasi](#7-pengujian-dan-validasi)
8. [Cara Menjalankan Proyek](#8-cara-menjalankan-proyek)
9. [Tech Stack](#9-tech-stack)

---

## 1. Analisis Kebutuhan

### Kategori Pertanyaan Common Information

Berdasarkan konteks e-commerce, pertanyaan yang masuk ke **Common Information Tool** dibagi menjadi 6 topik:

| Topik | Contoh Pertanyaan |
|-------|-------------------|
| **Shipping** | Berapa lama pengiriman? / Bagaimana cara melacak paket? / Apakah ada pengiriman gratis? |
| **Purchasing** | Bagaimana cara memesan? / Metode pembayaran apa yang diterima? / Bisakah saya membatalkan pesanan? |
| **Refund** | Apa kebijakan refund? / Bagaimana cara mengajukan refund? / Berapa lama refund diproses? |
| **Return** | Bagaimana cara mengembalikan barang? / Apa saja syarat return? / Apakah return gratis? |
| **Account** | Cara membuat akun? / Cara reset password? / Cara memperbarui profil? |
| **General** | Cara menghubungi support? / Apakah ada loyalty program? / Apakah produk 100% asli? |

### Batasan Routing antara Product vs Common Information

```
User Query
    │
    ▼
Google ADK Agent (LLM-based intent detection)
    │
    ├── [Intent: Product] ─────────────────────────────────────────┐
    │   Kata kunci: recommend, dress, tops, shoes, cotton, leather, │
    │               price range, brand, material, category, outfit  │
    │                         ↓                                     │
    │            FunctionTool: retrieve_and_generate_recommendation │
    │                                                               │
    └── [Intent: Common Info] ─────────────────────────────────────┘
        Kata kunci: shipping, delivery, track, refund, return,
                    payment, order, account, cancel, policy, how to
                         ↓
             FunctionTool: retrieve_common_information

    [Intent: Greeting/Off-topic] → Agent jawab langsung (no tool)
```

---

## 2. Penyusunan Dataset Common Information

Dataset Common Information dibuat menggunakan LLM (GPT-4) sebagai bantuan penyusunan, mencakup **29 dokumen FAQ** e-commerce yang disimpan di `data/common_information.json`.

### Struktur Dataset

```json
{
  "topic": "shipping",
  "question": "How long does shipping take?",
  "answer": "Shipping time depends on the destination..."
}
```

### Distribusi Topik

| Topik | Jumlah Dokumen |
|-------|---------------|
| Shipping | 6 |
| Purchasing | 6 |
| Refund | 4 |
| Return | 4 |
| Account | 3 |
| General | 6 |
| **Total** | **29** |

Dataset dirancang agar:
- Relevan dengan konteks e-commerce fashion
- Mudah diproses oleh AI Agent (satu Q&A per dokumen)
- Mencakup berbagai variasi pertanyaan yang mungkin diajukan user

---

## 3. Storing Data ke MongoDB Atlas

### Arsitektur Penyimpanan

Proyek ini menggunakan **dua collection terpisah** di database `depato_store`:

| Collection | Isi | Index |
|-----------|-----|-------|
| `products` | 1262 produk fashion dari Amazon | `vector_index` |
| `common_information` | 29 FAQ e-commerce | `common_info_vector_index` |
| `materials` | Daftar material unik | - |
| `categories` | Daftar kategori unik | - |

### Strategi Penyimpanan

#### Produk (`products`)
```
content : "<product_title>\n <product_description>"  ← teks untuk embedding
meta    :
  asin     : ID produk Amazon
  title    : nama produk
  brand    : merek
  price    : harga (float)
  gender   : female / male / unisex
  material : bahan (Cotton, Leather, Polyester, dll.)
  category : kategori (Tops, Dresses/Jumpsuits, Accessories, dll.)
embedding: [768 float values]  ← all-mpnet-base-v2
```

#### Common Information (`common_information`)
```
content : "Question: <pertanyaan>\nAnswer: <jawaban>"  ← gabungan Q&A untuk embedding kontekstual
meta    :
  topic    : kategori (shipping / purchasing / refund / return / account / general)
  question : pertanyaan asli (untuk referensi)
embedding: [768 float values]  ← all-mpnet-base-v2
```

**Mengapa content = Q+A digabung?**  
Menggabungkan pertanyaan dan jawaban dalam satu teks menghasilkan embedding yang lebih kontekstual — saat user bertanya dengan kalimat berbeda dari yang ada di dataset, vector search tetap bisa menemukan dokumen yang relevan karena embedding merepresentasikan makna keduanya.

**Duplicate Policy: OVERWRITE**  
Pipeline storing menggunakan `DuplicatePolicy.OVERWRITE`, sehingga notebook/script dapat dijalankan berulang kali tanpa duplikasi data.

### Program Storing

**Script Python (direkomendasikan):**
```bash
# Storing produk
python process/store_data_script.py

# Storing common information
python process/store_common_info_script.py
```

**Atau gunakan Jupyter Notebook:**
```bash
jupyter notebook process/store_data.ipynb
jupyter notebook process/store_common_info.ipynb
```

### Pipeline Storing (Haystack)

```
documents (List[Document])
    │
    ▼
SentenceTransformersDocumentEmbedder   ← Generate embedding 768 dimensi (all-mpnet-base-v2)
    │
    ▼
DocumentWriter (OVERWRITE policy)      ← Simpan ke MongoDB Atlas
```

### Vector Search Index Configuration

Setelah storing, buat Vector Search Index di MongoDB Atlas:

**Untuk collection `products`** (index name: `vector_index`):
```json
{
  "fields": [
    {
      "numDimensions": 768,
      "path": "embedding",
      "similarity": "cosine",
      "type": "vector"
    }
  ]
}
```

**Untuk collection `common_information`** (index name: `common_info_vector_index`):
```json
{
  "fields": [
    {
      "numDimensions": 768,
      "path": "embedding",
      "similarity": "cosine",
      "type": "vector"
    }
  ]
}
```

---

## 4. Tools Common Information (Google ADK FunctionTool)

Tools diimplementasikan sebagai **plain Python function** yang secara otomatis di-wrap menjadi `FunctionTool` oleh Google ADK saat dimasukkan ke dalam `Agent(tools=[...])`.

### `retrieve_common_information`

```python
def retrieve_common_information(query: str) -> str:
    """
    Answers general e-commerce questions about shopping processes and policies.

    Use this tool when the user asks about: shipping and delivery time, order tracking,
    how to place or cancel an order, refund policy, how to request a refund,
    how to return an item, return eligibility, payment methods, account management,
    loyalty programs, gift cards, or any general e-commerce inquiry.

    Do NOT use this tool for product recommendations or product searches.
    """
    common_info_pipeline = _pipelines.get("common_info")
    return common_info_pipeline.run(query)
```

**Kunci keberhasilan routing:** Docstring berfungsi sebagai deskripsi tool untuk LLM. ADK menggunakan docstring ini untuk memutuskan kapan tool harus dipanggil — sehingga docstring harus ditulis sejelas mungkin.

### `retrieve_and_generate_recommendation`

```python
def retrieve_and_generate_recommendation(query: str) -> str:
    """
    Retrieves products from the catalog and generates personalized recommendations.

    Use this tool when the user asks about specific products, wants product
    recommendations, or inquires about items such as dresses, tops, shoes,
    accessories, specific price ranges, materials, brands, or categories.
    """
    # Step 1: Extract metadata filters
    # Step 2: Retrieve + generate recommendation
```

---

## 5. Integrasi RAG

### Alur RAG — Product Recommendation

```
User Query
    │
    ▼
[1] MetaDataFilterPipeline
    ├── GetMaterials (dari MongoDB)
    ├── GetCategories (dari MongoDB)
    ├── ChatPromptBuilder + Jinja2 Template
    └── Groq LLM → JSON filter {category, material, price}
    │
    ▼
[2] RetrieveAndGenerateAnswerPipeline
    ├── SentenceTransformersTextEmbedder → query vector
    ├── MongoDBAtlasEmbeddingRetriever (top_k=10, dengan filter)
    ├── ChatPromptBuilder
    └── Groq LLM → formatted product recommendations
```

### Alur RAG — Common Information

```
User Query
    │
    ▼
CommonInfoRAGPipeline
    ├── SentenceTransformersTextEmbedder → query vector
    ├── MongoDBAtlasEmbeddingRetriever (top_k=5, collection: common_information)
    ├── ChatPromptBuilder + COMMON_INFO_TEMPLATE
    └── Groq LLM → friendly, contextual answer
```

### Embedding Model

Semua pipeline menggunakan **`all-mpnet-base-v2`** (768 dimensi) dari SentenceTransformers — konsisten antara storing dan querying untuk memastikan similarity search akurat.

---

## 6. Integrasi dengan AI Agent & Routing

### Google ADK Agent Setup

```python
agent = Agent(
    name="smartshopper_agent",
    model=LiteLlm(model="groq/llama-3.3-70b-versatile"),
    tools=[
        retrieve_and_generate_recommendation,   # FunctionTool: Product
        retrieve_common_information,             # FunctionTool: Common Info
    ],
    instruction="""
    ROUTING RULES:
    - Product query → retrieve_and_generate_recommendation
    - Shipping/refund/return/payment/account → retrieve_common_information
    - Greeting/off-topic → respond directly (no tool)
    """
)
```

### Mekanisme Routing

1. **LLM-based intent detection**: Agent menganalisis query user
2. **Tool selection via docstring**: ADK membaca docstring tiap FunctionTool sebagai panduan kapan tool dipanggil
3. **Instruction reinforcement**: System instruction memperkuat aturan routing
4. **Automatic execution**: ADK secara otomatis memanggil tool yang tepat, menunggu hasil, lalu menyusun final response

---

## 7. Pengujian dan Validasi

### Test Cases Routing

| Query | Tool yang Dipilih | Hasil yang Diharapkan |
|-------|-------------------|-----------------------|
| "I want a cotton dress under $50" | `retrieve_and_generate_recommendation` | Daftar produk dengan filter cotton + dress + price ≤ 50 |
| "Show me women's leather accessories" | `retrieve_and_generate_recommendation` | Daftar produk dengan filter leather + accessories |
| "How long does shipping take?" | `retrieve_common_information` | Penjelasan estimasi waktu pengiriman |
| "What is your refund policy?" | `retrieve_common_information` | Penjelasan kebijakan 30-hari refund |
| "How do I return an item?" | `retrieve_common_information` | Langkah-langkah proses return |
| "What payment methods do you accept?" | `retrieve_common_information` | Daftar metode pembayaran |
| "Hello!" | (langsung oleh agent) | Sapaan ramah tanpa memanggil tool |
| "Can I cancel my order?" | `retrieve_common_information` | Penjelasan kebijakan pembatalan |

---

## 8. Cara Menjalankan Proyek

### Prasyarat

- Python 3.10+
- Akun [MongoDB Atlas](https://cloud.mongodb.com) (free tier cukup)
- Akun [Groq](https://console.groq.com) (free tier cukup)

### Langkah 1 — Clone & Setup Environment

```bash
git clone https://github.com/<username>/SmartShopperAI.git
cd SmartShopperAI

python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

pip install -r requirements.txt
```

### Langkah 2 — Konfigurasi `.env`

Salin template dan isi dengan kredensial kamu:

```bash
cp .env.example .env
```

```env
MONGO_CONNECTION_STRING=mongodb+srv://<username>:<password>@<cluster>.mongodb.net/?retryWrites=true&w=majority
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

### Langkah 3 — Storing Data ke MongoDB Atlas

```bash
# Storing data produk (proses embedding ~5-15 menit)
python process/store_data_script.py

# Storing common information FAQ
python process/store_common_info_script.py
```

### Langkah 4 — Buat Vector Search Index di MongoDB Atlas

1. Login ke [cloud.mongodb.com](https://cloud.mongodb.com)
2. Masuk ke cluster → **Atlas Search** → **Create Search Index**
3. Pilih **Atlas Vector Search** → **JSON Editor**
4. Buat 2 index (lihat konfigurasi JSON di [Bagian 3](#3-storing-data-ke-mongodb-atlas))
5. Tunggu status index menjadi **Active** ✅

### Langkah 5 — Jalankan Streamlit App

```bash
streamlit run website/website.py
```

Buka browser di [http://localhost:8501](http://localhost:8501) 🎉

### Langkah 6 — (Opsional) Jalankan FastAPI

```bash
uvicorn website.api:app --host 0.0.0.0 --port 8000 --reload
```

Dokumentasi API: [http://localhost:8000/docs](http://localhost:8000/docs)

| Method | Endpoint | Deskripsi |
|--------|----------|-----------|
| `GET` | `/health` | Status API |
| `POST` | `/recommend` | Product recommendation |
| `POST` | `/common-info` | Common information |

---

## Struktur Repository

```
SmartShopperAI/
├── data/
│   ├── datasets.pkl                  # Dataset produk Amazon Fashion (1262 produk)
│   └── common_information.json       # Dataset FAQ e-commerce (29 dokumen)
├── process/
│   ├── store_data_script.py          # [UTAMA] Script storing produk ke MongoDB
│   ├── store_common_info_script.py   # [UTAMA] Script storing common info ke MongoDB
│   ├── store_data.ipynb              # Notebook storing produk (eksplorasi)
│   ├── store_common_info.ipynb       # Notebook storing common info (eksplorasi)
│   ├── retriever.ipynb               # Eksplorasi vector retrieval
│   ├── generator.ipynb               # Eksplorasi RAG pipeline
│   ├── generator_filter.ipynb        # Eksplorasi metadata filter
│   ├── chat_memory.ipynb             # Eksplorasi chat memory
│   └── shop_recommendation.ipynb     # Eksplorasi AI Agent lengkap
├── website/
│   ├── website.py                    # [UTAMA] Streamlit app (ADK Agent + Haystack RAG)
│   ├── api.py                        # FastAPI REST API
│   └── template.py                   # Jinja2 prompt templates
├── .env.example                      # Template environment variables
├── .gitignore                        # Exclude .env, venv, __pycache__, dll.
├── Dockerfile                        # Docker image untuk Streamlit UI
├── Dockerfile.api                    # Docker image untuk FastAPI
├── docker-compose.yml                # Orkestrasi Docker services
├── requirements.txt                  # Python dependencies
└── README.md                         # Dokumentasi ini
```

---

## 9. Tech Stack

| Komponen | Library / Service |
|----------|-------------------|
| AI Agent | [Google ADK](https://google.github.io/adk-docs/) v1.33+ |
| RAG Framework | [Haystack AI](https://haystack.deepset.ai/) v2.15+ |
| LLM | [Groq](https://groq.com/) — `llama-3.3-70b-versatile` |
| Embedding | SentenceTransformers (`all-mpnet-base-v2`, 768 dim) |
| Vector DB | MongoDB Atlas Vector Search |
| Frontend | Streamlit |
| API | FastAPI + Uvicorn |

---

## Troubleshooting

| Error | Solusi |
|-------|--------|
| `ModuleNotFoundError` | Aktifkan venv, lalu `pip install -r requirements.txt` |
| `ServerSelectionTimeoutError` | Whitelist IP di MongoDB Atlas → Network Access → `0.0.0.0/0` |
| `AuthenticationError` (Groq) | Cek `GROQ_API_KEY` di `.env` di [console.groq.com](https://console.groq.com) |
| Vector search kosong | Pastikan index `vector_index` / `common_info_vector_index` statusnya **Active** |
| ADK `GeneratorExit` error | Sudah diperbaiki di versi ini — jangan gunakan `break` di async generator |