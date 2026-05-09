# SmartShopper Assistant

Repository ini berisi implementasi **Personalized SmartShopper Assistant** — AI Agent yang mampu menjawab pertanyaan produk maupun pertanyaan umum e-commerce secara otomatis menggunakan **Google ADK**, **Haystack RAG**, dan **MongoDB Atlas**.

---

## Arsitektur Sistem

```
User Query
    │
    ▼
Google ADK Agent  (Routing otomatis berdasarkan intent)
    │
    ├── [Pertanyaan Produk]
    │       ↓
    │   FunctionTool: retrieve_and_generate_recommendation
    │       ↓
    │   Haystack RAG Pipeline
    │   ├── MetaDataFilterPipeline   → Extract filter (category, material, price)
    │   ├── SentenceTransformers     → Embed query
    │   ├── MongoDB Atlas (products) → Vector Search
    │   └── Groq LLM                 → Generate rekomendasi
    │
    └── [Pertanyaan Umum: shipping, refund, return, dll.]
            ↓
        FunctionTool: retrieve_common_information
            ↓
        Haystack RAG Pipeline
        ├── SentenceTransformers              → Embed query
        ├── MongoDB Atlas (common_information) → Vector Search
        └── Groq LLM                           → Generate jawaban
```

---

## Analisis Kebutuhan

### Kategori Pertanyaan Common Information
Pertanyaan yang masuk ke **Common Information Tool**:
- **Shipping**: estimasi pengiriman, tracking, kurir, pengiriman internasional
- **Purchasing**: cara beli, metode pembayaran, cancel order, keamanan transaksi
- **Refund**: kebijakan refund, cara request refund, timeline refund
- **Return**: cara return barang, syarat return, biaya return
- **Account**: buat akun, reset password, update profil
- **General**: loyalty program, gift card, keaslian produk, customer support

### Batasan Routing
| Tipe Pertanyaan | Tool yang Digunakan |
|---|---|
| "Recommend me a dress under $50" | `retrieve_and_generate_recommendation` |
| "Show me cotton tops for women" | `retrieve_and_generate_recommendation` |
| "How long does shipping take?" | `retrieve_common_information` |
| "What is your refund policy?" | `retrieve_common_information` |
| "Hello!" / Sapaan | Dijawab langsung oleh Agent (tanpa tool) |

---

## Struktur Repository

```
SmartShopperAI/
├── data/
│   ├── datasets.pkl                  # Dataset produk Amazon Fashion (1262 produk)
│   └── common_information.json       # Dataset FAQ e-commerce (29 dokumen)
├── process/
│   ├── store_data.ipynb              # Storing produk ke MongoDB Atlas
│   ├── store_common_info.ipynb       # Storing Common Info ke MongoDB Atlas
│   ├── retriever.ipynb               # Eksplorasi vector retrieval
│   ├── generator.ipynb               # Eksplorasi RAG pipeline
│   ├── generator_filter.ipynb        # Eksplorasi metadata filter
│   ├── chat_memory.ipynb             # Eksplorasi chat memory
│   └── shop_recommendation.ipynb     # Eksplorasi AI Agent lengkap
├── website/
│   ├── website.py                    # Streamlit app (Google ADK Agent + Haystack RAG)
│   ├── api.py                        # FastAPI REST API
│   └── template.py                   # Jinja2 prompt templates
├── .env.example                      # Template environment variables
├── .dockerignore                     # Exclude .env dari Docker build context
├── Dockerfile                        # Docker image untuk Streamlit UI
├── Dockerfile.api                    # Docker image untuk FastAPI
├── docker-compose.yml                # Orkestrasi kedua service
└── requirements.txt                  # Python dependencies
```

---

## Prasyarat

- Python 3.10+
- Akun [MongoDB Atlas](https://cloud.mongodb.com) (free tier cukup)
- Akun [Groq](https://console.groq.com) (free tier cukup)
- Docker & Docker Compose (opsional)

---

## Langkah 1 — Clone & Setup Environment

```bash
git clone https://github.com/alfiahzalfa/SmartShopperAI.git
cd SmartShopperAI

python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

python -m pip install -r requirements.txt
```

---

## Langkah 2 — Konfigurasi Environment Variables

Salin template dan isi dengan kredensial kamu:

```bash
cp .env.example .env
```

```env
MONGO_CONNECTION_STRING=mongodb+srv://<username>:<password>@<cluster>.mongodb.net/?retryWrites=true&w=majority
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

---

## Langkah 3 — Setup MongoDB Atlas: Storing Data Produk

### 3a. Jalankan Script Storing Produk

```bash
python process/store_data_script.py
```

Script ini akan:
1. Load dataset Amazon Fashion dari `data/datasets.pkl`
2. Generate embedding (SentenceTransformers, 768 dimensi)
3. Simpan 1262 produk ke collection `products` di MongoDB Atlas

### 3b. Buat Vector Search Index untuk `products`

Di MongoDB Atlas → **Atlas Search** → **Create Search Index** → **Atlas Vector Search**:
- Collection: `products`
- Nama index: `vector_index`

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

## Langkah 4 — Setup MongoDB Atlas: Storing Common Information

### Struktur dan Strategi Penyimpanan Common Information

Data Common Information disimpan di collection terpisah (`common_information`) dengan strategi:

- **Database**: `depato_store` (sama dengan produk)
- **Collection**: `common_information` (terpisah dari `products`)
- **Struktur Dokumen**:
  ```
  content : "Question: <pertanyaan>\nAnswer: <jawaban>"  ← untuk embedding kontekstual
  meta    :
    topic    : kategori topik (shipping / purchasing / refund / return / account / general)
    question : pertanyaan asli (untuk referensi)
  embedding: [768 float values]  ← vektor dari SentenceTransformers
  ```
- **Duplicate Policy**: OVERWRITE — aman untuk re-run script/notebook
- **Embedding Model**: `all-mpnet-base-v2` (768 dimensi) — konsisten dengan produk

### 4a. Jalankan Script Storing Common Info

```bash
python process/store_common_info_script.py
```

Script ini akan:
1. Load 29 FAQ dari `data/common_information.json`
2. Konversi ke Haystack Documents
3. Generate embedding dan simpan ke collection `common_information`

### 4b. Buat Vector Search Index untuk `common_information`

Di MongoDB Atlas → **Atlas Search** → **Create Search Index** → **Atlas Vector Search**:
- Collection: `common_information`
- Nama index: `common_info_vector_index`

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

Tunggu status index menjadi **Active** sebelum menjalankan aplikasi.

---

## Langkah 5 — Jalankan Streamlit App

```bash
streamlit run website/website.py
```

Buka browser di [http://localhost:8501](http://localhost:8501)

### Contoh Pengujian Routing Agent

| Query | Tool yang Dipilih Agent |
|---|---|
| "I want a cotton dress under $50" | `retrieve_and_generate_recommendation` |
| "Show me women's leather accessories" | `retrieve_and_generate_recommendation` |
| "How long does shipping take?" | `retrieve_common_information` |
| "What is your refund policy?" | `retrieve_common_information` |
| "How do I return an item?" | `retrieve_common_information` |
| "Hello!" | Dijawab langsung (no tool) |

---

## Langkah 6 — Jalankan FastAPI (Opsional)

```bash
uvicorn website.api:app --host 0.0.0.0 --port 8000 --reload
```

Dokumentasi API: [http://localhost:8000/docs](http://localhost:8000/docs)

**Endpoints:**

| Method | Endpoint | Deskripsi |
|--------|----------|-----------||
| `GET` | `/health` | Cek status API |
| `POST` | `/recommend` | Rekomendasi produk |
| `POST` | `/common-info` | Pertanyaan umum e-commerce |

---

## Langkah 7 — Deployment dengan Docker

```bash
# Pastikan .env sudah terisi
docker compose up --build
```

- Streamlit UI: [http://localhost:8501](http://localhost:8501)
- FastAPI: [http://localhost:8000](http://localhost:8000)

> **Catatan Keamanan**: File `.env` **tidak** di-copy ke dalam Docker image. Credentials disuntik saat runtime via `env_file` di `docker-compose.yml`.

---

## Tech Stack

| Komponen | Library / Service |
|----------|-------------------|
| AI Agent | [Google ADK](https://google.github.io/adk-docs/) |
| RAG Framework | [Haystack AI](https://haystack.deepset.ai/) |
| LLM | [Groq](https://groq.com/) — `llama-3.3-70b-versatile` |
| Embedding | SentenceTransformers (`all-mpnet-base-v2`) |
| Vector DB | MongoDB Atlas Vector Search |
| Frontend | Streamlit |
| API | FastAPI |
| Deployment | Docker + Docker Compose |

---

## Troubleshooting

**`ModuleNotFoundError`**
```bash
python -m pip install -r requirements.txt
```

**`ServerSelectionTimeoutError` (MongoDB)**
- Whitelist IP di MongoDB Atlas → Network Access → Add IP Address → `0.0.0.0/0`

**`AuthenticationError` (Groq)**
- Pastikan `GROQ_API_KEY` di `.env` sudah benar
- Cek di [console.groq.com](https://console.groq.com)

**Vector search tidak mengembalikan hasil**
- Pastikan index `vector_index` / `common_info_vector_index` statusnya **Active**
- Re-run script storing jika perlu