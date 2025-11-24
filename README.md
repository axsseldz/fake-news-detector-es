# Fake News Detector (Word2Vec + BETO)

Aplicación full-stack para evaluar noticias con cuatro modelos entrenados en español:
Word2Vec v1/v2 (TF-IDF + LinearSVC) y BETO v1/v2 (fine-tuning de BERT).

## Demo visual
Vista de la interfaz
- ![Interfaz](https://shorturl.at/0U6HR)

## Notebook de los modelos
- Entrenamiento y preparación de los cuatro modelos: https://colab.research.google.com/drive/1Ihj6gFfU8Dj3WanaLHlq3onYBtwHfN95?usp=sharing

## Tecnologías
- Frontend: Next.js 13+, React, TailwindCSS (estilos personalizados en `frontend/src/app`).
- Backend: FastAPI + Uvicorn.
- ML/NLP: Gensim (Word2Vec), scikit-learn (TF-IDF, LinearSVC), Transformers (BETO), Torch.

## Endpoints principales (backend)
- `POST /predict/word2vec_v1`
- `POST /predict/word2vec_v2`
- `POST /predict/beto_finetuned`
- `POST /predict/beto2_finetuned`
- `POST /generate/gemini` (genera ejemplos de noticias via Gemini; requiere `GEMINI_API_KEY` en `backend/.env`).

## Uso rápido
1) Backend: `cd backend && pip install -r requirements.txt && uvicorn main:app --reload --port 8000`
2) Frontend: `cd frontend && npm install && npm run dev`
3) Abre `http://localhost:3000`, elige un modelo y envía texto de noticia.

## Modelos locales (no se suben al repo)
- Crea la carpeta `backend/models/` y coloca allí los artefactos de los cuatro modelos (Word2Vec v1/v2 y BETO v1/v2).
- La carpeta está ignorada en Git; necesitas copiar los archivos localmente antes de ejecutar el backend.
