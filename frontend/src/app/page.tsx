"use client";

import { useEffect, useMemo, useRef, useState } from "react";

type ModelId = "word2vec_v1" | "word2vec_v2" | "beto_finetuned" | "beto2_finetuned";

type ModelOption = {
  id: ModelId;
  name: string;
  tagline: string;
  details: string;
  chip: string;
};

type Message = {
  id: string;
  role: "user" | "model";
  text: string;
  model?: ModelId;
  label?: string | number;
  scores?: number[];
};

const MODEL_OPTIONS: ModelOption[] = [
  {
    id: "word2vec_v1",
    name: "Word2Vec v1",
    tagline: "Base TF-IDF + LinearSVC",
    details: "Modelo base con TF-IDF y LinearSVC para clasificación de fake news en español.",
    chip: "V1",
  },
  {
    id: "word2vec_v2",
    name: "Word2Vec v2",
    tagline: "Back-translation + TF-IDF ponderado",
    details: "Modelo mejorado con back-translation, TF-IDF ponderado y LinearSVC optimizado.",
    chip: "V2",
  },
  {
    id: "beto_finetuned",
    name: "BETO v1 (Fine-tuned)",
    tagline: "BERT multilingüe en español",
    details: "Fine-tuning de BETO para clasificación de fake news en español.",
    chip: "V1",
  },
  {
    id: "beto2_finetuned",
    name: "BETO v2 (Fine-tuned)",
    tagline: "Back-translation en BETO",
    details: "Modelo mejorado con back-translation para clasificación de fake news.",
    chip: "V2",
  },
];

const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";

const formatLabel = (label?: string | number) => {
  if (label === undefined || label === null) return "Desconocido";
  const normalized = String(label).trim().toLowerCase();
  if (normalized === "0" || normalized.includes("fake") || normalized.includes("falso")) return "Falsa";
  if (normalized === "1" || normalized.includes("true") || normalized.includes("verdadero")) return "Verdadera";
  return String(label);
};

const isFake = (label?: string | number) => {
  const normalized = String(label ?? "").trim().toLowerCase();
  return normalized === "0" || normalized.includes("fake") || normalized.includes("falso");
};

const sanitizeText = (text: string) => {
  const noBold = text.replace(/\*\*/g, "");
  const noIntro = noBold.replace(/^aquí tienes (una opción|una propuesta):?\s*/i, "");
  const noLabels = noIntro.replace(/^\s*(titular|párrafo)\s*:\s*/gim, "");
  return noLabels.replace(/\s+\n/g, "\n").trim();
};

export default function Home() {
  const [input, setInput] = useState("");
  const [modelId, setModelId] = useState<ModelId>("beto_finetuned");
  const [messages, setMessages] = useState<Message[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [genLoading, setGenLoading] = useState(false);
  const [genFlavor, setGenFlavor] = useState<"fake" | "true">("fake");
  const feedRef = useRef<HTMLDivElement>(null);
  const lastMessageRef = useRef<HTMLDivElement | null>(null);

  const activeModel = useMemo(
    () => MODEL_OPTIONS.find((m) => m.id === modelId)!,
    [modelId],
  );

  useEffect(() => {
    if (lastMessageRef.current) {
      lastMessageRef.current.scrollIntoView({ behavior: "smooth", block: "end" });
    }
  }, [messages.length]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const trimmed = input.trim();
    if (!trimmed || loading) return;

    const userMessage: Message = {
      id: crypto.randomUUID(),
      role: "user",
      text: trimmed,
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setError(null);
    setLoading(true);

    try {
      const res = await fetch(`${API_BASE}/predict/${modelId}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: trimmed }),
      });

      if (!res.ok) {
        const detail = await res.text();
        throw new Error(detail || `Request failed with ${res.status}`);
      }

      const data = (await res.json()) as {
        label?: string | number;
        scores?: number[];
        proba?: number[];
      };

      const isWord2Vec = modelId.startsWith("word2vec");
      const scoreValues = data.proba ?? data.scores;
      const scoreText = scoreValues ? scoreValues.map((p) => p.toFixed(2)).join(" / ") : null;

      const modelMessage: Message = {
        id: crypto.randomUUID(),
        role: "model",
        text: `Predicción: ${formatLabel(data.label)}${scoreText ? ` · Puntajes: ${scoreText}` : ""}`,
        model: modelId,
        label: data.label,
        scores: scoreValues ?? undefined,
      };
      setMessages((prev) => [...prev, modelMessage]);
    } catch (err) {
      console.error(err);
      setError(err instanceof Error ? err.message : "Unexpected error");
    } finally {
      setLoading(false);
    }
  };

  const handleGenerate = async () => {
    if (genLoading) return;
    setGenLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/generate/gemini`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ flavor: genFlavor }),
      });
      if (!res.ok) {
        const detail = await res.text();
        throw new Error(detail || `Request failed with ${res.status}`);
      }
      const data = (await res.json()) as { text?: string };
      if (data.text) setInput(data.text);
    } catch (err) {
      console.error(err);
      setError(err instanceof Error ? err.message : "Error al generar texto");
    } finally {
      setGenLoading(false);
    }
  };
  return (
    <div className="min-h-screen bg-slate-950 text-slate-50">
      <div className="mx-auto max-w-6xl px-4 py-10 md:py-14">
        <header className="mb-10 space-y-3">
          <div className="inline-flex items-center gap-2 rounded-full bg-slate-800/70 px-3 py-1 text-xs uppercase tracking-[0.18em] text-slate-200">
            <span className="h-2 w-2 rounded-full bg-emerald-400 shadow-[0_0_0_6px_rgba(52,211,153,0.15)]" />
            Radar anti fake news
          </div>
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div>
              <h1 className="text-3xl font-semibold text-slate-50 sm:text-4xl">
                Evalúa noticias con cuatro modelos especializados
              </h1>
              <p className="mt-2 max-w-2xl text-sm text-slate-300 sm:text-base">
                Compara dos variantes Word2Vec (TF-IDF/LinearSVC) y dos BETO fine-tuned para español. Envía un titular o párrafo y alterna el modelo al instante.
              </p>
            </div>
            <div className="rounded-xl bg-slate-800/80 px-4 py-3 text-right shadow-lg ring-1 ring-slate-700/70 backdrop-blur">
              <p className="text-xs uppercase tracking-[0.14em] text-slate-400">
                Motor activo
              </p>
              <p className="text-lg font-medium text-emerald-200">{activeModel.name}</p>
              <p className="text-xs text-slate-400">{activeModel.tagline}</p>
            </div>
          </div>
        </header>

        <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_340px]">
          <main className="flex flex-col gap-4 rounded-2xl bg-slate-900/70 p-4 shadow-2xl ring-1 ring-slate-800/80 backdrop-blur">
            <div className="flex items-center justify-between border-b border-slate-800 pb-3">
              <div>
                <p className="text-sm font-medium text-slate-200">Historial de sesión</p>
                <p className="text-xs text-slate-400">
                  Los resultados se agregan en orden temporal. Cambia de modelo cuando quieras.
                </p>
              </div>
              <div className="flex items-center gap-2">
                <button
                  type="button"
                  onClick={() => setMessages([])}
                  className="rounded-lg border border-slate-700/70 bg-slate-800/60 px-3 py-1 text-xs font-semibold text-slate-200 shadow-sm shadow-slate-900/40 transition hover:-translate-y-[1px] hover:border-emerald-400/40 hover:text-emerald-100"
                >
                  Limpiar historial
                </button>
                <span className="rounded-full bg-slate-800 px-3 py-1 text-xs text-slate-300">
                  {messages.length} entradas
                </span>
              </div>
            </div>

            <section className="grid h-[55vh] gap-3 rounded-xl border border-slate-800/80 bg-gradient-to-b from-slate-900/60 to-slate-950/60 p-4 shadow-inner">
              <div
                ref={feedRef}
                className="scrollbar-thin flex h-full flex-col gap-3 overflow-y-auto pr-1 pb-12"
              >
                {messages.length === 0 && (
                  <div className="flex flex-col items-center justify-center gap-2 rounded-xl border border-dashed border-slate-800/80 px-6 py-12 text-center text-slate-400">
                    <p className="text-sm font-medium text-slate-200">Sin mensajes aún</p>
                    <p className="text-xs text-slate-400">
                      Pega un titular o párrafo, elige un modelo y obtén un veredicto al instante.
                    </p>
                  </div>
                )}
                {messages.map((msg, idx) => {
                  const isUser = msg.role === "user";
                  return (
                    <div
                      key={msg.id}
                      ref={idx === messages.length - 1 ? lastMessageRef : null}
                      className={`group rounded-xl border px-4 py-3 shadow transition ${isUser
                        ? "border-slate-800/80 bg-slate-900/70"
                        : "border-emerald-800/50 bg-emerald-950/40"
                        }`}
                    >
                      <div className="mb-1 flex items-center justify-between text-xs uppercase tracking-[0.14em] text-slate-400">
                        <span>{isUser ? "Usuario" : "Respuesta del modelo"}</span>
                        {!isUser && msg.model && (
                          <span className="rounded-full bg-emerald-500/10 px-2 py-1 text-[11px] font-semibold text-emerald-200">
                            {MODEL_OPTIONS.find((m) => m.id === msg.model)?.name ?? msg.model}
                          </span>
                        )}
                      </div>
                      <p className="text-sm leading-relaxed text-slate-100 whitespace-pre-line">
                        {sanitizeText(msg.text)}
                      </p>
                      {!isUser && msg.label && (
                        <div className="mt-2 flex items-center gap-2">
                          <span
                            className={`inline-flex items-center gap-1 rounded-full px-2 py-1 text-[11px] font-semibold ${isFake(msg.label)
                              ? "bg-rose-500/15 text-rose-100 ring-1 ring-rose-500/25"
                              : "bg-emerald-500/15 text-emerald-200 ring-1 ring-emerald-400/20"
                              }`}
                          >
                            <span
                              className={`h-1.5 w-1.5 rounded-full ${isFake(msg.label) ? "bg-rose-400" : "bg-emerald-400"
                                }`}
                            />
                            {formatLabel(msg.label)}
                          </span>
                          {msg.scores && (
                            <span className="text-[11px] text-slate-400">
                              puntajes: {msg.scores.map((s) => s.toFixed(2)).join(" / ")}
                            </span>
                          )}
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            </section>

            <form onSubmit={handleSubmit} className="space-y-3 rounded-xl border border-slate-800 bg-slate-900/70 p-4 shadow-lg">
              <div className="flex items-start gap-3">
                <div className="flex-1">
                  <label htmlFor="news-input" className="text-xs uppercase tracking-[0.14em] text-slate-400">
                    Pega el texto de la noticia
                  </label>
                  <textarea
                    id="news-input"
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    placeholder="Ejemplo: La NASA confirma que la Luna está hecha de queso..."
                    className="mt-2 h-28 w-full resize-none rounded-xl border border-slate-800 bg-slate-950/70 px-4 py-3 text-sm text-slate-100 outline-none transition focus:border-emerald-400/60 focus:ring-2 focus:ring-emerald-400/20"
                  />
                </div>
              </div>
              {error && <p className="text-sm text-rose-300">Error: {error}</p>}
              <div className="flex flex-wrap items-center justify-between gap-3">
                <div className="flex items-center gap-2 text-xs text-slate-400">
                  <span className="h-2 w-2 rounded-full bg-emerald-400 shadow-[0_0_0_6px_rgba(52,211,153,0.15)]" />
                  {loading ? "Analizando..." : "Listo"}
                </div>

                <div className="flex flex-wrap items-center gap-2">
                  <div className="flex items-center gap-2 rounded-lg border border-slate-800 bg-slate-900/70 px-3 py-2 text-xs text-slate-300">
                    <span>Gemini:</span>
                    <label className="flex items-center gap-1">
                      <input
                        type="radio"
                        name="gemini-flavor"
                        value="fake"
                        checked={genFlavor === "fake"}
                        onChange={() => setGenFlavor("fake")}
                        className="accent-emerald-400"
                      />
                      Falsa
                    </label>
                    <label className="flex items-center gap-1">
                      <input
                        type="radio"
                        name="gemini-flavor"
                        value="true"
                        checked={genFlavor === "true"}
                        onChange={() => setGenFlavor("true")}
                        className="accent-emerald-400"
                      />
                      Verdadera
                    </label>
                  </div>
                  <button
                    type="button"
                    onClick={handleGenerate}
                    disabled={genLoading}
                    className="inline-flex items-center gap-2 rounded-xl border border-slate-800 bg-slate-900 px-3 py-2 text-xs font-semibold text-slate-100 shadow hover:-translate-y-[1px] hover:border-emerald-400/40 disabled:cursor-not-allowed disabled:opacity-60"
                  >
                    {genLoading ? (
                      <>
                        <span className="h-3 w-3 animate-spin rounded-full border-2 border-slate-500 border-t-transparent" />
                        Generando
                      </>
                    ) : (
                      <>
                        <span className="h-2 w-2 rounded-full bg-emerald-400" />
                        Generar con Gemini
                      </>
                    )}
                  </button>
                </div>

                <button
                  type="submit"
                  disabled={loading}
                  className="inline-flex items-center gap-2 rounded-xl bg-emerald-500 px-4 py-2 text-sm font-semibold text-emerald-950 shadow-lg shadow-emerald-500/30 transition hover:-translate-y-[1px] hover:bg-emerald-400 disabled:cursor-not-allowed disabled:opacity-60"
                >
                  {loading ? (
                    <span className="inline-flex items-center gap-2">
                      <span className="h-4 w-4 animate-spin rounded-full border-2 border-emerald-900 border-t-transparent" />
                      Procesando
                    </span>
                  ) : (
                    <>
                      <span className="h-2 w-2 rounded-full bg-emerald-900" />
                      Enviar a revisión
                    </>
                  )}
                </button>
              </div>
            </form>
          </main>

          <aside className="space-y-4">
            <div className="rounded-2xl border border-slate-800/80 bg-slate-900/70 p-4 shadow-xl ring-1 ring-slate-800/80 backdrop-blur">
              <div className="mb-3 flex items-center justify-between">
                <div>
                  <p className="text-sm font-semibold text-slate-100">Selector de modelo</p>
                  <p className="text-xs text-slate-400">Cambia de motor sin perder el historial.</p>
                </div>
                <span className="rounded-full bg-slate-800 px-3 py-1 text-[11px] uppercase tracking-[0.14em] text-slate-300">
                  Selector de modelos
                </span>
              </div>
              <div className="space-y-3">
                {MODEL_OPTIONS.map((model) => {
                  const active = model.id === modelId;
                  return (
                    <button
                      key={model.id}
                      type="button"
                      onClick={() => setModelId(model.id)}
                      className={`w-full rounded-xl border px-4 py-3 text-left transition ${active
                        ? "border-emerald-500/60 bg-emerald-500/10 shadow-lg shadow-emerald-500/20"
                        : "border-slate-800 bg-slate-900/70 hover:border-slate-700 hover:bg-slate-900"
                        }`}
                    >
                      <div className="flex items-center justify-between">
                        <div>
                          <p className="text-sm font-semibold text-slate-100">{model.name}</p>
                          <p className="text-xs text-slate-400">{model.tagline}</p>
                        </div>
                        <span
                          className={`rounded-full px-2 py-1 text-[11px] font-semibold ${active ? "bg-emerald-400 text-emerald-950" : "bg-slate-800 text-slate-200"
                            }`}
                        >
                          {active ? "Activo" : model.chip}
                        </span>
                      </div>
                      <p className="mt-2 text-[13px] leading-relaxed text-slate-300">{model.details}</p>
                    </button>
                  );
                })}
              </div>
            </div>

            <div className="rounded-2xl border border-slate-800/80 bg-slate-900/70 p-4 shadow-lg ring-1 ring-slate-800/80 backdrop-blur">
              <div className="mb-3 flex items-center justify-between">
                <p className="text-sm font-semibold text-slate-100">Panel de señal</p>
                <span className="rounded-full bg-slate-800 px-3 py-1 text-[11px] uppercase tracking-[0.14em] text-slate-300">
                  En vivo
                </span>
              </div>
              <ul className="space-y-2 text-sm text-slate-300">
                <li className="flex items-center justify-between rounded-lg bg-slate-950/60 px-3 py-2">
                  <span>Latencia</span>
                  <span className="text-emerald-200">~ tiempo real</span>
                </li>
                <li className="flex items-center justify-between rounded-lg bg-slate-950/60 px-3 py-2">
                  <span>Entradas</span>
                  <span className="text-emerald-200">{messages.length}</span>
                </li>
                <li className="flex items-center justify-between rounded-lg bg-slate-950/60 px-3 py-2">
                  <span>Motor activo</span>
                  <span className="text-emerald-200">{activeModel.name}</span>
                </li>
              </ul>
              <p className="mt-3 text-xs text-slate-500">
                Usa BETO para matices y contexto; usa Word2Vec para triaje rápido en hardware limitado.
              </p>
            </div>
          </aside>
        </div>
      </div>
    </div>
  );
}