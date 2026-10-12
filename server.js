import express from 'express';
import dotenv from 'dotenv';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings } from '@langchain/google-genai';
import { ChatOpenAI } from '@langchain/openai';
import { MemoryVectorStore } from '@langchain/classic/vectorstores/memory';
import { createRetrievalChain } from '@langchain/classic/chains/retrieval';
import { createStuffDocumentsChain } from '@langchain/classic/chains/combine_documents';
import { ChatPromptTemplate } from '@langchain/core/prompts';
import cors from 'cors';
import { RESUME_EXPORT_URL, portfolioProjects, extraFacts } from './knowledge.js';

dotenv.config();

const app = express();

// Render (and other reverse proxies) terminate TLS and forward the client IP
app.set("trust proxy", 1);

const corsOrigin = process.env.CORS_ORIGIN;
app.use(
  cors(
    corsOrigin
      ? { origin: corsOrigin.split(",").map((origin) => origin.trim()).filter(Boolean) }
      : undefined
  )
);
app.use(express.json());

const TOKEN_EXPIRED_MESSAGE = "The token has expired and no more questions can be asked.";
const QUOTA_EXCEEDED_MESSAGE = "The API quota has been exceeded. No more questions can be asked right now. Please try again later.";
const GROQ_UNAVAILABLE_MESSAGE = "Gemini quota is exceeded and Groq could not answer. Please try again later.";
const GROQ_CHAT_MODEL = process.env.GROQ_CHAT_MODEL || "openai/gpt-oss-20b";
const GEMINI_CHAT_MODEL = "gemini-2.5-flash";
const GEMINI_QUOTA_FILE = path.join(path.dirname(fileURLToPath(import.meta.url)), ".gemini-quota.json");

let geminiChain;
let groqChain;
let tokenExpired = false;
let geminiQuotaExceeded = false;
let geminiQuotaResetAt = null;
let geminiQuotaResetTimer = null;

// Lightweight health check for Render (responds before RAG init finishes)
app.get(["/", "/health"], (req, res) => {
  res.status(200).json({
    status: "ok",
    ready: Boolean(geminiChain || groqChain),
  });
});

function isTokenExpiredError(error) {
  const status = error?.status || error?.statusCode || error?.response?.status;
  const message = String(error?.message || error || "").toLowerCase();
  return (
    status === 401 ||
    status === 403 ||
    message.includes("api key expired") ||
    message.includes("api_key_invalid") ||
    message.includes("invalid api key") ||
    message.includes("api key not valid") ||
    message.includes("unauthenticated") ||
    (message.includes("expired") && (message.includes("key") || message.includes("token")))
  );
}

function markTokenExpired() {
  tokenExpired = true;
  geminiChain = null;
}

function collectErrorText(error) {
  const parts = [];
  let current = error;
  for (let depth = 0; depth < 5 && current; depth += 1) {
    parts.push(String(current?.name || ""));
    parts.push(String(current?.message || current || ""));
    parts.push(String(current?.status || current?.statusCode || current?.statusText || ""));
    parts.push(String(current?.rateLimitReason || current?.rateLimitType || ""));
    if (current?.errorDetails) {
      parts.push(JSON.stringify(current.errorDetails));
    }
    current = current.cause || current.error || current.originalError;
  }
  return parts.join(" ").toLowerCase();
}

function isQuotaExceededError(error) {
  const status = error?.status || error?.statusCode || error?.response?.status || error?.cause?.status;
  const details = collectErrorText(error);
  return (
    status === 429 ||
    details.includes("429") ||
    details.includes("quota") ||
    details.includes("too many requests") ||
    details.includes("rate limit") ||
    details.includes("ratelimit") ||
    details.includes("resource exhausted")
  );
}

function parseGeminiRetryMs(error) {
  const details = error?.errorDetails || error?.cause?.errorDetails || [];
  for (const item of Array.isArray(details) ? details : []) {
    const delay = item?.retryDelay || item?.retry_delay;
    if (!delay) {
      continue;
    }
    const seconds = parseFloat(String(delay).replace(/s$/i, ""));
    if (Number.isFinite(seconds) && seconds > 0) {
      return Math.min(seconds * 1000, 20 * 60 * 60 * 1000);
    }
  }

  const match = collectErrorText(error).match(/retry in (\d+)h(\d+)m([\d.]+)?s/);
  if (match) {
    const hours = Number(match[1]);
    const minutes = Number(match[2]);
    const seconds = Number(match[3] || 0);
    return Math.min(((hours * 60 + minutes) * 60 + seconds) * 1000, 20 * 60 * 60 * 1000);
  }
  return 0;
}

function clearGeminiQuotaFile() {
  try {
    if (fs.existsSync(GEMINI_QUOTA_FILE)) {
      fs.unlinkSync(GEMINI_QUOTA_FILE);
    }
  } catch (error) {
    console.warn("Could not clear Gemini quota file:", error?.message || error);
  }
}

function persistGeminiQuota(resetAt) {
  try {
    fs.writeFileSync(GEMINI_QUOTA_FILE, JSON.stringify({ resetAt }, null, 2));
  } catch (error) {
    console.warn("Could not save Gemini quota file:", error?.message || error);
  }
}

function scheduleGeminiQuotaReset(retryMs) {
  if (geminiQuotaResetTimer) {
    clearTimeout(geminiQuotaResetTimer);
    geminiQuotaResetTimer = null;
  }
  if (!(retryMs > 0)) {
    return;
  }
  geminiQuotaResetTimer = setTimeout(() => {
    geminiQuotaExceeded = false;
    geminiQuotaResetAt = null;
    geminiQuotaResetTimer = null;
    clearGeminiQuotaFile();
    console.log("⏳ Gemini daily quota window ended. Gemini is primary again.");
  }, retryMs);
}

function loadGeminiQuotaState() {
  try {
    if (!fs.existsSync(GEMINI_QUOTA_FILE)) {
      return;
    }
    const saved = JSON.parse(fs.readFileSync(GEMINI_QUOTA_FILE, "utf8"));
    const resetAt = Number(saved.resetAt);
    if (!resetAt || resetAt <= Date.now()) {
      clearGeminiQuotaFile();
      geminiQuotaExceeded = false;
      geminiQuotaResetAt = null;
      return;
    }
    geminiQuotaExceeded = true;
    geminiQuotaResetAt = resetAt;
    scheduleGeminiQuotaReset(resetAt - Date.now());
    console.log("📉 Gemini daily quota is still exceeded. Header and chat will use Groq until it resets.");
  } catch (error) {
    console.warn("Could not read Gemini quota file:", error?.message || error);
  }
}

function markGeminiQuotaExceeded(error) {
  geminiQuotaExceeded = true;
  const retryMs = parseGeminiRetryMs(error) || 14 * 60 * 60 * 1000;
  geminiQuotaResetAt = Date.now() + retryMs;
  persistGeminiQuota(geminiQuotaResetAt);
  scheduleGeminiQuotaReset(retryMs);
}

function currentModel() {
  const provider = currentProvider();
  if (provider === "gemini") {
    return GEMINI_CHAT_MODEL;
  }
  if (provider === "groq") {
    return GROQ_CHAT_MODEL;
  }
  return null;
}

function providerStatusFields() {
  return {
    activeProvider: currentProvider(),
    activeModel: currentModel(),
    groqReady: Boolean(groqChain),
    geminiQuotaExceeded,
    geminiQuotaResetAt: geminiQuotaExceeded ? geminiQuotaResetAt : null
  };
}

function currentProvider() {
  if (!geminiQuotaExceeded && geminiChain) {
    return "gemini";
  }
  if (groqChain) {
    return "groq";
  }
  return null;
}

function sendQuotaExceeded(res) {
  return res.status(429).json({
    quotaExceeded: true,
    error: QUOTA_EXCEEDED_MESSAGE,
    ...providerStatusFields()
  });
}

function chunkText(text, maxLen = 700) {
  const parts = String(text || "")
    .replace(/^\uFEFF/, "")
    .replace(/_{3,}/g, "\n")
    .split(/\n{2,}/)
    .map((part) => part.replace(/\s+/g, " ").trim())
    .filter((part) => part.length > 20);

  const chunks = [];
  for (const part of parts) {
    if (part.length <= maxLen) {
      chunks.push(part);
      continue;
    }
    for (let index = 0; index < part.length; index += maxLen) {
      chunks.push(part.slice(index, index + maxLen).trim());
    }
  }
  return chunks;
}

async function loadResumeChunks() {
  const response = await fetch(RESUME_EXPORT_URL);
  if (!response.ok) {
    throw new Error(`Failed to fetch resume document (${response.status})`);
  }
  const resumeText = await response.text();
  const chunks = chunkText(resumeText);
  if (!chunks.length) {
    throw new Error("Resume document did not contain usable text");
  }
  return chunks;
}

function sendExpired(res) {
  return res.status(401).json({
    tokenExpired: true,
    error: TOKEN_EXPIRED_MESSAGE
  });
}

function formatHistory(history) {
  if (!Array.isArray(history) || history.length === 0) {
    return "None";
  }

  return history
    .slice(-16)
    .map((entry) => {
      const who = entry.role === "user" ? "User" : "Assistant";
      const text = String(entry.text || "").trim().slice(0, 600);
      return text ? `${who}: ${text}` : "";
    })
    .filter(Boolean)
    .join("\n") || "None";
}

function createRagPrompt() {
  return ChatPromptTemplate.fromTemplate(`
      You are an automated, friendly assistant on Meghsham Jade's portfolio website.
      Strictly follow these execution instructions:
      1. Answer the user's inquiry using the provided facts in the Context section below. Those facts come from Meghsham's resume and the projects on this portfolio.
      2. Also use Chat History to understand follow-up questions, pronouns, and references to earlier answers.
      3. If the answer cannot be found in the context (even with chat history), say exactly: "I don't have that specific information in Meghsham's resume or projects."
      4. Do not invent details, employers, dates, or project links. Keep responses short and conversational.

      Chat History:
      {chat_history}

      Context: {context}

      User Question: {input}
      Assistant Answer:
    `);
}

async function createRagChain(llm, retriever) {
  const documentChain = await createStuffDocumentsChain({
    llm,
    prompt: createRagPrompt()
  });
  return createRetrievalChain({
    combineDocsChain: documentChain,
    retriever
  });
}

function createGroqLlm() {
  return new ChatOpenAI({
    model: GROQ_CHAT_MODEL,
    apiKey: process.env.GROQ_API_KEY,
    temperature: 0.1,
    maxRetries: 1,
    configuration: {
      baseURL: "https://api.groq.com/openai/v1"
    }
  });
}

async function invokeRagChain(chain, message, history) {
  const response = await chain.invoke(
    {
      input: message,
      chat_history: formatHistory(history)
    },
    {
      callbacks: [
        {
          handleLLMEnd: (output) => {
            const tokenUsage = output.generations[0][0].message?.usage_metadata;

            console.log("\n📊 Token Usage:");
            if (tokenUsage) {
              console.log(`- Input Tokens:  ${tokenUsage.input_tokens}`);
              console.log(`- Output Tokens: ${tokenUsage.output_tokens}`);
              console.log(`- Total Tokens:  ${tokenUsage.total_tokens}\n`);
            } else {
              console.log(output.llmOutput || "Usage data not found.");
            }
          }
        }
      ]
    }
  );
  return response.answer;
}

// 1. Build knowledge from the public resume Google Doc plus portfolio projects
async function buildKnowledgeBase() {
  const resumeChunks = await loadResumeChunks();
  return [...resumeChunks, ...portfolioProjects, ...extraFacts];
}

// 2. Initialize Gemini first, then Groq as a quota fallback
async function initializeBot() {
  loadGeminiQuotaState();
  try {
    const knowledgeBase = await buildKnowledgeBase();

    const embeddings = new GoogleGenerativeAIEmbeddings({
      apiKey: process.env.GEMINI_API_KEY,
      model: "gemini-embedding-2"
    });

    const vectorStore = await MemoryVectorStore.fromTexts(
      knowledgeBase,
      knowledgeBase.map((_, index) => ({ id: index })),
      embeddings
    );
    const retriever = vectorStore.asRetriever(4);

    try {
      const geminiLlm = new ChatGoogleGenerativeAI({
        model: GEMINI_CHAT_MODEL,
        apiKey: process.env.GEMINI_API_KEY,
        temperature: 0.1,
        maxRetries: 0,
      });
      geminiChain = await createRagChain(geminiLlm, retriever);
      tokenExpired = false;
      console.log("🚀 Gemini RAG chain initialized.");
    } catch (error) {
      console.error("❌ Failed to initialize Gemini chat model:", error);
      if (isTokenExpiredError(error)) {
        markTokenExpired();
      }
      if (isQuotaExceededError(error)) {
        markGeminiQuotaExceeded(error);
      }
    }

    if (process.env.GROQ_API_KEY) {
      try {
        groqChain = await createRagChain(createGroqLlm(), retriever);
        console.log("⚡ Groq fallback chain initialized with " + GROQ_CHAT_MODEL);
      } catch (error) {
        console.error("❌ Failed to initialize Groq fallback:", error);
      }
    } else {
      console.warn("⚠️ GROQ_API_KEY is missing. Gemini quota fallback is disabled.");
    }
  } catch (error) {
    console.error("❌ Failed to initialize AI brain:", error);
    if (isTokenExpiredError(error)) {
      markTokenExpired();
      console.error("🔑 Gemini token has expired. Chat questions will be blocked unless Groq is available.");
    }
    if (isQuotaExceededError(error)) {
      markGeminiQuotaExceeded(error);
      console.error("📉 Gemini API quota has been exceeded.");
    }
  }
}

app.get('/api/chat/status', (req, res) => {
  if (tokenExpired && !groqChain) {
    return sendExpired(res);
  }
  if (geminiQuotaExceeded && !groqChain) {
    return sendQuotaExceeded(res);
  }
  return res.json({
    ready: Boolean(geminiChain || groqChain),
    tokenExpired: false,
    quotaExceeded: false,
    ...providerStatusFields()
  });
});

// 3. Create the endpoint for processing messages
app.post('/api/chat', async (req, res) => {
  const { message, history } = req.body;

  if (!message) {
    return res.status(400).json({ error: "Message property is required in the request body." });
  }

  if (!geminiChain && !groqChain) {
    if (tokenExpired) {
      return sendExpired(res);
    }
    if (geminiQuotaExceeded) {
      return sendQuotaExceeded(res);
    }
    return res.status(503).json({ error: "The AI system is still initializing. Please try again in a moment." });
  }

  if (!geminiQuotaExceeded && geminiChain) {
    try {
      const answer = await invokeRagChain(geminiChain, message, history);
      return res.json({ reply: answer, provider: "gemini", ...providerStatusFields() });
    } catch (error) {
      console.error("Error during Gemini generation:", error);
      if (isTokenExpiredError(error)) {
        markTokenExpired();
        if (!groqChain) {
          return sendExpired(res);
        }
      } else if (isQuotaExceededError(error)) {
        markGeminiQuotaExceeded(error);
        console.log("📉 Gemini quota exceeded. Switching to Groq fallback.");
        if (!groqChain) {
          return sendQuotaExceeded(res);
        }
      } else if (!groqChain) {
        return res.status(500).json({ error: "Internal server error during response generation." });
      }
    }
  }

  if (groqChain) {
    try {
      const answer = await invokeRagChain(groqChain, message, history);
      return res.json({ reply: answer, provider: "groq", ...providerStatusFields() });
    } catch (error) {
      console.error("Error during Groq generation:", error);
      const details = collectErrorText(error);
      if (details.includes("401") || details.includes("403") || details.includes("invalid api key") || details.includes("unauthorized")) {
        return res.status(401).json({
          error: "The Groq API key is invalid or expired."
        });
      }
      if (isQuotaExceededError(error)) {
        return res.status(429).json({
          quotaExceeded: true,
          groqReady: false,
          error: QUOTA_EXCEEDED_MESSAGE
        });
      }
      return res.status(500).json({ error: GROQ_UNAVAILABLE_MESSAGE });
    }
  }

  return sendQuotaExceeded(res);
});

const HOST = process.env.HOST || "0.0.0.0";
const PORT = Number(process.env.PORT) || 5000;

app.listen(PORT, HOST, async () => {
  console.log(`📡 Server listening on http://${HOST}:${PORT}`);
  await initializeBot();
});
