import { useState, useRef, useEffect, useMemo } from "react"
import ReactMarkdown from "react-markdown"
import { BookOpen, Zap, Sparkles, Send, Bot, User } from "lucide-react"
import CitationTag from "./CitationTag"

const API = (import.meta.env.VITE_API_URL || "http://localhost:8000").replace(/\/+$/, "")

// Helper to safely extract text from React children
const extractText = (children) => {
  if (typeof children === "string" || typeof children === "number") return String(children)
  if (Array.isArray(children)) return children.map(extractText).join("")
  if (children?.props?.children) return extractText(children.props.children)
  return ""
}

/**
 * Parses raw markdown output from the LLM:
 * 1. Strips any trailing "## Citations" / "Citations" raw list
 * 2. Collects all unique citations in order of appearance
 * 3. Replaces inline citations with special `cite:index:citation` code tokens
 *    which render as Wikipedia-style [1], [2] badges
 */
function processCitations(rawText) {
  if (!rawText) return { processedText: "", citations: [] }

  let body = rawText
  let footnotes = ""

  // Separate any trailing Citations section if present
  const headerRegex = /(?:^|\n)(?:##\s*|###\s*)?Citations[\s:]*\n([\s\S]*)$/i
  const match = headerRegex.exec(rawText)
  if (match) {
    body = rawText.substring(0, match.index).trim()
    footnotes = match[1]
  }

  const citationRegex = /(?:📎\s*)?([a-zA-Z0-9_./\\-]+\.[a-zA-Z0-9]+:\d+(?:-\d+)?)/g
  const citationMap = new Map()
  const citations = []

  // Collect citations in order of appearance from the body
  let m
  while ((m = citationRegex.exec(body)) !== null) {
    const cit = m[1]
    if (!citationMap.has(cit)) {
      citationMap.set(cit, citations.length + 1)
      citations.push(cit)
    }
  }

  // Also collect any citations mentioned only in the footer
  if (footnotes) {
    citationRegex.lastIndex = 0
    while ((m = citationRegex.exec(footnotes)) !== null) {
      const cit = m[1]
      if (!citationMap.has(cit)) {
        citationMap.set(cit, citations.length + 1)
        citations.push(cit)
      }
    }
  }

  // 1. Replace parenthesized citations: ( 📎 path:1-2 ) or (path:1-2) or [📎 path:1-2]
  let processed = body.replace(
    /[([]\s*(?:📎\s*)?([a-zA-Z0-9_./\\-]+\.[a-zA-Z0-9]+:\d+(?:-\d+)?)\s*[)\]]/g,
    (full, cit) => {
      const idx = citationMap.get(cit)
      return idx ? ` \`cite:${idx}:${cit}\`` : full
    }
  )

  // 2. Replace standalone backtick or clip citations: 📎 path:1-2 or `path:1-2`
  processed = processed.replace(
    /(?:📎\s*|(?<=[\s(]))`?([a-zA-Z0-9_./\\-]+\.[a-zA-Z0-9]+:\d+(?:-\d+)?)`?/g,
    (full, cit) => {
      const idx = citationMap.get(cit)
      return idx ? ` \`cite:${idx}:${cit}\`` : full
    }
  )

  return { processedText: processed, citations }
}

const markdownComponents = {
  p: ({ children }) => (
    <p className="mb-4 last:mb-0 leading-relaxed text-gray-200 text-[13.5px]">
      {children}
    </p>
  ),
  h1: ({ children }) => (
    <h1 className="text-base font-bold text-white mt-6 mb-3 pb-1 border-b border-gray-700/60">
      {children}
    </h1>
  ),
  h2: ({ children }) => (
    <h2 className="text-sm font-bold text-blue-300 mt-5 mb-2 flex items-center gap-1.5">
      {children}
    </h2>
  ),
  h3: ({ children }) => (
    <h3 className="text-xs font-bold text-gray-200 uppercase tracking-wider mt-4 mb-2">
      {children}
    </h3>
  ),
  ul: ({ children }) => (
    <ul className="list-disc list-outside pl-5 mb-4 space-y-1.5 text-gray-200 text-[13.5px]">
      {children}
    </ul>
  ),
  ol: ({ children }) => (
    <ol className="list-decimal list-outside pl-5 mb-4 space-y-2 text-gray-200 text-[13.5px]">
      {children}
    </ol>
  ),
  li: ({ children }) => (
    <li className="leading-relaxed pl-1">
      {children}
    </li>
  ),
  strong: ({ children }) => (
    <strong className="font-semibold text-white">
      {children}
    </strong>
  ),
  em: ({ children }) => (
    <em className="italic text-gray-300">
      {children}
    </em>
  ),
  blockquote: ({ children }) => (
    <blockquote className="border-l-4 border-blue-500/70 bg-blue-950/20 pl-3.5 py-1.5 my-3 italic text-gray-300 rounded-r text-xs">
      {children}
    </blockquote>
  ),
  hr: () => <hr className="my-5 border-gray-700/60" />,
  code: ({ className, children, ...props }) => {
    const text = extractText(children).trim()
    // Intercept citation token: cite:index:citation_string
    if (text.startsWith("cite:")) {
      const parts = text.split(":")
      const index = parts[1]
      const citation = parts.slice(2).join(":")
      return (
        <CitationTag
          index={index}
          citation={citation}
          mode="inline"
          onClick={() => {
            const el = document.getElementById(`citation-${index}`)
            if (el) {
              el.scrollIntoView({ behavior: "smooth", block: "center" })
              el.classList.add("ring-2", "ring-blue-400", "bg-blue-950/80")
              setTimeout(() => {
                el.classList.remove("ring-2", "ring-blue-400", "bg-blue-950/80")
              }, 2000)
            }
          }}
        />
      )
    }

    return (
      <code
        className={
          className ||
          "bg-gray-900/90 text-blue-300 px-1.5 py-0.5 rounded font-mono text-[12px] border border-gray-700/50"
        }
        {...props}
      >
        {children}
      </code>
    )
  },
  pre: ({ children }) => (
    <pre className="bg-gray-950/90 p-3.5 rounded-lg my-3.5 overflow-x-auto border border-gray-800 text-[12px] font-mono custom-scrollbar text-gray-200">
      {children}
    </pre>
  ),
}

function AssistantMessage({ text }) {
  const { processedText, citations } = useMemo(() => processCitations(text), [text])

  return (
    <div className="bg-gray-800/80 text-gray-100 rounded-2xl rounded-tl-xs p-5 border border-gray-700/50 shadow-xl max-w-full backdrop-blur-sm">
      <div className="flex items-center gap-2 mb-3.5 pb-2.5 border-b border-gray-700/40">
        <div className="w-5 h-5 rounded-full bg-blue-600/30 border border-blue-500/40 flex items-center justify-center text-blue-400">
          <Bot className="w-3 h-3" />
        </div>
        <span className="text-xs font-medium text-gray-300">CodeAsk Assistant</span>
      </div>

      <ReactMarkdown components={markdownComponents}>
        {processedText}
      </ReactMarkdown>

      {citations.length > 0 && (
        <div className="mt-6 pt-4 border-t border-gray-700/60">
          <div className="flex items-center gap-2 mb-3 text-[11px] font-semibold text-gray-400 uppercase tracking-wider">
            <BookOpen className="w-3.5 h-3.5 text-blue-400" />
            <span>References & Code Citations</span>
            <span className="text-[10px] font-mono bg-blue-900/50 text-blue-300 px-1.5 py-0.2 rounded-full border border-blue-700/40">
              {citations.length}
            </span>
          </div>

          <div className="grid grid-cols-1 gap-1.5">
            {citations.map((c, idx) => (
              <CitationTag
                key={idx}
                index={idx + 1}
                citation={c}
                mode="footnote"
              />
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

export default function ChatWindow({ collectionName, onGraphUpdate }) {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState("")
  const [streaming, setStreaming] = useState(false)
  const [mode, setMode] = useState("graph") // "graph" (fast) or "agent" (thorough)
  const bottomRef = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [messages])

  async function handleSend() {
    if (!input.trim() || streaming) return
    const question = input.trim()
    setInput("")
    setMessages((prev) => [...prev, { role: "user", text: question }])
    setStreaming(true)
    if (onGraphUpdate) onGraphUpdate(null)

    try {
      const res = await fetch(`${API}/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, collection_name: collectionName, mode }),
      })

      if (!res.ok) {
        let errorDetail = `Backend error ${res.status}`
        try {
          const errData = await res.json()
          if (errData?.detail) errorDetail = errData.detail
        } catch {
          // ignore json parse error
        }
        throw new Error(errorDetail)
      }

      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      let rawStream = ""
      let assistantText = ""
      let graphExtracted = false
      setMessages((prev) => [...prev, { role: "assistant", text: "" }])

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        const chunk = decoder.decode(value, { stream: true })
        rawStream += chunk

        if (!graphExtracted) {
          const startIdx = rawStream.indexOf("__GRAPH_START__\n")
          const endIdx = rawStream.indexOf("__GRAPH_END__\n")

          if (startIdx !== -1 && endIdx !== -1) {
            const graphJson = rawStream.substring(startIdx + 16, endIdx)
            try {
              if (onGraphUpdate) onGraphUpdate(JSON.parse(graphJson))
            } catch (e) {
              console.error("Failed to parse graph", e)
            }

            assistantText = rawStream.substring(endIdx + 14)
            graphExtracted = true
          }
        } else {
          assistantText += chunk
        }

        if (graphExtracted) {
          setMessages((prev) => {
            const updated = [...prev]
            updated[updated.length - 1] = { role: "assistant", text: assistantText }
            return updated
          })
        }
      }
    } catch (e) {
      setMessages((prev) => {
        const updated = [...prev]
        const errText = `⚠️ **Error:** ${e.message || "Error connecting to backend."}`
        if (updated.length > 0 && updated[updated.length - 1].role === "assistant") {
          updated[updated.length - 1] = { role: "assistant", text: errText }
          return updated
        }
        return [...prev, { role: "assistant", text: errText }]
      })
    } finally {
      setStreaming(false)
    }
  }

  return (
    <div className="flex flex-col h-full w-full max-w-4xl p-4">
      {/* Messages container */}
      <div className="flex-1 overflow-y-auto mb-4 space-y-5 custom-scrollbar pr-2">
        {messages.length === 0 && (
          <div className="flex flex-col items-center justify-center h-full text-center text-gray-500 py-12">
            <Bot className="w-10 h-10 mb-3 text-gray-600 opacity-60" />
            <p className="text-sm font-medium text-gray-400">Ask anything about the indexed codebase</p>
            <p className="text-xs text-gray-600 mt-1 max-w-md">
              Trace execution flows, ask about function definitions, or inspect system dependencies with interactive graphs.
            </p>
          </div>
        )}

        {messages.map((m, i) => (
          <div key={i} className="w-full">
            {m.role === "user" ? (
              <div className="flex items-start gap-2 ml-auto max-w-[80%] justify-end">
                <div className="bg-blue-600 text-white rounded-2xl rounded-tr-xs px-4 py-2.5 shadow-md text-sm leading-relaxed border border-blue-500/40">
                  {m.text}
                </div>
                <div className="w-6 h-6 rounded-full bg-blue-700 flex items-center justify-center text-white shrink-0 mt-1 text-xs">
                  <User className="w-3.5 h-3.5" />
                </div>
              </div>
            ) : (
              <AssistantMessage text={m.text} />
            )}
          </div>
        ))}

        {streaming && messages[messages.length - 1]?.text === "" && (
          <div className="bg-gray-800/80 rounded-2xl rounded-tl-xs p-4 border border-gray-700/50 shadow-md text-sm flex items-center gap-3 text-gray-400">
            <div className="flex space-x-1">
              <div className="w-2 h-2 bg-blue-400 rounded-full animate-bounce [animation-delay:-0.3s]"></div>
              <div className="w-2 h-2 bg-blue-400 rounded-full animate-bounce [animation-delay:-0.15s]"></div>
              <div className="w-2 h-2 bg-blue-400 rounded-full animate-bounce"></div>
            </div>
            <span className="text-xs font-mono">Analyzing codebase and retrieving context...</span>
          </div>
        )}

        <div ref={bottomRef} />
      </div>

      {/* Mode selectors */}
      <div className="flex items-center gap-2 mb-2 text-xs">
        <span className="text-[11px] text-gray-500 font-medium mr-1">Retrieval:</span>
        <button
          type="button"
          onClick={() => setMode("graph")}
          disabled={streaming}
          title="Vector search + one-hop graph traversal — fast 2s response"
          className={`flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-medium transition-all cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed ${
            mode === "graph"
              ? "bg-blue-600 text-white shadow-sm"
              : "bg-gray-800 text-gray-400 hover:text-white border border-gray-700/60"
          }`}
        >
          <Zap className="w-3 h-3" />
          Fast (graph)
        </button>
        <button
          type="button"
          onClick={() => setMode("agent")}
          disabled={streaming}
          title="LangGraph multi-step planner/critic loop — deep thorough reasoning"
          className={`flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-medium transition-all cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed ${
            mode === "agent"
              ? "bg-blue-600 text-white shadow-sm"
              : "bg-gray-800 text-gray-400 hover:text-white border border-gray-700/60"
          }`}
        >
          <Sparkles className="w-3 h-3" />
          Thorough (agent)
        </button>
      </div>

      {/* Input bar */}
      <div className="flex gap-2 items-center bg-gray-800/90 p-1.5 rounded-xl border border-gray-700/60 shadow-lg focus-within:border-blue-500/80 transition-all">
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && handleSend()}
          disabled={streaming}
          placeholder={streaming ? "Generating response…" : "Ask a question about the code…"}
          className="flex-1 bg-transparent px-3 py-1.5 text-white placeholder-gray-500 text-sm focus:outline-none disabled:opacity-50"
        />
        <button
          onClick={handleSend}
          disabled={streaming || !input.trim()}
          className="bg-blue-600 hover:bg-blue-500 px-3.5 py-1.5 rounded-lg text-white font-medium text-xs flex items-center gap-1.5 disabled:opacity-40 disabled:cursor-not-allowed transition-colors cursor-pointer shadow-sm"
        >
          <span>Send</span>
          <Send className="w-3.5 h-3.5" />
        </button>
      </div>
    </div>
  )
}