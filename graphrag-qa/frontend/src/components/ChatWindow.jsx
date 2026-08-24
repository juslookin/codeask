import { useState, useRef, useEffect } from "react"
import ReactMarkdown from "react-markdown"
import CitationTag from "./CitationTag"

const API = import.meta.env.VITE_API_URL || "http://localhost:8000"

// Intercept inline code blocks that look like citations (filepath:start-end)
// and render them as clickable tags instead of <code> elements.
const renderers = {
  code({ node, className, children, ...props }) {
    const match = /[\w/.-]+:\d+-\d+/.exec(String(children).trim())
    if (!className && match) {
      return <CitationTag citation={match[0]} />
    }
    return (
      <code className={className} {...props}>
        {children}
      </code>
    )
  },
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
    if (onGraphUpdate) onGraphUpdate(null) // Reset graph on new question

    try {
      const res = await fetch(`${API}/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, collection_name: collectionName, mode }),
      })

      if (!res.ok) throw new Error(`Backend error ${res.status}`)

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
            } catch (e) { console.error("Failed to parse graph", e) }
            
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
    } catch {
      setMessages((prev) => [
        ...prev,
        { role: "assistant", text: "Error connecting to backend." },
      ])
    } finally {
      setStreaming(false)
    }
  }

  return (
    <div className="flex flex-col h-full w-full max-w-4xl p-4">
      <div className="flex-1 overflow-y-auto mb-4 space-y-2 custom-scrollbar pr-2">
        {messages.length === 0 && (
          <p className="text-gray-600 text-sm text-center mt-8">
            Ask anything about the indexed codebase.
          </p>
        )}
        {messages.map((m, i) => (
          <div
            key={i}
            className={`p-3 rounded-lg text-white text-sm ${m.role === "user"
                ? "bg-blue-600 ml-auto max-w-[75%]"
                : "bg-gray-800 max-w-full"
              }`}
          >
            {m.role === "assistant" ? (
              <ReactMarkdown components={renderers}>{m.text}</ReactMarkdown>
            ) : (
              m.text
            )}
          </div>
        ))}
        <div ref={bottomRef} />
      </div>

      <div className="flex gap-2 mb-2 text-xs">
        <button
          type="button"
          onClick={() => setMode("graph")}
          disabled={streaming}
          title="Vector search + one-hop graph traversal — no extra LLM calls"
          className={`px-3 py-1 rounded-full transition-colors disabled:opacity-50 disabled:cursor-not-allowed ${mode === "graph"
              ? "bg-blue-600 text-white"
              : "bg-gray-800 text-gray-400 hover:text-white"
            }`}
        >
          Fast (graph)
        </button>
        <button
          type="button"
          onClick={() => setMode("agent")}
          disabled={streaming}
          title="LangGraph planner/critic loop — more LLM calls, more latency"
          className={`px-3 py-1 rounded-full transition-colors disabled:opacity-50 disabled:cursor-not-allowed ${mode === "agent"
              ? "bg-blue-600 text-white"
              : "bg-gray-800 text-gray-400 hover:text-white"
            }`}
        >
          Thorough (agent)
        </button>
      </div>

      <div className="flex gap-2">
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && handleSend()}
          disabled={streaming}
          placeholder={streaming ? "Thinking…" : "Ask a question about the code…"}
          className="flex-1 p-2 rounded bg-gray-800 text-white placeholder-gray-500 disabled:opacity-50"
        />
        <button
          onClick={handleSend}
          disabled={streaming || !input.trim()}
          className="bg-blue-600 px-4 py-2 rounded text-white hover:bg-blue-500 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          Send
        </button>
      </div>
    </div>
  )
}