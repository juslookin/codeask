import { useState, useRef, useEffect, useCallback } from "react"
import ReactMarkdown from 'react-markdown'
import CitationTag from "./CitationTag"

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000"

const renderers = {
  code({className, children, ...props}) {
    const match = /[\w/.-]+:\d+-\d+/.exec(String(children).trim())
    if (!className && match) {
      return <CitationTag citation={match[0]} />
    }
    return <code className={className} {...props}>{children}</code>
  }
}

export default function ChatWindow({ collectionName }) {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState("")
  const [streaming, setStreaming] = useState(false)
  const bottomRef = useRef(null)
  const abortRef = useRef(null)
  const msgIdRef = useRef(0)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [messages])

  useEffect(() => {
    return () => { abortRef.current?.abort() }
  }, [])

  const handleSend = useCallback(async () => {
    if (!input.trim() || streaming) return
    const question = input.trim()
    setInput("")
    const userMsgId = ++msgIdRef.current
    const assistantMsgId = ++msgIdRef.current
    setMessages(prev => [...prev, { id: userMsgId, role: "user", text: question }])
    setStreaming(true)

    const controller = new AbortController()
    abortRef.current = controller

    try {
      const res = await fetch(`${API_URL}/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, collection_name: collectionName }),
        signal: controller.signal
      })
      if (!res.ok) throw new Error(`Backend error: ${res.status}`)
      if (!res.body) throw new Error("No response body")

      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      let assistantText = ""
      let rafPending = false
      setMessages(prev => [...prev, { id: assistantMsgId, role: "assistant", text: "" }])

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        assistantText += decoder.decode(value, { stream: true })
        if (!rafPending) {
          rafPending = true
          requestAnimationFrame(() => {
            setMessages(prev => {
              const updated = [...prev]
              updated[updated.length - 1] = { id: assistantMsgId, role: "assistant", text: assistantText }
              return updated
            })
            rafPending = false
          })
        }
      }
      // Final flush
      setMessages(prev => {
        const updated = [...prev]
        updated[updated.length - 1] = { id: assistantMsgId, role: "assistant", text: assistantText }
        return updated
      })
    } catch (err) {
      if (err.name !== 'AbortError') {
        console.error("Query error:", err)
        setMessages(prev => [...prev, { id: ++msgIdRef.current, role: "assistant", text: `Error: ${err.message}` }])
      }
    } finally {
      setStreaming(false)
      abortRef.current = null
    }
  }, [input, streaming, collectionName])

  return (
    <div className="flex flex-col h-full min-h-[400px] bg-gray-900 rounded-xl p-4">
      <div className="flex-1 overflow-y-auto mb-4">
         {messages.length === 0 && (
           <div className="flex items-center justify-center h-full text-gray-500">
             <p>Ask a question about the codebase to get started.</p>
           </div>
         )}
         {messages.map((m) => (
           <div key={m.id} className={`p-3 rounded-lg my-2 text-white ${m.role === 'user' ? 'bg-blue-600 ml-auto w-3/4' : 'bg-gray-800'}`}>
             {m.role === "assistant" ? <ReactMarkdown components={renderers}>{m.text}</ReactMarkdown> : m.text}
           </div>
         ))}
         {streaming && messages[messages.length - 1]?.text === "" && (
           <div className="p-3 rounded-lg my-2 bg-gray-800 text-gray-400">
             <span className="animate-pulse">Thinking...</span>
           </div>
         )}
         <div ref={bottomRef} />
      </div>
      <div className="flex gap-2">
        <input
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={e => e.key==='Enter' && handleSend()}
          className="flex-1 p-2 rounded bg-gray-800 text-white placeholder-gray-500"
          placeholder="Ask about the codebase..."
          aria-label="Question input"
          disabled={streaming}
        />
        <button
          onClick={handleSend}
          disabled={streaming || !input.trim()}
          className={`px-4 py-2 rounded text-white transition-colors ${
            streaming || !input.trim() ? 'bg-gray-600 cursor-not-allowed' : 'bg-blue-600 hover:bg-blue-700'
          }`}
        >
          {streaming ? 'Sending...' : 'Send'}
        </button>
      </div>
    </div>
  )
}
