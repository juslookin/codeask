import { useState } from "react"

const API = (import.meta.env.VITE_API_URL || "http://localhost:8000").replace(/\/+$/, "")

const STATUS = {
    idle: null,
    loading: "loading",
    error: "error",
}

export default function IngestForm({ onIngested }) {
    const [url, setUrl] = useState("")
    const [status, setStatus] = useState(STATUS.idle)
    const [message, setMessage] = useState("")

    const isValidGithubUrl = (s) =>
        /^https:\/\/github\.com\/[\w.-]+\/[\w.-]+\/?$/.test(s.trim())

    async function handleSubmit() {
        const trimmed = url.trim()
        if (!trimmed) return

        if (!isValidGithubUrl(trimmed)) {
            setStatus(STATUS.error)
            setMessage("Please enter a valid GitHub repo URL (https://github.com/owner/repo)")
            return
        }

        setStatus(STATUS.loading)
        setMessage("Cloning and indexing — this takes 1–3 minutes for a typical repo…")

        try {
            const res = await fetch(`${API}/ingest`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ github_url: trimmed }),
            })

            if (!res.ok) {
                throw new Error(`Server returned ${res.status}`)
            }

            const data = await res.json()

            if (data.success) {
                onIngested(data.collection_name)
            } else {
                setStatus(STATUS.error)
                setMessage(data.error || "Ingestion failed — check the backend logs.")
            }
        } catch (e) {
            setStatus(STATUS.error)
            setMessage(
                e.message.startsWith("Server")
                    ? e.message
                    : "Could not reach the backend. Is uvicorn running?"
            )
        }
    }

    const loading = status === STATUS.loading

    return (
        <div className="w-full max-w-xl">
            <div className="flex gap-2">
                <input
                    value={url}
                    onChange={(e) => {
                        setUrl(e.target.value)
                        if (status === STATUS.error) setStatus(STATUS.idle)
                    }}
                    onKeyDown={(e) => e.key === "Enter" && !loading && handleSubmit()}
                    placeholder="https://github.com/owner/repo"
                    disabled={loading}
                    className="flex-1 p-3 rounded bg-gray-800 text-white placeholder-gray-500 disabled:opacity-50"
                />
                <button
                    onClick={handleSubmit}
                    disabled={loading || !url.trim()}
                    className="bg-blue-600 px-5 py-3 rounded text-white font-medium hover:bg-blue-500 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                >
                    {loading ? "Indexing…" : "Index"}
                </button>
            </div>

            {message && (
                <p
                    className={`mt-3 text-sm ${status === STATUS.error ? "text-red-400" : "text-gray-400"
                        }`}
                >
                    {status === STATUS.loading && (
                        <span className="inline-block mr-2 animate-pulse">⏳</span>
                    )}
                    {message}
                </p>
            )}

            <p className="mt-4 text-xs text-gray-600">
                Public repos only · Python, JS, TS, JSX, TSX · max 500 files / 50 MB
            </p>
        </div>
    )
}