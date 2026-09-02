import { useState } from "react"
import { Copy, Check, FileCode } from "lucide-react"

export function InlineCitation({ index, citation, onClick }) {
  return (
    <button
      type="button"
      onClick={onClick}
      title={`Source: ${citation}`}
      className="inline-flex items-center justify-center font-mono text-[11px] font-semibold text-blue-400 hover:text-blue-200 bg-blue-950/80 hover:bg-blue-900 border border-blue-700/60 rounded px-1.5 py-0.5 mx-0.5 align-super cursor-pointer transition-all hover:scale-105 shadow-sm"
    >
      [{index}]
    </button>
  )
}

export function CitationFootnote({ index, citation }) {
  const [copied, setCopied] = useState(false)

  const handleCopy = () => {
    navigator.clipboard.writeText(citation)
    setCopied(true)
    setTimeout(() => setCopied(false), 1800)
  }

  const parts = citation.split(":")
  const filePath = parts[0]
  const lineRange = parts[1] ? `L${parts[1]}` : ""

  return (
    <div
      id={`citation-${index}`}
      className="flex items-center justify-between text-xs py-1.5 px-2.5 rounded-md bg-gray-900/70 hover:bg-gray-900 border border-gray-800 hover:border-gray-700 transition-colors group"
    >
      <div className="flex items-center gap-2 overflow-hidden mr-2">
        <span className="font-mono text-blue-400 font-bold text-[11px]">
          [{index}]
        </span>
        <FileCode className="w-3.5 h-3.5 text-gray-500 shrink-0" />
        <span className="font-mono text-gray-300 truncate" title={filePath}>
          {filePath}
        </span>
        {lineRange && (
          <span className="shrink-0 text-[10px] font-mono bg-gray-800 text-blue-300 px-1.5 py-0.5 rounded border border-gray-700/60">
            {lineRange}
          </span>
        )}
      </div>

      <button
        type="button"
        onClick={handleCopy}
        className="text-gray-400 hover:text-white p-1 rounded hover:bg-gray-800 transition-colors shrink-0 flex items-center gap-1 cursor-pointer"
        title="Copy citation"
      >
        {copied ? (
          <>
            <Check className="w-3.5 h-3.5 text-green-400" />
            <span className="text-[10px] text-green-400">Copied</span>
          </>
        ) : (
          <Copy className="w-3.5 h-3.5 opacity-60 group-hover:opacity-100" />
        )}
      </button>
    </div>
  )
}

export default function CitationTag({ citation, index = 1, mode = "inline", onClick }) {
  if (mode === "footnote") {
    return <CitationFootnote index={index} citation={citation} />
  }
  return <InlineCitation index={index} citation={citation} onClick={onClick} />
}
