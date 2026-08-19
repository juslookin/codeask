export default function CitationTag({ citation }) {
  return (
    <button
      type="button"
      className="inline-flex items-center bg-blue-800 text-blue-200 text-xs px-2 py-0.5 rounded-full font-mono hover:bg-blue-700 transition-colors focus:outline-none focus:ring-2 focus:ring-blue-500"
      title={citation}
      onClick={() => navigator.clipboard.writeText(citation)}
    >
      📎 {citation}
    </button>
  )
}
