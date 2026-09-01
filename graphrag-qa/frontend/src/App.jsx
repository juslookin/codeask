import { useState, useEffect } from "react"
import IngestForm from "./components/IngestForm"
import ChatWindow from "./components/ChatWindow"
import FileExplorer from "./components/FileExplorer"
import GraphVisualizer from "./components/GraphVisualizer"

const API = (import.meta.env.VITE_API_URL || "http://localhost:8000").replace(/\/+$/, "")

export default function App() {
  const [collectionName, setCollectionName] = useState(null)
  const [fileTree, setFileTree] = useState([])
  const [retrievalGraph, setRetrievalGraph] = useState(null)

  useEffect(() => {
    if (collectionName) {
      // Fetch file tree from backend
      fetch(`${API}/api/files?collection=${collectionName}`)
        .then(res => res.json())
        .then(data => setFileTree(data.files || []))
        .catch(console.error)
    }
  }, [collectionName])

  return (
    <div className="h-screen w-screen bg-gray-950 text-white flex overflow-hidden">
      {!collectionName ? (
        <div className="flex-1 flex flex-col items-center justify-center p-8">
          <h1 className="text-4xl font-bold mb-2">CodeAsk</h1>
          <p className="text-gray-400 mb-8 text-sm">
            Paste a GitHub repo URL and ask questions about the code.
          </p>
          <IngestForm onIngested={setCollectionName} />
        </div>
      ) : (
        <div className="flex w-full h-full">
          {/* Left Panel: File Explorer */}
          <div className="w-64 border-r border-gray-800 flex flex-col bg-gray-900">
            <div className="p-3 border-b border-gray-800 flex items-center justify-between">
              <span className="text-xs font-semibold text-gray-400 tracking-wider">PROJECT EXPLORER</span>
              <button
                onClick={() => setCollectionName(null)}
                className="text-xs text-gray-500 hover:text-white"
                title="Index another repo"
              >
                Reset
              </button>
            </div>
            <div className="flex-1 overflow-hidden">
              <FileExplorer files={fileTree} />
            </div>
          </div>

          {/* Center Panel: Chat */}
          <div className="flex-1 border-r border-gray-800 flex flex-col min-w-0 bg-gray-950">
            <div className="p-3 border-b border-gray-800">
              <span className="text-xs font-semibold text-gray-400 tracking-wider">AI ASSISTANT CHAT</span>
            </div>
            <div className="flex-1 overflow-hidden flex justify-center w-full">
              <ChatWindow 
                collectionName={collectionName} 
                onGraphUpdate={setRetrievalGraph} 
              />
            </div>
          </div>

          {/* Right Panel: Graph Visualizer */}
          <div className="w-96 flex flex-col bg-gray-900">
            <div className="p-3 border-b border-gray-800">
              <span className="text-xs font-semibold text-gray-400 tracking-wider">ARCHITECTURE & DEPENDENCIES</span>
            </div>
            <div className="flex-1 overflow-hidden">
              <GraphVisualizer retrievalGraph={retrievalGraph} />
            </div>
          </div>
        </div>
      )}
    </div>
  )
}