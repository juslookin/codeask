import ChatWindow from "./components/ChatWindow"
import ErrorBoundary from "./components/ErrorBoundary"

export default function App() {
  return (
    <main className="max-w-3xl mx-auto p-6 h-screen flex flex-col">
      <h1 className="text-2xl font-bold text-white mb-4">GraphRAG Codebase Q&A</h1>
      <div className="flex-1 min-h-0">
        <ErrorBoundary>
          <ChatWindow collectionName="pallets_flask" />
        </ErrorBoundary>
      </div>
    </main>
  )
}
