import { useEffect } from 'react'
import { useAppStore } from './stores/appStore'
import Sidebar from './components/Layout/Sidebar'
import ChatWindow from './components/Chat/ChatWindow'
import InputBar from './components/Chat/InputBar'
import { useChat } from './hooks/useChat'

export default function App() {
  const { darkMode } = useAppStore()
  const { chatHistory, sendMessage, loading, error } = useChat()
  const documents = useAppStore((s) => s.documents)

  useEffect(() => {
    if (darkMode) {
      document.documentElement.classList.add('dark')
    } else {
      document.documentElement.classList.remove('dark')
    }
  }, [darkMode])

  return (
    <div className="flex h-screen bg-slate-950 text-slate-100 overflow-hidden">
      <Sidebar />

      <main className="flex-1 flex flex-col min-w-0">
        {/* Top bar */}
        <header className="border-b border-slate-700 px-6 py-3 flex items-center justify-between bg-slate-900 shrink-0">
          <div>
            <h2 className="text-sm font-semibold text-slate-200">Research Chat</h2>
            <p className="text-xs text-slate-500">Answers grounded in your uploaded documents</p>
          </div>
          {documents.length > 0 && (
            <span className="text-xs bg-emerald-900/40 text-emerald-400 border border-emerald-800 px-2 py-1 rounded-full">
              {documents.length} source{documents.length !== 1 ? 's' : ''} active
            </span>
          )}
        </header>

        {/* Error banner */}
        {error && (
          <div className="mx-4 mt-3 bg-red-900/30 border border-red-800 text-red-300 text-sm px-4 py-2 rounded-lg">
            {error}
          </div>
        )}

        <ChatWindow messages={chatHistory} loading={loading} />

        <InputBar
          onSend={sendMessage}
          loading={loading}
          hasDocuments={documents.length > 0}
        />
      </main>
    </div>
  )
}
