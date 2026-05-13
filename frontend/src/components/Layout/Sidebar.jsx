import { useEffect } from 'react'
import DropZone from '../Upload/DropZone'
import FileChip from '../Upload/FileChip'
import { useDocuments } from '../../hooks/useDocuments'
import { useAppStore } from '../../stores/appStore'

export default function Sidebar({ open, onClose }) {
  const { documents, fetchDocuments } = useDocuments()
  const { darkMode, toggleDarkMode, clearChat } = useAppStore()

  useEffect(() => {
    fetchDocuments()
  }, [])

  return (
    <aside
      className={`
        fixed md:relative inset-y-0 left-0 z-30 md:z-auto
        w-72 shrink-0
        bg-slate-900 border-r border-slate-700
        flex flex-col h-screen
        transition-transform duration-300 ease-in-out
        ${open ? 'translate-x-0' : '-translate-x-full md:translate-x-0'}
      `}
    >
      {/* Header */}
      <div className="px-4 py-4 border-b border-slate-700">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-lg font-bold text-white tracking-tight">AMANLM</h1>
            <p className="text-xs text-slate-500">AI Research Assistant</p>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={toggleDarkMode}
              className="text-slate-400 hover:text-slate-200 transition-colors text-lg"
              title="Toggle dark mode"
            >
              {darkMode ? '☀️' : '🌙'}
            </button>
            {/* Close button — mobile only */}
            <button
              onClick={onClose}
              className="md:hidden text-slate-400 hover:text-slate-200 transition-colors text-base"
              aria-label="Close sidebar"
            >
              ✕
            </button>
          </div>
        </div>
      </div>

      {/* Upload */}
      <div className="px-4 py-3 border-b border-slate-700">
        <p className="text-xs text-slate-500 font-medium uppercase tracking-wider mb-2">Sources</p>
        <DropZone />
      </div>

      {/* Document list */}
      <div className="flex-1 overflow-y-auto px-4 py-3">
        {documents.length === 0 ? (
          <p className="text-xs text-slate-600 text-center mt-4">No documents yet</p>
        ) : (
          <div className="flex flex-col gap-2">
            {documents.map((doc) => (
              <FileChip key={doc.doc_id} doc={doc} />
            ))}
          </div>
        )}
      </div>

      {/* Footer */}
      <div className="px-4 py-3 border-t border-slate-700">
        <div className="flex items-center justify-between text-xs text-slate-600">
          <span>{documents.length} doc{documents.length !== 1 ? 's' : ''} · {documents.reduce((s, d) => s + d.chunk_count, 0)} chunks</span>
          <button
            onClick={clearChat}
            className="text-slate-500 hover:text-slate-300 transition-colors"
            title="Clear chat history"
          >
            🗑 Clear chat
          </button>
        </div>
      </div>
    </aside>
  )
}
