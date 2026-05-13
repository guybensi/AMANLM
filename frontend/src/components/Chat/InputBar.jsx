import { useState, useRef } from 'react'
import { useAppStore } from '../../stores/appStore'

export default function InputBar({ onSend, loading, hasDocuments }) {
  const [text, setText] = useState('')
  const { answerMode, setAnswerMode } = useAppStore()
  const textareaRef = useRef(null)

  const handleSend = () => {
    const trimmed = text.trim()
    if (!trimmed || loading || !hasDocuments) return
    onSend(trimmed)
    setText('')
    textareaRef.current?.focus()
  }

  const handleKey = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  return (
    <div className="border-t border-slate-700 bg-slate-900 px-4 py-3">
      {/* Mode toggle */}
      <div className="flex items-center gap-2 mb-2">
        <span className="text-xs text-slate-500 font-medium hidden sm:inline">Answer mode:</span>
        <div className="flex bg-slate-800 rounded-lg p-0.5 gap-0.5">
          {['short', 'long'].map((m) => (
            <button
              key={m}
              onClick={() => setAnswerMode(m)}
              className={`px-3 py-1 rounded-md text-xs font-medium transition-all capitalize ${
                answerMode === m
                  ? 'bg-indigo-600 text-white shadow'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              {m === 'short' ? '⚡ Short' : '📝 Long'}
            </button>
          ))}
        </div>
      </div>

      {/* Input */}
      <div className="flex items-end gap-2">
        <textarea
          ref={textareaRef}
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={handleKey}
          rows={2}
          disabled={loading || !hasDocuments}
          placeholder={hasDocuments ? 'Ask a question…' : 'Upload documents first'}
          className="flex-1 bg-slate-800 border border-slate-600 rounded-xl px-3 sm:px-4 py-2.5 text-sm text-slate-100 placeholder-slate-500 resize-none focus:outline-none focus:border-indigo-500 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
        />
        <button
          onClick={handleSend}
          disabled={!text.trim() || loading || !hasDocuments}
          className="bg-indigo-600 hover:bg-indigo-500 disabled:opacity-40 disabled:cursor-not-allowed text-white rounded-xl px-3 sm:px-4 py-2.5 text-sm font-medium transition-colors h-[52px] flex items-center gap-1 shrink-0"
        >
          {loading ? (
            <span className="animate-spin">⟳</span>
          ) : (
            <><span className="hidden sm:inline">Send </span><span>↑</span></>
          )}
        </button>
      </div>
      <p className="text-xs text-slate-600 mt-1.5 hidden sm:block">Enter to send · Shift+Enter for new line</p>
    </div>
  )
}
