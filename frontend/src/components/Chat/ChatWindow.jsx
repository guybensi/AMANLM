import { useEffect, useRef } from 'react'
import MessageBubble from './MessageBubble'

export default function ChatWindow({ messages, loading }) {
  const bottomRef = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading])

  if (messages.length === 0) {
    return (
      <div className="flex-1 flex flex-col items-center justify-center text-center px-6 select-none">
        <div className="text-5xl mb-4">🔍</div>
        <h2 className="text-xl font-semibold text-slate-300 mb-2">Ask anything about your documents</h2>
        <p className="text-sm text-slate-500 max-w-sm">
          Upload files using the sidebar, then ask questions. Every answer is grounded in your sources with proof.
        </p>
      </div>
    )
  }

  return (
    <div className="flex-1 overflow-y-auto px-4 py-4 space-y-1 scrollbar-thin scrollbar-thumb-slate-700">
      {messages.map((msg) => (
        <MessageBubble key={msg.id} message={msg} />
      ))}
      {loading && (
        <div className="flex justify-start mb-4">
          <div className="bg-slate-800 border border-slate-700 rounded-2xl rounded-tl-sm px-4 py-3">
            <div className="flex items-center gap-2 text-slate-400 text-sm">
              <span className="animate-pulse">●</span>
              <span className="animate-pulse delay-75">●</span>
              <span className="animate-pulse delay-150">●</span>
            </div>
          </div>
        </div>
      )}
      <div ref={bottomRef} />
    </div>
  )
}
