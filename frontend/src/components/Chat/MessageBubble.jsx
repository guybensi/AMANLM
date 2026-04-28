import { useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import ConfidenceBar from './ConfidenceBar'
import SourceCard from '../Sources/SourceCard'

export default function MessageBubble({ message }) {
  const [showSources, setShowSources] = useState(false)
  const isUser = message.role === 'user'

  if (isUser) {
    return (
      <div className="flex justify-end mb-4">
        <div className="max-w-[75%] bg-indigo-600 text-white rounded-2xl rounded-tr-sm px-4 py-3 text-sm leading-relaxed shadow-lg">
          {message.content}
        </div>
      </div>
    )
  }

  return (
    <div className="flex justify-start mb-4">
      <div className="max-w-[85%] w-full">
        <div className="bg-slate-800 border border-slate-700 rounded-2xl rounded-tl-sm px-4 py-3 shadow-lg">
          {/* Answer */}
          <div className="text-sm text-slate-100 leading-relaxed prose prose-invert prose-sm max-w-none">
            <ReactMarkdown remarkPlugins={[remarkGfm]}>{message.content}</ReactMarkdown>
          </div>

          {/* Confidence bar */}
          {message.confidence !== undefined && (
            <ConfidenceBar confidence={message.confidence} label={message.confidence_label} />
          )}

          {/* Sources toggle */}
          {message.sources && message.sources.length > 0 && (
            <div className="mt-2">
              <button
                onClick={() => setShowSources((v) => !v)}
                className="flex items-center gap-1.5 text-xs text-indigo-400 hover:text-indigo-300 font-medium transition-colors"
              >
                <span>{showSources ? '▾' : '▸'}</span>
                {showSources ? 'Hide' : 'Show'} {message.sources.length} source{message.sources.length !== 1 ? 's' : ''}
              </button>

              {showSources && (
                <div className="mt-2 flex flex-col gap-2">
                  {message.sources.map((src, i) => (
                    <SourceCard key={i} source={src} index={i} />
                  ))}
                </div>
              )}
            </div>
          )}

          {/* Meta */}
          <div className="mt-2 flex items-center gap-2 text-xs text-slate-600">
            <span>{message.mode === 'long' ? '📝 Long' : '⚡ Short'}</span>
            {message.model_used && <span>· {message.model_used}</span>}
          </div>
        </div>
      </div>
    </div>
  )
}
