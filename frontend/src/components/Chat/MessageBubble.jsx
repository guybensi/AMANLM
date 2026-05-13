import { useState, useRef, useCallback } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import ConfidenceBar from './ConfidenceBar'
import SourceCard from '../Sources/SourceCard'

// Wrap [1], [2, 3] etc. in a sentinel so the custom code renderer can intercept them.
// Only matches purely-numeric citation patterns — safe against markdown link syntax.
function wrapCitations(text) {
  return text.replace(/\[(\d+(?:,\s*\d+)*)\]/g, (_, nums) => `\`[[${nums}]]\``)
}

export default function MessageBubble({ message }) {
  const [showSources, setShowSources] = useState(false)
  const [highlightedSources, setHighlightedSources] = useState(new Set())
  const cardRefs = useRef({})
  const isUser = message.role === 'user'

  const handleCitationClick = useCallback((rawNums) => {
    const indices = rawNums.split(',').map(n => parseInt(n.trim(), 10) - 1) // 0-based
    setShowSources(true)
    setHighlightedSources(new Set(indices))
    const firstValid = indices.find(i => cardRefs.current[i])
    if (firstValid !== undefined) {
      setTimeout(() => cardRefs.current[firstValid]?.scrollIntoView({ behavior: 'smooth', block: 'nearest' }), 60)
    }
  }, [])

  const markdownComponents = {
    code: ({ children, className }) => {
      const str = String(children)
      const match = str.match(/^\[\[(\d+(?:,\s*\d+)*)\]\]$/)
      if (match) {
        const rawNums = match[1]
        const labels = rawNums.split(',').map(n => n.trim())
        return (
          <button
            onClick={() => handleCitationClick(rawNums)}
            className="inline-flex items-center mx-0.5 px-1.5 py-0.5 bg-indigo-900/60 hover:bg-indigo-700/70 border border-indigo-600/50 text-indigo-300 text-xs font-mono rounded cursor-pointer transition-colors"
            title={`Jump to source${labels.length > 1 ? 's' : ''} ${labels.join(', ')}`}
          >
            {labels.map(l => `[${l}]`).join('')}
          </button>
        )
      }
      return <code className={className}>{children}</code>
    },
  }

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
          {/* Answer with clickable citation badges */}
          <div className="text-sm text-slate-100 leading-relaxed prose prose-invert prose-sm max-w-none">
            <ReactMarkdown remarkPlugins={[remarkGfm]} components={markdownComponents}>
              {wrapCitations(message.content)}
            </ReactMarkdown>
          </div>

          {/* Inference warning */}
          {message.contains_inference && (
            <div className="mt-3 flex items-start gap-2 text-xs text-amber-300 bg-amber-900/20 border border-amber-700/40 rounded-lg px-3 py-2">
              <span className="shrink-0 mt-px">⚠</span>
              <span>This response may contain inferences or information not directly found in your sources. Verify independently where needed.</span>
            </div>
          )}

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
                    <SourceCard
                      key={i}
                      source={src}
                      index={i}
                      highlighted={highlightedSources.has(i)}
                      cardRef={el => { cardRefs.current[i] = el }}
                    />
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
