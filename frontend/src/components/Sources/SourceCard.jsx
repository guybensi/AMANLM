import { useState, useEffect } from 'react'

export default function SourceCard({ source, index, highlighted = false, cardRef }) {
  const [expanded, setExpanded] = useState(false)

  useEffect(() => {
    if (highlighted) setExpanded(true)
  }, [highlighted])

  const pct = Math.round(source.relevance_score * 100)
  const scoreColor =
    pct >= 75 ? 'text-emerald-400' : pct >= 50 ? 'text-yellow-400' : 'text-orange-400'

  const borderClass = highlighted
    ? 'border-indigo-500 ring-1 ring-indigo-500/40'
    : 'border-slate-700'

  return (
    <div ref={cardRef} className={`border rounded-lg overflow-hidden bg-slate-800/50 transition-all ${borderClass}`}>
      <button
        onClick={() => setExpanded((v) => !v)}
        className="w-full flex items-center gap-2 px-3 py-2 text-left hover:bg-slate-700/40 transition-colors"
      >
        <span className={`text-xs font-mono ${highlighted ? 'text-indigo-400' : 'text-slate-500'}`}>[{index + 1}]</span>
        <div className="flex-1 min-w-0">
          <span className="text-xs text-slate-300 font-medium truncate block">{source.filename}</span>
          <span className="text-xs text-slate-500">Page {source.page_number}</span>
        </div>
        <span className={`text-xs font-bold ${scoreColor}`}>{pct}%</span>
        <span className="text-slate-500 text-xs ml-1">{expanded ? '▲' : '▼'}</span>
      </button>

      {expanded && (
        <div className="px-3 pb-3 border-t border-slate-700">
          <blockquote className="mt-2 text-xs text-slate-300 font-mono bg-slate-900/60 rounded-lg p-3 leading-relaxed whitespace-pre-wrap border-l-2 border-indigo-500">
            {source.quote}
          </blockquote>
        </div>
      )}
    </div>
  )
}
