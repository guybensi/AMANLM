import { useEffect, useState } from 'react'

const LABEL_COLORS = {
  High: 'text-emerald-400',
  Medium: 'text-yellow-400',
  Low: 'text-orange-400',
  'Very Low': 'text-red-400',
}

const BAR_COLORS = {
  High: 'bg-emerald-500',
  Medium: 'bg-yellow-500',
  Low: 'bg-orange-500',
  'Very Low': 'bg-red-500',
}

export default function ConfidenceBar({ confidence, label }) {
  const [width, setWidth] = useState(0)

  useEffect(() => {
    const t = setTimeout(() => setWidth(confidence), 80)
    return () => clearTimeout(t)
  }, [confidence])

  return (
    <div className="mt-2 mb-1">
      <div className="flex items-center justify-between mb-1">
        <span className="text-xs text-slate-500 font-medium">Confidence</span>
        <span className={`text-xs font-bold ${LABEL_COLORS[label] || 'text-slate-400'}`}>
          {label} · {confidence.toFixed(0)}%
        </span>
      </div>
      <div className="h-1.5 bg-slate-700 rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full transition-all duration-700 ease-out ${BAR_COLORS[label] || 'bg-slate-500'}`}
          style={{ width: `${width}%` }}
        />
      </div>
    </div>
  )
}
