import { useState } from 'react'
import { useDocuments } from '../../hooks/useDocuments'

const ICONS = {
  pdf: '📄',
  txt: '📝',
  docx: '📃',
  image: '🖼️',
}

export default function FileChip({ doc }) {
  const { deleteDocument } = useDocuments()
  const [deleting, setDeleting] = useState(false)

  const handleDelete = async () => {
    setDeleting(true)
    try {
      await deleteDocument(doc.doc_id)
    } finally {
      setDeleting(false)
    }
  }

  return (
    <div className="flex items-center gap-2 bg-slate-800 rounded-lg px-3 py-2 group">
      <span className="text-base">{ICONS[doc.file_type] || '📁'}</span>
      <div className="flex-1 min-w-0">
        <p className="text-xs text-slate-200 truncate font-medium">{doc.filename}</p>
        <p className="text-xs text-slate-500">{doc.chunk_count} chunks</p>
      </div>
      <button
        onClick={handleDelete}
        disabled={deleting}
        className="opacity-0 group-hover:opacity-100 text-slate-500 hover:text-red-400 transition-all text-sm ml-1 disabled:opacity-30"
        title="Remove document"
      >
        {deleting ? '…' : '✕'}
      </button>
    </div>
  )
}
