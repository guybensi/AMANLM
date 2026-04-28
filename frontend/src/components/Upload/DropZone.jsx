import { useCallback, useState } from 'react'
import { useDropzone } from 'react-dropzone'
import { useDocuments } from '../../hooks/useDocuments'

const ACCEPT = {
  'application/pdf': ['.pdf'],
  'text/plain': ['.txt', '.md'],
  'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx'],
  'image/jpeg': ['.jpg', '.jpeg'],
  'image/png': ['.png'],
}

export default function DropZone() {
  const { uploadFiles } = useDocuments()
  const [uploading, setUploading] = useState(false)
  const [progress, setProgress] = useState(0)
  const [toast, setToast] = useState(null)

  const showToast = (msg, type = 'error') => {
    setToast({ msg, type })
    setTimeout(() => setToast(null), 3500)
  }

  const onDrop = useCallback(async (accepted, rejected) => {
    if (rejected.length > 0) {
      showToast('Unsupported file type. Use PDF, TXT, DOCX, or images.')
      return
    }
    if (accepted.length === 0) return

    setUploading(true)
    setProgress(0)
    try {
      const result = await uploadFiles(accepted, setProgress)
      showToast(`Uploaded ${result.filenames.length} file(s) · ${result.total_chunks} chunks`, 'success')
    } catch (e) {
      showToast(e.response?.data?.detail || 'Upload failed.')
    } finally {
      setUploading(false)
      setProgress(0)
    }
  }, [uploadFiles])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: ACCEPT,
    disabled: uploading,
  })

  return (
    <div className="relative">
      <div
        {...getRootProps()}
        className={`border-2 border-dashed rounded-xl p-4 text-center cursor-pointer transition-all select-none
          ${isDragActive ? 'border-indigo-400 bg-indigo-900/20' : 'border-slate-600 hover:border-slate-400'}
          ${uploading ? 'opacity-60 cursor-not-allowed' : ''}`}
      >
        <input {...getInputProps()} />
        <div className="text-2xl mb-1">📂</div>
        {uploading ? (
          <div>
            <p className="text-xs text-slate-300">Uploading… {progress}%</p>
            <div className="mt-2 h-1 bg-slate-700 rounded-full overflow-hidden">
              <div
                className="h-full bg-indigo-500 transition-all duration-300"
                style={{ width: `${progress}%` }}
              />
            </div>
          </div>
        ) : (
          <p className="text-xs text-slate-400">
            {isDragActive ? 'Drop files here' : 'Drop files or click to upload'}
            <br />
            <span className="text-slate-500">PDF · TXT · DOCX · Images</span>
          </p>
        )}
      </div>

      {toast && (
        <div className={`mt-2 text-xs px-3 py-2 rounded-lg ${toast.type === 'success' ? 'bg-emerald-900/50 text-emerald-300' : 'bg-red-900/50 text-red-300'}`}>
          {toast.msg}
        </div>
      )}
    </div>
  )
}
