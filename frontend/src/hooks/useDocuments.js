import { useCallback } from 'react'
import client from '../api/client'
import { useAppStore } from '../stores/appStore'

export function useDocuments() {
  const { documents, setDocuments, addDocuments, removeDocument } = useAppStore()

  const fetchDocuments = useCallback(async () => {
    const res = await client.get('/documents')
    setDocuments(res.data)
  }, [setDocuments])

  const uploadFiles = useCallback(async (files, onProgress) => {
    const form = new FormData()
    for (const f of files) form.append('files', f)

    const res = await client.post('/documents/upload', form, {
      headers: { 'Content-Type': 'multipart/form-data' },
      onUploadProgress: (e) => {
        if (onProgress && e.total) onProgress(Math.round((e.loaded / e.total) * 100))
      },
    })

    // Fetch updated list after upload
    const updated = await client.get('/documents')
    setDocuments(updated.data)
    return res.data
  }, [setDocuments])

  const deleteDocument = useCallback(async (doc_id) => {
    await client.delete(`/documents/${doc_id}`)
    removeDocument(doc_id)
  }, [removeDocument])

  return { documents, fetchDocuments, uploadFiles, deleteDocument }
}
