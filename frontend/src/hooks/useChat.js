import { useState, useCallback } from 'react'
import client from '../api/client'
import { useAppStore } from '../stores/appStore'

export function useChat() {
  const { chatHistory, addMessage, clearChat, answerMode } = useAppStore()
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const sendMessage = useCallback(async (text) => {
    if (!text.trim() || loading) return

    const historyToSend = chatHistory
      .filter(m => m.role === 'user' || m.role === 'assistant')
      .map(m => ({ role: m.role, content: m.content }))

    const userMsg = { role: 'user', content: text, id: Date.now() }
    addMessage(userMsg)
    setLoading(true)
    setError(null)

    try {
      const res = await client.post('/chat', {
        message: text,
        mode: answerMode,
        top_k: 5,
        history: historyToSend,
      })
      const data = res.data
      addMessage({
        role: 'assistant',
        id: Date.now() + 1,
        content: data.answer,
        confidence: data.confidence,
        confidence_label: data.confidence_label,
        sources: data.sources,
        mode: data.mode,
        model_used: data.model_used,
      })
    } catch (err) {
      const msg = err.response?.data?.detail || 'Something went wrong. Please try again.'
      setError(msg)
    } finally {
      setLoading(false)
    }
  }, [loading, answerMode, addMessage, chatHistory])

  return { chatHistory, sendMessage, loading, error, clearChat }
}
