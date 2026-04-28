import { create } from 'zustand'
import { persist } from 'zustand/middleware'

export const useAppStore = create(
  persist(
    (set) => ({
      darkMode: true,
      toggleDarkMode: () => set((s) => ({ darkMode: !s.darkMode })),

      documents: [],
      setDocuments: (docs) => set({ documents: docs }),
      addDocuments: (newDocs) => set((s) => ({ documents: [...s.documents, ...newDocs] })),
      removeDocument: (doc_id) =>
        set((s) => ({ documents: s.documents.filter((d) => d.doc_id !== doc_id) })),

      chatHistory: [],
      addMessage: (msg) => set((s) => ({ chatHistory: [...s.chatHistory, msg] })),
      clearChat: () => set({ chatHistory: [] }),

      answerMode: 'short',
      setAnswerMode: (mode) => set({ answerMode: mode }),
    }),
    {
      name: 'amanlm-storage',
      partialize: (s) => ({ darkMode: s.darkMode, answerMode: s.answerMode }),
    }
  )
)
