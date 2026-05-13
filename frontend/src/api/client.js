import axios from 'axios'

// In production (Vercel) set VITE_API_URL to your Railway backend URL,
// e.g. https://your-app.up.railway.app/api
// In local dev the Vite proxy forwards /api → http://localhost:8000/api
const client = axios.create({
  baseURL: import.meta.env.VITE_API_URL ?? '/api',
  timeout: 120000,
})

export default client
