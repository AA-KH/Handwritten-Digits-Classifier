/**
 * components.tsx — NeuralVis: the entire frontend in one file.
 *
 * Sections (in order):
 *   1. Types & Zustand store
 *   2. API utilities
 *   3. UI Components  (ParticleBackground … PipelineView)
 *   4. App layout     (was App.tsx)
 *   5. Bootstrap      (was main.tsx)
 */

import React, { useEffect, useRef, useCallback, useState, type ReactNode } from 'react'
import ReactDOM from 'react-dom/client'
import { motion, AnimatePresence } from 'framer-motion'
import { create } from 'zustand'
import './index.css'

// ─────────────────────────────────────────────────────────────────────────────
// 1. Types & Zustand store
// ─────────────────────────────────────────────────────────────────────────────

export interface PredictionResult {
  probabilities: number[]
  predicted_digit: number
  confidence: number
  latency_ms: number
  feature_maps_raw: number[][][]
  feature_maps_relu: number[][][]
  feature_maps_pooled: number[][][]
  hidden_activations: number[]
  preprocessed_image: number[][]
  model_loaded: boolean
}

export type ActivePanel = 'draw' | 'features' | 'architecture' | 'pipeline'

interface AppStore {
  prediction: PredictionResult | null
  isRunning: boolean
  error: string | null
  setPrediction: (p: PredictionResult | null) => void
  setRunning: (v: boolean) => void
  setError: (e: string | null) => void

  brushSize: number
  isErasing: boolean
  canvasHasContent: boolean
  setBrushSize: (s: number) => void
  setErasing: (v: boolean) => void
  setCanvasHasContent: (v: boolean) => void

  activePanel: ActivePanel
  selectedFeatureMap: number
  selectedLayer: string | null
  setActivePanel: (p: ActivePanel) => void
  setSelectedFeatureMap: (i: number) => void
  setSelectedLayer: (l: string | null) => void

  totalPredictions: number
  avgLatency: number
  incrementPredictions: (latency: number) => void
}

export const useAppStore = create<AppStore>((set, get) => ({
  prediction: null,
  isRunning: false,
  error: null,
  setPrediction: (p) => set({ prediction: p }),
  setRunning: (v) => set({ isRunning: v }),
  setError: (e) => set({ error: e }),

  brushSize: 18,
  isErasing: false,
  canvasHasContent: false,
  setBrushSize: (s) => set({ brushSize: s }),
  setErasing: (v) => set({ isErasing: v }),
  setCanvasHasContent: (v) => set({ canvasHasContent: v }),

  activePanel: 'draw',
  selectedFeatureMap: 0,
  selectedLayer: null,
  setActivePanel: (p) => set({ activePanel: p }),
  setSelectedFeatureMap: (i) => set({ selectedFeatureMap: i }),
  setSelectedLayer: (l) => set({ selectedLayer: l }),

  totalPredictions: 0,
  avgLatency: 0,
  incrementPredictions: (latency) => {
    const { totalPredictions, avgLatency } = get()
    const next = totalPredictions + 1
    set({ totalPredictions: next, avgLatency: (avgLatency * totalPredictions + latency) / next })
  },
}))

// ─────────────────────────────────────────────────────────────────────────────
// 2. API utilities
// ─────────────────────────────────────────────────────────────────────────────

const BASE = '/api'

export async function predictDigit(imageBase64: string) {
  const res = await fetch(`${BASE}/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ image: imageBase64 }),
  })
  if (!res.ok) throw new Error(`Prediction failed: ${res.statusText}`)
  return res.json()
}

export async function fetchArchitecture() {
  const res = await fetch(`${BASE}/architecture`)
  if (!res.ok) throw new Error('Failed to load architecture')
  return res.json()
}

export async function fetchHealth() {
  const res = await fetch(`${BASE}/health`)
  if (!res.ok) throw new Error('Backend unreachable')
  return res.json()
}

function normalise2D(map: number[][]): number[][] {
  let min = Infinity, max = -Infinity
  for (const row of map) for (const v of row) { if (v < min) min = v; if (v > max) max = v }
  const range = max - min || 1
  return map.map((row) => row.map((v) => (v - min) / range))
}

export function renderFeatureMap(
  canvas: HTMLCanvasElement,
  map: number[][],
  colormap: 'hot' | 'cool' | 'viridis' = 'viridis',
) {
  const norm = normalise2D(map)
  const h = norm.length, w = norm[0]?.length ?? 0
  canvas.width = w; canvas.height = h
  const ctx = canvas.getContext('2d')!
  const img = ctx.createImageData(w, h)
  for (let r = 0; r < h; r++) {
    for (let c = 0; c < w; c++) {
      const [R, G, B] = toRGB(norm[r][c], colormap)
      const idx = (r * w + c) * 4
      img.data[idx] = R; img.data[idx + 1] = G; img.data[idx + 2] = B; img.data[idx + 3] = 255
    }
  }
  ctx.putImageData(img, 0, 0)
}

function toRGB(t: number, cm: string): [number, number, number] {
  if (cm === 'hot') {
    return [Math.min(1, t * 3) * 255, Math.min(1, Math.max(0, t * 3 - 1)) * 255, Math.min(1, Math.max(0, t * 3 - 2)) * 255]
  }
  if (cm === 'cool') return [t * 255, (1 - t) * 255, 255]
  return [
    Math.max(0, Math.min(255, (0.267 + t * 0.713) * 255)),
    Math.max(0, Math.min(255, (0.005 + t * 0.783 - t * t * 0.218) * 255)),
    Math.max(0, Math.min(255, (0.329 + t * 0.218 - t * 0.547) * 255)),
  ]
}

export function throttle<T extends (...args: never[]) => void>(fn: T, ms: number): T {
  let last = 0
  return ((...args) => { const now = Date.now(); if (now - last >= ms) { last = now; fn(...args) } }) as T
}

// ─────────────────────────────────────────────────────────────────────────────
// ParticleBackground
// ─────────────────────────────────────────────────────────────────────────────

interface Particle {
  x: number; y: number
  vx: number; vy: number
  radius: number
  alpha: number
  color: string
}

const PARTICLE_COLORS = ['#6c63ff', '#00d4ff', '#00ff9d', '#ff6b35']
const PARTICLE_COUNT = 80
const CONNECTION_DIST = 140

export function ParticleBackground() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const mouseRef = useRef({ x: -9999, y: -9999 })
  const particlesRef = useRef<Particle[]>([])
  const rafRef = useRef<number>(0)

  useEffect(() => {
    const canvas = canvasRef.current!
    const ctx = canvas.getContext('2d')!

    const resize = () => {
      canvas.width = window.innerWidth
      canvas.height = window.innerHeight
    }
    resize()
    window.addEventListener('resize', resize)

    const onMouse = (e: MouseEvent) => {
      mouseRef.current = { x: e.clientX, y: e.clientY }
    }
    window.addEventListener('mousemove', onMouse)

    particlesRef.current = Array.from({ length: PARTICLE_COUNT }, () => ({
      x: Math.random() * canvas.width,
      y: Math.random() * canvas.height,
      vx: (Math.random() - 0.5) * 0.4,
      vy: (Math.random() - 0.5) * 0.4,
      radius: Math.random() * 1.5 + 0.5,
      alpha: Math.random() * 0.6 + 0.2,
      color: PARTICLE_COLORS[Math.floor(Math.random() * PARTICLE_COLORS.length)],
    }))

    const draw = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height)
      const ps = particlesRef.current
      const mouse = mouseRef.current

      for (const p of ps) {
        p.x += p.vx
        p.y += p.vy
        if (p.x < 0 || p.x > canvas.width) p.vx *= -1
        if (p.y < 0 || p.y > canvas.height) p.vy *= -1

        const dx = p.x - mouse.x
        const dy = p.y - mouse.y
        const dist = Math.sqrt(dx * dx + dy * dy)
        if (dist < 80) {
          p.x += (dx / dist) * 0.8
          p.y += (dy / dist) * 0.8
        }

        ctx.beginPath()
        ctx.arc(p.x, p.y, p.radius, 0, Math.PI * 2)
        ctx.fillStyle = p.color + Math.round(p.alpha * 255).toString(16).padStart(2, '0')
        ctx.fill()
      }

      for (let i = 0; i < ps.length; i++) {
        for (let j = i + 1; j < ps.length; j++) {
          const dx = ps[i].x - ps[j].x
          const dy = ps[i].y - ps[j].y
          const d = Math.sqrt(dx * dx + dy * dy)
          if (d < CONNECTION_DIST) {
            const alpha = (1 - d / CONNECTION_DIST) * 0.15
            ctx.beginPath()
            ctx.moveTo(ps[i].x, ps[i].y)
            ctx.lineTo(ps[j].x, ps[j].y)
            ctx.strokeStyle = `rgba(108, 99, 255, ${alpha})`
            ctx.lineWidth = 0.5
            ctx.stroke()
          }
        }
      }

      rafRef.current = requestAnimationFrame(draw)
    }

    draw()

    return () => {
      cancelAnimationFrame(rafRef.current)
      window.removeEventListener('resize', resize)
      window.removeEventListener('mousemove', onMouse)
    }
  }, [])

  return (
    <canvas
      ref={canvasRef}
      className="fixed inset-0 z-0 pointer-events-none"
      style={{ opacity: 0.5 }}
    />
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// NavBar
// ─────────────────────────────────────────────────────────────────────────────

export function NavBar() {
  const [backendOk, setBackendOk] = useState<boolean | null>(null)
  const [modelLoaded, setModelLoaded] = useState(false)

  useEffect(() => {
    fetchHealth()
      .then((data) => {
        setBackendOk(true)
        setModelLoaded(data.model_loaded)
      })
      .catch(() => setBackendOk(false))
  }, [])

  return (
    <motion.nav
      initial={{ y: -60, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ duration: 0.6, ease: 'easeOut' }}
      className="fixed top-0 left-0 right-0 z-50 glass border-b border-border"
    >
      <div className="max-w-[1600px] mx-auto px-4 md:px-6 h-16 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="relative w-8 h-8">
            <div className="absolute inset-0 rounded-lg bg-accent/20 border border-accent/40 animate-pulse-slow" />
            <div className="absolute inset-0 flex items-center justify-center font-mono text-accent font-bold text-sm">N</div>
          </div>
          <span className="font-display font-bold text-lg">
            Neural<span className="text-accent">Vis</span>
          </span>
          <span className="hidden md:block font-mono text-xs text-muted border border-border rounded px-2 py-0.5">
            CNN Explorer
          </span>
        </div>

        <div className="flex items-center gap-4">
          <StatusDot label="API" status={backendOk === null ? 'pending' : backendOk ? 'ok' : 'err'} />
          <StatusDot label="Model" status={backendOk === null ? 'pending' : modelLoaded ? 'ok' : 'warn'} />
          <div className="hidden md:flex items-center gap-2 font-mono text-xs text-muted">
            <span>NumPy CNN</span>
            <span className="text-border">·</span>
            <span>16 kernels · 2704→64→10</span>
          </div>
        </div>
      </div>
    </motion.nav>
  )
}

function StatusDot({ label, status }: { label: string; status: 'ok' | 'err' | 'warn' | 'pending' }) {
  const color = { ok: 'bg-emerald-400', err: 'bg-red-500', warn: 'bg-yellow-500', pending: 'bg-muted' }[status]
  const glow = {
    ok: 'shadow-[0_0_8px_rgba(0,255,157,0.6)]',
    err: 'shadow-[0_0_8px_rgba(255,80,80,0.6)]',
    warn: 'shadow-[0_0_8px_rgba(255,200,0,0.6)]',
    pending: '',
  }[status]

  return (
    <div className="flex items-center gap-1.5">
      <div className={`w-2 h-2 rounded-full ${color} ${glow} ${status === 'ok' ? 'animate-pulse' : ''}`} />
      <span className="font-mono text-xs text-ghost">{label}</span>
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// StatusBar
// ─────────────────────────────────────────────────────────────────────────────

export function StatusBar() {
  const { prediction, isRunning, totalPredictions } = useAppStore()

  return (
    <footer className="fixed bottom-0 left-0 right-0 z-50 border-t border-border glass">
      <div className="max-w-[1600px] mx-auto px-4 md:px-6 h-8 flex items-center justify-between">
        <div className="flex items-center gap-4 font-mono text-[10px] text-muted">
          <span>NeuralVis v1.0</span>
          <span className="text-border">·</span>
          <span>NumPy CNN from scratch</span>
          <span className="text-border">·</span>
          <span>No ML frameworks</span>
        </div>
        <div className="flex items-center gap-4 font-mono text-[10px] text-muted">
          {isRunning && <span className="text-neon animate-pulse">● Inferring</span>}
          {prediction && (
            <>
              <span>Digit: <span className="text-text">{prediction.predicted_digit}</span></span>
              <span>·</span>
              <span>{prediction.latency_ms.toFixed(1)}ms</span>
            </>
          )}
          <span>{totalPredictions} total runs</span>
        </div>
      </div>
    </footer>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// DrawingCanvas
// ─────────────────────────────────────────────────────────────────────────────

const CANVAS_SIZE = 280

export function DrawingCanvas() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const isDrawingRef = useRef(false)
  const lastPosRef = useRef({ x: 0, y: 0 })
  const historyRef = useRef<ImageData[]>([])
  const historyIndexRef = useRef(-1)

  const {
    brushSize, isErasing, setBrushSize, setErasing,
    setPrediction, setRunning, setError, setCanvasHasContent,
    incrementPredictions
  } = useAppStore()

  const saveSnapshot = useCallback(() => {
    const canvas = canvasRef.current!
    const ctx = canvas.getContext('2d')!
    const snapshot = ctx.getImageData(0, 0, canvas.width, canvas.height)
    historyRef.current = historyRef.current.slice(0, historyIndexRef.current + 1)
    historyRef.current.push(snapshot)
    historyIndexRef.current++
    if (historyRef.current.length > 40) {
      historyRef.current.shift()
      historyIndexRef.current--
    }
  }, [])

  const runPredict = useCallback(
    throttle(async () => {
      const canvas = canvasRef.current!
      const dataUrl = canvas.toDataURL('image/png')
      setRunning(true)
      try {
        const result = await predictDigit(dataUrl)
        setPrediction(result)
        incrementPredictions(result.latency_ms)
        setCanvasHasContent(true)
        setError(null)
      } catch (e) {
        setError(String(e))
      } finally {
        setRunning(false)
      }
    }, 120),
    []
  )

  const getPos = (e: MouseEvent | TouchEvent, canvas: HTMLCanvasElement) => {
    const rect = canvas.getBoundingClientRect()
    const scaleX = canvas.width / rect.width
    const scaleY = canvas.height / rect.height
    if ('touches' in e) {
      return {
        x: (e.touches[0].clientX - rect.left) * scaleX,
        y: (e.touches[0].clientY - rect.top) * scaleY,
      }
    }
    return {
      x: (e.clientX - rect.left) * scaleX,
      y: (e.clientY - rect.top) * scaleY,
    }
  }

  const draw = useCallback((pos: { x: number; y: number }, canvas: HTMLCanvasElement) => {
    const ctx = canvas.getContext('2d')!
    ctx.lineWidth = brushSize
    ctx.lineCap = 'round'
    ctx.lineJoin = 'round'
    ctx.strokeStyle = isErasing ? '#030305' : '#ffffff'
    ctx.globalCompositeOperation = isErasing ? 'destination-out' : 'source-over'

    if (isErasing) {
      ctx.clearRect(pos.x - brushSize / 2, pos.y - brushSize / 2, brushSize, brushSize)
      ctx.globalCompositeOperation = 'source-over'
    } else {
      ctx.beginPath()
      ctx.moveTo(lastPosRef.current.x, lastPosRef.current.y)
      ctx.lineTo(pos.x, pos.y)
      ctx.stroke()
    }

    lastPosRef.current = pos
  }, [brushSize, isErasing])

  useEffect(() => {
    const canvas = canvasRef.current!
    const ctx = canvas.getContext('2d')!
    canvas.width = CANVAS_SIZE
    canvas.height = CANVAS_SIZE
    ctx.fillStyle = '#030305'
    ctx.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE)
    saveSnapshot()
  }, [])

  useEffect(() => {
    const canvas = canvasRef.current!

    const onStart = (e: MouseEvent | TouchEvent) => {
      e.preventDefault()
      isDrawingRef.current = true
      lastPosRef.current = getPos(e, canvas)
      saveSnapshot()
    }

    const onMove = (e: MouseEvent | TouchEvent) => {
      e.preventDefault()
      if (!isDrawingRef.current) return
      draw(getPos(e, canvas), canvas)
      runPredict()
    }

    const onEnd = () => { isDrawingRef.current = false }

    canvas.addEventListener('mousedown', onStart)
    canvas.addEventListener('mousemove', onMove)
    canvas.addEventListener('mouseup', onEnd)
    canvas.addEventListener('mouseleave', onEnd)
    canvas.addEventListener('touchstart', onStart, { passive: false })
    canvas.addEventListener('touchmove', onMove, { passive: false })
    canvas.addEventListener('touchend', onEnd)

    return () => {
      canvas.removeEventListener('mousedown', onStart)
      canvas.removeEventListener('mousemove', onMove)
      canvas.removeEventListener('mouseup', onEnd)
      canvas.removeEventListener('mouseleave', onEnd)
      canvas.removeEventListener('touchstart', onStart)
      canvas.removeEventListener('touchmove', onMove)
      canvas.removeEventListener('touchend', onEnd)
    }
  }, [draw, runPredict, saveSnapshot])

  const undo = useCallback(() => {
    if (historyIndexRef.current <= 0) return
    historyIndexRef.current--
    const canvas = canvasRef.current!
    const ctx = canvas.getContext('2d')!
    ctx.putImageData(historyRef.current[historyIndexRef.current], 0, 0)
    runPredict()
  }, [runPredict])

  const redo = useCallback(() => {
    if (historyIndexRef.current >= historyRef.current.length - 1) return
    historyIndexRef.current++
    const canvas = canvasRef.current!
    const ctx = canvas.getContext('2d')!
    ctx.putImageData(historyRef.current[historyIndexRef.current], 0, 0)
    runPredict()
  }, [runPredict])

  const clear = useCallback(() => {
    const canvas = canvasRef.current!
    const ctx = canvas.getContext('2d')!
    ctx.fillStyle = '#030305'
    ctx.fillRect(0, 0, canvas.width, canvas.height)
    saveSnapshot()
    setPrediction(null)
    setCanvasHasContent(false)
  }, [saveSnapshot, setPrediction, setCanvasHasContent])

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key === 'z') { e.preventDefault(); undo() }
      if ((e.ctrlKey || e.metaKey) && e.key === 'y') { e.preventDefault(); redo() }
      if (e.key === 'e') setErasing(true)
      if (e.key === 'b') setErasing(false)
      if (e.key === 'Delete' || e.key === 'Backspace') clear()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [undo, redo, clear, setErasing])

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.96 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.5 }}
      className="glass-bright rounded-2xl p-4 corner-bracket"
    >
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <div className="w-1.5 h-4 bg-accent rounded-full" />
          <span className="font-display font-semibold text-sm">Drawing Canvas</span>
        </div>
        <span className="font-mono text-xs text-muted">{CANVAS_SIZE}×{CANVAS_SIZE}px → 28×28</span>
      </div>

      <div
        className={`relative rounded-xl overflow-hidden border-2 transition-colors duration-300 ${
          isErasing ? 'border-ember/40' : 'border-accent/30'
        }`}
        style={{ boxShadow: isErasing ? '0 0 20px rgba(255,107,53,0.2)' : '0 0 20px rgba(108,99,255,0.15)' }}
      >
        <canvas
          ref={canvasRef}
          className={`block w-full aspect-square ${isErasing ? 'canvas-erase' : 'canvas-draw'}`}
          style={{ imageRendering: 'pixelated' }}
        />
        <div
          className="absolute inset-0 pointer-events-none opacity-[0.03]"
          style={{
            backgroundImage: 'linear-gradient(rgba(108,99,255,1) 1px, transparent 1px), linear-gradient(90deg, rgba(108,99,255,1) 1px, transparent 1px)',
            backgroundSize: `${CANVAS_SIZE / 28}px ${CANVAS_SIZE / 28}px`,
          }}
        />
      </div>

      <div className="mt-3 flex flex-col gap-2">
        <div className="flex gap-2">
          <ToolButton active={!isErasing} onClick={() => setErasing(false)} label="Draw" shortcut="B" color="accent" />
          <ToolButton active={isErasing} onClick={() => setErasing(true)} label="Erase" shortcut="E" color="ember" />
          <div className="flex-1" />
          <IconButton onClick={undo} title="Undo (Ctrl+Z)" icon="↩" />
          <IconButton onClick={redo} title="Redo (Ctrl+Y)" icon="↪" />
          <IconButton onClick={clear} title="Clear (Delete)" icon="✕" danger />
        </div>

        <div className="flex items-center gap-3">
          <span className="font-mono text-xs text-muted w-12">Size {brushSize}</span>
          <input
            type="range"
            min="4"
            max="40"
            value={brushSize}
            onChange={(e) => setBrushSize(Number(e.target.value))}
            className="flex-1 h-1 rounded-full appearance-none cursor-pointer"
            style={{
              background: `linear-gradient(to right, #6c63ff ${((brushSize - 4) / 36) * 100}%, #1e1e32 0%)`,
            }}
          />
        </div>
      </div>

      <div className="mt-2 flex gap-3 font-mono text-[10px] text-muted">
        <span>Draw any digit 0–9</span>
        <span>·</span>
        <span>Predictions update live</span>
      </div>
    </motion.div>
  )
}

function ToolButton({ active, onClick, label, shortcut, color }: {
  active: boolean; onClick: () => void; label: string; shortcut: string; color: string
}) {
  return (
    <button
      onClick={onClick}
      className={`
        neural-button relative flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-mono
        border transition-all duration-200
        ${active
          ? color === 'accent'
            ? 'bg-accent/20 border-accent/50 text-accent-glow'
            : 'bg-ember/20 border-ember/50 text-ember'
          : 'bg-surface border-border text-muted hover:text-ghost'
        }
      `}
    >
      {label}
      <span className="opacity-40">[{shortcut}]</span>
    </button>
  )
}

function IconButton({ onClick, title, icon, danger }: {
  onClick: () => void; title: string; icon: string; danger?: boolean
}) {
  return (
    <button
      onClick={onClick}
      title={title}
      className={`
        w-8 h-8 flex items-center justify-center rounded-lg text-sm
        border transition-all duration-200
        ${danger
          ? 'border-red-900/50 text-red-500/60 hover:bg-red-900/20 hover:text-red-400 hover:border-red-700/50'
          : 'border-border text-muted hover:bg-white/5 hover:text-ghost'
        }
      `}
    >
      {icon}
    </button>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// PredictionDisplay
// ─────────────────────────────────────────────────────────────────────────────

export function PredictionDisplay() {
  const { prediction, isRunning, canvasHasContent } = useAppStore()

  return (
    <div className="glass-bright rounded-2xl p-6 flex flex-col items-center justify-center min-h-[320px] corner-bracket">
      <AnimatePresence mode="wait">
        {!canvasHasContent ? (
          <EmptyState key="empty" />
        ) : isRunning && !prediction ? (
          <LoadingState key="loading" />
        ) : prediction ? (
          <PredictionResult key={`pred-${prediction.predicted_digit}`} />
        ) : null}
      </AnimatePresence>
    </div>
  )
}

function EmptyState() {
  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="flex flex-col items-center gap-4 text-center"
    >
      <div className="relative w-24 h-24">
        {[0, 1, 2].map((i) => (
          <motion.div
            key={i}
            className="absolute inset-0 rounded-full border border-accent/20"
            animate={{ scale: [1, 1.5, 1], opacity: [0.4, 0, 0.4] }}
            transition={{ duration: 2.5, delay: i * 0.8, repeat: Infinity }}
          />
        ))}
        <div className="absolute inset-0 flex items-center justify-center">
          <span className="text-4xl text-muted">✏️</span>
        </div>
      </div>
      <div>
        <p className="font-display font-semibold text-ghost">Draw a digit</p>
        <p className="font-mono text-xs text-muted mt-1">Inference runs automatically</p>
      </div>
    </motion.div>
  )
}

function LoadingState() {
  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="flex flex-col items-center gap-4"
    >
      <div className="relative w-20 h-20">
        <motion.div
          className="absolute inset-0 rounded-full border-2 border-accent/30"
          animate={{ rotate: 360 }}
          transition={{ duration: 1.5, repeat: Infinity, ease: 'linear' }}
        />
        <motion.div
          className="absolute inset-2 rounded-full border-2 border-neon/30 border-t-neon"
          animate={{ rotate: -360 }}
          transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}
        />
      </div>
      <span className="font-mono text-xs text-muted animate-pulse">Running inference…</span>
    </motion.div>
  )
}

function PredictionResult() {
  const { prediction, isRunning } = useAppStore()
  if (!prediction) return null

  const pct = Math.round(prediction.confidence * 100)
  const confidenceColor =
    pct >= 80 ? '#00ff9d' :
    pct >= 50 ? '#6c63ff' :
    pct >= 30 ? '#00d4ff' : '#ff6b35'

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.8 }}
      animate={{ opacity: 1, scale: 1 }}
      exit={{ opacity: 0, scale: 0.8 }}
      transition={{ type: 'spring', stiffness: 200, damping: 20 }}
      className="flex flex-col items-center gap-6 w-full"
    >
      <div className="relative">
        <motion.div
          key={prediction.predicted_digit}
          initial={{ scale: 0.5, opacity: 0, rotateX: -90 }}
          animate={{ scale: 1, opacity: 1, rotateX: 0 }}
          transition={{ type: 'spring', stiffness: 300, damping: 25 }}
          className="font-mono font-bold text-center leading-none select-none"
          style={{
            fontSize: 'clamp(5rem, 12vw, 8rem)',
            color: confidenceColor,
            textShadow: `0 0 40px ${confidenceColor}60, 0 0 80px ${confidenceColor}30`,
          }}
        >
          {prediction.predicted_digit}
        </motion.div>

        <svg
          className="absolute inset-0 w-full h-full pointer-events-none"
          viewBox="0 0 100 100"
          style={{ transform: 'scale(1.3)' }}
        >
          <circle cx="50" cy="50" r="40" fill="none" stroke={`${confidenceColor}20`} strokeWidth="1" />
          <motion.circle
            cx="50" cy="50" r="40"
            fill="none"
            stroke={confidenceColor}
            strokeWidth="1.5"
            strokeLinecap="round"
            strokeDasharray={`${251.2 * (pct / 100)} 251.2`}
            strokeDashoffset="62.8"
            initial={{ pathLength: 0 }}
            animate={{ pathLength: pct / 100 }}
            transition={{ duration: 0.8, ease: 'easeOut' }}
            style={{ filter: `drop-shadow(0 0 4px ${confidenceColor})` }}
          />
        </svg>
      </div>

      <div className="flex flex-col items-center gap-1">
        <div className="flex items-center gap-2">
          <span className="font-mono text-3xl font-bold" style={{ color: confidenceColor }}>
            {pct}%
          </span>
          <span className="font-mono text-xs text-muted">confidence</span>
        </div>
        <div className="flex items-center gap-3 font-mono text-xs text-muted">
          <span className={isRunning ? 'text-neon animate-pulse' : 'text-ghost'}>
            {isRunning ? '⟳ updating' : '✓ classified'}
          </span>
          <span>·</span>
          <span>{prediction.latency_ms.toFixed(1)} ms</span>
        </div>
      </div>

      <SecondBest probs={prediction.probabilities} best={prediction.predicted_digit} />
    </motion.div>
  )
}

function SecondBest({ probs, best }: { probs: number[]; best: number }) {
  const sorted = probs.map((p, i) => ({ digit: i, p })).sort((a, b) => b.p - a.p)
  const second = sorted[1]

  return (
    <div className="w-full flex items-center justify-between glass rounded-xl px-4 py-2">
      <div className="flex items-center gap-2">
        <span className="font-mono text-xs text-muted">Runner-up</span>
        <span className="font-mono font-bold text-ghost text-lg">{second.digit}</span>
      </div>
      <div className="flex items-center gap-2">
        <div className="w-20 h-1 bg-surface rounded-full overflow-hidden">
          <motion.div
            className="h-full bg-muted rounded-full"
            animate={{ width: `${second.p * 100}%` }}
            transition={{ duration: 0.5 }}
          />
        </div>
        <span className="font-mono text-xs text-muted">{Math.round(second.p * 100)}%</span>
      </div>
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// ConfidenceChart
// ─────────────────────────────────────────────────────────────────────────────

const DIGITS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

export function ConfidenceChart() {
  const { prediction } = useAppStore()
  const probs = prediction?.probabilities ?? Array(10).fill(0)
  const best = prediction?.predicted_digit ?? -1

  return (
    <div className="glass-bright rounded-2xl p-4 flex flex-col gap-3">
      <div className="flex items-center gap-2 mb-1">
        <div className="w-1.5 h-4 bg-neon rounded-full" />
        <span className="font-display font-semibold text-sm">Class Probabilities</span>
      </div>

      <div className="flex justify-center py-2">
        <RadialChart probs={probs} best={best} />
      </div>

      <div className="flex flex-col gap-1.5">
        {DIGITS.map((d) => {
          const p = probs[d] ?? 0
          const isBest = d === best
          const pct = Math.round(p * 100)

          return (
            <motion.div key={d} className="flex items-center gap-2" layout>
              <span className={`font-mono text-xs w-4 text-right transition-colors duration-300 ${isBest ? 'text-emerald-400 font-bold' : 'text-muted'}`}>
                {d}
              </span>
              <div className="flex-1 h-4 bg-surface rounded-full overflow-hidden relative">
                <motion.div
                  className="h-full rounded-full"
                  style={{
                    background: isBest ? 'linear-gradient(90deg, #6c63ff, #00ff9d)' : 'linear-gradient(90deg, #2a2a44, #3a3a5a)',
                    boxShadow: isBest ? '0 0 12px rgba(0, 255, 157, 0.4)' : 'none',
                  }}
                  animate={{ width: `${Math.max(p * 100, 0.5)}%` }}
                  transition={{ duration: 0.4, ease: 'easeOut' }}
                />
                {isBest && <div className="absolute inset-0 shimmer rounded-full" />}
              </div>
              <span className={`font-mono text-xs w-9 text-right transition-colors duration-300 ${isBest ? 'text-emerald-300' : 'text-muted'}`}>
                {pct}%
              </span>
            </motion.div>
          )
        })}
      </div>
    </div>
  )
}

function RadialChart({ probs, best }: { probs: number[]; best: number }) {
  const size = 160
  const cx = size / 2
  const cy = size / 2
  const rings = DIGITS.map((d, i) => {
    const r = 14 + i * 6.5
    const circumference = 2 * Math.PI * r
    const pct = probs[d] ?? 0
    const dash = circumference * pct
    const gap = circumference - dash
    const isBest = d === best
    const colors = ['#6c63ff','#7c6bff','#8c73ff','#9c7bff','#00d4ff','#00bfe0','#00aac0','#00ff9d','#ff6b35','#ff9d35']
    return { d, r, circumference, dash, gap, isBest, color: colors[i] }
  })

  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
      {rings.map(({ d, r }) => (
        <circle key={`bg-${d}`} cx={cx} cy={cy} r={r} fill="none" stroke="rgba(30,30,50,0.8)" strokeWidth="4" />
      ))}
      {rings.map(({ d, r, dash, gap, isBest, color }) => (
        <motion.circle
          key={`arc-${d}`}
          cx={cx} cy={cy} r={r}
          fill="none"
          stroke={color}
          strokeWidth={isBest ? 5 : 3}
          strokeLinecap="round"
          strokeDasharray={`${dash} ${gap}`}
          strokeDashoffset={`${(2 * Math.PI * r) * 0.25}`}
          opacity={isBest ? 1 : 0.5}
          style={{ filter: isBest ? `drop-shadow(0 0 6px ${color})` : 'none', transition: 'all 0.4s ease' }}
          animate={{ strokeDasharray: `${dash} ${gap}` }}
          transition={{ duration: 0.4, ease: 'easeOut' }}
        />
      ))}
      {best >= 0 && (
        <motion.text
          x={cx} y={cy}
          textAnchor="middle"
          dominantBaseline="central"
          fill="#e8e8f0"
          fontSize="28"
          fontFamily="'JetBrains Mono', monospace"
          fontWeight="bold"
          key={best}
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.3 }}
        >
          {best}
        </motion.text>
      )}
    </svg>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// FeatureMaps
// ─────────────────────────────────────────────────────────────────────────────

type Colormap = 'viridis' | 'hot' | 'cool'
type FMStage = 'raw' | 'relu' | 'pooled'

const FM_STAGE_LABELS: Record<FMStage, string> = {
  raw: 'Conv Output',
  relu: 'After ReLU',
  pooled: 'After Pooling',
}

export function FeatureMaps() {
  const { prediction, selectedFeatureMap, setSelectedFeatureMap } = useAppStore()
  const [stage, setStage] = useState<FMStage>('relu')
  const [colormap, setColormap] = useState<Colormap>('viridis')

  const maps = prediction
    ? stage === 'raw' ? prediction.feature_maps_raw
    : stage === 'relu' ? prediction.feature_maps_relu
    : prediction.feature_maps_pooled
    : null

  const selectedMap = maps?.[selectedFeatureMap] ?? null

  return (
    <div className="flex flex-col gap-4">
      <div className="glass-bright rounded-2xl p-4 flex flex-wrap gap-4 items-center">
        <div className="flex items-center gap-2">
          <div className="w-1.5 h-4 bg-emerald-400 rounded-full" />
          <span className="font-display font-semibold text-sm">Feature Maps</span>
          <span className="font-mono text-xs text-muted">16 kernels · 3×3</span>
        </div>

        <div className="flex gap-1 ml-auto">
          {(Object.keys(FM_STAGE_LABELS) as FMStage[]).map((s) => (
            <button
              key={s}
              onClick={() => setStage(s)}
              className={`px-3 py-1.5 rounded-lg font-mono text-xs transition-all duration-200 ${
                stage === s
                  ? 'bg-emerald-400/20 text-emerald-300 border border-emerald-400/30'
                  : 'text-muted hover:text-ghost border border-transparent hover:border-border'
              }`}
            >
              {FM_STAGE_LABELS[s]}
            </button>
          ))}
        </div>

        <div className="flex gap-1">
          {(['viridis', 'hot', 'cool'] as Colormap[]).map((c) => (
            <button
              key={c}
              onClick={() => setColormap(c)}
              className={`px-2.5 py-1.5 rounded-lg font-mono text-xs capitalize transition-all duration-200 ${
                colormap === c
                  ? 'bg-accent/20 text-accent-glow border border-accent/30'
                  : 'text-muted hover:text-ghost border border-transparent hover:border-border'
              }`}
            >
              {c}
            </button>
          ))}
        </div>
      </div>

      {!prediction ? (
        <EmptyFeatureState />
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-[1fr_280px] gap-4">
          <div className="glass-bright rounded-2xl p-4">
            <div className="grid grid-cols-4 gap-2">
              {Array.from({ length: 16 }, (_, i) => (
                <FeatureMapTile
                  key={i}
                  index={i}
                  map={maps?.[i] ?? null}
                  colormap={colormap}
                  selected={selectedFeatureMap === i}
                  onSelect={setSelectedFeatureMap}
                />
              ))}
            </div>
          </div>

          <div className="glass-bright rounded-2xl p-4 flex flex-col gap-4">
            <div className="flex items-center justify-between">
              <span className="font-display font-semibold text-sm">Filter #{selectedFeatureMap}</span>
              <span className="font-mono text-xs text-muted">{stage === 'pooled' ? '13×13' : '26×26'}</span>
            </div>
            {selectedMap && (
              <>
                <DetailCanvas map={selectedMap} colormap={colormap} />
                <ActivationStats map={selectedMap} />
              </>
            )}
          </div>
        </div>
      )}

      {prediction && <HiddenActivations activations={prediction.hidden_activations} />}
    </div>
  )
}

function FeatureMapTile({ index, map, colormap, selected, onSelect }: {
  index: number; map: number[][] | null; colormap: Colormap; selected: boolean; onSelect: (i: number) => void
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || !map) return
    renderFeatureMap(canvas, map, colormap)
  }, [map, colormap])

  return (
    <motion.button
      onClick={() => onSelect(index)}
      whileHover={{ scale: 1.05 }}
      whileTap={{ scale: 0.95 }}
      className={`relative aspect-square rounded-lg overflow-hidden border-2 transition-all duration-200 ${
        selected ? 'border-emerald-400 shadow-glow-emerald' : 'border-border hover:border-accent/50'
      }`}
    >
      <canvas ref={canvasRef} className="w-full h-full" style={{ imageRendering: 'pixelated' }} />
      <div className="absolute bottom-0 right-0 bg-black/60 px-1 py-0.5 font-mono text-[8px] text-muted">{index}</div>
      {selected && <div className="absolute inset-0 border-2 border-emerald-400/50 rounded-lg" />}
    </motion.button>
  )
}

function DetailCanvas({ map, colormap }: { map: number[][]; colormap: Colormap }) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    renderFeatureMap(canvas, map, colormap)
  }, [map, colormap])

  return (
    <div className="relative rounded-xl overflow-hidden border border-border aspect-square">
      <canvas ref={canvasRef} className="w-full h-full" style={{ imageRendering: 'pixelated' }} />
    </div>
  )
}

function ActivationStats({ map }: { map: number[][] }) {
  const flat = map.flat()
  const min = Math.min(...flat)
  const max = Math.max(...flat)
  const mean = flat.reduce((a, b) => a + b, 0) / flat.length
  const active = flat.filter((v) => v > 0).length

  return (
    <div className="grid grid-cols-2 gap-2">
      {[
        { label: 'Max', value: max.toFixed(3), color: '#00ff9d' },
        { label: 'Min', value: min.toFixed(3), color: '#ff6b35' },
        { label: 'Mean', value: mean.toFixed(3), color: '#6c63ff' },
        { label: 'Active', value: `${Math.round((active / flat.length) * 100)}%`, color: '#00d4ff' },
      ].map((stat) => (
        <div key={stat.label} className="bg-surface rounded-lg p-2 text-center">
          <div className="font-mono text-xs" style={{ color: stat.color }}>{stat.value}</div>
          <div className="font-mono text-[10px] text-muted mt-0.5">{stat.label}</div>
        </div>
      ))}
    </div>
  )
}

function HiddenActivations({ activations }: { activations: number[] }) {
  const max = Math.max(...activations)
  const topK = activations.map((v, i) => ({ v, i })).sort((a, b) => b.v - a.v).slice(0, 10)

  return (
    <div className="glass-bright rounded-2xl p-4">
      <div className="flex items-center gap-2 mb-3">
        <div className="w-1.5 h-4 bg-accent rounded-full" />
        <span className="font-display font-semibold text-sm">Hidden Layer Activations</span>
        <span className="font-mono text-xs text-muted">64 neurons</span>
      </div>

      <div className="grid grid-cols-[repeat(32,1fr)] gap-px mb-3">
        {activations.map((v, i) => {
          const norm = max > 0 ? v / max : 0
          return (
            <div
              key={i}
              className="aspect-square rounded-sm"
              style={{ background: `rgba(108, 99, 255, ${Math.max(0.05, norm)})` }}
              title={`Neuron ${i}: ${v.toFixed(3)}`}
            />
          )
        })}
      </div>

      <div className="flex flex-col gap-1">
        <span className="font-mono text-[10px] text-muted">Top active neurons</span>
        {topK.map(({ v, i }) => (
          <div key={i} className="flex items-center gap-2">
            <span className="font-mono text-[10px] text-muted w-8">#{i}</span>
            <div className="flex-1 h-1.5 bg-surface rounded-full overflow-hidden">
              <motion.div
                className="h-full bg-accent rounded-full"
                animate={{ width: `${(v / max) * 100}%` }}
                transition={{ duration: 0.4 }}
              />
            </div>
            <span className="font-mono text-[10px] text-ghost w-12 text-right">{v.toFixed(2)}</span>
          </div>
        ))}
      </div>
    </div>
  )
}

function EmptyFeatureState() {
  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="glass-bright rounded-2xl p-12 flex flex-col items-center gap-4 text-center"
    >
      <div className="grid grid-cols-4 gap-2 opacity-20">
        {Array.from({ length: 16 }, (_, i) => (
          <div key={i} className="aspect-square rounded-lg bg-surface shimmer" />
        ))}
      </div>
      <p className="font-mono text-sm text-muted">Draw a digit to see feature map activations</p>
    </motion.div>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// PerformanceDashboard
// ─────────────────────────────────────────────────────────────────────────────

export function PerformanceDashboard() {
  const { prediction, isRunning, totalPredictions, avgLatency } = useAppStore()

  const stats = [
    { label: 'Prediction', value: prediction ? String(prediction.predicted_digit) : '—', unit: '', color: '#00ff9d', large: true },
    { label: 'Confidence', value: prediction ? `${Math.round(prediction.confidence * 100)}` : '—', unit: '%', color: '#6c63ff' },
    { label: 'Latency', value: prediction ? prediction.latency_ms.toFixed(1) : '—', unit: 'ms', color: '#00d4ff' },
    { label: 'Avg Latency', value: avgLatency > 0 ? avgLatency.toFixed(1) : '—', unit: 'ms', color: '#9d97ff' },
    { label: 'Inferences', value: String(totalPredictions), unit: '', color: '#ff6b35' },
    { label: 'Model', value: prediction?.model_loaded ? 'Trained' : 'Random', unit: '', color: prediction?.model_loaded ? '#00ff9d' : '#ff6b35' },
  ]

  return (
    <div className="glass-bright rounded-2xl p-4">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <div className={`w-2 h-2 rounded-full ${isRunning ? 'bg-neon animate-pulse' : 'bg-muted'}`} />
          <span className="font-display font-semibold text-sm">Control Panel</span>
        </div>
        <span className="font-mono text-[10px] text-muted">NumPy CNN v1</span>
      </div>

      <div className="grid grid-cols-3 gap-2 mb-3">
        {stats.map((s) => <StatCard key={s.label} {...s} />)}
      </div>

      <div className="border-t border-border pt-3 grid grid-cols-2 gap-x-4 gap-y-1">
        {[
          ['Input', '28 × 28'],
          ['Conv', '16 × 3×3'],
          ['Pool', '2×2 / stride 2'],
          ['Flatten', '2704'],
          ['Dense', '2704 → 64 → 10'],
          ['Params', '~175K'],
        ].map(([k, v]) => (
          <div key={k} className="flex items-center justify-between">
            <span className="font-mono text-[10px] text-muted">{k}</span>
            <span className="font-mono text-[10px] text-ghost">{v}</span>
          </div>
        ))}
      </div>
    </div>
  )
}

function StatCard({ label, value, unit, color, large }: {
  label: string; value: string; unit: string; color: string; large?: boolean
}) {
  return (
    <motion.div
      className="bg-surface rounded-xl p-2.5 border border-border text-center"
      whileHover={{ borderColor: color, boxShadow: `0 0 12px ${color}30` }}
      transition={{ duration: 0.2 }}
    >
      <div className={`font-mono font-bold leading-none digit-display ${large ? 'text-2xl' : 'text-lg'}`} style={{ color }}>
        <motion.span key={value} initial={{ opacity: 0, y: -6 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.25 }}>
          {value}
        </motion.span>
        {unit && <span className="text-xs font-normal ml-0.5 text-muted">{unit}</span>}
      </div>
      <div className="font-mono text-[10px] text-muted mt-1">{label}</div>
    </motion.div>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// ArchitectureExplorer
// ─────────────────────────────────────────────────────────────────────────────

interface LayerInfo {
  id: string; name: string; type: string
  input_shape: number[]; output_shape: number[]
  description: string; params: number
  kernel_size?: number[]; num_kernels?: number; pool_size?: number
}

const TYPE_COLORS: Record<string, string> = {
  input: '#00d4ff', conv: '#6c63ff', relu: '#00ff9d',
  pool: '#ff6b35', flatten: '#9d97ff', dense: '#f472b6', softmax: '#fbbf24',
}

const TYPE_ICONS: Record<string, string> = {
  input: '⬡', conv: '◈', relu: '⚡',
  pool: '⊛', flatten: '≡', dense: '◎', softmax: '◉',
}

export function ArchitectureExplorer() {
  const [layers, setLayers] = useState<LayerInfo[]>([])
  const [totalParams, setTotalParams] = useState(0)
  const [selectedLayer, setSelectedLayer] = useState<string | null>(null)
  const { prediction } = useAppStore()

  useEffect(() => {
    fetchArchitecture().then((data) => {
      setLayers(data.layers)
      setTotalParams(data.total_params)
    }).catch(console.error)
  }, [])

  return (
    <div className="flex flex-col gap-4">
      <div className="glass-bright rounded-2xl p-4 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <div className="w-1.5 h-4 bg-accent rounded-full" />
          <span className="font-display font-semibold text-sm">CNN Architecture</span>
        </div>
        <div className="flex items-center gap-4 font-mono text-xs text-muted">
          <span>{layers.length} layers</span>
          <span>·</span>
          <span>{totalParams.toLocaleString()} params</span>
        </div>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-[1fr_380px] gap-4">
        <div className="glass-bright rounded-2xl p-4">
          <div className="flex flex-col gap-0">
            {layers.map((layer, i) => (
              <LayerNode
                key={layer.id}
                layer={layer}
                index={i}
                selected={selectedLayer === layer.id}
                onSelect={() => setSelectedLayer(selectedLayer === layer.id ? null : layer.id)}
                isLast={i === layers.length - 1}
                prediction={prediction}
              />
            ))}
          </div>
        </div>

        <AnimatePresence mode="wait">
          {selectedLayer ? (
            <LayerDetail key={selectedLayer} layer={layers.find((l) => l.id === selectedLayer)!} prediction={prediction} />
          ) : (
            <LayerDetailEmpty key="empty" />
          )}
        </AnimatePresence>
      </div>
    </div>
  )
}

function LayerNode({ layer, index, selected, onSelect, isLast, prediction }: {
  layer: LayerInfo; index: number; selected: boolean; onSelect: () => void; isLast: boolean; prediction: PredictionResult | null
}) {
  const color = TYPE_COLORS[layer.type] ?? '#6c63ff'
  const icon = TYPE_ICONS[layer.type] ?? '●'
  const shapeStr = (s: number[]) => s.join(' × ')

  return (
    <div className="flex">
      <div className="flex flex-col items-center mr-4">
        <motion.div
          className="w-8 h-8 rounded-lg border flex items-center justify-center text-sm flex-shrink-0"
          style={{
            borderColor: selected ? color : 'rgba(30,30,50,0.8)',
            background: selected ? `${color}20` : 'rgba(13,13,20,0.5)',
            color: selected ? color : '#4a4a6a',
            boxShadow: selected ? `0 0 16px ${color}40` : 'none',
          }}
          animate={{ borderColor: selected ? color : 'rgba(30,30,50,0.8)' }}
          transition={{ duration: 0.2 }}
        >
          {icon}
        </motion.div>
        {!isLast && (
          <div className="w-px flex-1 min-h-[20px] relative overflow-hidden" style={{ background: 'rgba(30,30,50,0.8)' }}>
            {prediction && (
              <motion.div
                className="absolute w-full"
                style={{ background: color, height: '3px', top: 0 }}
                animate={{ y: ['0%', '100%'] }}
                transition={{ duration: 1.5, repeat: Infinity, ease: 'linear', delay: index * 0.1 }}
              />
            )}
          </div>
        )}
      </div>

      <motion.button
        onClick={onSelect}
        className={`flex-1 mb-2 text-left rounded-xl p-3 border transition-all duration-200 ${
          selected ? 'border-opacity-60 bg-surface' : 'border-border hover:border-muted bg-surface/50 hover:bg-surface/80'
        }`}
        style={{ borderColor: selected ? color : undefined }}
        whileHover={{ x: 4 }}
        transition={{ duration: 0.15 }}
      >
        <div className="flex items-center justify-between">
          <span className="font-display font-semibold text-sm" style={{ color: selected ? color : '#e8e8f0' }}>
            {layer.name}
          </span>
          {layer.params > 0 && <span className="font-mono text-[10px] text-muted">{layer.params.toLocaleString()} params</span>}
        </div>
        <div className="flex items-center gap-2 mt-1 font-mono text-xs text-muted">
          <span>{shapeStr(layer.input_shape)}</span>
          <span className="text-border">→</span>
          <span style={{ color: selected ? `${color}cc` : undefined }}>{shapeStr(layer.output_shape)}</span>
        </div>
      </motion.button>
    </div>
  )
}

function LayerDetail({ layer, prediction }: { layer: LayerInfo; prediction: PredictionResult | null }) {
  if (!layer) return null
  const color = TYPE_COLORS[layer.type] ?? '#6c63ff'

  return (
    <motion.div
      initial={{ opacity: 0, x: 20 }}
      animate={{ opacity: 1, x: 0 }}
      exit={{ opacity: 0, x: 20 }}
      transition={{ duration: 0.3 }}
      className="glass-bright rounded-2xl p-5 flex flex-col gap-4"
    >
      <div className="flex items-start gap-3">
        <div className="w-10 h-10 rounded-xl border flex items-center justify-center text-lg flex-shrink-0"
          style={{ borderColor: color, background: `${color}15`, color }}>
          {TYPE_ICONS[layer.type]}
        </div>
        <div>
          <h3 className="font-display font-bold text-base" style={{ color }}>{layer.name}</h3>
          <span className="font-mono text-xs text-muted capitalize">{layer.type} layer</span>
        </div>
      </div>

      <p className="text-sm text-ghost leading-relaxed">{layer.description}</p>

      <div className="grid grid-cols-2 gap-2">
        <InfoCell label="Input shape" value={layer.input_shape.join(' × ')} color={color} />
        <InfoCell label="Output shape" value={layer.output_shape.join(' × ')} color={color} />
        {layer.params > 0 && <InfoCell label="Parameters" value={layer.params.toLocaleString()} color={color} />}
        {layer.kernel_size && <InfoCell label="Kernel size" value={layer.kernel_size.join('×')} color={color} />}
        {layer.num_kernels && <InfoCell label="Filters" value={String(layer.num_kernels)} color={color} />}
        {layer.pool_size && <InfoCell label="Pool size" value={`${layer.pool_size}×${layer.pool_size}`} color={color} />}
      </div>

      {layer.type === 'conv' && (
        <div className="bg-surface rounded-xl p-3 font-mono text-xs text-ghost border border-border">
          <div className="text-muted mb-1">Output size formula</div>
          <div>⌊(28 - 3) / 1⌋ + 1 = <span style={{ color }}>26</span></div>
          <div className="mt-1 text-muted">16 filters → <span style={{ color }}>16 × 26 × 26</span></div>
        </div>
      )}
      {layer.type === 'pool' && (
        <div className="bg-surface rounded-xl p-3 font-mono text-xs text-ghost border border-border">
          <div className="text-muted mb-1">Downsampling</div>
          <div>⌊(26 - 2) / 2⌋ + 1 = <span style={{ color }}>13</span></div>
          <div className="mt-1 text-muted">16 maps → <span style={{ color }}>16 × 13 × 13</span></div>
        </div>
      )}
      {layer.type === 'dense' && layer.id === 'dense1' && (
        <div className="bg-surface rounded-xl p-3 font-mono text-xs text-ghost border border-border">
          <div className="text-muted mb-1">Weight matrix</div>
          <div>W: <span style={{ color }}>2704 × 64</span> = 173,056</div>
          <div>b: <span style={{ color }}>64</span></div>
        </div>
      )}
    </motion.div>
  )
}

function LayerDetailEmpty() {
  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="glass-bright rounded-2xl p-8 flex flex-col items-center justify-center gap-3 text-center min-h-[200px]"
    >
      <span className="text-3xl text-muted">◈</span>
      <p className="font-mono text-sm text-muted">Click any layer to explore</p>
    </motion.div>
  )
}

function InfoCell({ label, value, color }: { label: string; value: string; color: string }) {
  return (
    <div className="bg-surface rounded-xl p-2.5 border border-border">
      <div className="font-mono text-[10px] text-muted mb-0.5">{label}</div>
      <div className="font-mono text-sm font-bold" style={{ color }}>{value}</div>
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// PipelineView
// ─────────────────────────────────────────────────────────────────────────────

interface PipelineStage {
  id: string; label: string; sublabel: string; color: string; icon: string
}

const PIPELINE_STAGES: PipelineStage[] = [
  { id: 'draw',       label: 'Drawing',     sublabel: 'User input',         color: '#00d4ff', icon: '✏️' },
  { id: 'preprocess', label: 'Preprocess',  sublabel: '→ 28×28 float32',    color: '#6c63ff', icon: '⬡' },
  { id: 'conv',       label: 'Convolution', sublabel: '16 × 3×3 kernels',   color: '#9d97ff', icon: '◈' },
  { id: 'relu',       label: 'ReLU',        sublabel: 'max(0, x)',           color: '#00ff9d', icon: '⚡' },
  { id: 'pool',       label: 'Max Pool',    sublabel: '2×2 / stride 2',     color: '#ff6b35', icon: '⊛' },
  { id: 'flatten',    label: 'Flatten',     sublabel: '2704 values',         color: '#fbbf24', icon: '≡' },
  { id: 'dense',      label: 'Dense × 2',  sublabel: '→ 64 → 10',           color: '#f472b6', icon: '◎' },
  { id: 'softmax',    label: 'Softmax',     sublabel: 'probabilities',       color: '#00d4ff', icon: '◉' },
  { id: 'output',     label: 'Prediction',  sublabel: 'argmax',              color: '#00ff9d', icon: '★' },
]

export function PipelineView() {
  const { prediction, isRunning } = useAppStore()
  const [activeStage, setActiveStage] = useState<number>(-1)

  useEffect(() => {
    if (!prediction) { setActiveStage(-1); return }
    let i = 0
    setActiveStage(0)
    const interval = setInterval(() => {
      i++
      if (i >= PIPELINE_STAGES.length) { clearInterval(interval); setActiveStage(PIPELINE_STAGES.length - 1); return }
      setActiveStage(i)
    }, 200)
    return () => clearInterval(interval)
  }, [prediction?.predicted_digit, prediction?.latency_ms])

  return (
    <div className="flex flex-col gap-4">
      <div className="glass-bright rounded-2xl p-5">
        <div className="flex items-center gap-2 mb-5">
          <div className="w-1.5 h-4 bg-neon rounded-full" />
          <span className="font-display font-semibold text-sm">Inference Pipeline</span>
          {isRunning && <span className="font-mono text-xs text-neon animate-pulse ml-2">● running</span>}
        </div>

        <div className="hidden lg:flex items-center gap-2 overflow-x-auto pb-2">
          {PIPELINE_STAGES.map((stage, i) => (
            <>
              <PipelineStageCard key={stage.id} stage={stage} active={activeStage >= i} current={activeStage === i} index={i} />
              {i < PIPELINE_STAGES.length - 1 && (
                <motion.div
                  key={`arrow-${i}`}
                  className="flex-shrink-0 w-6 h-0.5 rounded-full"
                  style={{ background: activeStage > i ? stage.color : '#1e1e32' }}
                  animate={{ opacity: activeStage > i ? 1 : 0.3 }}
                  transition={{ duration: 0.3 }}
                />
              )}
            </>
          ))}
        </div>

        <div className="lg:hidden flex flex-col gap-2">
          {PIPELINE_STAGES.map((stage, i) => (
            <div key={stage.id} className="flex items-center gap-3">
              <PipelineStageCard stage={stage} active={activeStage >= i} current={activeStage === i} index={i} compact />
              {i < PIPELINE_STAGES.length - 1 && <div className="w-px h-4 bg-border ml-4" />}
            </div>
          ))}
        </div>
      </div>

      {prediction && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
          <StageDataCard title="Input (preprocessed)" sublabel="28 × 28 · normalised" color="#00d4ff">
            <PixelGrid data={prediction.preprocessed_image} />
          </StageDataCard>
          <StageDataCard title="Conv → ReLU (filter #0)" sublabel="26 × 26 · activated" color="#6c63ff">
            <FeatureMapMini map={prediction.feature_maps_relu[0]} colormap="viridis" />
          </StageDataCard>
          <StageDataCard title="After Pooling (filter #0)" sublabel="13 × 13 · downsampled" color="#ff6b35">
            <FeatureMapMini map={prediction.feature_maps_pooled[0]} colormap="hot" />
          </StageDataCard>
        </div>
      )}

      {prediction && (
        <div className="glass-bright rounded-2xl p-4">
          <div className="flex items-center gap-2 mb-3">
            <div className="w-1.5 h-4 bg-fuchsia-400 rounded-full" />
            <span className="font-display font-semibold text-sm">Output Logits → Softmax</span>
          </div>
          <SoftmaxViz probs={prediction.probabilities} best={prediction.predicted_digit} />
        </div>
      )}

      {!prediction && (
        <div className="glass-bright rounded-2xl p-12 text-center">
          <p className="font-mono text-sm text-muted">Draw a digit to see the full inference pipeline</p>
        </div>
      )}
    </div>
  )
}

function PipelineStageCard({ stage, active, current, index, compact }: {
  stage: PipelineStage; active: boolean; current: boolean; index: number; compact?: boolean
}) {
  if (compact) {
    return (
      <motion.div
        className="flex items-center gap-3 rounded-xl p-3 border flex-1 transition-all duration-300"
        style={{
          borderColor: active ? stage.color : '#1e1e32',
          background: active ? `${stage.color}10` : 'rgba(13,13,20,0.5)',
          boxShadow: current ? `0 0 20px ${stage.color}40` : 'none',
        }}
        animate={{ scale: current ? 1.02 : 1 }}
        transition={{ duration: 0.2 }}
      >
        <span>{stage.icon}</span>
        <div>
          <div className="font-mono text-xs font-bold" style={{ color: active ? stage.color : '#4a4a6a' }}>{stage.label}</div>
          <div className="font-mono text-[10px] text-muted">{stage.sublabel}</div>
        </div>
      </motion.div>
    )
  }

  return (
    <motion.div
      className="flex-shrink-0 flex flex-col items-center gap-1.5 rounded-xl p-3 border min-w-[90px] text-center"
      style={{
        borderColor: active ? stage.color : '#1e1e32',
        background: active ? `${stage.color}10` : 'rgba(13,13,20,0.5)',
        boxShadow: current ? `0 0 20px ${stage.color}40` : 'none',
      }}
      animate={{ scale: current ? 1.08 : 1 }}
      transition={{ duration: 0.2 }}
    >
      <span className="text-xl">{stage.icon}</span>
      <div className="font-mono text-xs font-bold" style={{ color: active ? stage.color : '#4a4a6a' }}>{stage.label}</div>
      <div className="font-mono text-[9px] text-muted">{stage.sublabel}</div>
    </motion.div>
  )
}

function StageDataCard({ title, sublabel, color, children }: {
  title: string; sublabel: string; color: string; children: ReactNode
}) {
  return (
    <div className="glass-bright rounded-2xl p-4 flex flex-col gap-3">
      <div>
        <div className="font-display font-semibold text-sm" style={{ color }}>{title}</div>
        <div className="font-mono text-xs text-muted">{sublabel}</div>
      </div>
      <div className="flex-1">{children}</div>
    </div>
  )
}

function PixelGrid({ data }: { data: number[][] }) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  useEffect(() => {
    const canvas = canvasRef.current!
    const h = data.length
    const w = data[0]?.length ?? 28
    canvas.width = w
    canvas.height = h
    const ctx = canvas.getContext('2d')!
    const img = ctx.createImageData(w, h)
    for (let r = 0; r < h; r++) {
      for (let c = 0; c < w; c++) {
        const v = Math.round(data[r][c] * 255)
        const idx = (r * w + c) * 4
        img.data[idx] = v; img.data[idx + 1] = v; img.data[idx + 2] = v; img.data[idx + 3] = 255
      }
    }
    ctx.putImageData(img, 0, 0)
  }, [data])

  return (
    <div className="rounded-lg overflow-hidden border border-border aspect-square">
      <canvas ref={canvasRef} className="w-full h-full" style={{ imageRendering: 'pixelated' }} />
    </div>
  )
}

function FeatureMapMini({ map, colormap }: { map: number[][]; colormap: 'viridis' | 'hot' | 'cool' }) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  useEffect(() => {
    const canvas = canvasRef.current!
    if (!map) return
    renderFeatureMap(canvas, map, colormap)
  }, [map, colormap])

  return (
    <div className="rounded-lg overflow-hidden border border-border aspect-square">
      <canvas ref={canvasRef} className="w-full h-full" style={{ imageRendering: 'pixelated' }} />
    </div>
  )
}

function SoftmaxViz({ probs, best }: { probs: number[]; best: number }) {
  return (
    <div className="flex gap-2 items-end h-24">
      {probs.map((p, d) => {
        const isBest = d === best
        return (
          <div key={d} className="flex-1 flex flex-col items-center gap-1">
            <motion.div
              className="w-full rounded-t-sm"
              style={{
                background: isBest ? 'linear-gradient(180deg, #00ff9d, #6c63ff)' : '#1e1e32',
                boxShadow: isBest ? '0 0 12px rgba(0,255,157,0.4)' : 'none',
              }}
              animate={{ height: `${Math.max(p * 80, 2)}px` }}
              transition={{ duration: 0.5, ease: 'easeOut' }}
            />
            <span className={`font-mono text-[10px] ${isBest ? 'text-emerald-300' : 'text-muted'}`}>{d}</span>
          </div>
        )
      })}
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. App layout (was App.tsx)
// ─────────────────────────────────────────────────────────────────────────────

const TABS = [
  { id: 'draw', label: 'Prediction', icon: '◉' },
  { id: 'features', label: 'Feature Maps', icon: '⬡' },
  { id: 'architecture', label: 'Architecture', icon: '◈' },
  { id: 'pipeline', label: 'Pipeline', icon: '⟶' },
] as const

function HeroSection() {
  return (
    <motion.section
      initial={{ opacity: 0, y: -20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.8, ease: 'easeOut' }}
      className="pt-8 pb-2"
    >
      <div className="flex items-center gap-3 mb-2">
        <div className="w-2 h-2 rounded-full bg-accent animate-pulse" />
        <span className="font-mono text-xs text-muted tracking-widest uppercase">
          NumPy CNN · Live Inference
        </span>
        <div className="flex-1 h-px bg-border" />
      </div>
      <h1 className="font-display text-4xl md:text-5xl font-bold leading-tight">
        <span className="text-text">Neural</span>
        <span className="text-accent glow-text-accent">Vis</span>
      </h1>
      <p className="mt-2 text-ghost text-sm max-w-xl">
        Draw any digit 0–9. Watch a hand-built convolutional neural network — written entirely in NumPy, no frameworks — classify it in real time while exposing every internal activation.
      </p>
    </motion.section>
  )
}

function PanelTabs() {
  const { activePanel, setActivePanel } = useAppStore()
  return (
    <div className="flex gap-1 glass rounded-xl p-1">
      {TABS.map((tab) => (
        <button
          key={tab.id}
          onClick={() => setActivePanel(tab.id)}
          className={`
            relative flex-1 flex items-center justify-center gap-2 py-2 px-3 rounded-lg
            font-mono text-xs tracking-wide transition-all duration-300
            ${activePanel === tab.id
              ? 'bg-accent/20 text-accent-glow border border-accent/30 shadow-glow-accent'
              : 'text-ghost hover:text-text hover:bg-white/5'
            }
          `}
        >
          <span className="text-sm">{tab.icon}</span>
          <span className="hidden sm:inline">{tab.label}</span>
          {activePanel === tab.id && (
            <motion.div
              layoutId="tab-indicator"
              className="absolute inset-0 rounded-lg bg-accent/10 border border-accent/20"
              transition={{ type: 'spring', stiffness: 300, damping: 30 }}
            />
          )}
        </button>
      ))}
    </div>
  )
}

function App() {
  const { activePanel } = useAppStore()
  return (
    <div className="relative min-h-screen bg-void overflow-x-hidden font-body text-text">
      <ParticleBackground />
      <NavBar />
      <main className="relative z-10 pt-16 pb-8 px-4 md:px-6 lg:px-8 max-w-[1600px] mx-auto">
        <HeroSection />
        <div className="mt-8 grid grid-cols-1 xl:grid-cols-[420px_1fr] gap-6">
          <div className="flex flex-col gap-4">
            <DrawingCanvas />
            <PerformanceDashboard />
          </div>
          <div className="flex flex-col gap-4">
            <PanelTabs />
            <AnimatePresence mode="wait">
              <motion.div
                key={activePanel}
                initial={{ opacity: 0, y: 16 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -16 }}
                transition={{ duration: 0.3, ease: 'easeInOut' }}
                className="flex-1"
              >
                {activePanel === 'draw' && (
                  <div className="grid grid-cols-1 lg:grid-cols-[1fr_320px] gap-4">
                    <PredictionDisplay />
                    <ConfidenceChart />
                  </div>
                )}
                {activePanel === 'features' && <FeatureMaps />}
                {activePanel === 'architecture' && <ArchitectureExplorer />}
                {activePanel === 'pipeline' && <PipelineView />}
              </motion.div>
            </AnimatePresence>
          </div>
        </div>
      </main>
      <StatusBar />
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// 5. Bootstrap (was main.tsx)
// ─────────────────────────────────────────────────────────────────────────────

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
)
