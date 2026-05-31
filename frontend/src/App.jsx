import React, { useState, useEffect, useRef } from 'react'
import { Code } from 'lucide-react'

// Modular Components
import Header from './components/Header'
import ModelParameters from './components/ModelParameters'
import MediaVisualizer from './components/MediaVisualizer'
import IdentifiedObjectsList from './components/IdentifiedObjectsList'
import LearningTabs from './components/LearningTabs'

const API_URL = import.meta.env.DEV ? 'http://127.0.0.1:8000' : window.location.origin

// Sample Mock data
const SAMPLE_MOCKS = {
  workspace: {
    url: '/desk_sample.jpg',
    detections: [
      { class_id: 73, label: 'laptop', confidence: 0.89, bbox: [480, 360, 280, 360] },
      { class_id: 84, label: 'book', confidence: 0.82, bbox: [150, 720, 370, 300] },
      { class_id: 62, label: 'chair', confidence: 0.76, bbox: [900, 480, 200, 360] }
    ]
  },
  livingroom: {
    url: '/couch_sample.jpg',
    detections: [
      { class_id: 62, label: 'chair', confidence: 0.91, bbox: [0, 1538, 1326, 582] },
      { class_id: 67, label: 'dining table', confidence: 0.74, bbox: [280, 1950, 800, 500] }
    ]
  }
}

export default function App() {
  // Navigation tabs
  const [learningTab, setLearningTab] = useState('explore')
  
  // Settings & Params State
  const [activeSource, setActiveSource] = useState('upload') // 'upload' | 'webcam'
  const [activeModel, setActiveModel] = useState('coco') // 'coco' | 'custom'
  const [confThreshold, setConfThreshold] = useState(50)
  const [nmsThreshold, setNmsThreshold] = useState(40)
  const [checkedClasses, setCheckedClasses] = useState(['person', 'laptop', 'chair', 'book', 'cell phone', 'dining table'])
  
  // Server connectivity State
  const [serverStatus, setServerStatus] = useState('checking') // 'online' | 'offline' | 'checking'
  const [serverBackend, setServerBackend] = useState('None')
  
  // Media Input & Detections State
  const [imageSrc, setImageSrc] = useState(null)
  const [imageBlob, setImageBlob] = useState(null)
  const [detections, setDetections] = useState([])
  const [naturalWidth, setNaturalWidth] = useState(600)
  const [naturalHeight, setNaturalHeight] = useState(400)
  const [inferenceLatency, setInferenceLatency] = useState(0)
  const [webcamFps, setWebcamFps] = useState(0)
  const [isProcessing, setIsProcessing] = useState(false)
  const [hoveredIndex, setHoveredIndex] = useState(null)
  
  // Layout scaling refs
  const viewerContainerRef = useRef(null)
  const viewerImgRef = useRef(null)
  const videoRef = useRef(null)
  const webcamContainerRef = useRef(null)
  const fileInputRef = useRef(null)
  
  // Webcam running refs
  const [webcamActive, setWebcamActive] = useState(false)
  const webcamStreamRef = useRef(null)
  const webcamIntervalRef = useRef(null)

  // Keep refs up-to-date for webcam loop to avoid stale closures
  const checkedClassesRef = useRef(checkedClasses)
  const confThresholdRef = useRef(confThreshold)
  const nmsThresholdRef = useRef(nmsThreshold)
  const activeModelRef = useRef(activeModel)

  useEffect(() => {
    checkedClassesRef.current = checkedClasses
    confThresholdRef.current = confThreshold
    nmsThresholdRef.current = nmsThreshold
    activeModelRef.current = activeModel
  }, [checkedClasses, confThreshold, nmsThreshold, activeModel])

  // Reset checked classes when switching models to matching defaults
  useEffect(() => {
    if (activeModel === 'custom') {
      setCheckedClasses(['person', 'laptop', 'chair', 'book', 'dining table', 'cell phone', 'cup', 'keyboard', 'mouse'])
    } else {
      setCheckedClasses(['person', 'laptop', 'chair', 'book', 'cell phone', 'dining table'])
    }
  }, [activeModel])
  
  // Resizing state triggers redraw
  const [windowDimensions, setWindowDimensions] = useState({ width: 0, height: 0 })

  // 1. Initial server check
  useEffect(() => {
    async function checkStatus() {
      try {
        const res = await fetch(`${API_URL}/api/status`)
        if (!res.ok) {
          setServerStatus('offline')
          return
        }
        const contentType = res.headers.get("content-type")
        if (!contentType || !contentType.includes("application/json")) {
          setServerStatus('offline')
          return
        }
        const data = await res.json()
        if (data.status === 'active') {
          setServerStatus('online')
          setServerBackend(data.backend)
        } else {
          setServerStatus('offline')
        }
      } catch (err) {
        setServerStatus('offline')
      }
    }
    checkStatus()
    
    // Resize listener
    const handleResize = () => {
      setWindowDimensions({ width: window.innerWidth, height: window.innerHeight })
    }
    window.addEventListener('resize', handleResize)
    return () => {
      window.removeEventListener('resize', handleResize)
      stopWebcam()
    }
  }, [])

  // Re-run detection on setting changes (for uploaded images/mock images)
  useEffect(() => {
    if (imageBlob && activeSource === 'upload') {
      runDetection(imageBlob)
    } else if (imageSrc && !imageBlob && activeSource === 'upload') {
      // It is a sample mock image, filter the static results
      const key = Object.keys(SAMPLE_MOCKS).find(k => SAMPLE_MOCKS[k].url === imageSrc)
      if (key) {
        let dets = SAMPLE_MOCKS[key].detections.filter(d => d.confidence >= confThreshold / 100)
        if (checkedClasses.length > 0) {
          dets = dets.filter(d => checkedClasses.includes(d.label))
        }
        if (activeModel === 'custom') {
          // Boost fine-tuned targets confidence representationally
          dets = dets.map(d => {
            if (['person', 'laptop', 'chair', 'book'].includes(d.label)) {
              return { ...d, confidence: Math.min(1.0, d.confidence + 0.08) }
            }
            return d
          }).sort((a, b) => b.confidence - a.confidence)
        }
        setDetections(dets)
      }
    }
  }, [confThreshold, nmsThreshold, checkedClasses, activeModel, imageBlob, activeSource])

  // Process uploaded/dragged images
  const handleImageFile = (file) => {
    if (!file || !file.type.startsWith('image/')) return
    setImageBlob(file)
    const reader = new FileReader()
    reader.onload = (e) => {
      setImageSrc(e.target.result)
    }
    reader.readAsDataURL(file)
  }

  // Handle image load coordinates
  const handleImageLoaded = (e) => {
    setNaturalWidth(e.target.naturalWidth)
    setNaturalHeight(e.target.naturalHeight)
    if (imageBlob) {
      runDetection(imageBlob)
    }
  }

  // Clear current image
  const clearActiveImage = () => {
    setImageSrc(null)
    setImageBlob(null)
    setDetections([])
    setInferenceLatency(0)
  }

  // API Call for detection
  const runDetection = async (blob) => {
    setIsProcessing(true)
    const formData = new FormData()
    formData.append('file', blob)
    
    const conf = confThreshold / 100
    const nms = nmsThreshold / 100
    const allowed = checkedClasses.join(',')
    
    let url = `${API_URL}/api/detect?conf_threshold=${conf}&nms_threshold=${nms}&model_variant=${activeModel}`
    if (allowed) {
      url += `&allowed_classes=${allowed}`
    }
    
    try {
      const res = await fetch(url, {
        method: 'POST',
        body: formData
      })
      if (!res.ok) {
        throw new Error(`HTTP status ${res.status}`)
      }
      const contentType = res.headers.get("content-type")
      if (!contentType || !contentType.includes("application/json")) {
        throw new TypeError("Response is not JSON")
      }
      const data = await res.json()
      
      setInferenceLatency(data.latency_ms)
      setDetections(data.detections)
      setServerStatus('online')
    } catch (err) {
      console.error("Detection error:", err)
      setServerStatus('offline')
    } finally {
      setIsProcessing(false)
    }
  }

  // --- WEBCAM STREAM CONTROLS ---
  const startWebcam = async () => {
    if (imageSrc) clearActiveImage()
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480, facingMode: 'user' }
      })
      if (videoRef.current) {
        videoRef.current.srcObject = stream
      }
      webcamStreamRef.current = stream
      setWebcamActive(true)
      
      // Start processing loop
      startWebcamProcessingLoop()
    } catch (err) {
      alert("Could not access camera. Ensure permission is granted.")
    }
  }

  const stopWebcam = () => {
    if (webcamIntervalRef.current) {
      clearInterval(webcamIntervalRef.current)
      webcamIntervalRef.current = null
    }
    if (webcamStreamRef.current) {
      webcamStreamRef.current.getTracks().forEach(track => track.stop())
      webcamStreamRef.current = null
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null
    }
    setWebcamActive(false)
    setWebcamFps(0)
  }

  const startWebcamProcessingLoop = () => {
    const canvas = document.createElement('canvas')
    const ctx = canvas.getContext('2d')
    let busy = false
    let frameCount = 0
    let lastTime = performance.now()
    
    webcamIntervalRef.current = setInterval(() => {
      const video = videoRef.current
      if (!video || video.paused || video.ended || busy) return
      
      busy = true
      canvas.width = video.videoWidth
      canvas.height = video.videoHeight
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height)
      
      // Capture size
      setNaturalWidth(video.videoWidth)
      setNaturalHeight(video.videoHeight)
      
      canvas.toBlob(async (blob) => {
        if (!blob) {
          busy = false
          return
        }
        
        const conf = confThresholdRef.current / 100
        const nms = nmsThresholdRef.current / 100
        const allowed = checkedClassesRef.current.join(',')
        
        let url = `${API_URL}/api/detect?conf_threshold=${conf}&nms_threshold=${nms}&model_variant=${activeModelRef.current}`
        if (allowed) {
          url += `&allowed_classes=${allowed}`
        }
        
        const startTime = performance.now()
        try {
          const res = await fetch(url, { method: 'POST', body: formData(blob) })
          if (!res.ok) {
            throw new Error(`HTTP status ${res.status}`)
          }
          const contentType = res.headers.get("content-type")
          if (!contentType || !contentType.includes("application/json")) {
            throw new TypeError("Response is not JSON")
          }
          const data = await res.json()
          
          setInferenceLatency(data.latency_ms)
          setDetections(data.detections)
          setServerStatus('online')
          
          // FPS counter
          frameCount++
          const elapsed = performance.now() - lastTime
          if (elapsed >= 1000) {
            setWebcamFps(Math.round((frameCount * 1000) / elapsed))
            frameCount = 0
            lastTime = performance.now()
          }
        } catch (e) {
          console.warn("Webcam inference frame dropped:", e.message || e)
          setServerStatus('offline')
        } finally {
          busy = false
        }
      }, 'image/jpeg', 0.65)
      
    }, 180) // Inference every 180ms
  }

  const formData = (blob) => {
    const fd = new FormData()
    fd.append('file', blob, 'webcam.jpg')
    return fd
  }

  // --- BOX LAYOUT SCALING CALCULATOR ---
  const getScaledCoordinates = (bbox) => {
    const [x, y, w, h] = bbox
    
    let containerWidth = 0
    let containerHeight = 0
    let dispW = 0
    let dispH = 0
    
    if (activeSource === 'webcam') {
      const video = videoRef.current
      if (!video) return { left: 0, top: 0, width: 0, height: 0 }
      containerWidth = webcamContainerRef.current?.clientWidth || 640
      containerHeight = webcamContainerRef.current?.clientHeight || 480
      dispW = video.clientWidth
      dispH = video.clientHeight
    } else {
      const img = viewerImgRef.current
      if (!img) return { left: 0, top: 0, width: 0, height: 0 }
      containerWidth = viewerContainerRef.current?.clientWidth || 600
      containerHeight = viewerContainerRef.current?.clientHeight || 400
      dispW = img.clientWidth
      dispH = img.clientHeight
    }
    
    const scaleX = dispW / naturalWidth
    const scaleY = dispH / naturalHeight
    
    const leftOffset = (containerWidth - dispW) / 2
    const topOffset = (containerHeight - dispH) / 2
    
    return {
      left: (x * scaleX) + leftOffset,
      top: (y * scaleY) + topOffset,
      width: w * scaleX,
      height: h * scaleY
    }
  }

  const handleCheckboxChange = (cls, checked) => {
    if (checked) {
      setCheckedClasses([...checkedClasses, cls])
    } else {
      setCheckedClasses(checkedClasses.filter(c => c !== cls))
    }
  }

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col items-center p-4 md:p-8 font-sans">
      {/* Decorative Orbs */}
      <div className="absolute top-[-10%] left-[-10%] w-[35vw] h-[35vw] rounded-full bg-violet-600/10 blur-[120px] pointer-events-none" />
      <div className="absolute bottom-[-10%] right-[-10%] w-[40vw] h-[40vw] rounded-full bg-cyan-600/8 blur-[120px] pointer-events-none" />
      
      <div className="w-full max-w-7xl flex flex-col gap-6">
        
        {/* HEADER SECTION */}
        <Header serverStatus={serverStatus} serverBackend={serverBackend} />

        {/* MAIN SPLIT-VIEW INTERFACE */}
        <main className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
          
          {/* LEFT SIDE: CONTROLS & SELECTION */}
          <ModelParameters 
            activeModel={activeModel}
            setActiveModel={setActiveModel}
            confThreshold={confThreshold}
            setConfThreshold={setConfThreshold}
            nmsThreshold={nmsThreshold}
            setNmsThreshold={setNmsThreshold}
            checkedClasses={checkedClasses}
            handleCheckboxChange={handleCheckboxChange}
            setCheckedClasses={setCheckedClasses}
          />
          
          {/* CENTER VIEW: DETECTIONS CANVAS & VIDEO STREAM */}
          <MediaVisualizer 
            activeSource={activeSource}
            setActiveSource={setActiveSource}
            imageSrc={imageSrc}
            detections={detections}
            hoveredIndex={hoveredIndex}
            setHoveredIndex={setHoveredIndex}
            webcamActive={webcamActive}
            webcamFps={webcamFps}
            inferenceLatency={inferenceLatency}
            startWebcam={startWebcam}
            stopWebcam={stopWebcam}
            handleImageFile={handleImageFile}
            handleImageLoaded={handleImageLoaded}
            clearActiveImage={clearActiveImage}
            fileInputRef={fileInputRef}
            viewerContainerRef={viewerContainerRef}
            viewerImgRef={viewerImgRef}
            videoRef={videoRef}
            webcamContainerRef={webcamContainerRef}
            getScaledCoordinates={getScaledCoordinates}
          />
          
          {/* RIGHT SIDE: RESULTS LIST */}
          <IdentifiedObjectsList 
            detections={detections}
            hoveredIndex={hoveredIndex}
            setHoveredIndex={setHoveredIndex}
          />

        </main>

        {/* BOTTOM LEARNING ACCORDION TABS */}
        <LearningTabs learningTab={learningTab} setLearningTab={setLearningTab} />

        {/* FOOTER */}
        <footer className="flex items-center justify-between text-[11px] text-slate-600 mt-4 border-t border-white/5 pt-4 pb-8">
          <span>InsightSSD Workplace Lab Dashboard &copy; 2026.</span>
          <div className="flex items-center gap-1 hover:text-slate-400 cursor-pointer">
            <Code className="w-3.5 h-3.5" />
            <span>GitHub repository</span>
          </div>
        </footer>

      </div>
    </div>
  )
}
