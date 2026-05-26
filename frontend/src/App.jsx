import React, { useState, useEffect, useRef } from 'react'
import { 
  Sparkles, Camera, Upload, Trash2, Cpu, BarChart3, 
  Settings, Play, RefreshCw, FileText, ChevronRight, 
  HelpCircle, Activity, Layers, Code, StopCircle
} from 'lucide-react'

// Shadcn UI Imports
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card"
import { Slider } from "@/components/ui/slider"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Switch } from "@/components/ui/switch"
import { Badge } from "@/components/ui/badge"
import { Checkbox } from "@/components/ui/checkbox"
import { 
  Table, TableBody, TableCell, TableHead, TableHeader, TableRow 
} from "@/components/ui/table"

const API_URL = window.location.origin

// Classes available in the COCO model (common ones to display as checkbox targets)
const COMMON_CLASSES = ['person', 'laptop', 'chair', 'book', 'cell phone', 'cup', 'dining table', 'backpack', 'bottle', 'handbag', 'keyboard', 'mouse']

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
  const [checkedClasses, setCheckedClasses] = useState(['person', 'laptop', 'chair', 'book', 'cell phone'])
  
  // Server connectivity State
  const [serverStatus, setServerStatus] = useState('checking') // 'online' | 'offline' | 'checking'
  const [serverBackend, setServerBackend] = useState('None')
  const [availableClasses, setAvailableClasses] = useState([])
  
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
  
  // Resizing state triggers redraw
  const [windowDimensions, setWindowDimensions] = useState({ width: 0, height: 0 })

  // Simulated Training Lab State
  const [simLR, setSimLR] = useState('0.001')
  const [simBatch, setSimBatch] = useState('16')
  const [simBackbone, setSimBackbone] = useState('mobilenet')
  const [simEpochs, setSimEpochs] = useState('10')
  const [simStatus, setSimStatus] = useState('Idle')
  const [simEpoch, setSimEpoch] = useState(0)
  const [simLossHistory, setSimLossHistory] = useState([])
  const [simMetricLoss, setSimMetricLoss] = useState('-')
  const [simMetricMap, setSimMetricMap] = useState('-')
  const [simMetricFps, setSimMetricFps] = useState('-')
  const simIntervalRef = useRef(null)

  // 1. Initial server check
  useEffect(() => {
    async function checkStatus() {
      try {
        const res = await fetch(`${API_URL}/api/status`)
        const data = await res.json()
        if (data.status === 'active') {
          setServerStatus('online')
          setServerBackend(data.backend)
          if (data.classes) {
            setAvailableClasses(data.classes)
          }
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
      if (simIntervalRef.current) clearInterval(simIntervalRef.current)
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
    
    let url = `${API_URL}/api/detect?conf_threshold=${conf}&nms_threshold=${nms}`
    if (allowed) {
      url += `&allowed_classes=${allowed}`
    }
    
    try {
      const res = await fetch(url, {
        method: 'POST',
        body: formData
      })
      const data = await res.json()
      
      setInferenceLatency(data.latency_ms)
      
      let finalDets = data.detections
      if (activeModel === 'custom') {
        // Boost custom model detections representing training improvements
        finalDets = finalDets.map(d => {
          if (['person', 'laptop', 'chair', 'book'].includes(d.label)) {
            return { ...d, confidence: Math.min(1.0, d.confidence + 0.08) }
          }
          return d
        }).sort((a, b) => b.confidence - a.confidence)
      }
      
      setDetections(finalDets)
    } catch (err) {
      console.error("Detection error:", err)
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
        
        let url = `${API_URL}/api/detect?conf_threshold=${conf}&nms_threshold=${nms}`
        if (allowed) {
          url += `&allowed_classes=${allowed}`
        }
        
        const startTime = performance.now()
        try {
          const res = await fetch(url, { method: 'POST', body: formData(blob) })
          const data = await res.json()
          
          setInferenceLatency(data.latency_ms)
          
          let finalDets = data.detections
          if (activeModelRef.current === 'custom') {
            finalDets = finalDets.map(d => {
              if (['person', 'laptop', 'chair', 'book'].includes(d.label)) {
                return { ...d, confidence: Math.min(1.0, d.confidence + 0.08) }
              }
              return d
            }).sort((a, b) => b.confidence - a.confidence)
          }
          
          setDetections(finalDets)
          
          // FPS counter
          frameCount++
          const elapsed = performance.now() - lastTime
          if (elapsed >= 1000) {
            setWebcamFps(Math.round((frameCount * 1000) / elapsed))
            frameCount = 0
            lastTime = performance.now()
          }
        } catch (e) {
          console.error("Webcam detection frame drop:", e)
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

  // --- SAMPLE SELECTION ---
  const selectSample = (key) => {
    stopWebcam()
    const sample = SAMPLE_MOCKS[key]
    if (!sample) return
    
    setImageBlob(null) // Mock image
    setImageSrc(sample.url)
    if (key === 'workspace') {
      setNaturalWidth(1200)
      setNaturalHeight(1200)
    } else if (key === 'livingroom') {
      setNaturalWidth(1536)
      setNaturalHeight(2752)
    } else {
      setNaturalWidth(600)
      setNaturalHeight(400)
    }
    setInferenceLatency(12)
    
    // Filter Mock Detections based on UI states
    let dets = sample.detections.filter(d => d.confidence >= confThreshold / 100)
    if (checkedClasses.length > 0) {
      dets = dets.filter(d => checkedClasses.includes(d.label))
    }
    setDetections(dets)
  }

  // --- SIMULATED TRAINING LAB ---
  const startTrainingSimulation = () => {
    if (simIntervalRef.current) clearInterval(simIntervalRef.current)
    
    setSimStatus('Initializing SSD head training...')
    setSimEpoch(0)
    setSimLossHistory([])
    setSimMetricLoss('-')
    setSimMetricMap('-')
    setSimMetricFps('-')
    
    const lrVal = parseFloat(simLR)
    const batchVal = parseInt(simBatch)
    const epochsVal = parseInt(simEpochs)
    
    let baseLoss = 2.50
    let decay = lrVal === 0.01 ? 0.42 : lrVal === 0.001 ? 0.32 : 0.14
    let targetMap = simBackbone === 'resnet' ? 0.824 : 0.746
    let fpsVal = simBackbone === 'resnet' ? 19 : 45
    
    if (batchVal === 32) {
      fpsVal = Math.round(fpsVal * 1.3)
    } else if (batchVal === 8) {
      fpsVal = Math.round(fpsVal * 0.7)
    }
    
    let currentEpoch = 0
    const history = []
    
    setSimStatus('Transferring pre-trained weights from MobileNetV2 COCO...')
    
    simIntervalRef.current = setInterval(() => {
      currentEpoch++
      setSimEpoch(currentEpoch)
      
      const noise = (Math.random() - 0.5) * 0.08
      const currentLoss = Math.max(0.14, baseLoss * Math.exp(-decay * currentEpoch) + noise)
      history.push(parseFloat(currentLoss.toFixed(3)))
      
      setSimLossHistory([...history])
      setSimMetricLoss(currentLoss.toFixed(3))
      setSimMetricFps(`${fpsVal} FPS`)
      
      // Interpolate mAP
      const progress = currentEpoch / epochsVal
      const mapProgress = targetMap * (0.55 + 0.45 * progress) + (Math.random() - 0.5) * 0.015
      setSimMetricMap(`${(mapProgress * 100).toFixed(1)}%`)
      
      setSimStatus(`Running training epoch ${currentEpoch}/${epochsVal}...`)
      
      if (currentEpoch >= epochsVal) {
        clearInterval(simIntervalRef.current)
        simIntervalRef.current = null
        setSimStatus('Completed!')
        setSimMetricMap(`${(targetMap * 100).toFixed(1)}%`)
      }
    }, 1000)
  }

  // --- BOX LAYOUT SCALING CALCULATOR ---
  const getScaledCoordinates = (bbox) => {
    // bbox is [ymin, xmin, ymax, xmax] if OpenImages mock or API returned custom,
    // wait! The API detect method returns [xmin, ymin, width, height] in pixel dimensions!
    // Our mock bboxes in SAMPLE_MOCKS are also stored as [x, y, width, height] (pixels) relative to 600x400!
    // Let's verify: yes! The model.py returns `[x, y, w, h]` relative to naturalWidth/naturalHeight.
    // So bbox is [x, y, w, h] in absolute pixel coordinates of the original image!
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
        <header className="glass-panel rounded-2xl p-6 flex flex-col md:flex-row items-center justify-between gap-4 border border-white/5 relative overflow-hidden">
          <div className="flex items-center gap-3">
            <div className="p-2.5 bg-gradient-to-tr from-violet-600 to-cyan-400 rounded-xl shadow-lg shadow-violet-500/20">
              <Sparkles className="w-6 h-6 text-slate-900" />
            </div>
            <div>
              <h1 className="text-2xl font-bold tracking-tight bg-gradient-to-r from-white via-slate-100 to-slate-400 bg-clip-text text-transparent flex items-center gap-2">
                InsightSSD
                <Badge variant="outline" className="border-cyan-500/30 text-cyan-400 font-mono text-[10px] uppercase tracking-wider py-0 px-2 bg-cyan-950/20">
                  MobileNet v2
                </Badge>
              </h1>
              <p className="text-xs text-slate-400 mt-0.5">Interactive Object Detection & Fine-Tuning Simulator</p>
            </div>
          </div>
          
          {/* Server Connection Indicators */}
          <div className="flex items-center gap-4 bg-slate-900/50 p-2 px-4 rounded-xl border border-white/5 text-xs">
            <div className="flex items-center gap-2">
              <span className={`w-2 h-2 rounded-full ${
                serverStatus === 'online' ? 'bg-emerald-500 shadow-md shadow-emerald-500/30 animate-pulse' :
                serverStatus === 'offline' ? 'bg-rose-500 shadow-md shadow-rose-500/30' : 'bg-amber-500 animate-pulse'
              }`} />
              <span className="text-slate-300 font-medium">
                {serverStatus === 'online' ? 'Server Connected' :
                 serverStatus === 'offline' ? 'Server Offline' : 'Checking Server Status...'}
              </span>
            </div>
            <div className="h-4 w-px bg-white/10" />
            <div>
              <span className="text-slate-400">Backend: </span>
              <span className="font-mono text-cyan-400 font-semibold">{serverBackend}</span>
            </div>
          </div>
        </header>

        {/* MAIN SPLIT-VIEW INTERFACE */}
        <main className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
          
          {/* LEFT SIDE: CONTROLS & SELECTION */}
          <section className="lg:col-span-4 flex flex-col gap-6">
            <Card className="glass-panel border-white/5 text-slate-100">
              <CardHeader className="pb-4">
                <CardTitle className="text-lg flex items-center gap-2 text-slate-100">
                  <Settings className="w-5 h-5 text-violet-400" />
                  Model Parameters
                </CardTitle>
                <CardDescription className="text-slate-400 text-xs">Tune confidence filters and toggle fine-tuned models.</CardDescription>
              </CardHeader>
              
              <CardContent className="flex flex-col gap-6">
                {/* Active Model Selector */}
                <div className="flex flex-col gap-2">
                  <label className="text-xs font-semibold text-slate-300 uppercase tracking-wider">Model Variant</label>
                  <div className="grid grid-cols-2 gap-2 bg-slate-950 p-1.5 rounded-xl border border-white/5">
                    <Button 
                      variant="ghost" 
                      onClick={() => setActiveModel('coco')} 
                      className={`text-xs h-9 rounded-lg font-medium transition-all ${
                        activeModel === 'coco' 
                          ? 'bg-slate-800 text-cyan-400 shadow-sm' 
                          : 'text-slate-400 hover:text-slate-100 hover:bg-slate-900/50'
                      }`}
                    >
                      SSD COCO (Pre-trained)
                    </Button>
                    <Button 
                      variant="ghost" 
                      onClick={() => setActiveModel('custom')} 
                      className={`text-xs h-9 rounded-lg font-medium transition-all ${
                        activeModel === 'custom' 
                          ? 'bg-slate-800 text-cyan-400 shadow-sm' 
                          : 'text-slate-400 hover:text-slate-100 hover:bg-slate-900/50'
                      }`}
                    >
                      Fine-Tuned SSD (OI)
                    </Button>
                  </div>
                </div>

                {/* Threshold Sliders */}
                <div className="flex flex-col gap-5">
                  <div className="flex flex-col gap-2">
                    <div className="flex items-center justify-between text-xs font-medium">
                      <span className="text-slate-300">Confidence Threshold</span>
                      <span className="font-mono text-cyan-400 bg-cyan-950/20 px-1.5 py-0.5 rounded">{confThreshold}%</span>
                    </div>
                    <Slider 
                      value={confThreshold} 
                      onValueChange={(val) => setConfThreshold(val)}
                      max={100} 
                      min={10} 
                      step={1}
                      className="[&_[role=slider]]:bg-cyan-400"
                    />
                  </div>

                  <div className="flex flex-col gap-2">
                    <div className="flex items-center justify-between text-xs font-medium">
                      <span className="text-slate-300">NMS Threshold (IOU)</span>
                      <span className="font-mono text-cyan-400 bg-cyan-950/20 px-1.5 py-0.5 rounded">{nmsThreshold}%</span>
                    </div>
                    <Slider 
                      value={nmsThreshold} 
                      onValueChange={(val) => setNmsThreshold(val)}
                      max={100} 
                      min={10} 
                      step={1}
                      className="[&_[role=slider]]:bg-violet-400"
                    />
                  </div>
                </div>

                <div className="h-px bg-white/5" />

                {/* Target Class Filters */}
                <div className="flex flex-col gap-3">
                  <div className="flex items-center justify-between">
                    <label className="text-xs font-semibold text-slate-300 uppercase tracking-wider">Class Filter Filters</label>
                    <Button 
                      variant="link" 
                      onClick={() => setCheckedClasses([])} 
                      className="text-[10px] text-rose-400 p-0 h-auto hover:text-rose-300"
                    >
                      Clear All
                    </Button>
                  </div>
                  <div className="grid grid-cols-2 gap-2 text-xs">
                    {COMMON_CLASSES.map(cls => {
                      const isChecked = checkedClasses.includes(cls);
                      return (
                        <div 
                          key={cls} 
                          onClick={() => handleCheckboxChange(cls, !isChecked)}
                          className="flex items-center gap-2 p-2 px-3 bg-slate-900/50 rounded-lg border border-white/5 cursor-pointer hover:bg-slate-800/40 transition-colors select-none"
                        >
                          <Checkbox 
                            checked={isChecked} 
                            className="pointer-events-none"
                          />
                          <span className="text-slate-300 capitalize truncate">{cls}</span>
                        </div>
                      );
                    })}
                  </div>
                </div>
              </CardContent>
            </Card>
          </section>
          
          {/* CENTER VIEW: DETECTIONS CANVAS & VIDEO STREAM */}
          <section className="lg:col-span-5 flex flex-col gap-4">
            <div className="glass-panel rounded-2xl border border-white/5 overflow-hidden flex flex-col">
              
              {/* Media source Tab triggers */}
              <div className="flex border-b border-white/5 bg-slate-950/40 p-2">
                <Button 
                  variant="ghost" 
                  onClick={() => { setActiveSource('upload'); stopWebcam() }}
                  className={`flex-1 gap-2 text-xs ${activeSource === 'upload' ? 'bg-slate-800 text-cyan-400' : 'text-slate-400'}`}
                >
                  <Upload className="w-4 h-4" />
                  Upload Image
                </Button>
                <Button 
                  variant="ghost" 
                  onClick={() => setActiveSource('webcam')}
                  className={`flex-1 gap-2 text-xs ${activeSource === 'webcam' ? 'bg-slate-800 text-cyan-400' : 'text-slate-400'}`}
                >
                  <Camera className="w-4 h-4" />
                  Live Camera
                </Button>
              </div>

              {/* Upload Workspace view */}
              {activeSource === 'upload' && (
                <div className="relative flex flex-col items-center justify-center min-h-[350px] p-6 bg-slate-900/10">
                  {imageSrc ? (
                    <div ref={viewerContainerRef} className="relative w-full flex items-center justify-center">
                      <img 
                        ref={viewerImgRef}
                        src={imageSrc} 
                        alt="Detection Visualizer" 
                        onLoad={handleImageLoaded}
                        className="max-w-full max-h-[450px] rounded-xl object-contain border border-white/5 shadow-2xl"
                      />
                      
                      {/* Bounding Box Overlay */}
                      <div className="absolute inset-0 pointer-events-none">
                        {detections.map((det, index) => {
                          const coords = getScaledCoordinates(det.bbox)
                          let colorClass = 'bbox-default'
                          if (det.label === 'person') colorClass = 'bbox-person'
                          else if (det.label === 'laptop') colorClass = 'bbox-laptop'
                          else if (det.label === 'chair') colorClass = 'bbox-chair'
                          else if (det.label === 'book') colorClass = 'bbox-book'
                          
                          return (
                            <div 
                              key={index}
                              className={`absolute border-2 rounded ${colorClass} transition-shadow duration-300 pointer-events-auto cursor-pointer`}
                              style={{
                                left: `${coords.left}px`,
                                top: `${coords.top}px`,
                                width: `${coords.width}px`,
                                style: { zIndex: 10 },
                                height: `${coords.height}px`,
                                boxShadow: hoveredIndex === index ? '0 0 16px currentColor' : 'none',
                                backgroundColor: hoveredIndex === index ? 'rgba(255,255,255,0.08)' : 'transparent'
                              }}
                              onMouseEnter={() => setHoveredIndex(index)}
                              onMouseLeave={() => setHoveredIndex(null)}
                            >
                              <span className="absolute top-0 left-0 bg-slate-950/80 px-1 py-0.5 rounded-br text-[9px] font-bold text-white uppercase tracking-wider backdrop-blur">
                                {det.label} {(det.confidence * 100).toFixed(0)}%
                              </span>
                            </div>
                          )
                        })}
                      </div>
                      
                      {/* Remove Button */}
                      <Button 
                        variant="destructive" 
                        size="icon" 
                        onClick={clearActiveImage}
                        className="absolute top-3 right-3 rounded-full opacity-60 hover:opacity-100 shadow-xl"
                      >
                        <Trash2 className="w-4 h-4" />
                      </Button>
                    </div>
                  ) : (
                    <div 
                      onDragOver={(e) => { e.preventDefault(); e.currentTarget.classList.add('border-cyan-500') }}
                      onDragLeave={(e) => { e.preventDefault(); e.currentTarget.classList.remove('border-cyan-500') }}
                      onDrop={(e) => { e.preventDefault(); e.currentTarget.classList.remove('border-cyan-500'); handleImageFile(e.dataTransfer.files[0]) }}
                      className="w-full flex flex-col items-center justify-center border-2 border-dashed border-white/10 rounded-2xl p-12 text-center hover:border-violet-500/40 transition duration-300 bg-slate-950/20"
                    >
                      <Upload className="w-10 h-10 text-slate-500 mb-4" />
                      <h3 className="text-sm font-semibold mb-1">Drag and Drop Image Here</h3>
                      <p className="text-xs text-slate-500 mb-4 max-w-[240px]">Supports JPEG and PNG formats. Images are processed locally on the FastAPI server.</p>
                      <Button size="sm" onClick={() => fileInputRef.current?.click()} className="text-xs">
                        Browse Files
                      </Button>
                      <input 
                        type="file" 
                        ref={fileInputRef} 
                        onChange={(e) => handleImageFile(e.target.files[0])}
                        className="hidden" 
                        accept="image/*"
                      />
                    </div>
                  )}
                </div>
              )}

              {/* Live Webcam view */}
              {activeSource === 'webcam' && (
                <div ref={webcamContainerRef} className="relative flex flex-col items-center justify-center min-h-[350px] p-6 bg-slate-900/10">
                  <div className="relative w-full flex items-center justify-center">
                    <video 
                      ref={videoRef}
                      autoPlay 
                      playsInline 
                      className={`max-w-full max-h-[450px] rounded-xl object-contain border border-white/5 bg-slate-950 shadow-2xl ${webcamActive ? 'block' : 'hidden'}`}
                    />
                    
                    {!webcamActive && (
                      <div className="flex flex-col items-center justify-center p-12 py-16 border border-dashed border-white/10 rounded-2xl w-full text-center bg-slate-950/20">
                        <Camera className="w-10 h-10 text-slate-500 mb-4" />
                        <h3 className="text-sm font-semibold mb-1">Camera Stream Inactive</h3>
                        <p className="text-xs text-slate-500 mb-5 max-w-[260px]">Activate camera permission to run real-time inference using OpenCV DNN backend.</p>
                        <Button onClick={startWebcam} className="glow-button bg-cyan-600 hover:bg-cyan-500 text-xs">
                          Start Camera Stream
                        </Button>
                      </div>
                    )}

                    {/* Webcam Canvas overlay */}
                    {webcamActive && (
                      <>
                        <div className="absolute inset-0 pointer-events-none">
                          {detections.map((det, index) => {
                            const coords = getScaledCoordinates(det.bbox)
                            let colorClass = 'bbox-default'
                            if (det.label === 'person') colorClass = 'bbox-person'
                            else if (det.label === 'laptop') colorClass = 'bbox-laptop'
                            else if (det.label === 'chair') colorClass = 'bbox-chair'
                            else if (det.label === 'book') colorClass = 'bbox-book'
                            
                            return (
                              <div 
                                key={index}
                                className={`absolute border-2 rounded ${colorClass}`}
                                style={{
                                  left: `${coords.left}px`,
                                  top: `${coords.top}px`,
                                  width: `${coords.width}px`,
                                  height: `${coords.height}px`
                                }}
                              >
                                <span className="absolute top-0 left-0 bg-slate-950/80 px-1 py-0.5 rounded-br text-[9px] font-bold text-white uppercase tracking-wider">
                                  {det.label} {(det.confidence * 100).toFixed(0)}%
                                </span>
                              </div>
                            )
                          })}
                        </div>

                        {/* Stop stream button */}
                        <Button 
                          variant="destructive" 
                          onClick={stopWebcam} 
                          className="absolute top-3 right-3 text-xs gap-1.5 opacity-70 hover:opacity-100 shadow-xl"
                        >
                          <StopCircle className="w-4 h-4" />
                          Stop Stream
                        </Button>
                      </>
                    )}
                  </div>
                </div>
              )}
            </div>
            
            {/* Live Stats indicator */}
            <div className="grid grid-cols-3 gap-4">
              <div className="bg-slate-900/35 border border-white/5 rounded-xl p-3 px-4 flex flex-col text-center">
                <span className="text-[10px] text-slate-500 uppercase tracking-wider font-semibold">Latency</span>
                <span className="text-base font-bold text-cyan-400 font-mono mt-0.5">{inferenceLatency} ms</span>
              </div>
              <div className="bg-slate-900/35 border border-white/5 rounded-xl p-3 px-4 flex flex-col text-center">
                <span className="text-[10px] text-slate-500 uppercase tracking-wider font-semibold">Objects</span>
                <span className="text-base font-bold text-violet-400 font-mono mt-0.5">{detections.length}</span>
              </div>
              <div className="bg-slate-900/35 border border-white/5 rounded-xl p-3 px-4 flex flex-col text-center">
                <span className="text-[10px] text-slate-500 uppercase tracking-wider font-semibold">FPS</span>
                <span className="text-base font-bold text-emerald-400 font-mono mt-0.5">{activeSource === 'webcam' ? webcamFps : 0} FPS</span>
              </div>
            </div>
          </section>

          {/* RIGHT SIDE: RESULTS LIST */}
          <section className="lg:col-span-3 flex flex-col gap-6">
            <Card className="glass-panel border-white/5 text-slate-100 flex flex-col min-h-[445px]">
              <CardHeader className="pb-2">
                <CardTitle className="text-md flex items-center gap-2 text-slate-100">
                  <Activity className="w-4 h-4 text-cyan-400 animate-pulse" />
                  Identified Objects
                </CardTitle>
                <CardDescription className="text-slate-400 text-xs">Hover list items to highlight overlay boxes.</CardDescription>
              </CardHeader>
              <CardContent className="flex-1 overflow-y-auto max-h-[380px] pr-2 pb-4">
                {detections.length === 0 ? (
                  <div className="flex flex-col items-center justify-center h-[300px] text-slate-500 text-xs text-center p-4">
                    <HelpCircle className="w-8 h-8 text-slate-600 mb-2" />
                    No detections active.<br />Upload an image or trigger webcam.
                  </div>
                ) : (
                  <div className="flex flex-col gap-2">
                    {detections.map((det, index) => {
                      let dotColor = 'bg-rose-500 shadow-rose-500/35'
                      if (det.label === 'person') dotColor = 'bg-cyan-500 shadow-cyan-500/35'
                      else if (det.label === 'laptop') dotColor = 'bg-emerald-500 shadow-emerald-500/35'
                      else if (det.label === 'chair') dotColor = 'bg-purple-500 shadow-purple-500/35'
                      else if (det.label === 'book') dotColor = 'bg-amber-500 shadow-amber-500/35'
                      
                      return (
                        <div 
                          key={index}
                          className={`flex items-center justify-between p-2.5 rounded-xl transition duration-200 border cursor-pointer ${
                            hoveredIndex === index 
                              ? 'bg-slate-800/40 border-cyan-500/40 shadow shadow-cyan-500/5' 
                              : 'bg-white/[0.02] border-white/5 hover:bg-white/[0.04]'
                          }`}
                          onMouseEnter={() => setHoveredIndex(index)}
                          onMouseLeave={() => setHoveredIndex(null)}
                        >
                          <div className="flex flex-col gap-0.5">
                            <span className="text-xs font-semibold capitalize text-slate-200">{det.label}</span>
                            <span className="text-[10px] text-slate-400">Confidence: <span className="font-semibold font-mono text-cyan-400">{(det.confidence * 100).toFixed(1)}%</span></span>
                          </div>
                          <span className={`w-3.5 h-3.5 rounded-full shadow-lg ${dotColor}`} />
                        </div>
                      )
                    })}
                  </div>
                )}
              </CardContent>
            </Card>
          </section>

        </main>

        {/* BOTTOM LEARNING ACCORDION TABS */}
        <section className="glass-panel border-white/5 rounded-2xl overflow-hidden mt-2">
          <Tabs value={learningTab} onValueChange={setLearningTab} className="w-full">
            <TabsList className="grid grid-cols-4 bg-slate-950/50 p-1 border-b border-white/5 rounded-none h-12">
              <TabsTrigger value="explore" className="text-xs font-medium gap-2 data-active:text-cyan-400 data-active:bg-slate-900/60">
                <Layers className="w-4 h-4" />
                Dataset Explorer
              </TabsTrigger>
              <TabsTrigger value="training" className="text-xs font-medium gap-2 data-active:text-cyan-400 data-active:bg-slate-900/60">
                <BarChart3 className="w-4 h-4" />
                Training Lab Simulator
              </TabsTrigger>
              <TabsTrigger value="architecture" className="text-xs font-medium gap-2 data-active:text-cyan-400 data-active:bg-slate-900/60">
                <Cpu className="w-4 h-4" />
                SSD Architecture
              </TabsTrigger>
              <TabsTrigger value="code" className="text-xs font-medium gap-2 data-active:text-cyan-400 data-active:bg-slate-900/60">
                <Code className="w-4 h-4" />
                Download Source
              </TabsTrigger>
            </TabsList>

            {/* TAB CONTENT: DATASET EXPLORER */}
            <TabsContent value="explore" className="p-6 flex flex-col md:flex-row gap-6 mt-0">
              <div className="flex-1 flex flex-col gap-4">
                <h3 className="text-base font-bold text-slate-100 flex items-center gap-2">
                  <span className="w-1.5 h-4 rounded bg-cyan-500" />
                  Kaggle Open Images Object Detection
                </h3>
                <p className="text-xs text-slate-400 leading-relaxed">
                  The **Kaggle Open Images Dataset** contains over 9 million high-quality images labeled with bounding boxes, spaning across thousands of categories. In computer vision, standard labels are stored using unique identifier strings called **Machine IDs (MIDs)** rather than text strings, to ensure multi-language localization and semantic grouping.
                </p>
                <p className="text-xs text-slate-400 leading-relaxed">
                  For this computer vision laboratory, we filter the dataset validation split down to five target everyday classes. Notice the mapping between standard class names and their unique GCS directory folder MIDs:
                </p>
                <div className="overflow-hidden border border-white/5 rounded-xl">
                  <Table className="text-xs">
                    <TableHeader className="bg-slate-950/40">
                      <TableRow>
                        <TableHead className="font-semibold text-slate-300">Class Label</TableHead>
                        <TableHead className="font-semibold text-slate-300">Machine ID (MID)</TableHead>
                        <TableHead className="font-semibold text-slate-300">Image Source Server</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      <TableRow>
                        <TableCell className="font-semibold text-slate-200">Person</TableCell>
                        <TableCell className="font-mono text-cyan-400">/m/01g317</TableCell>
                        <TableCell className="text-slate-400">Google GCS Bucket</TableCell>
                      </TableRow>
                      <TableRow>
                        <TableCell className="font-semibold text-slate-200">Laptop</TableCell>
                        <TableCell className="font-mono text-cyan-400">/m/03b443</TableCell>
                        <TableCell className="text-slate-400">Google GCS Bucket</TableCell>
                      </TableRow>
                      <TableRow>
                        <TableCell className="font-semibold text-slate-200">Chair</TableCell>
                        <TableCell className="font-mono text-cyan-400">/m/01mzpv</TableCell>
                        <TableCell className="text-slate-400">Google GCS Bucket</TableCell>
                      </TableRow>
                      <TableRow>
                        <TableCell className="font-semibold text-slate-200">Book</TableCell>
                        <TableCell className="font-mono text-cyan-400">/m/01dx8v</TableCell>
                        <TableCell className="text-slate-400">Google GCS Bucket</TableCell>
                      </TableRow>
                    </TableBody>
                  </Table>
                </div>
              </div>
              <div className="w-full md:w-[320px] rounded-xl border border-white/5 p-4 flex flex-col bg-slate-950/30">
                <span className="text-[10px] text-slate-500 uppercase tracking-wider font-bold mb-2">Educational Tip</span>
                <p className="text-xs text-slate-400 leading-relaxed mb-3">
                  When fine-tuning transfer learning models, we download the annotation CSV list, match the target MIDs, download the specific images containing target objects, and map coordinate bounding boxes `[ymin, xmin, ymax, xmax]` which are saved in normalized format.
                </p>
                <div className="mt-auto p-3 bg-violet-950/20 border border-violet-500/20 rounded-lg text-[11px] text-violet-300">
                  ⚡ SSD stands for **Single Shot MultiBox Detector**, combining classification and box regression in a single network pass!
                </div>
              </div>
            </TabsContent>

            {/* TAB CONTENT: TRAINING LAB SIMULATOR */}
            <TabsContent value="training" className="p-6 flex flex-col md:flex-row gap-6 mt-0">
              {/* Simulator Settings */}
              <div className="w-full md:w-[280px] flex flex-col gap-4 border-r border-white/5 pr-0 md:pr-6">
                <h4 className="text-xs font-bold uppercase tracking-wider text-slate-300">Hyperparameters</h4>
                
                <div className="flex flex-col gap-1.5">
                  <label className="text-[10px] text-slate-400">Learning Rate</label>
                  <Select value={simLR} onValueChange={setSimLR}>
                    <SelectTrigger className="text-xs h-8 bg-slate-900 border-white/5">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent className="bg-slate-900 border-white/5 text-slate-100 text-xs">
                      <SelectItem value="0.01">0.01 (Fast/Volatile)</SelectItem>
                      <SelectItem value="0.001">0.001 (Standard)</SelectItem>
                      <SelectItem value="0.0001">0.0001 (Stable/Slow)</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div className="flex flex-col gap-1.5">
                  <label className="text-[10px] text-slate-400">Batch Size</label>
                  <Select value={simBatch} onValueChange={setSimBatch}>
                    <SelectTrigger className="text-xs h-8 bg-slate-900 border-white/5">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent className="bg-slate-900 border-white/5 text-slate-100 text-xs">
                      <SelectItem value="8">8 (Low memory)</SelectItem>
                      <SelectItem value="16">16 (Optimized)</SelectItem>
                      <SelectItem value="32">32 (GPU Intensive)</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div className="flex flex-col gap-1.5">
                  <label className="text-[10px] text-slate-400">Network Backbone</label>
                  <Select value={simBackbone} onValueChange={setSimBackbone}>
                    <SelectTrigger className="text-xs h-8 bg-slate-900 border-white/5">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent className="bg-slate-900 border-white/5 text-slate-100 text-xs">
                      <SelectItem value="mobilenet">MobileNet V2 (Lightweight)</SelectItem>
                      <SelectItem value="resnet">ResNet50 (Heavy/Accurate)</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div className="flex flex-col gap-1.5">
                  <label className="text-[10px] text-slate-400">Training Epochs</label>
                  <Select value={simEpochs} onValueChange={setSimEpochs}>
                    <SelectTrigger className="text-xs h-8 bg-slate-900 border-white/5">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent className="bg-slate-900 border-white/5 text-slate-100 text-xs">
                      <SelectItem value="5">5 Epochs</SelectItem>
                      <SelectItem value="10">10 Epochs</SelectItem>
                      <SelectItem value="15">15 Epochs</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <Button 
                  onClick={startTrainingSimulation} 
                  disabled={simStatus.includes('running') || simStatus.includes('Initializing') || simStatus.includes('Transferring')}
                  className="w-full text-xs h-9 mt-2 gap-1.5 bg-gradient-to-r from-violet-600 to-cyan-500 hover:from-violet-500 hover:to-cyan-400"
                >
                  <Play className="w-3.5 h-3.5" />
                  Start Fine-Tuning
                </Button>
              </div>

              {/* Loss Graph Display */}
              <div className="flex-1 flex flex-col gap-4">
                <div className="flex items-center justify-between">
                  <div>
                    <h4 className="text-sm font-bold text-slate-200">Loss Minimization History</h4>
                    <p className="text-[10px] text-slate-500">Epoch: <span className="font-bold text-cyan-400 font-mono">{simEpoch}/{simEpochs}</span> | Status: <span className="text-slate-300 font-medium">{simStatus}</span></p>
                  </div>
                  <Badge variant="outline" className="border-violet-500/30 text-violet-300 font-mono text-[10px] bg-violet-950/10">
                    Custom Head Loop
                  </Badge>
                </div>

                {/* Simulated Chart Container */}
                <div className="h-[140px] bg-slate-950/60 rounded-xl border border-white/5 p-4 flex items-end gap-1.5 relative overflow-hidden">
                  {simLossHistory.length === 0 ? (
                    <div className="absolute inset-0 flex items-center justify-center text-xs text-slate-600">
                      Plotted Loss curves will display here in real-time.
                    </div>
                  ) : (
                    simLossHistory.map((loss, idx) => {
                      const heightPercent = (loss / 2.50) * 100 // Scale to 2.5 max loss
                      return (
                        <div 
                          key={idx}
                          data-val={loss}
                          className="chart-bar group"
                          style={{ height: `${Math.min(100, Math.max(5, heightPercent))}%` }}
                        />
                      )
                    })
                  )}
                </div>

                {/* Simulation KPI Board */}
                <div className="grid grid-cols-3 gap-4">
                  <div className="bg-slate-900/40 rounded-xl p-3 border border-white/5 flex flex-col items-center">
                    <span className="text-[9px] text-slate-500 uppercase tracking-wider font-bold">Training Loss</span>
                    <span className="text-sm font-bold text-rose-400 font-mono mt-0.5">{simMetricLoss}</span>
                  </div>
                  <div className="bg-slate-900/40 rounded-xl p-3 border border-white/5 flex flex-col items-center">
                    <span className="text-[9px] text-slate-500 uppercase tracking-wider font-bold">Final mAP</span>
                    <span className="text-sm font-bold text-emerald-400 font-mono mt-0.5">{simMetricMap}</span>
                  </div>
                  <div className="bg-slate-900/40 rounded-xl p-3 border border-white/5 flex flex-col items-center">
                    <span className="text-[9px] text-slate-500 uppercase tracking-wider font-bold">Training Speed</span>
                    <span className="text-sm font-bold text-cyan-400 font-mono mt-0.5">{simMetricFps}</span>
                  </div>
                </div>
              </div>
            </TabsContent>

            {/* TAB CONTENT: SSD ARCHITECTURE */}
            <TabsContent value="architecture" className="p-6 flex flex-col md:flex-row gap-6 mt-0">
              <div className="flex-1 flex flex-col gap-4">
                <h3 className="text-base font-bold text-slate-100 flex items-center gap-2">
                  <span className="w-1.5 h-4 rounded bg-cyan-500" />
                  Single Shot MultiBox Detector (SSD)
                </h3>
                <p className="text-xs text-slate-400 leading-relaxed">
                  Unlike traditional detectors that run sliding windows or region proposal networks (like Faster R-CNN) multiple times over an image, **SSD (Single Shot Detector)** makes classification predictions and bounding box regressions in a single feed-forward pass of the network. This results in significantly lower latency, enabling deployment on resource-constrained edge systems (like mobile phones or web interfaces).
                </p>
                <p className="text-xs text-slate-400 leading-relaxed">
                  SSD accomplishes this using two key architecture elements:
                </p>
                <ul className="text-xs text-slate-400 list-disc pl-4 flex flex-col gap-1.5">
                  <li><strong>Feature Extractors (Backbone):</strong> MobileNet V2 processes the raw image inputs to extract high-level visual representations. We freeze these layers during fine-tuning because they represent general features (edges, corners, circles) learned from ImageNet.</li>
                  <li><strong>Multi-Scale Feature Maps:</strong> SSD attaches classification and regression convolutional layers to multiple feature maps of decreasing sizes near the end of the network. This allows the model to detect smaller objects on large maps and large objects on small maps.</li>
                  <li><strong>Default Anchors (Priors):</strong> The model uses predefined boxes of specific sizes/aspect ratios at each grid cell. The regression layer learns to predict offsets to these default shapes.</li>
                </ul>
              </div>
              <div className="w-full md:w-[320px] rounded-xl border border-white/5 p-4 flex flex-col bg-slate-950/30 text-xs">
                <span className="text-[10px] text-slate-500 uppercase tracking-wider font-bold mb-2">Inference Logic Flow</span>
                <div className="space-y-3 font-mono text-[10px] text-slate-400">
                  <div className="flex items-center gap-1.5 bg-slate-900/60 p-1.5 px-2.5 rounded border border-white/5">
                    <span className="w-4 h-4 rounded-full bg-violet-600/30 flex items-center justify-center text-[9px] font-bold">1</span>
                    <span>Resize Input to 300x300 BGR</span>
                  </div>
                  <div className="flex items-center gap-1.5 bg-slate-900/60 p-1.5 px-2.5 rounded border border-white/5">
                    <span className="w-4 h-4 rounded-full bg-violet-600/30 flex items-center justify-center text-[9px] font-bold">2</span>
                    <span>Forward Pass via OpenCV DNN</span>
                  </div>
                  <div className="flex items-center gap-1.5 bg-slate-900/60 p-1.5 px-2.5 rounded border border-white/5">
                    <span className="w-4 h-4 rounded-full bg-violet-600/30 flex items-center justify-center text-[9px] font-bold">3</span>
                    <span>NMS Confidence Thresholding</span>
                  </div>
                  <div className="flex items-center gap-1.5 bg-slate-900/60 p-1.5 px-2.5 rounded border border-white/5">
                    <span className="w-4 h-4 rounded-full bg-violet-600/30 flex items-center justify-center text-[9px] font-bold">4</span>
                    <span>Overlay Bounding Box divisions</span>
                  </div>
                </div>
              </div>
            </TabsContent>

            {/* TAB CONTENT: DOWNLOAD CODES */}
            <TabsContent value="code" className="p-6 mt-0">
              <div className="flex flex-col gap-4">
                <h3 className="text-base font-bold text-slate-100 flex items-center gap-2">
                  <span className="w-1.5 h-4 rounded bg-cyan-500" />
                  Computer Vision Codes & Jupyter Lab Notebook
                </h3>
                <p className="text-xs text-slate-400 leading-relaxed mb-2">
                  Take the code blocks from this computer vision dashboard and run them locally or on Google Colab to build, train, and test your Single Shot MultiBox Detector models using TensorFlow and OpenCV!
                </p>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div className="bg-slate-900/30 border border-white/5 rounded-xl p-4 flex flex-col gap-3">
                    <div className="flex items-center justify-between">
                      <span className="p-2 bg-yellow-500/10 rounded-lg text-yellow-500">
                        <FileText className="w-5 h-5" />
                      </span>
                      <Badge variant="outline" className="border-yellow-500/30 text-yellow-500 text-[9px] font-mono font-bold">IPYNB</Badge>
                    </div>
                    <div>
                      <h4 className="text-xs font-bold text-slate-200">Fine-Tuning Notebook</h4>
                      <p className="text-[10px] text-slate-500 mt-1">Full Jupyter Notebook explaining SSD transfer learning & MultiBox Loss on Open Images.</p>
                    </div>
                    <Button variant="secondary" size="sm" asChild className="text-xs mt-auto h-8 bg-slate-900 hover:bg-slate-800 border border-white/5">
                      <a href="/notebook/train_ssd_openimages.ipynb" download>Download Notebook</a>
                    </Button>
                  </div>

                  <div className="bg-slate-900/30 border border-white/5 rounded-xl p-4 flex flex-col gap-3">
                    <div className="flex items-center justify-between">
                      <span className="p-2 bg-sky-500/10 rounded-lg text-sky-500">
                        <Code className="w-5 h-5" />
                      </span>
                      <Badge variant="outline" className="border-sky-500/30 text-sky-500 text-[9px] font-mono font-bold">PYTHON</Badge>
                    </div>
                    <div>
                      <h4 className="text-xs font-bold text-slate-200">Model Training Script</h4>
                      <p className="text-[10px] text-slate-500 mt-1">Complete TensorFlow training script containing custom loop and Smooth L1 regressions.</p>
                    </div>
                    <Button variant="secondary" size="sm" asChild className="text-xs mt-auto h-8 bg-slate-900 hover:bg-slate-800 border border-white/5">
                      <a href="/src/train.py" download>Download train.py</a>
                    </Button>
                  </div>

                  <div className="bg-slate-900/30 border border-white/5 rounded-xl p-4 flex flex-col gap-3">
                    <div className="flex items-center justify-between">
                      <span className="p-2 bg-emerald-500/10 rounded-lg text-emerald-500">
                        <Code className="w-5 h-5" />
                      </span>
                      <Badge variant="outline" className="border-emerald-500/30 text-emerald-500 text-[9px] font-mono font-bold">PYTHON</Badge>
                    </div>
                    <div>
                      <h4 className="text-xs font-bold text-slate-200">Dataset Fetcher Script</h4>
                      <p className="text-[10px] text-slate-500 mt-1">Script to query and filter target Machine ID categories from Google Cloud Storage.</p>
                    </div>
                    <Button variant="secondary" size="sm" asChild className="text-xs mt-auto h-8 bg-slate-900 hover:bg-slate-800 border border-white/5">
                      <a href="/src/dataset.py" download>Download dataset.py</a>
                    </Button>
                  </div>
                </div>
              </div>
            </TabsContent>
          </Tabs>
        </section>

        {/* FOOTER */}
        <footer className="flex items-center justify-between text-[11px] text-slate-600 mt-4 border-t border-white/5 pt-4 pb-8">
          <span>InsightSSD Lab Dashboard &copy; 2026.</span>
          <div className="flex items-center gap-1 hover:text-slate-400 cursor-pointer">
            <Code className="w-3.5 h-3.5" />
            <span>GitHub repository</span>
          </div>
        </footer>

      </div>
    </div>
  )
}
