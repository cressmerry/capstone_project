import React from 'react'
import { Upload, Camera, Trash2, StopCircle } from 'lucide-react'
import { Button } from "@/components/ui/button"

export default function MediaVisualizer({
  activeSource,
  setActiveSource,
  imageSrc,
  detections,
  hoveredIndex,
  setHoveredIndex,
  webcamActive,
  webcamFps,
  inferenceLatency,
  startWebcam,
  stopWebcam,
  handleImageFile,
  handleImageLoaded,
  clearActiveImage,
  fileInputRef,
  viewerContainerRef,
  viewerImgRef,
  videoRef,
  webcamContainerRef,
  getScaledCoordinates
}) {
  return (
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
                          height: `${coords.height}px`,
                          zIndex: 10,
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
  )
}
