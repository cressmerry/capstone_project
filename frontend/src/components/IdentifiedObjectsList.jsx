import React from 'react'
import { Activity, HelpCircle } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card"

export default function IdentifiedObjectsList({
  detections,
  hoveredIndex,
  setHoveredIndex
}) {
  return (
    <section className="lg:col-span-3 flex flex-col gap-6">
      <Card className="glass-panel border-white/5 text-slate-100 flex flex-col min-h-[445px]">
        <CardHeader className="pb-2">
          <CardTitle className="text-md flex items-center gap-2 text-slate-100">
            <Activity className="w-4 h-4 text-cyan-400 animate-pulse" />
            Identified Objects
          </CardTitle>
          <CardDescription className="text-slate-400 text-xs">
            Hover list items to highlight overlay boxes.
          </CardDescription>
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
                      <span className="text-[10px] text-slate-400">
                        Confidence:{' '}
                        <span className="font-semibold font-mono text-cyan-400">
                          {(det.confidence * 100).toFixed(1)}%
                        </span>
                      </span>
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
  )
}
