import React from 'react'
import { Settings } from 'lucide-react'
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card"
import { Slider } from "@/components/ui/slider"
import { Checkbox } from "@/components/ui/checkbox"

const COCO_CLASSES = [
  'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
  'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
  'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
  'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
  'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
  'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
  'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
  'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
  'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

const CUSTOM_CLASSES = ['person', 'laptop', 'chair', 'book', 'dining table', 'cell phone', 'cup', 'keyboard', 'mouse']

export default function ModelParameters({
  activeModel,
  setActiveModel,
  confThreshold,
  setConfThreshold,
  nmsThreshold,
  setNmsThreshold,
  checkedClasses,
  handleCheckboxChange,
  setCheckedClasses,
  hasCustomModel = true
}) {
  const classesToDisplay = activeModel === 'custom' ? CUSTOM_CLASSES : COCO_CLASSES;
  return (
    <section className="lg:col-span-4 flex flex-col gap-6">
      <Card className="glass-panel border-white/5 text-slate-100">
        <CardHeader className="pb-4">
          <CardTitle className="text-lg flex items-center gap-2 text-slate-100">
            <Settings className="w-5 h-5 text-violet-400" />
            Model Parameters
          </CardTitle>
          <CardDescription className="text-slate-400 text-xs">
            Tune confidence filters and toggle fine-tuned models.
          </CardDescription>
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
                disabled={!hasCustomModel}
                className={`text-xs h-9 rounded-lg font-medium transition-all ${
                  activeModel === 'custom' 
                    ? 'bg-slate-800 text-cyan-400 shadow-sm' 
                    : 'text-slate-400 hover:text-slate-100 hover:bg-slate-900/50'
                } ${!hasCustomModel ? 'opacity-40 cursor-not-allowed pointer-events-none' : ''}`}
              >
                Fine-Tuned SSD (OI)
              </Button>
            </div>
            {!hasCustomModel && (
              <div className="mt-1.5 p-3 bg-amber-950/25 border border-amber-500/20 rounded-xl text-amber-300 text-[11px] leading-relaxed flex flex-col gap-1 shadow-sm">
                <div className="font-semibold flex items-center gap-1.5 text-amber-400 text-xs">
                  <span className="inline-block w-1.5 h-1.5 rounded-full bg-amber-400 animate-pulse" />
                  Fine-Tuned Model Offline
                </div>
                <span>The custom weights file <code>ssd_fine_tuned_head.weights.h5</code> is missing on the server. The backend will fall back to the base COCO model. Run <code>python src/train.py</code> to train it.</span>
              </div>
            )}
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
                onValueChange={(val) => {
                  if (val && typeof val[0] === 'number' && !isNaN(val[0])) {
                    setConfThreshold(val[0]);
                  }
                }}
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
                onValueChange={(val) => {
                  if (val && typeof val[0] === 'number' && !isNaN(val[0])) {
                    setNmsThreshold(val[0]);
                  }
                }}
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
              <label className="text-xs font-semibold text-slate-300 uppercase tracking-wider">Class Filters</label>
              <div className="flex gap-2">
                <Button 
                  variant="link" 
                  onClick={() => setCheckedClasses(classesToDisplay)} 
                  className="text-[10px] text-cyan-400 p-0 h-auto hover:text-cyan-300"
                >
                  Select All
                </Button>
                <span className="text-[10px] text-slate-500">|</span>
                <Button 
                  variant="link" 
                  onClick={() => setCheckedClasses([])} 
                  className="text-[10px] text-rose-400 p-0 h-auto hover:text-rose-300"
                >
                  Clear All
                </Button>
              </div>
            </div>
            <div className="grid grid-cols-2 gap-2 text-xs max-h-[220px] overflow-y-auto pr-1">
              {classesToDisplay.map(cls => {
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
  )
}
