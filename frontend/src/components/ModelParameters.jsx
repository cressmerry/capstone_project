import React from 'react'
import { Settings } from 'lucide-react'
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card"
import { Slider } from "@/components/ui/slider"
import { Checkbox } from "@/components/ui/checkbox"

const COMMON_CLASSES = ['person', 'laptop', 'chair', 'book', 'cell phone', 'cup', 'dining table', 'backpack', 'bottle', 'handbag', 'keyboard', 'mouse']

export default function ModelParameters({
  activeModel,
  setActiveModel,
  confThreshold,
  setConfThreshold,
  nmsThreshold,
  setNmsThreshold,
  checkedClasses,
  handleCheckboxChange,
  setCheckedClasses
}) {
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
  )
}
