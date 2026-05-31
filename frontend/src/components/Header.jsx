import React from 'react'
import { Sparkles } from 'lucide-react'
import { Badge } from "@/components/ui/badge"

export default function Header({ serverStatus, serverBackend }) {
  return (
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
  )
}
