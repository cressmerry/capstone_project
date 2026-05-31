import React from 'react'
import { Cpu, FileText, Layers, Code } from 'lucide-react'
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"

export default function LearningTabs({ learningTab, setLearningTab }) {
  return (
    <section className="glass-panel border-white/5 rounded-2xl overflow-hidden mt-2">
      <Tabs value={learningTab} onValueChange={setLearningTab} className="w-full">
        <TabsList className="grid grid-cols-3 bg-slate-950/50 p-1 border-b border-white/5 rounded-none h-12">
          <TabsTrigger value="explore" className="text-xs font-medium gap-2 data-active:text-cyan-400 data-active:bg-slate-900/60">
            <Layers className="w-4 h-4" />
            Dataset Explorer
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
  )
}
