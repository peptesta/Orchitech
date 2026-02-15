import { 
  AlertTriangle, ArrowRight, ChevronDown, ChevronRight, 
  Hash, ScanLine, Binary, PenLine, RefreshCcw, Trash2 
} from "lucide-react";
import { useState } from "react";
import { getScoreColor, Conflict } from "@/utils/editorUtils";

interface EditorSidebarProps {
  boxes: number[][];
  scores: number[];
  eliminated: boolean[];
  modified: boolean[];
  isManual: boolean[];
  activeBoxIndex: number | null;
  mergeSelection: number[];
  isMergeMode: boolean;
  suggestions: Conflict[];
  onToggleBox: (index: number) => void;
  onToggleElimination: (e: React.MouseEvent, index: number) => void;
  onApplySuggestion: (indices: number[]) => void;
  onClearSelection: () => void;
  onStartDrawing: () => void;
}

export default function EditorSidebar({
  boxes,
  scores,
  eliminated,
  modified,
  isManual,
  activeBoxIndex,
  mergeSelection,
  isMergeMode,
  suggestions,
  onToggleBox,
  onToggleElimination,
  onApplySuggestion,
  onClearSelection,
  onStartDrawing
}: EditorSidebarProps) {
  
  // UI State moved from parent
  const [expandedItems, setExpandedItems] = useState<number[]>([]);
  const [isSuggestionsOpen, setIsSuggestionsOpen] = useState(false);

  const toggleItem = (index: number, e: React.MouseEvent) => {
    e.stopPropagation();
    setExpandedItems(prev => 
      prev.includes(index) ? prev.filter(i => i !== index) : [...prev, index]
    );
  };

  const activeCount = boxes.filter((_, i) => !eliminated[i]).length;

  return (
    <div className="lg:w-[320px] flex flex-col gap-4 flex-shrink-0">
      
      {/* SUGGESTIONS DROPDOWN */}
      {suggestions.length > 0 && (
        <div className="border border-amber-200 rounded-xl overflow-hidden bg-amber-50 animate-in fade-in slide-in-from-right-4 shadow-sm">
          <button 
            onClick={() => setIsSuggestionsOpen(!isSuggestionsOpen)}
            className="w-full flex items-center justify-between p-4 bg-amber-100/50 hover:bg-amber-100 transition text-amber-900"
          >
            <div className="flex items-center gap-2">
              <AlertTriangle size={16} className="fill-amber-500 text-white"/>
              <span className="font-bold text-sm uppercase tracking-wide">
                Suggestions ({suggestions.length})
              </span>
            </div>
            <ChevronDown 
              size={18} 
              className={`transition-transform duration-200 text-amber-600 ${isSuggestionsOpen ? 'rotate-180' : ''}`} 
            />
          </button>

          {isSuggestionsOpen && (
            <div className="p-3 space-y-2 max-h-[200px] overflow-y-auto custom-scrollbar border-t border-amber-200/50">
              {suggestions.map((s, i) => (
                <div key={i} className="bg-white/80 p-2.5 rounded-lg border border-amber-100 text-xs flex flex-col gap-2 shadow-sm">
                  <span className="font-medium text-stone-700">{s.message}</span>
                  <div className="flex items-center justify-between">
                    <span className={`px-1.5 py-0.5 rounded text-[10px] uppercase font-bold ${s.type === 'containment' ? 'bg-red-100 text-red-600' : 'bg-orange-100 text-orange-600'}`}>
                      {s.type}
                    </span>
                    <button 
                      onClick={() => onApplySuggestion(s.indices)}
                      className="flex items-center gap-1 text-indigo-600 font-bold hover:underline"
                    >
                      Review <ArrowRight size={10} />
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* STATS */}
      <div className="bg-white p-4 rounded-xl shadow-sm border border-stone-200">
        <div className="flex items-center gap-3 mb-1">
          <div className="p-2 bg-emerald-50 rounded-lg text-emerald-600"><Hash size={20} /></div>
          <div>
            <p className="text-xs text-stone-500 font-bold uppercase">Detections</p>
            <div className="flex items-baseline gap-2">
              <p className="text-xl font-extrabold text-stone-800">
                {activeCount}
                <span className="text-sm font-normal text-stone-400 ml-1">/ {boxes.length}</span>
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* BOX LIST */}
      {boxes.length > 0 ? (
        <div className="bg-white p-4 rounded-xl shadow-sm border border-stone-200 flex-1 overflow-hidden flex flex-col h-[400px] lg:h-auto">
          <h4 className="text-sm font-bold text-stone-700 mb-3 border-b border-stone-100 pb-2 flex justify-between items-center">
            <span>Detailed Results</span>
            {activeBoxIndex !== null && (
              <button onClick={onClearSelection} className="text-[10px] text-emerald-600 bg-emerald-50 px-2 py-1 rounded cursor-pointer hover:bg-emerald-100">
                Show All
              </button>
            )}
          </h4>
          <div className="overflow-y-auto pr-2 space-y-3 custom-scrollbar flex-1">
            {scores.map((score, idx) => {
              const isElim = eliminated[idx];
              const isMan = isManual[idx];
              const color = isElim ? '#9ca3af' : getScoreColor(score, isMan);
              const isActive = activeBoxIndex === idx;
              const isSelectedForMerge = mergeSelection.includes(idx);
              const isDimmed = activeBoxIndex !== null && !isActive;
              const currentBox = boxes[idx];
              const isMod = modified[idx];
              const isExpanded = expandedItems.includes(idx);

              return (
                <div 
                  key={idx} 
                  onClick={() => onToggleBox(idx)} 
                  className={`p-3 rounded-lg border transition relative cursor-pointer
                    ${isMergeMode && isSelectedForMerge ? 'bg-indigo-50 border-indigo-500 ring-1 ring-indigo-500' : ''}
                    ${!isMergeMode && isActive ? 'bg-emerald-50 border-emerald-500 ring-1 ring-emerald-500 shadow-sm' : ''} 
                    ${!isActive && !isSelectedForMerge ? 'bg-stone-50 border-stone-100 hover:border-emerald-200' : ''}
                    ${isDimmed && !isMergeMode ? 'opacity-40 grayscale-[0.5]' : ''}
                    ${isElim && !isActive ? 'opacity-60 bg-stone-100 border-stone-200' : ''}
                  `}
                >
                  {/* ITEM HEADER */}
                  <div className="flex justify-between items-center mb-1">
                    <div className="flex items-center gap-2">
                      <span className={`text-[10px] font-bold text-white px-1.5 rounded ${isElim ? 'line-through bg-stone-400' : ''}`} style={{ backgroundColor: isElim ? undefined : color }}>
                        #{idx + 1}
                      </span>
                      {isElim ? (
                        <span className="text-[9px] font-bold text-red-500 uppercase tracking-wider border border-red-200 bg-red-50 px-1 rounded">Eliminated</span>
                      ) : (
                        <div className="flex items-center gap-2">
                          {isMan ? (
                            <span className="text-xs font-bold text-blue-600 bg-blue-50 px-1.5 py-0.5 rounded border border-blue-100">Manual</span>
                          ) : (
                            <span className="text-xs font-mono font-bold" style={{ color: color }}>{(score * 100).toFixed(2)}%</span>
                          )}
                          {isMod && !isMan && <div className="text-[9px] bg-amber-100 text-amber-700 px-1.5 py-0.5 rounded-full border border-amber-200 font-medium"><PenLine size={8} /></div>}
                        </div>
                      )}
                    </div>
                    
                    <div className="flex items-center gap-1">
                      <button onClick={(e) => onToggleElimination(e, idx)} className={`p-1 rounded hover:bg-stone-200 ${isElim ? 'text-emerald-600' : 'text-stone-400 hover:text-red-500'}`}>
                        {isElim ? <RefreshCcw size={12} /> : <Trash2 size={12} />}
                      </button>
                      <button onClick={(e) => toggleItem(idx, e)} className="p-1 text-stone-400 hover:text-stone-600">
                        {isExpanded ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
                      </button>
                    </div>
                  </div>
                  
                  {/* COORDINATES */}
                  {isExpanded && !isElim && (
                    <div className="mt-2 pt-2 border-t border-stone-200/50 animate-in slide-in-from-top-1 fade-in duration-200">
                      <div className="text-[10px] text-stone-500 font-mono flex flex-col gap-1">
                        <div className="flex items-center gap-2 mb-1"><ScanLine size={12} className="text-stone-400" /><span className="font-semibold text-stone-600">Box Coordinates:</span></div>
                        {currentBox ? (
                          <div className="grid grid-cols-2 gap-x-2 gap-y-1 pl-1 bg-white/50 p-2 rounded border border-stone-100">
                            <span>x1: {Math.round(currentBox[0])}</span><span>y1: {Math.round(currentBox[1])}</span>
                            <span>x2: {Math.round(currentBox[2])}</span><span>y2: {Math.round(currentBox[3])}</span>
                          </div>
                        ) : <span className="pl-1 text-stone-400 italic">No box data</span>}
                      </div>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      ) : (
        <div className="bg-white p-8 rounded-xl border border-stone-200 text-center text-stone-400 flex flex-col items-center justify-center h-full">
          <Binary size={32} className="mb-2 opacity-50" />
          <p className="text-sm">No analysis data available</p>
          <button onClick={onStartDrawing} className="mt-4 text-xs bg-stone-100 hover:bg-stone-200 px-3 py-2 rounded-lg text-stone-600 font-medium transition">
            Start drawing
          </button>
        </div>
      )}
    </div>
  );
}