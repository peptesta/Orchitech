import { 
  Combine, Plus, PenLine, CheckSquare, Trash2, X 
} from "lucide-react";

interface EditorToolbarProps {
  isMergeMode: boolean;
  isDrawingMode: boolean;
  mergeSelectionCount: number;
  
  // Actions
  onStartDrawing: () => void;
  onStartMerge: () => void;
  onCancelMode: () => void;
  onBatchDelete: () => void;
  onMergeSelected: () => void;
}

export default function EditorToolbar({
  isMergeMode,
  isDrawingMode,
  mergeSelectionCount,
  onStartDrawing,
  onStartMerge,
  onCancelMode,
  onBatchDelete,
  onMergeSelected
}: EditorToolbarProps) {

  return (
    <div className="bg-white border border-stone-200 p-3 rounded-lg shadow-sm flex items-center justify-between gap-4 min-h-[64px]">
      
      {/* DEFAULT TOOLBAR */}
      {!isMergeMode && !isDrawingMode && (
        <div className="flex items-center gap-3 w-full justify-between">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-stone-100 rounded text-stone-500"><Combine size={20}/></div>
            <div className="hidden sm:block">
              <p className="text-sm font-bold text-stone-700">Actions</p>
              <p className="text-xs text-stone-400">Modify multiple boxes</p>
            </div>
          </div>
          <div className="flex gap-2">
            <button 
              onClick={onStartDrawing}
              className="px-4 py-2 bg-white border border-stone-200 text-stone-700 hover:bg-stone-50 rounded-lg text-sm font-medium transition flex items-center gap-2"
            >
              <Plus size={16} /> Draw Box
            </button>
            <button 
              onClick={onStartMerge}
              className="px-4 py-2 bg-indigo-50 text-indigo-700 hover:bg-indigo-100 rounded-lg text-sm font-medium transition"
            >
              Select Boxes
            </button>
          </div>
        </div>
      )}

      {/* DRAWING MODE TOOLBAR */}
      {isDrawingMode && (
        <div className="flex items-center gap-3 w-full">
          <div className="p-2 bg-blue-100 text-blue-600 rounded animate-pulse"><PenLine size={20}/></div>
          <div className="flex-1">
            <p className="text-sm font-bold text-blue-700">Drawing Mode</p>
            <p className="text-xs text-blue-400">Click and drag on the image to create boxes.</p>
          </div>
          <button onClick={onCancelMode} className="px-4 py-2 bg-stone-100 hover:bg-stone-200 text-stone-600 rounded-lg font-bold text-sm">
            Done
          </button>
        </div>
      )}

      {/* MERGE MODE TOOLBAR */}
      {isMergeMode && (
        <div className="flex items-center gap-3 w-full">
          <div className="p-2 bg-indigo-100 text-indigo-600 rounded animate-pulse"><CheckSquare size={20}/></div>
          <div className="flex-1">
            <p className="text-sm font-bold text-indigo-700">{mergeSelectionCount} Selected</p>
            <p className="text-xs text-indigo-400">Select boxes to apply actions.</p>
          </div>
          
          <div className="flex items-center gap-2">
            <button onClick={onBatchDelete} disabled={mergeSelectionCount === 0} className="px-3 py-2 bg-red-50 text-red-600 hover:bg-red-100 rounded-lg text-xs font-bold uppercase disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2">
              <Trash2 size={14} /> Delete
            </button>
            <button onClick={onMergeSelected} disabled={mergeSelectionCount < 2} className="px-3 py-2 bg-indigo-600 text-white hover:bg-indigo-700 rounded-lg text-xs font-bold uppercase disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2">
              <Combine size={14} /> Merge
            </button>
            <button onClick={onCancelMode} className="p-2 hover:bg-stone-100 text-stone-500 rounded-lg">
              <X size={16} />
            </button>
          </div>
        </div>
      )}
    </div>
  );
}