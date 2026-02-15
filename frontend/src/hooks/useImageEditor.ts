/**
 * useImageEditor Hook
 * * Acts as the central "Controller" for the Image Editor modal. 
 * It separates complex logic from the UI components.
 * * Key Responsibilities:
 * 1. State Management: Maintains local copies of detection data (boxes, scores) and history for resets.
 * 2. Canvas Interaction: Handles heavy math for coordinate mapping, resizing, and drawing new boxes.
 * 3. Tool Logic: Implements business logic for merging boxes, batch deletion, and conflict detection.
 * 4. Persistence: Prepares and formats the final data structure for saving back to the parent.
 */

import { useState, useRef, useEffect, useCallback } from "react";
import { ImageFile } from "@/types"; 
import { detectConflicts, Conflict } from "@/utils/editorUtils";

interface ExtendedAnalysis {
    boxes: number[][];
    scores: number[];
    labels: string[];
    count: number;
    modified?: boolean[];
    eliminated?: boolean[];
    isManual?: boolean[]; 
}

// --- GLOBAL CACHE ---
const originalStateCache = new Map<string, { 
  boxes: number[][], 
  modified: boolean[],
  eliminated: boolean[],
  isManual: boolean[],
  scores: number[]
}>();

export function useImageEditor(
    selectedImage: ImageFile, 
    onSaveParent: (updatedImage: ImageFile) => void
) {
  const currentImageKey = useRef(selectedImage.url);
  const analysis = selectedImage.analysis as ExtendedAnalysis | undefined;
  const imgElementRef = useRef<HTMLImageElement>(null);

  // --- INITIALIZATION HELPER ---
  const getCachedOriginals = (img: ImageFile) => {
    const imgAnalysis = img.analysis as ExtendedAnalysis | undefined;
    if (!originalStateCache.has(img.url)) {
      const count = imgAnalysis?.count || 0;
      originalStateCache.set(img.url, {
        boxes: imgAnalysis?.boxes.map(b => [...b]) || [],
        modified: imgAnalysis?.modified ? [...imgAnalysis.modified] : new Array(count).fill(false),
        eliminated: imgAnalysis?.eliminated ? [...imgAnalysis.eliminated] : new Array(count).fill(false),
        isManual: imgAnalysis?.isManual ? [...imgAnalysis.isManual] : new Array(count).fill(false),
        scores: imgAnalysis?.scores ? [...imgAnalysis.scores] : []
      });
    }
    return originalStateCache.get(img.url)!;
  };

  const defaults = getCachedOriginals(selectedImage);

  // --- STATE ---
  const [originalBoxesSnapshot, setOriginalBoxesSnapshot] = useState<number[][]>(defaults.boxes);
  const [originalModifiedSnapshot, setOriginalModifiedSnapshot] = useState<boolean[]>(defaults.modified);
  const [originalEliminatedSnapshot, setOriginalEliminatedSnapshot] = useState<boolean[]>(defaults.eliminated);
  const [originalIsManualSnapshot, setOriginalIsManualSnapshot] = useState<boolean[]>(defaults.isManual);
  const [originalScoresSnapshot, setOriginalScoresSnapshot] = useState<number[]>(defaults.scores);

  const [localBoxes, setLocalBoxes] = useState<number[][]>(() => analysis?.boxes.map(b => [...b]) || []);
  const [localModified, setLocalModified] = useState<boolean[]>(() => analysis?.modified ? [...analysis.modified] : new Array(analysis?.count || 0).fill(false));
  const [localEliminated, setLocalEliminated] = useState<boolean[]>(() => analysis?.eliminated ? [...analysis.eliminated] : new Array(analysis?.count || 0).fill(false));
  const [localIsManual, setLocalIsManual] = useState<boolean[]>(() => analysis?.isManual ? [...analysis.isManual] : new Array(analysis?.count || 0).fill(false));
  const [localScores, setLocalScores] = useState<number[]>(() => analysis?.scores ? [...analysis.scores] : []);

  // UI States
  const [activeBoxIndex, setActiveBoxIndex] = useState<number | null>(null);
  const [suggestions, setSuggestions] = useState<Conflict[]>([]); 
  
  // Modes
  const [isMergeMode, setIsMergeMode] = useState(false);
  const [mergeSelection, setMergeSelection] = useState<number[]>([]); 
  const [isDrawingMode, setIsDrawingMode] = useState(false);
  const [currentDrawingBox, setCurrentDrawingBox] = useState<number[] | null>(null);
  const drawingStartRef = useRef<{x: number, y: number} | null>(null);

  // Image & Resizing
  const [imgNaturalSize, setImgNaturalSize] = useState<{w: number, h: number} | null>(null);
  const [isResizing, setIsResizing] = useState(false);
  const [isSaving, setIsSaving] = useState(false); 

  const resizeRef = useRef<{ handle: string } | null>(null); 

  // --- IMAGE SWITCHING EFFECT ---
  if (selectedImage.url !== currentImageKey.current) {
    currentImageKey.current = selectedImage.url;
    const originals = getCachedOriginals(selectedImage);
    setOriginalBoxesSnapshot(originals.boxes);
    setOriginalModifiedSnapshot(originals.modified);
    setOriginalEliminatedSnapshot(originals.eliminated);
    setOriginalIsManualSnapshot(originals.isManual);
    setOriginalScoresSnapshot(originals.scores);

    const currentAnalysis = selectedImage.analysis as ExtendedAnalysis | undefined;
    const count = currentAnalysis?.count || 0;

    setLocalBoxes(currentAnalysis?.boxes.map(b => [...b]) || []);
    setLocalModified(currentAnalysis?.modified ? [...currentAnalysis.modified] : new Array(count).fill(false));
    setLocalEliminated(currentAnalysis?.eliminated ? [...currentAnalysis.eliminated] : new Array(count).fill(false));
    setLocalIsManual(currentAnalysis?.isManual ? [...currentAnalysis.isManual] : new Array(count).fill(false));
    setLocalScores(currentAnalysis?.scores ? [...currentAnalysis.scores] : []);
    
    setImgNaturalSize(null);
    setActiveBoxIndex(null);
    setIsMergeMode(false);
    setIsDrawingMode(false);
    setMergeSelection([]);
  }

  // --- CONFLICT DETECTION ---
  useEffect(() => {
    if (isResizing || currentDrawingBox) return;
    const conflicts = detectConflicts(localBoxes, localEliminated);
    setSuggestions(conflicts);
  }, [localBoxes, localEliminated, isResizing, currentDrawingBox]);


  // --- HANDLERS ---

  const toggleBox = (index: number) => {
    if (isMergeMode) {
      if (localEliminated[index]) return; 
      setMergeSelection(prev => {
        if (prev.includes(index)) return prev.filter(i => i !== index);
        return [...prev, index];
      });
    } else {
      setActiveBoxIndex((prev) => (prev === index ? null : index));
    }
  };

  const handleApplySuggestion = (indices: number[]) => {
      setIsMergeMode(true);
      setMergeSelection(indices);
      setActiveBoxIndex(null);
  };

  const handleMergeSelected = () => {
    if (mergeSelection.length < 2) return;
    const boxesToMerge = mergeSelection.map(idx => localBoxes[idx]);
    const x1 = Math.min(...boxesToMerge.map(b => b[0]));
    const y1 = Math.min(...boxesToMerge.map(b => b[1]));
    const x2 = Math.max(...boxesToMerge.map(b => b[2]));
    const y2 = Math.max(...boxesToMerge.map(b => b[3]));
    const masterIndex = mergeSelection[0];
    const indexesToEliminate = mergeSelection.slice(1);

    setLocalBoxes(prev => { const next = [...prev]; next[masterIndex] = [x1, y1, x2, y2]; return next; });
    setLocalModified(prev => { const next = [...prev]; next[masterIndex] = true; return next; });
    setLocalEliminated(prev => { const next = [...prev]; indexesToEliminate.forEach(idx => next[idx] = true); return next; });
    setMergeSelection([]);
    setIsMergeMode(false);
  };

  const handleBatchDelete = () => {
    if (mergeSelection.length === 0) return;
    setLocalEliminated(prev => { const next = [...prev]; mergeSelection.forEach(idx => next[idx] = true); return next; });
    setMergeSelection([]);
    setIsMergeMode(false);
  };

  const handleEliminateAll = () => {
    if (confirm("Are you sure you want to eliminate ALL boxes for this image?")) {
        setLocalEliminated(new Array(localBoxes.length).fill(true));
        setMergeSelection([]);
        setIsMergeMode(false);
        setActiveBoxIndex(null);
    }
  };

  const toggleElimination = (e: React.MouseEvent, index: number) => {
    e.stopPropagation(); 
    const willBeEliminated = !localEliminated[index];
    setLocalEliminated(prev => { const next = [...prev]; next[index] = !next[index]; return next; });
    if (willBeEliminated && activeBoxIndex === index) setActiveBoxIndex(null);
  };

  const handleResizeStart = (e: React.MouseEvent, index: number, handle: string) => {
    e.stopPropagation(); 
    e.preventDefault(); 
    if (localEliminated[index]) return; 
    
    if (activeBoxIndex !== index) setActiveBoxIndex(index);
    setIsResizing(true);
    resizeRef.current = { handle };
  };

  const handleContainerMouseMove = useCallback((e: React.MouseEvent) => {
    if (!imgElementRef.current || !imgNaturalSize) return;

    const rect = imgElementRef.current.getBoundingClientRect();
    const scaleX = imgNaturalSize.w / rect.width;
    const scaleY = imgNaturalSize.h / rect.height;
    const relativeX = (e.clientX - rect.left) * scaleX;
    const relativeY = (e.clientY - rect.top) * scaleY;
    const curX = Math.max(0, Math.min(imgNaturalSize.w, relativeX));
    const curY = Math.max(0, Math.min(imgNaturalSize.h, relativeY));

    // Resize
    if (isResizing && resizeRef.current && activeBoxIndex !== null) {
        const { handle } = resizeRef.current;
        const currentBox = [...localBoxes[activeBoxIndex]];
        const newBox = [...currentBox];
        if (handle.includes('w')) newBox[0] = Math.min(curX, currentBox[2] - 5); 
        if (handle.includes('e')) newBox[2] = Math.max(curX, currentBox[0] + 5);
        if (handle.includes('n')) newBox[1] = Math.min(curY, currentBox[3] - 5); 
        if (handle.includes('s')) newBox[3] = Math.max(curY, currentBox[1] + 5);

        setLocalBoxes(prev => { const next = [...prev]; next[activeBoxIndex] = newBox; return next; });
        setLocalModified(prev => { const next = [...prev]; next[activeBoxIndex] = true; return next; });
    }

    // Draw
    if (isDrawingMode && drawingStartRef.current) {
        const startX = drawingStartRef.current.x;
        const startY = drawingStartRef.current.y;
        setCurrentDrawingBox([
            Math.min(startX, curX), Math.min(startY, curY),
            Math.max(startX, curX), Math.max(startY, curY)
        ]);
    }
  }, [isResizing, activeBoxIndex, isDrawingMode, imgNaturalSize, localBoxes]);

  const handleContainerMouseDown = (e: React.MouseEvent) => {
    if (isDrawingMode && imgElementRef.current && imgNaturalSize) {
        const rect = imgElementRef.current.getBoundingClientRect();
        const scaleX = imgNaturalSize.w / rect.width;
        const scaleY = imgNaturalSize.h / rect.height;
        const relativeX = (e.clientX - rect.left) * scaleX;
        const relativeY = (e.clientY - rect.top) * scaleY;
        drawingStartRef.current = { x: relativeX, y: relativeY };
        setCurrentDrawingBox([relativeX, relativeY, relativeX, relativeY]);
    }
  };

  const handleContainerMouseUp = () => {
    if (isResizing) {
        setIsResizing(false);
        resizeRef.current = null;
    }
    if (isDrawingMode && currentDrawingBox) {
        const [x1, y1, x2, y2] = currentDrawingBox;
        if ((x2 - x1) > 5 && (y2 - y1) > 5) {
            setLocalBoxes(prev => [...prev, currentDrawingBox]);
            setLocalModified(prev => [...prev, false]); 
            setLocalEliminated(prev => [...prev, false]);
            setLocalIsManual(prev => [...prev, true]); 
            setLocalScores(prev => [...prev, 1.0]); 
            setActiveBoxIndex(localBoxes.length); 
        }
        setCurrentDrawingBox(null);
        drawingStartRef.current = null;
    }
  };

  const handleDiscardChanges = () => {
    if (activeBoxIndex !== null) {
      if (activeBoxIndex >= originalBoxesSnapshot.length) {
         setActiveBoxIndex(null);
      } else {
         const snapshotBox = originalBoxesSnapshot[activeBoxIndex];
         const snapshotMod = originalModifiedSnapshot[activeBoxIndex];
         const snapshotElim = originalEliminatedSnapshot[activeBoxIndex];
         setLocalBoxes(prev => { const next = [...prev]; next[activeBoxIndex] = [...snapshotBox]; return next; });
         setLocalModified(prev => { const next = [...prev]; next[activeBoxIndex] = snapshotMod; return next; });
         setLocalEliminated(prev => { const next = [...prev]; next[activeBoxIndex] = snapshotElim; return next; });
      }
    } else {
      setLocalBoxes(originalBoxesSnapshot.map(b => [...b]));
      setLocalModified([...originalModifiedSnapshot]);
      setLocalEliminated([...originalEliminatedSnapshot]);
      setLocalIsManual([...originalIsManualSnapshot]);
      setLocalScores([...originalScoresSnapshot]);
      setMergeSelection([]);
      setIsDrawingMode(false);
    }
  };

  const handleSave = () => {
    setIsSaving(true);
    setTimeout(() => {
        const updatedImage = {
            ...selectedImage,
            analysis: {
              ...selectedImage.analysis!,
              boxes: localBoxes,
              modified: localModified,
              eliminated: localEliminated,
              isManual: localIsManual,
              scores: localScores,
              count: localBoxes.length
            }
        };
        setIsSaving(false);
        setTimeout(() => { onSaveParent(updatedImage); }, 0);
    }, 500);
  };

  return {
    // State
    localBoxes,
    localScores,
    localEliminated,
    localModified,
    localIsManual,
    activeBoxIndex,
    suggestions,
    isMergeMode,
    mergeSelection,
    isDrawingMode,
    currentDrawingBox,
    imgNaturalSize,
    isSaving,
    imgElementRef, // The Ref
    
    // Setters / Actions
    setActiveBoxIndex,
    setIsDrawingMode,
    setIsMergeMode,
    setMergeSelection,
    setImgNaturalSize,
    
    // Complex Handlers
    toggleBox,
    handleApplySuggestion,
    handleMergeSelected,
    handleBatchDelete,
    handleEliminateAll,
    toggleElimination,
    handleResizeStart,
    handleContainerMouseMove,
    handleContainerMouseDown,
    handleContainerMouseUp,
    handleDiscardChanges,
    handleSave
  };
}