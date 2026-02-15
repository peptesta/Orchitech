'use client';

import { useState } from "react";
import Image from "next/image";
import { ApiResponse } from "@/types";
import { getConfidenceColor } from "@/components/PredictionCard";
import ProbabilityDropdown from "@/components/ProbabilityDropdown";
import { useImageMetadata } from "@/hooks/useImageMetadata";

interface SingleViewProps {
  result: ApiResponse;
  preview: string;
  mode: string; 
  useGpu: boolean;
  strategyName: string;
}

export default function SingleView({ result, preview, mode, useGpu, strategyName }: SingleViewProps) {
  const [showMetadata, setShowMetadata] = useState(false);
  
  const isExternal = mode === "external";
  
  // Immagine da mostrare (può essere il crop)
  const displayImageSrc = isExternal && result.image_cropped ? `data:image/jpeg;base64,${result.image_cropped}` : preview;
  const displayImageLabel = isExternal && result.image_cropped ? "Focused Crop Input" : "Original Input";

  // IMPORTANTE: Passiamo sempre e solo 'preview' all'hook dei metadati.
  // Questo assicura che i dati EXIF/GPS siano letti dal file originale.
  const { metadata, address, loading } = useImageMetadata(preview);

  const lat = metadata?.latitude;
  const lon = metadata?.longitude;
  const hasGps = lat !== undefined && lon !== undefined;

  return (
    <div className="grid grid-cols-1 lg:grid-cols-5 gap-8">
      
      {/* --- COLONNA SINISTRA: Analisi e Dati --- */}
      <div className="lg:col-span-2 flex flex-col gap-6 order-2 lg:order-1">

        {/* 1. Card Risultato Principale */}
        <div className="bg-white p-8 rounded-2xl shadow-lg shadow-pink-100/50 border-l-8 border-pink-500">
          <div className="flex justify-between items-start mb-4">
            <span className="text-xs font-bold text-stone-400 uppercase tracking-widest">Identified Species</span>
            <span className="text-[10px] font-bold uppercase bg-stone-100 text-stone-600 px-2 py-1 rounded border border-stone-200">
              {strategyName}
            </span>
          </div>
          <h1 className={`text-4xl font-extrabold mt-3 mb-2 leading-tight ${getConfidenceColor(result.confidence)}`}>
            {result.predicted_class}
          </h1>
          <div className="flex items-center gap-2 mb-6">
            <span className={`inline-block w-3 h-3 rounded-full ${result.confidence > 80 ? 'bg-emerald-500' : 'bg-amber-500'}`}></span>
            <p className="text-stone-500 font-medium">{result.confidence.toFixed(2)}% Confidence Score</p>
          </div>
          
          <div className="pt-4 border-t border-stone-100">
            <h4 className="text-[10px] font-bold text-stone-400 uppercase mb-3 tracking-widest">Analysis Details</h4>
            <ProbabilityDropdown probs={result.all_classes_probs} />
          </div>
        </div>

        {/* 2. Sezione GPS & Mappa (Sempre dall'originale) */}
        <div className="bg-white p-6 rounded-2xl shadow-md shadow-stone-200/50 border border-stone-200 overflow-hidden">
          <div className="flex justify-between items-center mb-4">
            <h3 className="font-bold text-stone-800 flex items-center gap-2 text-sm uppercase">📍 Original Location</h3>
            {address && <span className="text-[9px] bg-emerald-50 text-emerald-600 px-2 py-0.5 rounded-full font-bold border border-emerald-100 uppercase">Geocoded</span>}
          </div>
          
          {loading ? (
            <div className="h-48 bg-stone-100 animate-pulse rounded-xl flex items-center justify-center text-stone-400 text-xs italic">Estrazione dati GPS dal file originale...</div>
          ) : hasGps ? (
            <div className="space-y-4">
              <div className="bg-stone-50 p-3 rounded-lg border border-stone-100">
                <span className="block text-[9px] uppercase text-stone-400 font-bold mb-0.5">Detected Area</span>
                <p className="text-stone-800 font-semibold text-sm leading-tight">{address}</p>
              </div>
              <div className="w-full h-44 rounded-xl overflow-hidden border border-stone-200 shadow-inner relative">
                <iframe 
                  width="100%" height="100%" frameBorder="0" 
                  src={`https://www.openstreetmap.org/export/embed.html?bbox=${lon-0.005}%2C${lat-0.005}%2C${lon+0.005}%2C${lat+0.005}&layer=mapnik&marker=${lat}%2C${lon}`}
                ></iframe>
              </div>
              <div className="flex flex-col gap-2">
                <div className="flex justify-between text-[11px] py-1 border-b border-stone-50 text-stone-600">
                  <span className="font-bold uppercase text-stone-400">Coordinates</span>
                  <span className="font-mono">{lat.toFixed(5)}, {lon.toFixed(5)}</span>
                </div>
                <a href={`https://www.google.com/maps/search/?api=1&query=${lat},${lon}`} target="_blank" rel="noopener noreferrer" className="flex items-center justify-center w-full py-2 bg-stone-900 text-white text-[11px] font-bold rounded-lg hover:bg-stone-800 transition-colors">View on Google Maps</a>
              </div>
            </div>
          ) : (
            <div className="h-24 bg-stone-50 rounded-xl flex flex-col items-center justify-center border border-dashed border-stone-200 p-4 text-center text-stone-400 italic text-xs">Gps Coords not found in original image</div>
          )}
        </div>

        {/* 3. Sezione Metadati EXIF (Sempre dall'originale) */}
        <div className="bg-white p-6 rounded-2xl shadow-md shadow-stone-200/50 border border-stone-200">
          <div className="flex justify-between items-center mb-4">
            <h3 className="font-bold text-stone-800 text-sm uppercase">ℹ️ Technical Meta (Original)</h3>
            {metadata && (
              <button onClick={() => setShowMetadata(!showMetadata)} className="text-xs font-bold text-pink-600 underline">
                {showMetadata ? "Hide" : "Show All"}
              </button>
            )}
          </div>
          {metadata ? (
            <div className="space-y-2 text-[11px]">
              <div className="grid grid-cols-2 gap-2">
                <div className="bg-stone-50 p-2 rounded">
                  <span className="block text-stone-400 font-bold uppercase text-[9px]">Model</span>
                  <span className="text-stone-700 font-medium truncate block">{metadata.Model || 'Unknown'}</span>
                </div>
                <div className="bg-stone-50 p-2 rounded">
                  <span className="block text-stone-400 font-bold uppercase text-[9px]">Date Taken</span>
                  <span className="text-stone-700 font-medium truncate block">
                    {metadata.DateTimeOriginal ? new Date(metadata.DateTimeOriginal).toLocaleDateString() : 'N/A'}
                  </span>
                </div>
              </div>
              {showMetadata && (
                <div className="mt-4 max-h-48 overflow-y-auto custom-scrollbar bg-stone-50 p-3 rounded-lg border border-stone-100 font-mono text-[10px]">
                  <pre className="whitespace-pre-wrap">{JSON.stringify(metadata, null, 2)}</pre>
                </div>
              )}
            </div>
          ) : <p className="text-xs text-stone-400 italic font-medium">Dati tecnici non disponibili.</p>}
        </div>
      </div>

      {/* --- COLONNA DESTRA: Visualizzazioni --- */}
      <div className="lg:col-span-3 space-y-6 order-1 lg:order-2">
        {/* Preview Immagine (Può essere il Crop) */}
        <div className="bg-white p-4 rounded-xl shadow-md border border-stone-200 relative">
          <h3 className="font-bold text-[10px] text-stone-400 uppercase mb-3 tracking-widest">{displayImageLabel}</h3>
          <div className="relative w-full h-[550px] bg-stone-100 rounded-lg overflow-hidden border border-stone-50">
            <Image src={displayImageSrc} alt="Input" fill className="object-contain" unoptimized />
          </div>
        </div>

        {/* Visual Explanation (Heatmaps) */}
        {useGpu && (result.integrated_gradients || result.occlusion) && (
          <div className="bg-stone-50 p-6 rounded-xl border border-stone-200 shadow-inner">
            <h3 className="font-bold text-emerald-900 mb-6 flex items-center justify-between">
               <span className="flex items-center gap-2">🧠 Visual Explanation</span>
               <span className="text-[9px] uppercase font-bold text-stone-400 bg-white px-3 py-1 rounded-full border border-stone-200 shadow-sm">Red = High Importance</span>
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {result.integrated_gradients && (
                <div className="relative w-full h-[350px] bg-white rounded-lg overflow-hidden shadow-sm border border-stone-200 group">
                  <p className="absolute top-2 left-2 z-10 bg-stone-900/80 text-white px-3 py-1 text-[10px] rounded font-bold backdrop-blur-sm">Integrated Gradients</p>
                  <Image src={`data:image/jpeg;base64,${result.integrated_gradients}`} alt="IG" fill className="object-contain p-2" />
                </div>
              )}
              {result.occlusion && (
                <div className="relative w-full h-[350px] bg-white rounded-lg overflow-hidden shadow-sm border border-stone-200 group">
                  <p className="absolute top-2 left-2 z-10 bg-stone-900/80 text-white px-3 py-1 text-[10px] rounded font-bold backdrop-blur-sm">Occlusion Map</p>
                  <Image src={`data:image/jpeg;base64,${result.occlusion}`} alt="Occ" fill className="object-contain p-2" />
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}