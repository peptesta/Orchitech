import React from 'react';

interface Props {
  title: string;
  className: string;
  confidence: number;
}

export const getConfidenceColor = (conf: number) => {
  if (conf >= 85) return "text-green-600";
  if (conf >= 70) return "text-teal-500";
  if (conf >= 50) return "text-yellow-500";
  return "text-red-600";
};

export default function PredictionCard({ title, className, confidence }: Props) {
  return (
    <div className="bg-slate-100 rounded-xl p-6 text-center shadow-inner border border-slate-200 h-full flex flex-col justify-center min-h-[160px]">
      <h4 className="text-gray-500 font-bold uppercase text-xs tracking-widest mb-2">{title}</h4>
      <h3 className="text-3xl font-extrabold my-1 text-slate-800 break-words leading-tight">
        {className || "Unknown"}
      </h3>
      <div className="mt-3">
        <span className={`text-5xl font-black ${getConfidenceColor(confidence)}`}>
          {confidence.toFixed(1)}%
        </span>
      </div>
      <p className="text-gray-400 text-xs mt-2 uppercase font-semibold">Confidence Score</p>
    </div>
  );
}