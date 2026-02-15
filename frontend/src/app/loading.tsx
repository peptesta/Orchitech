import { Loader2 } from "lucide-react";

export default function Loading() {
  return (
    <div className="flex h-screen w-full items-center justify-center bg-[#F6F4EF]">
      <div className="flex flex-col items-center gap-4">
        <Loader2 className="h-12 w-12 animate-spin text-emerald-600" />
        <p className="text-lg font-medium text-stone-600">Loading...</p>
      </div>
    </div>
  );
}