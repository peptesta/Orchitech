import type { Metadata } from "next";
import localFont from "next/font/local";
import "./globals.css";

import NavBar from "@/components/NavBar";

const geistSans = localFont({
  src: "./fonts/GeistVF.woff",
  variable: "--font-geist-sans",
  weight: "100 900",
});
const geistMono = localFont({
  src: "./fonts/GeistMonoVF.woff",
  variable: "--font-geist-mono",
  weight: "100 900",
});

export const metadata: Metadata = {
  title: "Orchid Species Classifier",
  description: "An interactive dashboard for orchid species classification using deep learning models.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased`}
      >
        {/* The header must be inside the body */}
        <header>
          <NavBar />
        </header>
        
        {children}
        {/* Footer */}
        <footer className="w-full bg-[#F0F7F3] border-t border-[#D8D2C8] p-6 text-center text-stone-600 text-sm">
          © {new Date().getFullYear()} Orchid Tech. All rights reserved.
        </footer>
      </body>
    </html>
  );
}