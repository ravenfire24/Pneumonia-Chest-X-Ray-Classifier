import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Chest X-Ray Classifier",
  description: "Browser-based ONNX chest X-ray classifier for pneumonia triage."
};

export default function RootLayout({
  children
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
