import { dirname } from "node:path";
import { fileURLToPath } from "node:url";

const root = dirname(fileURLToPath(import.meta.url));

/** @type {import('next').NextConfig} */
const nextConfig = {
  agentRules: false,
  output: "export",
  turbopack: {
    root
  }
};

export default nextConfig;
