import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
export default defineConfig({plugins:[react()],base:'./',server:{host:'0.0.0.0',port:4173,allowedHosts:['terminal.local']},build:{rollupOptions:{output:{manualChunks(id){if(id.includes('/node_modules/katex/'))return 'math';if(id.includes('/node_modules/highlight.js/'))return 'code';}}}}});
