import { Pinecone } from '@pinecone-database/pinecone';
import fetch from 'node-fetch';
import 'dotenv/config';
import fs from 'fs';
import { createRequire } from 'module';
const require = createRequire(import.meta.url);
const pdf = require('pdf-parse');
import { knowledgeBase } from '../knowledge_base.js';

const pc = new Pinecone({ apiKey: process.env.PINECONE_API_KEY });
const index = pc.index(process.env.PINECONE_INDEX_NAME || 'chatbot');

// Helper to extract text from PDF
async function extractResumeText(filePath) {
    console.log(`Extracting text from PDF: ${filePath}...`);
    try {
        const dataBuffer = fs.readFileSync(filePath);
        const data = await pdf(dataBuffer);
        // Clean and chunk the text simply by paragraphs/newlines
        return data.text.split('\n\n').filter(chunk => chunk.trim().length > 20);
    } catch (error) {
        console.error("PDF Extraction failed, skipping PDF data:", error.message);
        return [];
    }
}

function prepareTextChunks(kb) {
    const chunks = [];
    if (kb.personalInfo) chunks.push(`Ashoka K U is a ${kb.personalInfo.role} based in ${kb.personalInfo.location}. ${kb.personalInfo.summary}`);
    if (kb.skills) Object.entries(kb.skills).forEach(([c, s]) => chunks.push(`${c} Skills: ${s.join(', ')}`));
    if (Array.isArray(kb.experience)) kb.experience.forEach(exp => chunks.push(`Experience as ${exp.role} at ${exp.company} (${exp.period}): ${exp.description}`));
    if (Array.isArray(kb.projects)) kb.projects.forEach(p => chunks.push(`Project: ${p.title}. Tech: ${p.tech}. Details: ${p.details}. Result: ${p.result}`));
    if (Array.isArray(kb.qa)) kb.qa.forEach(qa => chunks.push(`Question: ${qa.question} Answer: ${qa.answer}`));
    return chunks;
}

async function generateEmbeddings(textArray) {
    console.log(`Vectorizing ${textArray.length} chunks...`);
    const res = await fetch('https://api.pinecone.io/embed', {
        method: 'POST',
        headers: {
            'Api-Key': process.env.PINECONE_API_KEY,
            'X-Pinecone-Api-Version': '2024-10',
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            model: 'multilingual-e5-large',
            inputs: textArray.map(text => ({ text })),
            parameters: { input_type: 'passage' }
        })
    });
    
    if (!res.ok) {
        const errBody = await res.text();
        throw new Error(`Pinecone Inference Failed (${res.status}): ${errBody}`);
    }
    
    const data = await res.json();
    return data.data;
}

async function ingest() {
    console.log("🚀 Deep Vectorization Started...");
    
    // 1. Get knowledge base chunks
    let allChunks = prepareTextChunks(knowledgeBase);
    console.log(`${allChunks.length} chunks from knowledge_base.js ready.`);

    // 2. Extract and add PDF resume text
    const pdfPath = '../data/ASHOKA K U.pdf'; // Adjust path if needed
    if (fs.existsSync(pdfPath)) {
        const pdfChunks = await extractResumeText(pdfPath);
        console.log(`Added ${pdfChunks.length} chunks from PDF resume.`);
        allChunks = [...allChunks, ...pdfChunks];
    }

    // 3. Generate Embeddings 
    const embeddings = await generateEmbeddings(allChunks);
    console.log(`Embeddings generated for all data.`);

    // 4. Format for Pinecone Upsert
    const vectors = embeddings.map((emb, i) => ({
        id: `full_chunk_${Date.now()}_${i}`,
        values: emb.values,
        metadata: { text: allChunks[i] }
    }));

    // 5. Upsert to Index
    console.log(`Upserting ${vectors.length} vectors to index [${process.env.PINECONE_INDEX_NAME || 'chatbot'}]...`);
    await index.upsert({ records: vectors });
    console.log("✅ COMPLETE! Both your Website Data and PDF Resume are now in the Vector DB.");
}

ingest().catch(err => {
    console.error("INGESTION CRITICAL ERROR:");
    console.error(err);
});
