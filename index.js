import express from 'express';
import cors from 'cors';
import { Pinecone } from '@pinecone-database/pinecone';
import Groq from 'groq-sdk';
import fetch from 'node-fetch';
import 'dotenv/config';

const app = express();
app.use(cors());
app.use(express.json());

const pc = new Pinecone({ apiKey: process.env.PINECONE_API_KEY });
const index = pc.index(process.env.PINECONE_INDEX_NAME || 'chatbot');
const groq = new Groq({ apiKey: process.env.GROQ_API_KEY });

const PORT = process.env.PORT || 3000;

// Custom embedder because the SDK has a version pinning issue
async function generateQueryEmbedding(query) {
    const res = await fetch('https://api.pinecone.io/embed', {
        method: 'POST',
        headers: {
            'Api-Key': process.env.PINECONE_API_KEY,
            'X-Pinecone-Api-Version': '2024-10',
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            model: 'multilingual-e5-large',
            inputs: [{ text: query }],
            parameters: { input_type: 'query' }
        })
    });
    
    if (!res.ok) {
        const err = await res.json();
        throw new Error(`Embedding Failed: ${JSON.stringify(err)}`);
    }
    
    const data = await res.json();
    return data.data[0].values;
}

app.post('/api/chat', async (req, res) => {
    try {
        const { query } = req.body;
        if (!query) return res.status(400).json({ error: "Missing query" });

        console.log(`User Query: ${query}`);

        // 1. Generate Query Vector
        const queryVector = await generateQueryEmbedding(query);

        // 2. Search Pinecone
        const queryResponse = await index.query({
            vector: queryVector,
            topK: 4,
            includeMetadata: true
        });

        // 3. Prepare Context
        const context = queryResponse.matches
            .map(m => `[RELEVANT RESUME INFO]: ${m.metadata.text}`)
            .join('\n\n');

        console.log(`Matched ${queryResponse.matches.length} context pieces.`);

        // 4. Generate AI response
        const systemPrompt = `
            You are "Ashoka's Assistant", the AI ambassador for Ashoka K.U's portfolio.
            Ashoka is an AI dev specializing in RAG systems, document extraction, and analytics.
            
            Use the provided context to answer questions. 
            If data is missing, suggest contacting him at ashokaku04@gmail.com.
            
            STRICT GUIDELINES:
            - Be professional, brief, and highly accurate.
            - Mention his results: (e.g., 95% OCR accuracy, 100% fraud precision).
            - Keep answers under 3-4 sentences.
            
            CONTEXT:
            ${context || "No context found. Speak generally about Ashoka's profile if possible."}
        `;

        const completion = await groq.chat.completions.create({
            messages: [
                { role: "system", content: systemPrompt },
                { role: "user", content: query }
            ],
            model: "llama3-8b-8192",
            temperature: 0.6,
            max_tokens: 400
        });

        res.json({ response: completion.choices[0].message.content });

    } catch (error) {
        console.error("Backend Error:", error);
        res.status(500).json({ error: "Brain malfunction!", details: error.message });
    }
});

app.get('/', (req, res) => {
    res.send("Ashoka's AI Backend (Vector RAG) is online!");
});

app.listen(PORT, () => {
    console.log(`Server is running on port ${PORT}`);
});
