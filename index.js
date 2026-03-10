const express = require('express');
const cors = require('cors');
const Groq = require('groq-sdk');
require('dotenv').config();

const app = express();
app.use(cors()); // In production, restrict this to your domain!
app.use(express.json());

const groq = new Groq({
    apiKey: process.env.GROQ_API_KEY
});

// Port for Railway
const PORT = process.env.PORT || 3000;

app.post('/api/chat', async (req, res) => {
    try {
        const { query, context } = req.body;

        if (!query) {
            return res.status(400).json({ error: "Missing query" });
        }

        const systemPrompt = `
            You are "Ashoka's Assistant", a professional and helpful AI agent for Ashoka K U's portfolio.
            Use the provided context to answer. Be concise and friendly.
            
            CONTEXT:
            ${context}
        `;

        const completion = await groq.chat.completions.create({
            messages: [
                { role: "system", content: systemPrompt },
                { role: "user", content: query }
            ],
            model: "llama3-8b-8192",
            temperature: 0.7,
            max_tokens: 500
        });

        res.json({ response: completion.choices[0].message.content });

    } catch (error) {
        console.error("Backend Error:", error);
        res.status(500).json({ error: "Failed to process with Groq", details: error.message });
    }
});

app.get('/', (req, res) => {
    res.send("Ashoka's AI Backend is running!");
});

app.listen(PORT, () => {
    console.log(`Server is running on port ${PORT}`);
});
