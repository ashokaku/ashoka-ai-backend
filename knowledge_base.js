export const knowledgeBase = {
    personalInfo: {
        name: "Ashoka K U",
        role: "Data Scientist & AI/ML Engineer",
        location: "Bengaluru, India",
        email: "ashokaku04@gmail.com",
        phone: "+91 7259615116",
        resume: "./data/ASHOKA K U.pdf",
        summary: "Data Scientist and AI/ML Engineer with 6+ years of professional experience. Specialized in Customer Lifetime Value (CLV), Media Mix Modeling (MMM), Predictive Analytics, and Generative AI (RAG). Highly proficient in Python, SQL, and BI tools like Power BI and Tableau.",
        about: "Ashoka is an AI developer specializing in RAG systems, AI automation, and data analytics. He crafts intelligent systems that bridge technical complexity and business value."
    },
    qa: [
        {
            question: "Who is Ashok?",
            answer: "Ashok is an AI developer and Data Scientist specializing in RAG systems, AI automation, and marketing analytics."
        },
        {
            question: "What AI projects has Ashok built?",
            answer: "Ashok has built AI-powered logistics extractors, multi-document RAG assistants, and predictive analytics suites for marketing."
        },
        {
            question: "Does he know Power BI?",
            answer: "Yes, Ashoka is an expert in Power BI and Tableau. He has built several high-impact interactive dashboards for tracking CAC, ROAS, and customer purchasing behaviors."
        }
    ],
    experience: [
        {
            role: "AI/ML Engineer",
            company: "Big Wings LLC",
            period: "Oct 2025 - Present",
            description: "Designed end-to-end AI pipelines for OCR + LLM document extraction. Built scalable workflows using Python and Spark."
        },
        {
            role: "Data Scientist – Media & Marketing Analytics (Trainee)",
            company: "Spinnaker Analytics",
            period: "Oct 2025 - Feb 2026",
            description: "Developed sophisticated CLV, segmentation, and MMM models. Built time-series forecasting to predict campaign ROI."
        }
    ],
    skills: {
        AI: ["RAG Pipelines", "LLMs (Llama, GPT)", "Prompt Engineering", "FAISS", "Pinecone", "Transformers"],
        Programming: ["Python (Scikit-learn, PyTorch, TensorFlow)", "SQL", "Spark", "JavaScript"],
        Data: ["Data Storytelling", "Power BI", "Tableau", "Marketing Analytics", "Time-Series Forecasting"],
        AITools: ["OpenAI API", "Gemini API", "Groq", "Databricks", "Snowflake"]
    },
    projects: [
        {
            title: "Advanced Analytics Suite (CLV & MMM)",
            tech: "Python, BG/NBD, Scikit-learn",
            github: "https://github.com/ashokaku/Advanced-Analytics-CLV-Media-Mix-Modeling",
            details: "Predictive suite for Customer Lifetime Value and Media Mix Modeling. Uses Adstock transformations to measure marketing lag.",
            result: "Developed 12+ premium visualizations and provided growth strategies that optimized multi-channel spend.",
            image: "https://raw.githubusercontent.com/ashokaku/Advanced-Analytics-CLV-Media-Mix-Modeling/main/output_images/clv_output.png"
        },
        {
            title: "AI-Powered Logistics Extraction",
            tech: "OCR, LLM (LoRA Fine-tuned)",
            github: "https://github.com/ashokaku",
            details: "Builds a pipeline to extract 50+ business fields from messy logistics PDFs.",
            result: "Achieved 95% accuracy on critical fields, automating workflows and saving 100+ manual hours per week.",
            image: ""
        },
        {
            title: "E-commerce Segmentation (Power BI Integrated)",
            tech: "Python, K-Means, Power BI",
            github: "https://github.com/ashokaku/E-commerce-Customer-Segmentation-and-Prediction",
            details: "Clusters customers based on RFM (Recency, Frequency, Monetary) data. Fully integrated with an interactive Power BI dashboard.",
            result: "Mapped high-value customer behavior, allowing for targeted marketing that boosts retention and ROAS.",
            image: "https://raw.githubusercontent.com/ashokaku/E-commerce-Customer-Segmentation-and-Prediction/main/Dashboard_Preview.png" 
        },
        {
            title: "Credit Card Fraud Detection",
            tech: "XGBoost, Isolation Forest, SMOTE",
            github: "https://github.com/ashokaku/Fraud-Detection-Using-Machine-Learning",
            details: "Detects fraudulent activity in massive transaction datasets using advanced resampling.",
            result: "Achieved 100% Precision and 96% Recall in identifying valid fraud cases.",
            image: ""
        },
        {
            title: "Store Performance Forecaster",
            tech: "Random Forest, Python",
            github: "https://github.com/ashokaku/store_performance_analysis",
            details: "Analyzes historical store datasets to predict future revenue performance.",
            result: "The final Random Forest model achieved an exceptionally high R² score of 0.9781.",
            image: ""
        },
        {
            title: "Interactive Power BI Analytics Dashboards",
            tech: "Power BI, DAX, SQL",
            github: "https://github.com/ashokaku",
            details: "Created specialized dashboards for tracking CAC, ROAS, LTV, and customer buying trends.",
            result: "Delivered actionable insights that drove revenue growth through clear, executive-level visualization.",
            image: ""
        }
    ]
};
