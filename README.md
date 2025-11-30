# AI News Detector - Misinformation Analysis System

A sophisticated misinformation detection platform using multi-algorithm AI analysis with chain-of-thought reasoning and interactive investigation capabilities.

## 🚀 Live Demo

**Frontend:** Deployed on Lovable  
**Backend API:** `https://yosemite000-misinformation-detector.hf.space` (Legacy PyTorch model)

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        FRONTEND                                  │
│                   React + TypeScript + Tailwind                  │
│                                                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │   Index     │  │   History   │  │   Insights              │  │
│  │   Page      │  │   Page      │  │   Page                  │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
│         │                │                     │                 │
│         └────────────────┼─────────────────────┘                 │
│                          │                                       │
│              ┌───────────▼───────────┐                          │
│              │  AdvancedAnalyzer     │                          │
│              │  Component            │                          │
│              └───────────┬───────────┘                          │
└──────────────────────────┼──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│                    BACKEND (Supabase)                            │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              Edge Functions (Deno/TypeScript)            │    │
│  │                                                          │    │
│  │  ┌──────────────────┐  ┌──────────────────────────────┐ │    │
│  │  │ analyze-news-    │  │ investigate-chat             │ │    │
│  │  │ advanced         │  │                              │ │    │
│  │  │                  │  │ Interactive Q&A about        │ │    │
│  │  │ 5 AI Algorithms: │  │ analysis results             │ │    │
│  │  │ • Factual        │  └──────────────────────────────┘ │    │
│  │  │ • Linguistic     │                                   │    │
│  │  │ • Sentiment      │  ┌──────────────────────────────┐ │    │
│  │  │ • Source         │  │ analyze-news (legacy)        │ │    │
│  │  │ • Propaganda     │  │ Simple single-pass analysis  │ │    │
│  │  │                  │  └──────────────────────────────┘ │    │
│  │  │ + Synthesis      │                                   │    │
│  │  └──────────────────┘                                   │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              Lovable AI Gateway                          │    │
│  │              (Google Gemini 2.5 Flash)                   │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              PostgreSQL Database                         │    │
│  │              • analysis_history table                    │    │
│  │              • User authentication (Supabase Auth)       │    │
│  └─────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────┘
```

## 🛠️ Tech Stack

### Frontend
| Technology | Purpose |
|------------|---------|
| **React 18** | UI Framework |
| **TypeScript** | Type-safe JavaScript |
| **Vite** | Build tool & dev server |
| **Tailwind CSS** | Utility-first styling |
| **shadcn/ui** | Component library |
| **React Router** | Client-side routing |
| **TanStack Query** | Server state management |
| **Lucide React** | Icon library |

### Backend
| Technology | Purpose |
|------------|---------|
| **Supabase** | Backend-as-a-Service |
| **Deno** | Edge function runtime |
| **PostgreSQL** | Database |
| **Supabase Auth** | User authentication |
| **Lovable AI** | AI Gateway (Gemini 2.5 Flash) |

### Legacy ML Service (Optional)
| Technology | Purpose |
|------------|---------|
| **Python 3.11** | ML runtime |
| **PyTorch** | Deep learning framework |
| **FastAPI** | API framework |
| **scikit-learn** | TF-IDF vectorization |

## 🧠 Multi-Algorithm Analysis System

The system employs **5 specialized AI algorithms** that analyze text from different perspectives:

### 1. Factual Analysis
- Verifiable claims and statistics
- Named sources and citations
- Logical consistency
- Historical & scientific accuracy

### 2. Linguistic Analysis
- Sensationalist language patterns
- Clickbait detection
- Grammatical quality
- Professional vs manipulative tone

### 3. Sentiment & Bias Analysis
- Political bias indicators
- Emotional loading
- One-sided presentation
- Fear/anger/outrage triggers

### 4. Source Credibility Analysis
- Attribution to named sources
- Expert credentials
- Document/report citations
- Journalistic standards

### 5. Propaganda Detection
- Appeal to authority/emotion/fear
- Bandwagon effect
- Card stacking (selective facts)
- Name calling/labeling

### Chain-of-Thought Synthesis
All algorithm results are synthesized using chain-of-thought reasoning to produce:
- Overall credibility score (0-1)
- Confidence rating
- Executive summary
- Key concerns & strengths
- Actionable recommendations

## ✨ Features

- **Multi-Algorithm Analysis** - 5 specialized AI perspectives
- **Chain-of-Thought Reasoning** - Detailed synthesis with explanations
- **Interactive Investigation Chat** - Ask follow-up questions about results
- **Analysis History** - Track previous analyses (authenticated users)
- **Batch Processing** - Analyze multiple articles
- **Model Insights** - View system performance metrics
- **Responsive Design** - Works on all devices
- **Dark/Light Theme** - Automatic theme detection

## 📁 Project Structure

```
├── src/
│   ├── components/
│   │   ├── analysis/
│   │   │   ├── AdvancedAnalyzer.tsx    # Main analysis component
│   │   │   ├── AlgorithmCard.tsx       # Individual algorithm results
│   │   │   ├── SynthesisCard.tsx       # Overall synthesis display
│   │   │   └── InvestigationChat.tsx   # Interactive Q&A chat
│   │   ├── ui/                         # shadcn/ui components
│   │   └── ...
│   ├── pages/
│   │   ├── Index.tsx                   # Home page
│   │   ├── History.tsx                 # Analysis history
│   │   ├── Batch.tsx                   # Batch processing
│   │   └── Insights.tsx                # Model insights
│   ├── types/
│   │   └── analysis.ts                 # TypeScript interfaces
│   ├── integrations/
│   │   └── supabase/                   # Supabase client & types
│   └── lib/
│       └── utils.ts                    # Utility functions
├── supabase/
│   ├── functions/
│   │   ├── analyze-news-advanced/      # Multi-algorithm analysis
│   │   ├── investigate-chat/           # Interactive Q&A
│   │   └── analyze-news/               # Legacy simple analysis
│   └── config.toml                     # Supabase configuration
├── api/
│   └── ml_service/                     # Legacy PyTorch backend
│       ├── app.py                      # FastAPI server
│       ├── model.py                    # Neural network definition
│       ├── train.py                    # Training script
│       └── requirements.txt            # Python dependencies
└── models/                             # Trained model artifacts
    ├── fake_news_model.pth             # PyTorch weights
    └── tfidf_vectorizer.pkl            # TF-IDF vectorizer
```

## 🚀 Getting Started

### Prerequisites
- Node.js 18+
- npm or bun

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd <project-directory>

# Install dependencies
npm install

# Start development server
npm run dev
```

The app will be available at `http://localhost:8080`

### Environment Variables

The following environment variables are automatically configured:
- `VITE_SUPABASE_URL` - Supabase project URL
- `VITE_SUPABASE_PUBLISHABLE_KEY` - Supabase anon key

## 📊 Model Performance

### Current AI System (Lovable AI + Gemini)
- **Model**: Google Gemini 2.5 Flash
- **Approach**: Multi-perspective LLM analysis
- **Latency**: ~3-5 seconds for full analysis
- **Accuracy**: Context-dependent, high reasoning capability

### Legacy PyTorch Model
- **Architecture**: 2-layer neural network (1000 → 64 → 1)
- **Input**: TF-IDF vectors (max 1000 features)
- **Output**: Binary classification (fake/real)
- **Accuracy**: ~67.5% on test set
- **Inference**: <50ms

## 🔒 Security

- Row Level Security (RLS) on all database tables
- User authentication via Supabase Auth
- API keys secured as environment secrets
- CORS configured for allowed origins

## 📚 API Reference

### Analyze News (Advanced)
```bash
POST /functions/v1/analyze-news-advanced
Content-Type: application/json

{
  "text": "News article text to analyze..."
}
```

### Investigation Chat
```bash
POST /functions/v1/investigate-chat
Content-Type: application/json

{
  "text": "Original article text",
  "analyses": { /* Previous analysis results */ },
  "synthesis": { /* Synthesis results */ },
  "question": "Why is this considered misinformation?"
}
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is part of an academic research initiative for misinformation detection.

## 🙏 Acknowledgments

- [Lovable](https://lovable.dev) - AI-powered development platform
- [Supabase](https://supabase.com) - Backend infrastructure
- [shadcn/ui](https://ui.shadcn.com) - UI components
- [Google Gemini](https://deepmind.google/technologies/gemini/) - AI model
