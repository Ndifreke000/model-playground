export interface AlgorithmResult {
  score: number;
  verdict: string;
  reasoning: string;
  findings?: string[];
  patterns?: string[];
  emotional_triggers?: string[];
  bias_direction?: string;
  sources_identified?: string[];
  credibility_issues?: string[];
  techniques_found?: string[];
}

export interface AnalysisSynthesis {
  overall_score: number;
  overall_verdict: string;
  confidence: number;
  executive_summary: string;
  key_concerns: string[];
  strengths: string[];
  recommendation: string;
  detailed_reasoning: string;
}

export interface AdvancedAnalysisResult {
  analyses: {
    factual: AlgorithmResult;
    linguistic: AlgorithmResult;
    sentiment: AlgorithmResult;
    source: AlgorithmResult;
    propaganda: AlgorithmResult;
  };
  synthesis: AnalysisSynthesis;
  timestamp: string;
}

export interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
}
