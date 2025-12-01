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

export interface MLModelResult {
  prediction: 'fake' | 'real';
  confidence: number;
  raw_score: number;
  model_version?: string;
  processing_time_ms?: number;
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
  mlModel?: MLModelResult;
  timestamp: string;
}

export interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
}

export interface ModelMetrics {
  accuracy: number;
  precision: number;
  recall: number;
  f1_score: number;
  total_samples: number;
  last_trained: string;
}

export interface RetrainingStatus {
  status: 'idle' | 'preparing' | 'training' | 'evaluating' | 'complete' | 'error';
  progress: number;
  message: string;
  metrics?: ModelMetrics;
}
