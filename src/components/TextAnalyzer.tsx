import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Card } from "@/components/ui/card";
import { Loader2, ScanSearch, Sparkles } from "lucide-react";
import { AnalysisResult } from "./AnalysisResult";
import { toast } from "sonner";

interface PredictionResult {
  prediction: number;
  confidence: number;
}

// Mock prediction function - simulates the PyTorch model behavior
const mockPredict = (text: string): Promise<PredictionResult> => {
  return new Promise((resolve) => {
    setTimeout(() => {
      // Simple heuristic for demo: longer, more formal text is more likely "real"
      const wordCount = text.trim().split(/\s+/).length;
      const hasProperPunctuation = /[.!?]/.test(text);
      const hasCapitalization = /[A-Z]/.test(text);
      const hasShortSentences = text.split(/[.!?]/).some(s => s.trim().split(/\s+/).length < 5);
      
      let score = 0.5;
      if (wordCount > 50) score += 0.15;
      if (hasProperPunctuation) score += 0.1;
      if (hasCapitalization) score += 0.1;
      if (hasShortSentences) score -= 0.15;
      
      // Add some randomness
      score += (Math.random() - 0.5) * 0.2;
      score = Math.max(0.1, Math.min(0.9, score));
      
      resolve({
        prediction: score,
        confidence: 0.7 + Math.random() * 0.25
      });
    }, 1500);
  });
};

const exampleTexts = [
  "Breaking: Scientists discover that drinking coffee can cure all diseases overnight. No medical research needed!",
  "The Federal Reserve announced today that it will maintain current interest rates amid ongoing economic uncertainty. The decision was made following extensive review of inflation data and employment statistics.",
  "You won't believe what this celebrity did! Doctors hate them for this one weird trick!"
];

export const TextAnalyzer = () => {
  const [text, setText] = useState("");
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState<PredictionResult | null>(null);

  const handleAnalyze = async () => {
    if (!text.trim()) {
      toast.error("Please enter some text to analyze");
      return;
    }

    if (text.trim().split(/\s+/).length < 10) {
      toast.error("Please enter at least 10 words for accurate analysis");
      return;
    }

    setIsAnalyzing(true);
    setResult(null);

    try {
      // In production, this would call your backend API with the trained model
      const prediction = await mockPredict(text);
      setResult(prediction);
      toast.success("Analysis complete!");
    } catch (error) {
      toast.error("Failed to analyze text. Please try again.");
    } finally {
      setIsAnalyzing(false);
    }
  };

  const loadExample = (exampleText: string) => {
    setText(exampleText);
    setResult(null);
  };

  return (
    <div className="space-y-6">
      <Card className="p-6 bg-gradient-card border-border/50 shadow-lg">
        <div className="flex items-center gap-3 mb-4">
          <ScanSearch className="w-6 h-6 text-primary" />
          <h2 className="text-xl font-semibold text-card-foreground">Enter News Text</h2>
        </div>
        
        <Textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Paste a news article, headline, or any text you want to verify..."
          className="min-h-[200px] mb-4 text-base resize-none"
          disabled={isAnalyzing}
        />

        <div className="flex flex-wrap gap-2 mb-4">
          <span className="text-sm text-muted-foreground mr-2">Try an example:</span>
          {exampleTexts.map((example, idx) => (
            <Button
              key={idx}
              variant="outline"
              size="sm"
              onClick={() => loadExample(example)}
              disabled={isAnalyzing}
              className="text-xs"
            >
              Example {idx + 1}
            </Button>
          ))}
        </div>

        <Button
          onClick={handleAnalyze}
          disabled={isAnalyzing || !text.trim()}
          className="w-full bg-gradient-hero hover:opacity-90 transition-opacity text-primary-foreground font-semibold"
          size="lg"
        >
          {isAnalyzing ? (
            <>
              <Loader2 className="w-5 h-5 mr-2 animate-spin" />
              Analyzing...
            </>
          ) : (
            <>
              <Sparkles className="w-5 h-5 mr-2" />
              Analyze Text
            </>
          )}
        </Button>
      </Card>

      {result && <AnalysisResult prediction={result.prediction} confidence={result.confidence} />}
    </div>
  );
};
