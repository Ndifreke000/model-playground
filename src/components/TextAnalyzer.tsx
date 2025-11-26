import { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Card } from "@/components/ui/card";
import { Loader2, ScanSearch, Sparkles } from "lucide-react";
import { AnalysisResult } from "./AnalysisResult";
import { toast } from "sonner";
import { supabase } from "@/integrations/supabase/client";
import { useNavigate } from "react-router-dom";

interface PredictionResult {
  prediction: number;
  confidence: number;
  reasoning?: string;
  red_flags?: string[];
  authenticity_indicators?: string[];
}

const exampleTexts = [
  "Breaking: President announces new economic reforms aimed at reducing inflation and boosting local manufacturing. Finance Minister confirms details in official statement.",
  "BREAKING!!! You won't believe what this politician did! Click now before it's deleted! No evidence needed, just trust us!",
  "The Central Bank of Nigeria has raised the monetary policy rate by 50 basis points to 18.75% to curb rising inflation, according to an official press release today."
];

export const TextAnalyzer = () => {
  const navigate = useNavigate();
  const [text, setText] = useState("");
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [user, setUser] = useState<any>(null);

  useEffect(() => {
    // Check auth status
    supabase.auth.getSession().then(({ data: { session } }) => {
      setUser(session?.user ?? null);
    });

    const {
      data: { subscription },
    } = supabase.auth.onAuthStateChange((_event, session) => {
      setUser(session?.user ?? null);
    });

    return () => subscription.unsubscribe();
  }, []);

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
      // Call the edge function
      const { data, error } = await supabase.functions.invoke('analyze-news', {
        body: { text }
      });

      if (error) throw error;

      if (data.error) {
        if (data.error.includes('Rate limit')) {
          toast.error("Rate limit exceeded. Please try again later.");
        } else if (data.error.includes('credits')) {
          toast.error("AI service credits depleted. Please contact support.");
        } else {
          toast.error(data.error);
        }
        return;
      }

      setResult(data);
      
      // Save to history if user is logged in
      if (user) {
        const { error: saveError } = await supabase
          .from("analysis_history")
          .insert({
            user_id: user.id,
            text: text,
            prediction: data.prediction,
            confidence: data.confidence,
            is_authentic: data.prediction >= 0.5
          });

        if (saveError) {
          console.error("Error saving to history:", saveError);
        }
      }

      toast.success("Analysis complete!");
    } catch (error: any) {
      console.error("Analysis error:", error);
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
          placeholder="Paste a news article, headline, or any text you want to verify for authenticity..."
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

        {!user && (
          <div className="mb-4 p-3 bg-accent/30 border border-accent rounded-md">
            <p className="text-sm text-accent-foreground">
              💡 <strong>Tip:</strong> Sign in to save your analysis history and track your results over time.
            </p>
          </div>
        )}

        <Button
          onClick={handleAnalyze}
          disabled={isAnalyzing || !text.trim()}
          className="w-full bg-gradient-hero hover:opacity-90 transition-opacity text-primary-foreground font-semibold"
          size="lg"
        >
          {isAnalyzing ? (
            <>
              <Loader2 className="w-5 h-5 mr-2 animate-spin" />
              Analyzing with AI...
            </>
          ) : (
            <>
              <Sparkles className="w-5 h-5 mr-2" />
              Analyze Text
            </>
          )}
        </Button>
      </Card>

      {result && <AnalysisResult prediction={result.prediction} confidence={result.confidence} reasoning={result.reasoning} redFlags={result.red_flags} authenticityIndicators={result.authenticity_indicators} />}
    </div>
  );
};
