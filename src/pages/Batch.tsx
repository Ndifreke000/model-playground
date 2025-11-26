import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { supabase } from "@/integrations/supabase/client";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Shield, ArrowLeft, Upload, Loader2, CheckCircle, XCircle } from "lucide-react";
import { toast } from "sonner";
import { Progress } from "@/components/ui/progress";

interface BatchResult {
  text: string;
  prediction: number;
  confidence: number;
  isAuthentic: boolean;
}

const Batch = () => {
  const navigate = useNavigate();
  const [texts, setTexts] = useState("");
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [results, setResults] = useState<BatchResult[]>([]);
  const [progress, setProgress] = useState(0);

  const handleBatchAnalyze = async () => {
    const textArray = texts
      .split("\n\n")
      .map((t) => t.trim())
      .filter((t) => t.length > 0);

    if (textArray.length === 0) {
      toast.error("Please enter at least one news text to analyze");
      return;
    }

    if (textArray.length > 10) {
      toast.error("Maximum 10 texts can be analyzed at once");
      return;
    }

    const { data: { session } } = await supabase.auth.getSession();
    if (!session) {
      toast.error("Please sign in to use batch analysis");
      navigate("/auth");
      return;
    }

    setIsAnalyzing(true);
    setResults([]);
    setProgress(0);

    const batchResults: BatchResult[] = [];

    try {
      for (let i = 0; i < textArray.length; i++) {
        const text = textArray[i];
        
        // Analyze text
        const { data, error } = await supabase.functions.invoke('analyze-news', {
          body: { text }
        });

        if (error || data.error) {
          console.error("Analysis error:", error || data.error);
          continue;
        }

        const result: BatchResult = {
          text,
          prediction: data.prediction,
          confidence: data.confidence,
          isAuthentic: data.prediction >= 0.5,
        };

        batchResults.push(result);

        // Save to history
        await supabase
          .from("analysis_history")
          .insert({
            user_id: session.user.id,
            text: text,
            prediction: data.prediction,
            confidence: data.confidence,
            is_authentic: data.prediction >= 0.5
          });

        setProgress(((i + 1) / textArray.length) * 100);
        setResults([...batchResults]);

        // Small delay to avoid rate limiting
        if (i < textArray.length - 1) {
          await new Promise((resolve) => setTimeout(resolve, 500));
        }
      }

      toast.success(`Analyzed ${batchResults.length} texts successfully!`);
    } catch (error) {
      console.error("Batch analysis error:", error);
      toast.error("Some analyses failed. Check results below.");
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-subtle">
      <div className="container mx-auto max-w-6xl px-4 py-8">
        <Button
          variant="ghost"
          onClick={() => navigate("/")}
          className="mb-6"
        >
          <ArrowLeft className="w-4 h-4 mr-2" />
          Back
        </Button>

        <div className="flex items-center gap-3 mb-8">
          <Upload className="w-8 h-8 text-primary" />
          <h1 className="text-3xl font-bold">Batch Analysis</h1>
        </div>

        <Card className="p-6 mb-6">
          <h3 className="text-lg font-semibold mb-4">Instructions</h3>
          <ul className="space-y-2 text-sm text-muted-foreground mb-4">
            <li>• Separate each news text with a blank line (double enter)</li>
            <li>• Maximum 10 texts per batch</li>
            <li>• Each text should be at least 10 words</li>
            <li>• Results will be saved to your history automatically</li>
          </ul>

          <Textarea
            value={texts}
            onChange={(e) => setTexts(e.target.value)}
            placeholder="Enter your first news text here...

Then add a blank line and enter the next news text...

And so on (up to 10 texts)"
            className="min-h-[300px] mb-4 text-base resize-none"
            disabled={isAnalyzing}
          />

          <Button
            onClick={handleBatchAnalyze}
            disabled={isAnalyzing || !texts.trim()}
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
                <Upload className="w-5 h-5 mr-2" />
                Analyze Batch
              </>
            )}
          </Button>

          {isAnalyzing && (
            <div className="mt-4">
              <Progress value={progress} className="h-2" />
              <p className="text-sm text-muted-foreground mt-2 text-center">
                {Math.round(progress)}% complete
              </p>
            </div>
          )}
        </Card>

        {results.length > 0 && (
          <div className="space-y-4">
            <h2 className="text-2xl font-bold">Results ({results.length})</h2>
            {results.map((result, idx) => (
              <Card key={idx} className="p-6">
                <div className="flex items-start gap-4">
                  {result.isAuthentic ? (
                    <CheckCircle className="w-6 h-6 text-success flex-shrink-0 mt-1" />
                  ) : (
                    <XCircle className="w-6 h-6 text-destructive flex-shrink-0 mt-1" />
                  )}
                  <div className="flex-1">
                    <div className="flex items-center gap-3 mb-2">
                      <span
                        className={`px-3 py-1 rounded-full text-sm font-semibold ${
                          result.isAuthentic
                            ? "bg-success/20 text-success"
                            : "bg-destructive/20 text-destructive"
                        }`}
                      >
                        {result.isAuthentic ? "Authentic" : "Misinformation"}
                      </span>
                      <span className="text-sm text-muted-foreground">
                        Confidence: {Math.round(result.confidence * 100)}%
                      </span>
                    </div>
                    <p className="text-sm text-muted-foreground line-clamp-3">
                      {result.text}
                    </p>
                  </div>
                </div>
              </Card>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

export default Batch;
