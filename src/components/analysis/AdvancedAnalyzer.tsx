import { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Card } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Loader2, ScanSearch, Sparkles, FileSearch, Scale, MessageCircle, Eye, Shield } from "lucide-react";
import { toast } from "sonner";
import { supabase } from "@/integrations/supabase/client";
import { AdvancedAnalysisResult } from "@/types/analysis";
import { AlgorithmCard } from "./AlgorithmCard";
import { SynthesisCard } from "./SynthesisCard";
import { InvestigationChat } from "./InvestigationChat";

const exampleTexts = [
  "Breaking: President announces new economic reforms aimed at reducing inflation and boosting local manufacturing. Finance Minister confirms details in official statement.",
  "BREAKING!!! You won't believe what this politician did! Click now before it's deleted! No evidence needed, just trust us! Share before they take this down!",
  "A new study published in Nature Medicine found that the experimental treatment showed a 45% improvement in patient outcomes compared to the control group, according to researchers at Johns Hopkins University."
];

export const AdvancedAnalyzer = () => {
  const [text, setText] = useState("");
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState<AdvancedAnalysisResult | null>(null);
  const [user, setUser] = useState<any>(null);
  const [activeTab, setActiveTab] = useState("algorithms");

  useEffect(() => {
    supabase.auth.getSession().then(({ data: { session } }) => {
      setUser(session?.user ?? null);
    });

    const { data: { subscription } } = supabase.auth.onAuthStateChange((_event, session) => {
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
      const { data, error } = await supabase.functions.invoke('analyze-news-advanced', {
        body: { text }
      });

      if (error) throw error;

      if (data.error) {
        if (data.error.includes('Rate limit')) {
          toast.error("Rate limit exceeded. Please try again later.");
        } else {
          toast.error(data.error);
        }
        return;
      }

      setResult(data);
      setActiveTab("synthesis");

      // Save to history if user is logged in
      if (user) {
        await supabase.from("analysis_history").insert({
          user_id: user.id,
          text: text,
          prediction: data.synthesis.overall_score,
          confidence: data.synthesis.confidence,
          is_authentic: data.synthesis.overall_score >= 0.5
        });
      }

      toast.success("Advanced analysis complete!");
    } catch (error: any) {
      console.error("Analysis error:", error);
      toast.error("Failed to analyze text. Please try again.");
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <div className="space-y-6">
      <Card className="p-6 bg-gradient-card border-border/50 shadow-lg">
        <div className="flex items-center gap-3 mb-4">
          <ScanSearch className="w-6 h-6 text-primary" />
          <div>
            <h2 className="text-xl font-semibold text-card-foreground">Advanced Multi-Algorithm Analysis</h2>
            <p className="text-sm text-muted-foreground">5 specialized AI algorithms analyze your text from different perspectives</p>
          </div>
        </div>

        <Textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Paste a news article, headline, or any text you want to deeply analyze for authenticity..."
          className="min-h-[180px] mb-4 text-base resize-none"
          disabled={isAnalyzing}
        />

        <div className="flex flex-wrap gap-2 mb-4">
          <span className="text-sm text-muted-foreground mr-2">Try an example:</span>
          {exampleTexts.map((example, idx) => (
            <Button
              key={idx}
              variant="outline"
              size="sm"
              onClick={() => { setText(example); setResult(null); }}
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
              💡 Sign in to save your analysis history.
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
              Running 5 AI Algorithms...
            </>
          ) : (
            <>
              <Sparkles className="w-5 h-5 mr-2" />
              Run Advanced Analysis
            </>
          )}
        </Button>
      </Card>

      {result && (
        <Tabs value={activeTab} onValueChange={setActiveTab} className="animate-fade-in">
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="synthesis" className="flex items-center gap-2">
              <Shield className="w-4 h-4" />
              Synthesis
            </TabsTrigger>
            <TabsTrigger value="algorithms" className="flex items-center gap-2">
              <Scale className="w-4 h-4" />
              Algorithms
            </TabsTrigger>
            <TabsTrigger value="investigate" className="flex items-center gap-2">
              <MessageCircle className="w-4 h-4" />
              Investigate
            </TabsTrigger>
          </TabsList>

          <TabsContent value="synthesis" className="mt-4">
            <SynthesisCard synthesis={result.synthesis} />
          </TabsContent>

          <TabsContent value="algorithms" className="mt-4 space-y-4">
            <AlgorithmCard
              title="Factual Analysis"
              icon={<FileSearch className="w-5 h-5 text-blue-100" />}
              result={result.analyses.factual}
              color="bg-blue-500/20"
            />
            <AlgorithmCard
              title="Linguistic Analysis"
              icon={<Eye className="w-5 h-5 text-purple-100" />}
              result={result.analyses.linguistic}
              color="bg-purple-500/20"
            />
            <AlgorithmCard
              title="Sentiment & Bias"
              icon={<Scale className="w-5 h-5 text-amber-100" />}
              result={result.analyses.sentiment}
              color="bg-amber-500/20"
            />
            <AlgorithmCard
              title="Source Credibility"
              icon={<Shield className="w-5 h-5 text-emerald-100" />}
              result={result.analyses.source}
              color="bg-emerald-500/20"
            />
            <AlgorithmCard
              title="Propaganda Detection"
              icon={<MessageCircle className="w-5 h-5 text-rose-100" />}
              result={result.analyses.propaganda}
              color="bg-rose-500/20"
            />
          </TabsContent>

          <TabsContent value="investigate" className="mt-4">
            <InvestigationChat originalText={text} analysisResult={result} />
          </TabsContent>
        </Tabs>
      )}
    </div>
  );
};
