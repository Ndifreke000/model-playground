import { Shield, AlertTriangle, TrendingUp } from "lucide-react";
import { Card } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";

interface AnalysisResultProps {
  prediction: number;
  confidence: number;
}

export const AnalysisResult = ({ prediction, confidence }: AnalysisResultProps) => {
  const isReal = prediction >= 0.5;
  const displayConfidence = Math.round(confidence * 100);

  return (
    <div className="animate-fade-in space-y-6">
      <Card 
        className={`p-8 border-2 ${
          isReal 
            ? 'bg-gradient-success border-success shadow-glow-success' 
            : 'bg-gradient-danger border-destructive shadow-glow-danger'
        }`}
      >
        <div className="flex items-center gap-4 mb-4">
          {isReal ? (
            <Shield className="w-12 h-12 text-success-foreground" />
          ) : (
            <AlertTriangle className="w-12 h-12 text-destructive-foreground" />
          )}
          <div>
            <h3 className={`text-2xl font-bold ${
              isReal ? 'text-success-foreground' : 'text-destructive-foreground'
            }`}>
              {isReal ? 'Likely Authentic' : 'Likely Misinformation'}
            </h3>
            <p className={`text-sm ${
              isReal ? 'text-success-foreground/90' : 'text-destructive-foreground/90'
            }`}>
              {isReal 
                ? 'This content appears to be credible news' 
                : 'This content shows signs of misinformation'}
            </p>
          </div>
        </div>
      </Card>

      <Card className="p-6 bg-card">
        <div className="flex items-center gap-3 mb-3">
          <TrendingUp className="w-5 h-5 text-primary" />
          <h4 className="font-semibold text-card-foreground">Confidence Score</h4>
        </div>
        <div className="space-y-2">
          <div className="flex justify-between text-sm">
            <span className="text-muted-foreground">Model Confidence</span>
            <span className="font-bold text-card-foreground">{displayConfidence}%</span>
          </div>
          <Progress 
            value={displayConfidence} 
            className="h-3"
          />
          <p className="text-xs text-muted-foreground mt-2">
            Based on text analysis using TF-IDF features and neural network classification
          </p>
        </div>
      </Card>

      <Card className="p-6 bg-accent/30 border-accent">
        <h4 className="font-semibold text-accent-foreground mb-2">Important Notice</h4>
        <p className="text-sm text-accent-foreground/80">
          This is an AI-powered prediction tool. Always verify information through multiple 
          trusted sources before drawing conclusions. No automated system is 100% accurate.
        </p>
      </Card>
    </div>
  );
};
