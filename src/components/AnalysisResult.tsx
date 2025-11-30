import { Shield, AlertTriangle, TrendingUp, CheckCircle, XCircle } from "lucide-react";
import { Card } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";

interface AnalysisResultProps {
  prediction: number;
  confidence: number;
  reasoning?: string;
  redFlags?: string[];
  authenticityIndicators?: string[];
}

export const AnalysisResult = ({ 
  prediction, 
  confidence, 
  reasoning,
  redFlags,
  authenticityIndicators 
}: AnalysisResultProps) => {
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
            <span className="text-muted-foreground">AI Confidence</span>
            <span className="font-bold text-card-foreground">{displayConfidence}%</span>
          </div>
          <Progress 
            value={displayConfidence} 
            className="h-3"
          />
        </div>
      </Card>

      {reasoning && (
        <Card className="p-6 bg-card">
          <h4 className="font-semibold text-card-foreground mb-3">AI Analysis</h4>
          <p className="text-sm text-muted-foreground">{reasoning}</p>
        </Card>
      )}

      {redFlags && redFlags.length > 0 && (
        <Card className="p-6 bg-destructive/10 border-destructive/30">
          <div className="flex items-center gap-2 mb-3">
            <XCircle className="w-5 h-5 text-destructive" />
            <h4 className="font-semibold text-destructive">Red Flags</h4>
          </div>
          <ul className="space-y-2">
            {redFlags.map((flag, idx) => (
              <li key={idx} className="text-sm text-destructive/90 flex items-start gap-2">
                <span className="text-destructive mt-1">•</span>
                <span>{flag}</span>
              </li>
            ))}
          </ul>
        </Card>
      )}

      {authenticityIndicators && authenticityIndicators.length > 0 && (
        <Card className="p-6 bg-success/10 border-success/30">
          <div className="flex items-center gap-2 mb-3">
            <CheckCircle className="w-5 h-5 text-success" />
            <h4 className="font-semibold text-success">Authenticity Indicators</h4>
          </div>
          <ul className="space-y-2">
            {authenticityIndicators.map((indicator, idx) => (
              <li key={idx} className="text-sm text-success/90 flex items-start gap-2">
                <span className="text-success mt-1">•</span>
                <span>{indicator}</span>
              </li>
            ))}
          </ul>
        </Card>
      )}

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
