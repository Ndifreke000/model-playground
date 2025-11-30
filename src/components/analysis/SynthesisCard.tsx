import { Card } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { Shield, AlertTriangle, CheckCircle, XCircle, Brain } from "lucide-react";
import { AnalysisSynthesis } from "@/types/analysis";

interface SynthesisCardProps {
  synthesis: AnalysisSynthesis;
}

export const SynthesisCard = ({ synthesis }: SynthesisCardProps) => {
  const isAuthentic = synthesis.overall_score >= 0.5;
  const scorePercent = Math.round(synthesis.overall_score * 100);
  const confidencePercent = Math.round(synthesis.confidence * 100);

  const getVerdictStyle = () => {
    if (synthesis.overall_score >= 0.7) return {
      bg: "bg-gradient-success border-success",
      icon: <Shield className="w-10 h-10 text-success-foreground" />,
      textColor: "text-success-foreground"
    };
    if (synthesis.overall_score >= 0.5) return {
      bg: "bg-gradient-to-br from-success/20 to-warning/20 border-warning",
      icon: <Shield className="w-10 h-10 text-warning-foreground" />,
      textColor: "text-warning-foreground"
    };
    if (synthesis.overall_score >= 0.3) return {
      bg: "bg-gradient-to-br from-warning/20 to-destructive/20 border-warning",
      icon: <AlertTriangle className="w-10 h-10 text-warning-foreground" />,
      textColor: "text-warning-foreground"
    };
    return {
      bg: "bg-gradient-danger border-destructive",
      icon: <AlertTriangle className="w-10 h-10 text-destructive-foreground" />,
      textColor: "text-destructive-foreground"
    };
  };

  const style = getVerdictStyle();

  return (
    <div className="space-y-4">
      <Card className={`p-6 border-2 ${style.bg}`}>
        <div className="flex items-center gap-4 mb-4">
          {style.icon}
          <div>
            <h3 className={`text-2xl font-bold ${style.textColor}`}>
              {synthesis.overall_verdict}
            </h3>
            <p className={`text-sm ${style.textColor}/80`}>
              {synthesis.executive_summary}
            </p>
          </div>
        </div>

        <div className="grid grid-cols-2 gap-4">
          <div>
            <span className="text-sm text-muted-foreground">Overall Score</span>
            <div className="flex items-center gap-2">
              <Progress value={scorePercent} className="h-2 flex-1" />
              <span className="font-bold">{scorePercent}%</span>
            </div>
          </div>
          <div>
            <span className="text-sm text-muted-foreground">Confidence</span>
            <div className="flex items-center gap-2">
              <Progress value={confidencePercent} className="h-2 flex-1" />
              <span className="font-bold">{confidencePercent}%</span>
            </div>
          </div>
        </div>
      </Card>

      <div className="grid md:grid-cols-2 gap-4">
        {synthesis.key_concerns.length > 0 && (
          <Card className="p-4 bg-destructive/10 border-destructive/30">
            <div className="flex items-center gap-2 mb-3">
              <XCircle className="w-5 h-5 text-destructive" />
              <h4 className="font-semibold text-destructive">Key Concerns</h4>
            </div>
            <ul className="space-y-2">
              {synthesis.key_concerns.map((concern, idx) => (
                <li key={idx} className="text-sm text-destructive/90 flex items-start gap-2">
                  <span className="text-destructive mt-1">•</span>
                  <span>{concern}</span>
                </li>
              ))}
            </ul>
          </Card>
        )}

        {synthesis.strengths.length > 0 && (
          <Card className="p-4 bg-success/10 border-success/30">
            <div className="flex items-center gap-2 mb-3">
              <CheckCircle className="w-5 h-5 text-success" />
              <h4 className="font-semibold text-success">Strengths</h4>
            </div>
            <ul className="space-y-2">
              {synthesis.strengths.map((strength, idx) => (
                <li key={idx} className="text-sm text-success/90 flex items-start gap-2">
                  <span className="text-success mt-1">•</span>
                  <span>{strength}</span>
                </li>
              ))}
            </ul>
          </Card>
        )}
      </div>

      <Card className="p-4 bg-primary/10 border-primary/30">
        <div className="flex items-center gap-2 mb-3">
          <Brain className="w-5 h-5 text-primary" />
          <h4 className="font-semibold text-primary">Recommendation</h4>
        </div>
        <p className="text-sm text-card-foreground">{synthesis.recommendation}</p>
      </Card>

      <Card className="p-4 bg-card">
        <h4 className="font-semibold text-card-foreground mb-2">Detailed Analysis</h4>
        <p className="text-sm text-muted-foreground whitespace-pre-wrap">
          {synthesis.detailed_reasoning}
        </p>
      </Card>
    </div>
  );
};
