import { Card } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { ChevronDown, ChevronUp } from "lucide-react";
import { useState } from "react";
import { AlgorithmResult } from "@/types/analysis";

interface AlgorithmCardProps {
  title: string;
  icon: React.ReactNode;
  result: AlgorithmResult;
  color: string;
}

export const AlgorithmCard = ({ title, icon, result, color }: AlgorithmCardProps) => {
  const [expanded, setExpanded] = useState(false);
  const scorePercent = Math.round(result.score * 100);

  const getVerdictColor = (score: number) => {
    if (score >= 0.7) return "bg-success text-success-foreground";
    if (score >= 0.4) return "bg-warning text-warning-foreground";
    return "bg-destructive text-destructive-foreground";
  };

  const details = [
    ...(result.findings || []),
    ...(result.patterns || []),
    ...(result.emotional_triggers || []),
    ...(result.sources_identified || []),
    ...(result.credibility_issues || []),
    ...(result.techniques_found || []),
  ];

  return (
    <Card className="p-4 bg-card border-border/50 hover:border-primary/30 transition-colors">
      <div 
        className="flex items-center justify-between cursor-pointer"
        onClick={() => setExpanded(!expanded)}
      >
        <div className="flex items-center gap-3">
          <div className={`p-2 rounded-lg ${color}`}>
            {icon}
          </div>
          <div>
            <h4 className="font-semibold text-card-foreground">{title}</h4>
            <Badge className={getVerdictColor(result.score)}>
              {result.verdict}
            </Badge>
          </div>
        </div>
        <div className="flex items-center gap-3">
          <div className="text-right">
            <span className="text-2xl font-bold text-card-foreground">{scorePercent}%</span>
          </div>
          {expanded ? (
            <ChevronUp className="w-5 h-5 text-muted-foreground" />
          ) : (
            <ChevronDown className="w-5 h-5 text-muted-foreground" />
          )}
        </div>
      </div>

      <div className="mt-3">
        <Progress value={scorePercent} className="h-2" />
      </div>

      {expanded && (
        <div className="mt-4 pt-4 border-t border-border/50 space-y-3 animate-fade-in">
          <div>
            <h5 className="text-sm font-medium text-muted-foreground mb-1">Reasoning</h5>
            <p className="text-sm text-card-foreground">{result.reasoning}</p>
          </div>
          
          {details.length > 0 && (
            <div>
              <h5 className="text-sm font-medium text-muted-foreground mb-1">Details</h5>
              <ul className="space-y-1">
                {details.map((item, idx) => (
                  <li key={idx} className="text-sm text-card-foreground flex items-start gap-2">
                    <span className="text-primary mt-1">•</span>
                    <span>{item}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {result.bias_direction && (
            <div>
              <h5 className="text-sm font-medium text-muted-foreground mb-1">Bias Direction</h5>
              <Badge variant="outline">{result.bias_direction}</Badge>
            </div>
          )}
        </div>
      )}
    </Card>
  );
};
