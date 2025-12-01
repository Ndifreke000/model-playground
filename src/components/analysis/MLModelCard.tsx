import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Brain, CheckCircle2, AlertTriangle, Zap, Clock } from "lucide-react";
import { MLModelResult } from "@/types/analysis";

interface MLModelCardProps {
  result: MLModelResult;
}

export const MLModelCard = ({ result }: MLModelCardProps) => {
  const isFake = result.prediction === 'fake';
  const confidencePercent = Math.round(result.confidence * 100);

  return (
    <Card className="border-border/50 bg-gradient-to-br from-indigo-500/10 to-purple-500/10">
      <CardHeader className="pb-2">
        <CardTitle className="flex items-center gap-2 text-lg">
          <div className="p-2 rounded-lg bg-indigo-500/20">
            <Brain className="w-5 h-5 text-indigo-300" />
          </div>
          PyTorch ML Model Prediction
          <Badge variant="outline" className="ml-auto text-xs">
            Neural Network + TF-IDF
          </Badge>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            {isFake ? (
              <AlertTriangle className="w-8 h-8 text-destructive" />
            ) : (
              <CheckCircle2 className="w-8 h-8 text-emerald-500" />
            )}
            <div>
              <p className="text-sm text-muted-foreground">Prediction</p>
              <p className="text-2xl font-bold">
                {isFake ? 'Likely Misinformation' : 'Likely Authentic'}
              </p>
            </div>
          </div>
          <Badge 
            variant={isFake ? "destructive" : "default"}
            className="text-lg px-4 py-2"
          >
            {result.prediction.toUpperCase()}
          </Badge>
        </div>

        <div className="space-y-2">
          <div className="flex justify-between text-sm">
            <span className="text-muted-foreground">Model Confidence</span>
            <span className="font-mono font-semibold">{confidencePercent}%</span>
          </div>
          <Progress 
            value={confidencePercent} 
            className={`h-3 ${isFake ? '[&>div]:bg-destructive' : '[&>div]:bg-emerald-500'}`}
          />
        </div>

        <div className="grid grid-cols-2 gap-4 pt-2">
          <div className="p-3 rounded-lg bg-background/50">
            <div className="flex items-center gap-2 text-muted-foreground text-xs mb-1">
              <Zap className="w-3 h-3" />
              Raw Score
            </div>
            <p className="font-mono text-lg">{result.raw_score.toFixed(4)}</p>
          </div>
          {result.processing_time_ms && (
            <div className="p-3 rounded-lg bg-background/50">
              <div className="flex items-center gap-2 text-muted-foreground text-xs mb-1">
                <Clock className="w-3 h-3" />
                Processing Time
              </div>
              <p className="font-mono text-lg">{result.processing_time_ms}ms</p>
            </div>
          )}
        </div>

        <div className="text-xs text-muted-foreground pt-2 border-t border-border/50">
          <p>
            <strong>Model Architecture:</strong> 2-layer Neural Network (1000 → 64 → 1) with TF-IDF vectorization
          </p>
          <p className="mt-1">
            <strong>Training Data:</strong> 79K news articles from misinformation dataset
          </p>
        </div>
      </CardContent>
    </Card>
  );
};
