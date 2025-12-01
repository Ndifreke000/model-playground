import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { 
  RefreshCw, 
  CheckCircle2, 
  AlertTriangle, 
  Loader2,
  Database,
  Cpu,
  BarChart3,
  Info
} from "lucide-react";
import { toast } from "sonner";
import { misinformationAPI } from "@/lib/api/misinformation.ts";

interface ModelMetrics {
  accuracy: number;
  precision: number;
  recall: number;
  model_loaded: boolean;
  device: string;
}

export const ModelRetraining = () => {
  const [isChecking, setIsChecking] = useState(false);
  const [metrics, setMetrics] = useState<ModelMetrics | null>(null);
  const [error, setError] = useState<string | null>(null);

  const checkModelHealth = async () => {
    setIsChecking(true);
    setError(null);
    
    try {
      const health = await misinformationAPI.healthCheck();
      setMetrics({
        accuracy: 0.695,
        precision: 0.992,
        recall: 0.666,
        model_loaded: health.model_loaded,
        device: health.device
      });
      toast.success("Model health check complete!");
    } catch (err: any) {
      setError(err.message || "Failed to connect to ML service");
      toast.error("Failed to check model health");
    } finally {
      setIsChecking(false);
    }
  };

  return (
    <div className="space-y-6">
      <Card className="border-border/50">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Cpu className="w-5 h-5 text-primary" />
            PyTorch Model Status
          </CardTitle>
          <CardDescription>
            Monitor and manage the deployed machine learning model on Hugging Face Spaces
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Badge variant="outline" className="gap-1">
                <div className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
                Deployed
              </Badge>
              <span className="text-sm text-muted-foreground">
                Hugging Face Spaces
              </span>
            </div>
            <Button 
              variant="outline" 
              size="sm"
              onClick={checkModelHealth}
              disabled={isChecking}
            >
              {isChecking ? (
                <>
                  <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                  Checking...
                </>
              ) : (
                <>
                  <RefreshCw className="w-4 h-4 mr-2" />
                  Check Health
                </>
              )}
            </Button>
          </div>

          {error && (
            <Alert variant="destructive">
              <AlertTriangle className="w-4 h-4" />
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}

          {metrics && (
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 pt-4">
              <MetricCard 
                label="Accuracy" 
                value={`${(metrics.accuracy * 100).toFixed(1)}%`}
                icon={<BarChart3 className="w-4 h-4" />}
              />
              <MetricCard 
                label="Precision" 
                value={`${(metrics.precision * 100).toFixed(1)}%`}
                icon={<CheckCircle2 className="w-4 h-4" />}
              />
              <MetricCard 
                label="Recall" 
                value={`${(metrics.recall * 100).toFixed(1)}%`}
                icon={<Database className="w-4 h-4" />}
              />
              <MetricCard 
                label="Device" 
                value={metrics.device}
                icon={<Cpu className="w-4 h-4" />}
              />
            </div>
          )}
        </CardContent>
      </Card>

      <Card className="border-border/50">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <RefreshCw className="w-5 h-5 text-primary" />
            Model Retraining
          </CardTitle>
          <CardDescription>
            Retrain the model with updated data to improve accuracy
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <Alert>
            <Info className="w-4 h-4" />
            <AlertDescription>
              Model retraining requires access to the training infrastructure. The model is 
              currently deployed on Hugging Face Spaces and can be retrained by pushing 
              updated model weights to the repository.
            </AlertDescription>
          </Alert>

          <div className="space-y-3">
            <h4 className="font-medium text-sm">Retraining Steps:</h4>
            <ol className="list-decimal list-inside space-y-2 text-sm text-muted-foreground">
              <li>Clone the Hugging Face repository</li>
              <li>Add new training data to the <code className="px-1 py-0.5 bg-muted rounded">data/</code> folder</li>
              <li>Run <code className="px-1 py-0.5 bg-muted rounded">python train.py</code> locally</li>
              <li>Push updated model weights to Hugging Face</li>
              <li>The API will automatically use the new model</li>
            </ol>
          </div>

          <div className="flex gap-2">
            <Button 
              variant="outline" 
              className="flex-1"
              onClick={() => window.open('https://huggingface.co/spaces/yosemite000/misinformation-detector', '_blank')}
            >
              View on Hugging Face
            </Button>
            <Button 
              variant="outline"
              className="flex-1"
              onClick={() => window.open('https://huggingface.co/spaces/yosemite000/misinformation-detector/tree/main', '_blank')}
            >
              View Repository
            </Button>
          </div>
        </CardContent>
      </Card>

      <Card className="border-border/50">
        <CardHeader>
          <CardTitle className="text-lg">Model Architecture</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid gap-4 md:grid-cols-2">
            <div className="p-4 rounded-lg bg-muted/50">
              <h4 className="font-medium mb-2">Text Preprocessing</h4>
              <ul className="text-sm text-muted-foreground space-y-1">
                <li>• TF-IDF Vectorization (1000 features)</li>
                <li>• Lowercase normalization</li>
                <li>• Punctuation removal</li>
                <li>• Stop word filtering</li>
              </ul>
            </div>
            <div className="p-4 rounded-lg bg-muted/50">
              <h4 className="font-medium mb-2">Neural Network</h4>
              <ul className="text-sm text-muted-foreground space-y-1">
                <li>• Input Layer: 1000 neurons</li>
                <li>• Hidden Layer: 64 neurons (ReLU)</li>
                <li>• Output Layer: 1 neuron (Sigmoid)</li>
                <li>• Optimizer: Adam (lr=0.001)</li>
              </ul>
            </div>
          </div>
          
          <div className="p-4 rounded-lg bg-muted/50">
            <h4 className="font-medium mb-2">Training Configuration</h4>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
              <div>
                <p className="text-muted-foreground">Epochs</p>
                <p className="font-mono">10</p>
              </div>
              <div>
                <p className="text-muted-foreground">Batch Size</p>
                <p className="font-mono">64</p>
              </div>
              <div>
                <p className="text-muted-foreground">Learning Rate</p>
                <p className="font-mono">0.001</p>
              </div>
              <div>
                <p className="text-muted-foreground">Dataset Size</p>
                <p className="font-mono">79K articles</p>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

const MetricCard = ({ 
  label, 
  value, 
  icon 
}: { 
  label: string; 
  value: string; 
  icon: React.ReactNode;
}) => (
  <div className="p-3 rounded-lg bg-muted/50">
    <div className="flex items-center gap-2 text-muted-foreground text-xs mb-1">
      {icon}
      {label}
    </div>
    <p className="font-mono text-lg font-semibold">{value}</p>
  </div>
);
